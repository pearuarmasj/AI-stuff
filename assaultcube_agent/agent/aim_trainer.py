"""
Aim Trainer Agent

Simple agent that:
1. Reads player state from memory
2. Detects enemies via memory reading
3. Snaps crosshair to nearest enemy in FOV
4. (Optional) Shoots

No movement, no game sense, just aiming practice.

CONTROLS:
    F10 = STOP
    F9  = PAUSE/RESUME
"""

import time

from ..memory import ACMemoryReader
from ..raycast import EnemyDetector
from ..control import MouseController, emergency_stop, check_stop, wait_if_paused


class AimTrainer:
    """
    Basic aim training agent using memory-based enemy detection.

    Continuously reads game state, detects enemies, and snaps
    the crosshair toward the closest visible enemy.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        aim_speed: float = 0.3,
        shoot: bool = False,
        fov_h: float = 90.0,
        fov_v: float = 60.0,
        max_distance: float = 300.0,
    ):
        """
        Args:
            sensitivity: Mouse sensitivity multiplier
            aim_speed: 0-1, how much of the angle offset to correct per frame
                       1.0 = instant snap, 0.1 = slow tracking
            shoot: Whether to auto-fire when on target
            fov_h: Horizontal field of view in degrees
            fov_v: Vertical field of view in degrees
            max_distance: Max distance to track enemies
        """
        self.memory = ACMemoryReader()
        self.detector = EnemyDetector(fov_h=fov_h, fov_v=fov_v)
        self.mouse = MouseController(sensitivity=sensitivity)

        self.aim_speed = aim_speed
        self.shoot = shoot
        self.max_distance = max_distance
        self.running = False

        # Stats
        self.frames_processed = 0
        self.targets_found = 0
        self.shots_fired = 0

        # Attach to game
        if not self.memory.attach():
            raise RuntimeError("Failed to attach to AssaultCube! Make sure it's running.")
        if not self.detector.attach():
            raise RuntimeError("Failed to attach enemy detector!")

        print("[+] Aim trainer initialized")

    def step(self) -> bool:
        """
        Process one frame.

        Returns:
            True if a target was found and aimed at
        """
        # Check emergency stop
        if check_stop():
            self.running = False
            return False

        # Wait if paused
        if not wait_if_paused():
            self.running = False
            return False

        # Read player state
        state = self.memory.read_state()
        own_pos = (state.pos_x, state.pos_y, state.pos_z)

        self.frames_processed += 1
        do_debug = self.frames_processed % 60 == 0

        # Debug output every 60 frames
        if do_debug:
            array_ptr, player_count = self.detector._read_players_array()
            own_team = self.detector._get_own_team()
            print(f"[DEBUG] Frame {self.frames_processed}: players={player_count}, own_team={own_team}, pos=({state.pos_x:.0f},{state.pos_y:.0f},{state.pos_z:.0f})")

        # Detect enemies
        enemies = self.detector.detect_enemies(
            own_pos,
            state.yaw,
            state.pitch,
            max_distance=self.max_distance,
            filter_team=True,
            debug=do_debug,
        )

        if do_debug:
            print(f"  => enemies_found={len(enemies)}")

        if not enemies:
            return False

        # Find closest enemy in FOV (or closest overall if none in FOV)
        in_fov = [e for e in enemies if e.in_fov]
        if in_fov:
            target = min(in_fov, key=lambda e: e.distance)
        else:
            target = min(enemies, key=lambda e: abs(e.angle_h))

        self.targets_found += 1

        # Calculate mouse movement needed
        # angle_h is degrees offset from center, positive = right
        # angle_v is degrees offset from center, positive = up
        dx = target.angle_h * self.aim_speed
        dy = -target.angle_v * self.aim_speed  # Invert for mouse (down = positive)

        # Convert degrees to pixels (rough approximation)
        # This depends on game sensitivity - may need tuning
        pixels_per_degree = 15.0  # Adjust based on your in-game sensitivity
        dx_pixels = int(dx * pixels_per_degree)
        dy_pixels = int(dy * pixels_per_degree)

        # Only move if there's meaningful offset
        if abs(dx_pixels) > 1 or abs(dy_pixels) > 1:
            self.mouse.move(dx_pixels, dy_pixels)
        elif self.shoot and target.in_fov and abs(target.angle_h) < 5:
            # On target (within 5 degrees), shoot
            self.mouse.click()
            self.shots_fired += 1

        return True

    def run(self, fps_limit: int = 60):
        """
        Main loop. Run until stopped.

        Args:
            fps_limit: Max frames per second
        """
        self.running = True
        frame_time = 1.0 / fps_limit

        # Start emergency stop listener
        emergency_stop.start()

        print("=" * 50)
        print("AIM TRAINER STARTED")
        print(f"  aim_speed={self.aim_speed}, shoot={self.shoot}")
        print("  F10 = STOP | F9 = PAUSE/RESUME")
        print("=" * 50)

        try:
            while self.running:
                start = time.perf_counter()

                self.step()

                # Check if we should stop
                if check_stop():
                    break

                # Frame limiting
                elapsed = time.perf_counter() - start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.running = False
            emergency_stop.stop_listener()
            self._print_stats()

    def stop(self):
        """Stop the main loop."""
        self.running = False

    def _print_stats(self):
        """Print session statistics."""
        print(f"\nSession stats:")
        print(f"  Frames processed: {self.frames_processed}")
        print(f"  Targets found: {self.targets_found}")
        print(f"  Shots fired: {self.shots_fired}")
        if self.frames_processed > 0:
            hit_rate = self.targets_found / self.frames_processed * 100
            print(f"  Target acquisition rate: {hit_rate:.1f}%")


def main():
    """Run aim trainer from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="AssaultCube Aim Trainer")
    parser.add_argument(
        "--sensitivity", "-s",
        type=float,
        default=1.0,
        help="Mouse sensitivity multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--aim-speed", "-a",
        type=float,
        default=0.3,
        help="Aim correction speed 0-1 (default: 0.3, 1.0=instant snap)",
    )
    parser.add_argument(
        "--shoot",
        action="store_true",
        help="Auto-fire when on target",
    )
    parser.add_argument(
        "--distance", "-d",
        type=float,
        default=300.0,
        help="Max tracking distance (default: 300)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS limit (default: 60)",
    )

    args = parser.parse_args()

    trainer = AimTrainer(
        sensitivity=args.sensitivity,
        aim_speed=args.aim_speed,
        shoot=args.shoot,
        max_distance=args.distance,
    )
    trainer.run(fps_limit=args.fps)


if __name__ == "__main__":
    main()
