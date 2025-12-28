"""
Aim Trainer Agent - Using EnemyDetector with LiDAR visualization.

Uses memory-based enemy detection (EnemyDetector) for accurate angle calculation,
with optional LiDAR visualization for spatial awareness.

CONTROLS:
    F10 = STOP
    F9  = PAUSE/RESUME
"""

import time
from typing import Optional

from ..memory import ACMemoryReader
from ..raycast import OmniRaycastObserver
from ..raycast.enemy_detector import EnemyDetector, EnemyInfo
from ..control import MouseController, emergency_stop, check_stop, wait_if_paused
from ..visualize import LidarVisualizer


class AimTrainer:
    """
    Aim training agent using memory-based enemy detection.

    Uses EnemyDetector for accurate angle calculation (reads actual enemy positions
    and computes proper view-relative angles). Optional LiDAR visualization shows
    spatial awareness data.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        aim_speed: float = 0.3,
        shoot: bool = False,
        game_fov: float = 120.0,
        max_distance: float = 350.0,   # Match numba_raycast fov_max_dist
        horizontal_rays: int = 41,     # Match numba_raycast fov_rays_h
        vertical_layers: int = 17,     # Match numba_raycast fov_rays_v
        visualize: bool = False,
    ):
        """
        Args:
            sensitivity: Mouse sensitivity multiplier
            aim_speed: 0-1, how much of the angle offset to correct per frame
                       1.0 = instant snap, 0.1 = slow tracking
            shoot: Whether to auto-fire when on target
            game_fov: Game's field of view in degrees (for FOV filtering)
            max_distance: Max detection distance
            horizontal_rays: Horizontal rays for LiDAR visualization
            vertical_layers: Vertical layers for LiDAR visualization
            visualize: Whether to show real-time LiDAR visualization window
        """
        # Memory reader for player state
        self.memory = ACMemoryReader()

        # Enemy detector - uses memory reading with proper angle calculation
        self.detector = EnemyDetector(fov_h=game_fov, fov_v=90.0, use_los=True)

        # Optional LiDAR for visualization only (NOT for enemy detection!)
        self._visualize = visualize
        self._visualizer: Optional[LidarVisualizer] = None
        self._raycast: Optional[OmniRaycastObserver] = None

        if visualize:
            self._raycast = OmniRaycastObserver(
                horizontal_rays=horizontal_rays,
                vertical_layers=vertical_layers,
                max_distance=max_distance,
            )
            self._visualizer = LidarVisualizer(
                h_rays=horizontal_rays,
                v_layers=vertical_layers,
                fov_degrees=game_fov,
                max_dist=max_distance,
            )

        self.mouse = MouseController(sensitivity=sensitivity)

        self.aim_speed = aim_speed
        self.shoot = shoot
        self.game_fov = game_fov
        self.half_fov = game_fov / 2.0
        self.max_distance = max_distance
        self.running = False

        # Stats
        self.frames_processed = 0
        self.targets_found = 0
        self.shots_fired = 0

        # Shooting state
        self._is_shooting = False
        self._last_shot_time = 0
        self._shot_cooldown = 0.1

        # Attach to game
        if not self.memory.attach():
            raise RuntimeError("Failed to attach to AssaultCube! Make sure it's running.")
        if not self.detector.attach():
            raise RuntimeError("Failed to attach enemy detector!")
        if self._raycast and not self._raycast.attach():
            raise RuntimeError("Failed to attach raycast observer!")

        print("[+] Aim trainer initialized")
        print(f"    Detection: Memory-based (EnemyDetector with LOS)")
        print(f"    Game FOV: {game_fov}° (±{self.half_fov}°)")
        print(f"    Visualize: {visualize}")

    def _select_target(self, enemies: list) -> Optional[EnemyInfo]:
        """Select best target from detected enemies."""
        if not enemies:
            return None

        # Only consider enemies in FOV with clear LOS
        valid = [e for e in enemies if e.in_fov and e.has_los]
        if not valid:
            return None

        # Pick closest enemy
        return min(valid, key=lambda e: e.distance)

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

        self.frames_processed += 1
        do_debug = self.frames_processed % 60 == 0

        # Read player state
        state = self.memory.read_state()
        own_pos = state.position
        own_yaw = state.yaw
        own_pitch = state.pitch

        # Detect enemies using EnemyDetector (proper angle calculation!)
        enemies = self.detector.detect_enemies(
            own_pos, own_yaw, own_pitch,
            max_distance=self.max_distance,
            filter_team=True,
            filter_los=True,
        )

        # Update LiDAR visualization if enabled
        if self._visualizer and self._visualizer.is_running() and self._raycast:
            rays = self._raycast.get_observation()
            # Convert enemies to dict format for radar overlay
            enemy_data = [
                {'angle_h': e.angle_h, 'distance': e.distance, 'has_los': e.has_los}
                for e in enemies
            ]
            self._visualizer.update(rays, {
                'health': state.health,
                'frags': state.frags,
                'damage': state.damage_dealt,
                'enemies': enemy_data,  # Pass enemies for radar overlay
            })

        if do_debug:
            enemy_count = len(enemies)
            in_fov_count = len([e for e in enemies if e.in_fov])
            print(f"[Frame {self.frames_processed}] HP:{state.health} Enemies:{enemy_count} (FOV:{in_fov_count})")

        if not enemies:
            # No enemies visible - stop shooting
            if self._is_shooting:
                self.mouse.release('left')
                self._is_shooting = False
            return False

        # Select target
        target = self._select_target(enemies)
        if not target:
            if self._is_shooting:
                self.mouse.release('left')
                self._is_shooting = False
            return False

        self.targets_found += 1

        # Calculate mouse movement needed
        # angle_h: positive = enemy is to the right, need to move mouse right
        # angle_v: positive = enemy is above, need to move mouse up (which is negative dy)
        dx = target.angle_h * self.aim_speed
        dy = -target.angle_v * self.aim_speed  # Invert for mouse

        # Convert degrees to pixels (tune based on in-game sensitivity)
        pixels_per_degree = 15.0
        dx_pixels = int(dx * pixels_per_degree)
        dy_pixels = int(dy * pixels_per_degree)

        # Debug target info occasionally
        if do_debug:
            print(f"  Target: h={target.angle_h:.1f}° v={target.angle_v:.1f}° dist={target.distance:.1f} hp={target.health}")

        # Apply aim correction
        if abs(dx_pixels) > 1 or abs(dy_pixels) > 1:
            self.mouse.move(dx_pixels, dy_pixels)
            # Not on target - stop shooting
            if self._is_shooting:
                self.mouse.release('left')
                self._is_shooting = False
        elif self.shoot and abs(target.angle_h) < 5 and abs(target.angle_v) < 10:
            # On target (within 5° horizontal, 10° vertical), shoot
            current_time = time.perf_counter()
            if current_time - self._last_shot_time >= self._shot_cooldown:
                if not self._is_shooting:
                    self.mouse.press('left')
                    self._is_shooting = True
                self._last_shot_time = current_time
                self.shots_fired += 1
        else:
            if self._is_shooting:
                self.mouse.release('left')
                self._is_shooting = False

        return True

    def run(self, fps_limit: int = 60):
        """
        Main loop. Run until stopped.
        """
        self.running = True
        frame_time = 1.0 / fps_limit

        emergency_stop.start()

        # Start visualization if enabled
        if self._visualizer:
            self._visualizer.start()

        print("=" * 50)
        print("AIM TRAINER STARTED (EnemyDetector + LOS)")
        print(f"  aim_speed={self.aim_speed}, shoot={self.shoot}")
        print(f"  max_distance={self.max_distance}, fov={self.game_fov}°")
        print(f"  visualize={self._visualize}")
        print("  F10 = STOP | F9 = PAUSE/RESUME")
        print("=" * 50)

        try:
            while self.running:
                start = time.perf_counter()

                self.step()

                if check_stop():
                    break

                # Stop if visualization window was closed
                if self._visualizer and not self._visualizer.is_running():
                    print("[*] Visualization closed, stopping...")
                    break

                elapsed = time.perf_counter() - start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.running = False
            if self._is_shooting:
                self.mouse.release('left')
                self._is_shooting = False
            if self._visualizer:
                self._visualizer.stop()
            if self.detector:
                self.detector.detach()
            if self._raycast:
                self._raycast.detach()
            emergency_stop.stop_listener()
            self._print_stats()

    def stop(self):
        """Stop and release inputs."""
        self.running = False
        if self._is_shooting:
            self.mouse.release('left')
            self._is_shooting = False
        if self._visualizer:
            self._visualizer.stop()
        if self.detector:
            self.detector.detach()
        if self._raycast:
            self._raycast.detach()

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

    parser = argparse.ArgumentParser(description="AssaultCube Aim Trainer (Memory-based with LOS)")
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
        "--fov",
        type=float,
        default=120.0,
        help="Game's field of view in degrees (default: 120). Enemy detection is limited to this FOV.",
    )
    parser.add_argument(
        "--distance", "-d",
        type=float,
        default=250.0,
        help="Max enemy detection distance (default: 250)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS limit (default: 60)",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show real-time LiDAR visualization window",
    )

    args = parser.parse_args()

    trainer = AimTrainer(
        sensitivity=args.sensitivity,
        aim_speed=args.aim_speed,
        shoot=args.shoot,
        game_fov=args.fov,
        max_distance=args.distance,
        visualize=args.visualize,
    )
    trainer.run(fps_limit=args.fps)


if __name__ == "__main__":
    main()
