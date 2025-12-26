"""
Phase 1: Aim Trainer Agent

Simple agent that:
1. Captures screen
2. Detects enemy
3. Moves crosshair to enemy
4. (Optional) Shoots

No movement, no game sense, just aiming.
"""

import time
from ..vision import ScreenCapture, EnemyDetector
from ..control import MouseController


class AimTrainer:
    """
    Basic aim training agent.

    Continuously captures frames, detects enemies, and moves
    the crosshair toward them.
    """

    def __init__(
        self,
        target_color: str = "red",
        sensitivity: float = 1.0,
        aim_speed: float = 0.5,
        shoot: bool = False,
        capture_region: dict | None = None,
    ):
        """
        Args:
            target_color: "red" or "blue" enemy team
            sensitivity: Mouse sensitivity multiplier
            aim_speed: 0-1, how much of the offset to correct per frame
                       1.0 = instant snap, 0.1 = slow tracking
            shoot: Whether to auto-fire when on target
            capture_region: Optional screen region to capture
        """
        self.capture = ScreenCapture(region=capture_region)
        self.detector = EnemyDetector(target_color=target_color)
        self.mouse = MouseController(sensitivity=sensitivity)

        self.aim_speed = aim_speed
        self.shoot = shoot
        self.running = False

        # Stats
        self.frames_processed = 0
        self.targets_found = 0

    def step(self) -> bool:
        """
        Process one frame.

        Returns:
            True if a target was found and aimed at
        """
        frame = self.capture.capture()
        offset = self.detector.get_offset_from_center(frame)

        self.frames_processed += 1

        if offset is None:
            return False

        self.targets_found += 1

        dx, dy = offset

        # Apply aim speed (don't move full offset at once for smoother tracking)
        dx = int(dx * self.aim_speed)
        dy = int(dy * self.aim_speed)

        # Only move if there's meaningful offset
        if abs(dx) > 2 or abs(dy) > 2:
            self.mouse.move(dx, dy)
        elif self.shoot:
            # On target, shoot
            self.mouse.click()

        return True

    def run(self, fps_limit: int = 60):
        """
        Main loop. Run until stopped.

        Args:
            fps_limit: Max frames per second (to not murder CPU)
        """
        self.running = True
        frame_time = 1.0 / fps_limit

        print(f"Aim trainer started. Press Ctrl+C to stop.")
        print(f"Settings: aim_speed={self.aim_speed}, shoot={self.shoot}")

        try:
            while self.running:
                start = time.perf_counter()

                self.step()

                # Frame limiting
                elapsed = time.perf_counter() - start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.running = False
            self._print_stats()

    def stop(self):
        """Stop the main loop."""
        self.running = False

    def _print_stats(self):
        """Print session statistics."""
        print(f"\nSession stats:")
        print(f"  Frames processed: {self.frames_processed}")
        print(f"  Targets found: {self.targets_found}")
        if self.frames_processed > 0:
            hit_rate = self.targets_found / self.frames_processed * 100
            print(f"  Target acquisition rate: {hit_rate:.1f}%")
