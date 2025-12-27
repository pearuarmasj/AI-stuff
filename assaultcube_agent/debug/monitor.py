"""
Real-time monitoring and debugging for training.

Run alongside training to visualize:
- Raycast data (radar view)
- Player state
- Enemy positions
- Training stats from TensorBoard logs

Usage:
    python -m assaultcube_agent.debug.monitor
"""

import time
import os
import sys
from pathlib import Path

import numpy as np


def clear_screen():
    """Clear console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


class TrainingMonitor:
    """Real-time visualization of game state during training."""

    def __init__(self, refresh_rate: float = 10.0):
        self.refresh_rate = refresh_rate
        self.refresh_interval = 1.0 / refresh_rate

        # Components
        self.memory_reader = None
        self.raycast = None

        # Stats
        self.frame_count = 0
        self.start_time = None
        self.last_health = 100
        self.kills = 0
        self.deaths = 0

    def attach(self) -> bool:
        """Attach to game process."""
        try:
            from ..memory import ACMemoryReader
            from ..raycast.omni_raycast import OmniRaycastObserver

            print("[Monitor] Attaching to game...")

            self.memory_reader = ACMemoryReader()
            if not self.memory_reader.attach():
                return False

            self.raycast = OmniRaycastObserver(
                horizontal_rays=36,  # Fewer rays for faster viz
                vertical_layers=5,
            )
            if not self.raycast.attach():
                return False

            self.start_time = time.time()
            print("[Monitor] Ready!")
            return True

        except Exception as e:
            print(f"[Monitor] Error: {e}")
            return False

    def detach(self):
        """Detach from game."""
        if self.memory_reader:
            self.memory_reader.detach()
        if self.raycast:
            self.raycast.detach()

    def run(self):
        """Main monitoring loop."""
        print("\n[Monitor] Starting real-time visualization...")
        print("[Monitor] Press Ctrl+C to stop\n")
        time.sleep(1)

        try:
            while True:
                start = time.perf_counter()

                self._update()
                self._render()

                self.frame_count += 1

                # Rate limiting
                elapsed = time.perf_counter() - start
                sleep_time = self.refresh_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Monitor] Stopped.")

    def _update(self):
        """Update state from game."""
        # Read game state
        gs = self.memory_reader.read_state()

        # Track kills/deaths
        if gs.health <= 0 and self.last_health > 0:
            self.deaths += 1
        if gs.frags > getattr(self, '_last_frags', 0):
            self.kills += gs.frags - getattr(self, '_last_frags', 0)

        self._last_frags = gs.frags
        self.last_health = gs.health
        self._last_state = gs

    def _render(self):
        """Render visualization."""
        clear_screen()

        gs = self._last_state
        uptime = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / uptime if uptime > 0 else 0

        # Header
        print("=" * 60)
        print("  ASSAULTCUBE TRAINING MONITOR")
        print("=" * 60)

        # Player state
        hp_bar = "#" * (gs.health // 5) + "-" * (20 - gs.health // 5)
        ar_bar = "#" * (gs.armor // 5) + "-" * (20 - gs.armor // 5)

        print(f"\n  PLAYER STATE")
        print(f"  Health: [{hp_bar}] {gs.health:3d}")
        print(f"  Armor:  [{ar_bar}] {gs.armor:3d}")
        print(f"  Ammo:   {gs.ammo_mag}")
        print(f"  Pos:    ({gs.pos_x:.0f}, {gs.pos_y:.0f}, {gs.pos_z:.0f})")
        print(f"  Angles: yaw={gs.yaw:.1f}, pitch={gs.pitch:.1f}")

        # Stats
        print(f"\n  SESSION STATS")
        print(f"  Kills: {self.kills}  Deaths: {self.deaths}  K/D: {self.kills/max(self.deaths,1):.2f}")
        print(f"  Uptime: {uptime:.0f}s  Monitor FPS: {fps:.0f}")

        # Raycast visualization (radar)
        obs = self.raycast.get_observation()
        radar = self.raycast.visualize_radar(obs)

        print(f"\n  RADAR (360° view)")
        print("  " + "-" * 23)
        for line in radar.split('\n'):
            print(f"  |{line}|")
        print("  " + "-" * 23)
        print("  @ = You  E = Enemy  . = Wall")

        # Layer visualization
        print(f"\n  HORIZONTAL LAYERS (top to bottom)")
        for i in range(self.raycast.v_layers):
            pitch = self.raycast.v_min + i * (self.raycast.v_max - self.raycast.v_min) / (self.raycast.v_layers - 1)
            layer_vis = self.raycast.visualize_layer(i, obs)
            print(f"  {pitch:+5.0f}°: [{layer_vis}]")

        # Enemy count
        hit_types = obs[:, 1] * 2
        enemy_rays = np.sum(hit_types > 1.5)
        wall_rays = np.sum((hit_types > 0.4) & (hit_types < 1.5))
        print(f"\n  Ray hits: {wall_rays} walls, {enemy_rays} enemies")

        print("\n" + "=" * 60)
        print("  Press Ctrl+C to stop")
        print("=" * 60)


def main():
    monitor = TrainingMonitor(refresh_rate=10)

    if not monitor.attach():
        print("[Monitor] Failed to attach. Is AssaultCube running?")
        return 1

    try:
        monitor.run()
    finally:
        monitor.detach()

    return 0


if __name__ == "__main__":
    sys.exit(main())
