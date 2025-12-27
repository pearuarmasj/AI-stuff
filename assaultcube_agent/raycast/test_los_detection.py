"""
Test enemy detection with LOS checking.

Run: python -m assaultcube_agent.raycast.test_los_detection
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from assaultcube_agent.memory import ACMemoryReader
from assaultcube_agent.raycast import EnemyDetector


def main():
    print("=" * 70)
    print("  Enemy Detection with LOS Test")
    print("=" * 70)

    memory = ACMemoryReader()
    if not memory.attach():
        print("ERROR: Could not attach to AC")
        return 1

    detector = EnemyDetector(use_los=True)
    if not detector.attach():
        print("ERROR: Could not attach detector")
        return 1

    print("\n[+] Running detection loop (Ctrl+C to stop)...")
    print("    Enemies behind walls will be filtered out.\n")

    try:
        while True:
            state = memory.read_state()
            own_pos = (state.pos_x, state.pos_y, state.pos_z)

            # Detect enemies with LOS filtering
            enemies_los = detector.detect_enemies(
                own_pos, state.yaw, state.pitch,
                max_distance=300,
                filter_los=True,
                debug=False
            )

            # Detect all enemies (no LOS filtering) for comparison
            enemies_all = detector.detect_enemies(
                own_pos, state.yaw, state.pitch,
                max_distance=300,
                filter_los=False,
                debug=False
            )

            # Count how many are blocked
            blocked = len(enemies_all) - len(enemies_los)

            if enemies_all:
                print(f"\rEnemies: {len(enemies_los)} visible, {blocked} blocked by walls", end="")

                # Show details for visible enemies
                for e in enemies_los[:3]:  # Show first 3
                    fov_str = "FOV" if e.in_fov else "   "
                    print(f"  | #{e.index}: {e.distance:.0f}m {fov_str}", end="")

            else:
                print("\rNo enemies detected                               ", end="")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped.")

    detector.detach()
    memory.detach()
    return 0


if __name__ == "__main__":
    sys.exit(main())
