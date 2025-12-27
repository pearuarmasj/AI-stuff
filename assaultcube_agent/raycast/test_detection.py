"""
Test script for enemy detection.

Run: python -m assaultcube_agent.raycast.test_detection
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from assaultcube_agent.memory.reader import ACMemoryReader
from assaultcube_agent.raycast.enemy_detector import EnemyDetector


def main():
    print("=" * 70)
    print("  AssaultCube Enemy Detection")
    print("=" * 70)

    reader = ACMemoryReader()
    if not reader.attach():
        print("\nERROR: Could not attach!")
        return 1

    detector = EnemyDetector(fov_h=90.0, fov_v=60.0)
    if not detector.attach():
        reader.detach()
        return 1

    print("\nPress Ctrl+C to stop.\n")

    try:
        while True:
            state = reader.read_state()
            own_pos = (state.pos_x, state.pos_y, state.pos_z)

            # Get own team
            own_team = detector._get_own_team()
            team_name = "CLA" if own_team == 0 else "RVSF" if own_team == 1 else "?"

            # Detect enemies (filters teammates)
            enemies = detector.detect_enemies(
                own_pos, state.yaw, state.pitch,
                max_distance=500.0,
                filter_team=True
            )

            os.system('cls' if os.name == 'nt' else 'clear')

            print("=" * 70)
            print(f"  ENEMY DETECTION  |  Your team: {team_name}")
            print("=" * 70)
            print(f"  HP: {state.health}  Armor: {state.armor}  Yaw: {state.yaw:.1f}")
            print("-" * 70)

            if not enemies:
                print("  No enemies detected")
            else:
                for e in enemies:
                    fov = "[IN FOV]" if e.in_fov else ""
                    print(
                        f"  Enemy: dist={e.distance:6.1f}  "
                        f"angle=({e.angle_h:+6.1f}h, {e.angle_v:+5.1f}v)  "
                        f"hp={e.health:3d}  {fov}"
                    )

                closest = detector.get_closest_enemy_in_fov(
                    own_pos, state.yaw, state.pitch
                )
                if closest:
                    print()
                    print(f"  >> TARGET at {closest.distance:.1f} units")
                    print(f"     Turn: {closest.angle_h:+.1f}h, {closest.angle_v:+.1f}v to aim")

            print("-" * 70)
            print("  Ctrl+C to stop")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")

    reader.detach()
    detector.detach()
    return 0


if __name__ == "__main__":
    sys.exit(main())
