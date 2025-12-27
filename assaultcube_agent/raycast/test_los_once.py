"""
Quick one-shot test of LOS detection.

Run: python -m assaultcube_agent.raycast.test_los_once
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from assaultcube_agent.memory import ACMemoryReader
from assaultcube_agent.raycast import EnemyDetector


def main():
    print("=" * 70)
    print("  LOS Detection Quick Test")
    print("=" * 70)

    memory = ACMemoryReader()
    if not memory.attach():
        print("ERROR: Could not attach to AC")
        return 1

    detector = EnemyDetector(use_los=True)
    if not detector.attach():
        print("ERROR: Could not attach detector")
        return 1

    state = memory.read_state()
    own_pos = (state.pos_x, state.pos_y, state.pos_z)

    print(f"\n[+] Player pos: ({own_pos[0]:.1f}, {own_pos[1]:.1f}, {own_pos[2]:.1f})")
    print(f"[+] Player yaw: {state.yaw:.1f}, pitch: {state.pitch:.1f}")

    # Detect all enemies (no LOS filter)
    print("\n[*] Detecting all enemies (no LOS filter)...")
    enemies_all = detector.detect_enemies(
        own_pos, state.yaw, state.pitch,
        max_distance=300,
        filter_los=False,
        debug=True
    )

    print(f"\n[+] Total enemies: {len(enemies_all)}")

    # Now with LOS filter
    print("\n[*] Detecting with LOS filter...")
    enemies_los = detector.detect_enemies(
        own_pos, state.yaw, state.pitch,
        max_distance=300,
        filter_los=True,
        debug=True
    )

    print(f"\n[+] Visible enemies: {len(enemies_los)}")
    print(f"[+] Blocked by walls: {len(enemies_all) - len(enemies_los)}")

    if enemies_los:
        print("\nVisible enemies:")
        for e in enemies_los:
            fov_str = "IN FOV" if e.in_fov else "      "
            print(f"  #{e.index}: dist={e.distance:.0f}, angle=({e.angle_h:.1f},{e.angle_v:.1f}), {fov_str}")

    detector.detach()
    memory.detach()
    return 0


if __name__ == "__main__":
    sys.exit(main())
