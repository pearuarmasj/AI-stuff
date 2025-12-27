"""
Quick test for memory reading.

Run this with AssaultCube running to verify memory reading works.
"""

import time
import sys

from .reader import ACMemoryReader


def main():
    print("=" * 60)
    print("AssaultCube Memory Reader Test")
    print("AC Version: 1.3.0.2 (32-bit)")
    print("=" * 60)

    reader = ACMemoryReader()

    if not reader.attach():
        print("\nMake sure AssaultCube is running!")
        print("Start a game (not just menu) and try again.")
        return 1

    print("\nSuccessfully attached!")
    print("\nReading state every second. Press Ctrl+C to stop.\n")

    try:
        while True:
            state = reader.read_state()

            # Check if readings look valid
            valid = (
                0 <= state.health <= 100 and
                -10000 < state.pos_x < 10000 and
                -10000 < state.pos_y < 10000
            )

            status = "OK" if valid else "INVALID - need to find correct offsets"

            print(f"\rHealth: {state.health:3d} | "
                  f"Armor: {state.armor:3d} | "
                  f"Pos: ({state.pos_x:7.1f}, {state.pos_y:7.1f}, {state.pos_z:7.1f}) | "
                  f"Yaw: {state.yaw:6.1f} | [{status}]", end="")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped.")

    reader.detach()
    return 0


if __name__ == "__main__":
    sys.exit(main())
