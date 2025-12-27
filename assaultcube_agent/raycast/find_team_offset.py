"""
Find the team offset using differential scan.

1. Note your current team (CLA=0 or RVSF=1)
2. Switch teams
3. Find what changed

Run: python -m assaultcube_agent.raycast.find_team_offset

After finding, update TEAM in: memory/offsets.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymem
import pymem.process

from ..memory.offsets import PLAYER1_PTR_OFFSET


def main():
    print("=" * 70)
    print("  Team Offset Finder")
    print("=" * 70)

    try:
        pm = pymem.Pymem("ac_client.exe")
        module_base = pymem.process.module_from_name(
            pm.process_handle, "ac_client.exe"
        ).lpBaseOfDll
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"[+] Attached")

    # Read player1 pointer
    player1_ptr = pm.read_int(module_base + PLAYER1_PTR_OFFSET)
    print(f"[+] player1: 0x{player1_ptr:X}")

    # Step 1: Note current team
    print()
    print("What team are you on?")
    print("  0 = CLA (brown/tan)")
    print("  1 = RVSF (blue)")
    team1 = int(input("Enter team number: "))

    # Scan player struct for this value
    print(f"\n[*] Scanning player struct for value {team1}...")
    matches1 = []
    for offset in range(0, 0x300, 4):
        try:
            val = pm.read_int(player1_ptr + offset)
            if val == team1:
                matches1.append(offset)
        except:
            pass

    print(f"[+] Found {len(matches1)} locations with value {team1}")

    # Step 2: Switch teams
    print()
    print("Now switch teams using: /team CLA  or  /team RVSF")
    input("Press ENTER after switching teams...")

    team2 = int(input("What team are you on now (0 or 1)? "))

    # Find which locations changed
    print(f"\n[*] Finding locations that changed from {team1} to {team2}...")
    candidates = []
    for offset in matches1:
        try:
            val = pm.read_int(player1_ptr + offset)
            if val == team2:
                candidates.append(offset)
        except:
            pass

    print(f"[+] Found {len(candidates)} candidates!")
    print()

    if candidates:
        for offset in candidates:
            print(f"  Team offset: 0x{offset:X}")

        print()
        print(f"Most likely team offset: 0x{candidates[0]:X}")
        print()
        print("Update memory/offsets.py with:")
        print(f"    TEAM = 0x{candidates[0]:X}")

    pm.close_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
