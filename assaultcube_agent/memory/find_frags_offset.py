"""
Find the frags offset in player structure.

After finding, update FRAGS in: memory/offsets.py
"""

import pymem
import pymem.process

from .offsets import PLAYER1_PTR_OFFSET, HEALTH, ARMOR

PROCESS_NAME = "ac_client.exe"


def main():
    print("Attach to AC and get a kill, then run this.")
    print("Enter your current frag count: ", end="")
    frags = int(input().strip())

    pm = pymem.Pymem(PROCESS_NAME)
    module_base = pymem.process.module_from_name(
        pm.process_handle, PROCESS_NAME
    ).lpBaseOfDll

    player_ptr = pm.read_int(module_base + PLAYER1_PTR_OFFSET)
    print(f"Player pointer: 0x{player_ptr:X}")

    # Scan for frags value
    print(f"\nScanning for value {frags}...")
    matches = []

    for offset in range(0, 0x300, 4):
        try:
            val = pm.read_int(player_ptr + offset)
            if val == frags:
                matches.append((offset, val))
        except:
            pass

    if matches:
        print(f"\nFound {len(matches)} potential frags offsets:")
        for offset, val in matches:
            print(f"  0x{offset:04X}: {val}")
        print("\nUpdate FRAGS in memory/offsets.py with the correct offset.")
    else:
        print("No matches found!")

    # Also show values around known offsets
    print(f"\n\nValues around health (0x{HEALTH:X}):")
    for offset in range(0xE0, 0x200, 4):
        try:
            val = pm.read_int(player_ptr + offset)
            label = ""
            if offset == HEALTH:
                label = " <- health"
            elif offset == ARMOR:
                label = " <- armor"
            print(f"  0x{offset:04X}: {val:10d}{label}")
        except:
            print(f"  0x{offset:04X}: ERROR")

    pm.close_process()

if __name__ == "__main__":
    main()
