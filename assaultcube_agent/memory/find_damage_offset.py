"""
Find pstatdamage offset by scanning memory after dealing damage.

Instructions:
1. Run this script
2. Deal some damage to an enemy (shoot them)
3. Note the total damage dealt
4. The script will search for that value near the player struct
"""

import time
import struct
import pymem
import pymem.process

from .offsets import PLAYER1_PTR_OFFSET, HEALTH, FRAGS

PROCESS_NAME = "ac_client.exe"

def main():
    print("=" * 50)
    print("  DAMAGE OFFSET FINDER")
    print("=" * 50)

    try:
        pm = pymem.Pymem(PROCESS_NAME)
        module_base = pymem.process.module_from_name(
            pm.process_handle, PROCESS_NAME
        ).lpBaseOfDll
    except:
        print(f"ERROR: {PROCESS_NAME} not found")
        return

    player_ptr = pm.read_int(module_base + PLAYER1_PTR_OFFSET)
    print(f"Player ptr: 0x{player_ptr:X}")

    # Read a large chunk of the player struct
    print("\nDeal some damage to enemies, then enter the total damage you dealt.")
    print("(Check the scoreboard or count your hits)")

    try:
        damage_to_find = int(input("\nEnter damage dealt (e.g., 50): "))
    except:
        print("Invalid input")
        return

    print(f"\nSearching for value {damage_to_find} near player struct...")
    print("-" * 50)

    # Scan range: 0x0 to 0x500 from player pointer
    scan_range = 0x500

    found = []
    for offset in range(0, scan_range, 4):
        try:
            val = pm.read_int(player_ptr + offset)
            if val == damage_to_find:
                found.append(offset)
                print(f"  FOUND at offset 0x{offset:03X} = {val}")
        except:
            pass

    if not found:
        print("  No exact matches found.")
        print("\n  Trying to find values close to the target...")
        for offset in range(0, scan_range, 4):
            try:
                val = pm.read_int(player_ptr + offset)
                if damage_to_find - 10 <= val <= damage_to_find + 10 and val > 0:
                    print(f"  Offset 0x{offset:03X} = {val}")
            except:
                pass

    # Also scan the pstatdamage array area specifically
    print("\n" + "-" * 50)
    print("Scanning expected pstatdamage area (0x170 - 0x1C0):")
    for offset in range(0x170, 0x1C0, 4):
        try:
            val = pm.read_int(player_ptr + offset)
            if val != 0:
                print(f"  Offset 0x{offset:03X} = {val}")
        except:
            pass

    # Try reading the full pstatdamage array at different potential offsets
    print("\n" + "-" * 50)
    print("Trying different base offsets for pstatdamage[9] array:")

    potential_bases = [0x170, 0x194, 0x1A8, 0x1B8, 0x1BC, 0x1C0, 0x1C4, 0x1D0]

    for base in potential_bases:
        total = 0
        values = []
        for i in range(9):  # NUMGUNS = 9
            try:
                val = pm.read_int(player_ptr + base + i * 4)
                if val > 0:
                    total += val
                    values.append(val)
            except:
                pass
        if total > 0:
            print(f"  Base 0x{base:03X}: sum={total}, values={values}")

    pm.close_process()
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
