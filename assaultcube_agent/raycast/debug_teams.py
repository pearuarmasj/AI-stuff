"""
Debug script to check team values for all players.

Run: python -m assaultcube_agent.raycast.debug_teams
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymem
import pymem.process


def main():
    print("=" * 70)
    print("  Team Debug")
    print("=" * 70)

    try:
        pm = pymem.Pymem("ac_client.exe")
        module_base = pymem.process.module_from_name(
            pm.process_handle, "ac_client.exe"
        ).lpBaseOfDll
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"[+] Attached, module base: 0x{module_base:X}")

    # Read player1 pointer
    player1_ptr = pm.read_int(module_base + 0x18AC00)
    array_ptr = pm.read_int(module_base + 0x18AC04)
    player_count = pm.read_int(module_base + 0x18AC0C)

    print(f"[+] player1: 0x{player1_ptr:X}")
    print(f"[+] players array: 0x{array_ptr:X}")
    print(f"[+] player count: {player_count}")
    print()

    # Check team value at different offsets for player1
    print(f"Player1 (you) team value at various offsets:")
    offsets_to_check = [0x1D4, 0x1D8, 0x1DC, 0x1E0, 0x30C, 0x32C, 0x33C]
    for offset in offsets_to_check:
        try:
            val = pm.read_int(player1_ptr + offset)
            print(f"  0x{offset:X}: {val}")
        except:
            print(f"  0x{offset:X}: ERROR")
    print()

    # Now check all players in array
    print("All players in array:")
    for i in range(player_count):
        try:
            player_ptr = pm.read_int(array_ptr + i * 4)
            if player_ptr == 0:
                print(f"  Player {i}: NULL")
                continue

            is_self = player_ptr == player1_ptr

            # Read values
            pos_x = pm.read_float(player_ptr + 0x04)
            pos_y = pm.read_float(player_ptr + 0x08)
            pos_z = pm.read_float(player_ptr + 0x0C)
            health = pm.read_int(player_ptr + 0xEC)
            team_1d8 = pm.read_int(player_ptr + 0x1D8)

            # Try other potential team offsets
            team_32c = pm.read_int(player_ptr + 0x32C)
            team_30c = pm.read_int(player_ptr + 0x30C)

            print(f"  Player {i}: ptr=0x{player_ptr:X}, hp={health:3d}, team@1D8={team_1d8}, team@30C={team_30c}, team@32C={team_32c}, pos=({pos_x:.0f},{pos_y:.0f},{pos_z:.0f}) {'<- YOU' if is_self else ''}")
        except Exception as e:
            print(f"  Player {i}: ERROR {e}")

    print()

    # Also check if there's a different structure - maybe bots use different offsets
    print("Reading name strings to verify player identity:")
    name_offset = 0x205  # Common name offset in AC
    for i in range(player_count):
        try:
            player_ptr = pm.read_int(array_ptr + i * 4)
            if player_ptr == 0:
                continue
            # Read name (null-terminated string)
            name_bytes = pm.read_bytes(player_ptr + name_offset, 16)
            name = name_bytes.split(b'\x00')[0].decode('utf-8', errors='replace')
            print(f"  Player {i}: {name}")
        except Exception as e:
            print(f"  Player {i}: name error {e}")

    pm.close_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
