"""
Find the players vector offset using differential scanning.

1. Scan with X bots
2. Add/remove a bot
3. Scan again - find what changed

Run: python -m assaultcube_agent.raycast.find_players_offset

After finding, update PLAYERS_ARRAY_OFFSET in: memory/offsets.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymem
import pymem.process

from ..memory.offsets import PLAYER1_PTR_OFFSET, HEALTH, POS_X, POS_Y, POS_Z


def scan_for_value(pm, module_base, value, scan_start=0x100000, scan_end=0x200000):
    """Scan memory for a specific integer value."""
    matches = []
    for offset in range(scan_start, scan_end, 4):
        try:
            val = pm.read_int(module_base + offset)
            if val == value:
                matches.append(offset)
        except:
            continue
    return set(matches)


def main():
    print("=" * 70)
    print("  Players Vector Offset Finder (Differential Scan)")
    print("  AC 1.3.0.2 (32-bit)")
    print("=" * 70)
    print()

    try:
        pm = pymem.Pymem("ac_client.exe")
        module_base = pymem.process.module_from_name(
            pm.process_handle, "ac_client.exe"
        ).lpBaseOfDll
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure AssaultCube is running!")
        return 1

    print(f"[+] Attached to ac_client.exe")
    print(f"[+] Module base: 0x{module_base:X}")
    print()

    # Step 1: Get initial bot count
    print("=" * 70)
    input("Step 1: Start a bot match. Press ENTER when ready...")

    bot_count_1 = int(input("How many bots are in the game (excluding yourself)? "))
    total_players_1 = bot_count_1 + 1  # +1 for player1

    print(f"\n[*] Scanning for value {total_players_1} (total players including you)...")
    matches_1 = scan_for_value(pm, module_base, total_players_1)
    print(f"[+] Found {len(matches_1)} locations with value {total_players_1}")

    # Step 2: Change bot count
    print()
    print("=" * 70)
    print("Step 2: Add or remove a bot using console commands:")
    print("  To add bot:    /addbot 1")
    print("  To remove bot: /delbot")
    print()
    input("Press ENTER after changing the bot count...")

    bot_count_2 = int(input("How many bots now (excluding yourself)? "))
    total_players_2 = bot_count_2 + 1

    print(f"\n[*] Scanning for value {total_players_2}...")
    matches_2 = scan_for_value(pm, module_base, total_players_2)
    print(f"[+] Found {len(matches_2)} locations with value {total_players_2}")

    # Find locations that changed correctly
    # These are locations that had value X, now have value Y
    print()
    print("=" * 70)
    print("[*] Finding locations that changed from {} to {}...".format(
        total_players_1, total_players_2
    ))

    # Re-read the first set of matches to see which now have the new value
    candidates = []
    for offset in matches_1:
        try:
            current_val = pm.read_int(module_base + offset)
            if current_val == total_players_2:
                candidates.append(offset)
        except:
            continue

    print(f"[+] Found {len(candidates)} candidates that changed correctly!")
    print()

    if not candidates:
        print("[!] No candidates found. Try again with different bot counts.")
        print("[!] Make sure you're changing bots with /addbot or /delbot")
    else:
        # For each candidate, check if it looks like a vector count
        # Vector count would be at offset+4 relative to data pointer
        # Or it could be (end_ptr - data_ptr) / 4

        print("Candidates (potential player count locations):")
        print()

        verified = []

        for offset in sorted(candidates):
            # Check nearby memory for vector-like structure
            # If this is the count derived from (end-start)/4,
            # then start should be at offset-4 or offset-8

            try:
                # Try interpreting as count inside vector struct
                # MSVC vector: data_ptr, end_ptr, cap_ptr
                # count = (end - data) / 4

                # Check if offset-4 and offset-8 look like pointers
                val_m8 = pm.read_int(module_base + offset - 8)
                val_m4 = pm.read_int(module_base + offset - 4)
                val_0 = pm.read_int(module_base + offset)
                val_p4 = pm.read_int(module_base + offset + 4)

                print(f"  0x{offset:X}: value={val_0}")
                print(f"    [-8]=0x{val_m8:X}  [-4]=0x{val_m4:X}  [+4]=0x{val_p4:X}")

                # Check if this could be a vector by looking for data pointer before it
                # The actual vector start would be 8 bytes before the "end" pointer
                # Let's check if (val_m4 - val_m8) / 4 == val_0 for any interpretation

                # Or check offset-8 as potential vector start
                for vec_offset in [offset - 8, offset - 4, offset]:
                    try:
                        data_ptr = pm.read_int(module_base + vec_offset)
                        end_ptr = pm.read_int(module_base + vec_offset + 4)

                        if data_ptr > 0 and end_ptr > data_ptr:
                            count = (end_ptr - data_ptr) // 4
                            if count == total_players_2:
                                print(f"    -> Vector at 0x{vec_offset:X}: count={count}")

                                # Verify by reading player pointers
                                player1_ptr = pm.read_int(module_base + PLAYER1_PTR_OFFSET)
                                valid_players = 0
                                has_player1 = False

                                for i in range(count):
                                    p_ptr = pm.read_int(data_ptr + i * 4)
                                    if p_ptr == player1_ptr:
                                        has_player1 = True
                                    # Check if has valid health
                                    try:
                                        hp = pm.read_int(p_ptr + HEALTH)
                                        if 0 <= hp <= 100:
                                            valid_players += 1
                                    except:
                                        pass

                                if has_player1 and valid_players == count:
                                    print(f"    -> VERIFIED! Has player1, all {count} have valid health")
                                    verified.append(vec_offset)
                    except:
                        pass

                print()

            except Exception as e:
                print(f"  0x{offset:X}: read error - {e}")

        if verified:
            print("=" * 70)
            print(f"[+] VERIFIED PLAYERS VECTOR OFFSET: 0x{verified[0]:X}")
            print("=" * 70)
            print()
            print("Update memory/offsets.py with:")
            print(f"    PLAYERS_ARRAY_OFFSET = 0x{verified[0]:X}")
            print()

            # Dump players
            vec_offset = verified[0]
            data_ptr = pm.read_int(module_base + vec_offset)
            end_ptr = pm.read_int(module_base + vec_offset + 4)
            count = (end_ptr - data_ptr) // 4
            player1_ptr = pm.read_int(module_base + PLAYER1_PTR_OFFSET)

            print("Players:")
            for i in range(count):
                p_ptr = pm.read_int(data_ptr + i * 4)
                try:
                    hp = pm.read_int(p_ptr + HEALTH)
                    px = pm.read_float(p_ptr + POS_X)
                    py = pm.read_float(p_ptr + POS_Y)
                    pz = pm.read_float(p_ptr + POS_Z)
                    me = " <- YOU" if p_ptr == player1_ptr else ""
                    print(f"  [{i}] hp={hp:3d} pos=({px:.1f}, {py:.1f}, {pz:.1f}){me}")
                except:
                    print(f"  [{i}] read error")

    pm.close_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
