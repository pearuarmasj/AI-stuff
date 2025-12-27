"""
Verify world pointer offset based on found sfactor/ssize.

sfactor=8, ssize=256 found at offset 0x182930.
Layout should be: world(4), sfactor(4), ssize(4), cubicsize(4), mipsize(4)

Run: python -m assaultcube_agent.raycast.verify_world_offset
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymem
import pymem.process


# sqr struct types from world.h
SOLID = 0
CORNER = 1
FHF = 2
CHF = 3
SPACE = 4
SEMISOLID = 5

SQR_SIZE = 16  # sqr struct is 16 bytes


def main():
    print("=" * 70)
    print("  Verify World Offset")
    print("=" * 70)

    try:
        pm = pymem.Pymem("ac_client.exe")
        module = pymem.process.module_from_name(pm.process_handle, "ac_client.exe")
        module_base = module.lpBaseOfDll
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"[+] Attached, module base: 0x{module_base:X}")

    # Found sfactor at 0x182930
    sfactor_offset = 0x182930

    # Read values around sfactor
    print(f"\n[*] Reading values around offset 0x{sfactor_offset:X}:")

    for delta in range(-16, 24, 4):
        offset = sfactor_offset + delta
        try:
            val = pm.read_int(module_base + offset)
            label = ""
            if delta == -4:
                label = " <- world?"
            elif delta == 0:
                label = " <- sfactor"
            elif delta == 4:
                label = " <- ssize"
            elif delta == 8:
                label = " <- cubicsize"
            elif delta == 12:
                label = " <- mipsize"
            print(f"  0x{offset:X}: {val} (0x{val & 0xFFFFFFFF:08X}){label}")
        except Exception as e:
            print(f"  0x{offset:X}: ERROR {e}")

    # Now read world pointer (should be at sfactor_offset - 4)
    world_ptr_offset = sfactor_offset - 4
    world_ptr = pm.read_int(module_base + world_ptr_offset)
    ssize = pm.read_int(module_base + sfactor_offset + 4)

    print(f"\n[+] World pointer: 0x{world_ptr:X}")
    print(f"[+] ssize: {ssize}")

    if world_ptr < 0x00400000 or world_ptr > 0x7FFFFFFF:
        print("[-] World pointer doesn't look valid, trying other offsets...")

        # Maybe the layout is different - scan around for valid heap pointer
        for delta in range(-32, 32, 4):
            offset = sfactor_offset + delta
            try:
                ptr = pm.read_int(module_base + offset)
                if 0x01000000 < ptr < 0x7FFFFFFF:
                    # Try to validate as world pointer
                    valid = 0
                    for i in range(100):
                        sqr_type = pm.read_uchar(ptr + i * SQR_SIZE)
                        if sqr_type <= SEMISOLID:
                            valid += 1
                    if valid >= 80:
                        print(f"  0x{offset:X}: 0x{ptr:X} - {valid}/100 valid sqrs - LIKELY WORLD")
            except:
                pass
    else:
        # Verify world data
        print(f"\n[*] Verifying world data at 0x{world_ptr:X}...")

        valid_count = 0
        solid_count = 0
        space_count = 0

        for i in range(256):
            try:
                sqr_type = pm.read_uchar(world_ptr + i * SQR_SIZE)
                if sqr_type <= SEMISOLID:
                    valid_count += 1
                    if sqr_type == SOLID:
                        solid_count += 1
                    elif sqr_type == SPACE:
                        space_count += 1
            except:
                pass

        print(f"  Valid sqrs: {valid_count}/256")
        print(f"  SOLID: {solid_count}, SPACE: {space_count}")

        if valid_count >= 200:
            print("\n" + "=" * 50)
            print("SUCCESS! World offsets found:")
            print("=" * 50)
            print(f"WORLD_PTR_OFFSET = 0x{world_ptr_offset:X}")
            print(f"SFACTOR_OFFSET = 0x{sfactor_offset:X}")
            print(f"SSIZE_OFFSET = 0x{sfactor_offset + 4:X}")

            # Test reading a specific position
            print(f"\n[*] Testing position read...")
            # Read center of map
            cx, cy = ssize // 2, ssize // 2
            idx = cy * ssize + cx
            sqr_addr = world_ptr + idx * SQR_SIZE

            sqr_type = pm.read_uchar(sqr_addr)
            floor = pm.read_char(sqr_addr + 1)
            ceil = pm.read_char(sqr_addr + 2)

            type_names = {0: "SOLID", 1: "CORNER", 2: "FHF", 3: "CHF", 4: "SPACE", 5: "SEMISOLID"}
            print(f"  Position ({cx},{cy}): type={type_names.get(sqr_type, sqr_type)}, floor={floor}, ceil={ceil}")

    pm.close_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
