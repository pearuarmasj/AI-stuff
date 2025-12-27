"""
Find world pointer by scanning for valid sqr arrays.

The world pointer points to an array of sqr structs (16 bytes each).
We look for pointers that point to memory with valid sqr patterns.

Run: python -m assaultcube_agent.raycast.find_world_offset2
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


def validate_world_ptr(pm, ptr: int, expected_size: int = 0) -> tuple[bool, int]:
    """
    Check if ptr looks like a world pointer.
    Returns (is_valid, detected_ssize)
    """
    # sqr struct is 16 bytes
    SQR_SIZE = 16

    # Try to read first few sqr structs
    try:
        valid_sqrs = 0
        total_sqrs = 256  # Check first 256 sqrs

        for i in range(total_sqrs):
            sqr_type = pm.read_uchar(ptr + i * SQR_SIZE)
            floor = pm.read_char(ptr + i * SQR_SIZE + 1)
            ceil = pm.read_char(ptr + i * SQR_SIZE + 2)

            # Valid type check
            if sqr_type > SEMISOLID:
                continue

            # Floor should be below ceil
            if sqr_type != SOLID and floor >= ceil:
                continue

            valid_sqrs += 1

        if valid_sqrs >= total_sqrs * 0.8:  # 80% should be valid
            return True, 0

    except:
        pass

    return False, 0


def main():
    print("=" * 70)
    print("  World Pointer Finder v2")
    print("=" * 70)

    try:
        pm = pymem.Pymem("ac_client.exe")
        module = pymem.process.module_from_name(pm.process_handle, "ac_client.exe")
        module_base = module.lpBaseOfDll
        module_size = module.SizeOfImage
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"[+] Attached, module base: 0x{module_base:X}")

    # Known offsets for reference
    player1_offset = 0x18AC00
    print(f"[+] player1 at offset 0x{player1_offset:X}")

    # Scan the data segment for heap pointers
    print("\n[*] Scanning data segment for world pointer candidates...")

    # Data segment is typically 0x150000 - 0x1A0000
    data_start = module_base + 0x150000
    data_end = module_base + 0x1A0000

    world_candidates = []

    for offset in range(0, data_end - data_start, 4):
        addr = data_start + offset
        try:
            ptr = pm.read_int(addr)

            # Check if it looks like a heap pointer
            if not (0x01000000 < ptr < 0x7FFFFFFF):
                continue

            # Check if it points to valid world data
            is_valid, detected_size = validate_world_ptr(pm, ptr)
            if is_valid:
                rel_offset = addr - module_base
                world_candidates.append((rel_offset, ptr))

        except:
            pass

    print(f"\n[+] Found {len(world_candidates)} world pointer candidates")

    # For each candidate, look for ssize/sfactor nearby
    print("\n[*] Looking for ssize/sfactor near each candidate...")

    valid_ssizes = {128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11, 4096: 12}

    results = []

    for world_offset, world_ptr in world_candidates:
        # Look within +/- 64 bytes for ssize
        for delta in range(-64, 68, 4):
            try:
                addr = module_base + world_offset + delta
                val = pm.read_int(addr)

                if val in valid_ssizes:
                    sfactor = valid_ssizes[val]
                    ssize = val
                    cubicsize = ssize * ssize
                    mipsize = cubicsize * 134 // 100

                    # Verify cubicsize follows
                    next_val = pm.read_int(addr + 4)
                    if next_val == cubicsize:
                        # Check mipsize
                        mip_val = pm.read_int(addr + 8)
                        if abs(mip_val - mipsize) < mipsize * 0.1:
                            results.append({
                                'world_offset': world_offset,
                                'world_ptr': world_ptr,
                                'ssize_offset': world_offset + delta,
                                'ssize': ssize,
                                'sfactor': sfactor,
                                'cubicsize': cubicsize
                            })

            except:
                pass

    if results:
        print(f"\n[+] Found {len(results)} valid configurations!")

        for r in results:
            print(f"\n  World offset: 0x{r['world_offset']:X} -> 0x{r['world_ptr']:X}")
            print(f"  ssize offset: 0x{r['ssize_offset']:X} = {r['ssize']}")
            print(f"  sfactor = {r['sfactor']}, cubicsize = {r['cubicsize']}")

        # Output the offsets to use
        best = results[0]
        print("\n" + "=" * 50)
        print("OFFSETS TO USE IN LOS CODE:")
        print("=" * 50)
        print(f"WORLD_PTR_OFFSET = 0x{best['world_offset']:X}")
        print(f"SSIZE_OFFSET = 0x{best['ssize_offset']:X}")
        print(f"Current map: ssize={best['ssize']} ({best['ssize']}x{best['ssize']})")

    else:
        print("\n[-] No ssize found near world candidates")
        print("\n[*] Dumping world candidates for manual analysis:")
        for offset, ptr in world_candidates[:10]:
            print(f"  0x{offset:X} -> 0x{ptr:X}")

        # Just scan for ssize values anywhere
        print("\n[*] Scanning for ssize values anywhere in data segment...")
        for ssize_val in valid_ssizes:
            sfactor_val = valid_ssizes[ssize_val]
            for offset in range(0, data_end - data_start, 4):
                addr = data_start + offset
                try:
                    val = pm.read_int(addr)
                    if val == ssize_val:
                        # Check sfactor before
                        prev = pm.read_int(addr - 4)
                        if prev == sfactor_val:
                            rel_offset = addr - 4 - module_base
                            print(f"  Found sfactor={sfactor_val}, ssize={ssize_val} at offset 0x{rel_offset:X}")
                except:
                    pass

    pm.close_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
