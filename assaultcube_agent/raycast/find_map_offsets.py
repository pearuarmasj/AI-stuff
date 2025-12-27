"""
Find map/world geometry offsets for line-of-sight raycasting.

AssaultCube uses the Cube engine which stores maps as an octree of cubes.
We need to find:
1. worldroot - pointer to the octree root
2. worldsize/hdr - map dimensions
3. Any traceline/raycube function we can call

Run: python -m assaultcube_agent.raycast.find_map_offsets

After finding, update WORLD_PTR_OFFSET, SSIZE_OFFSET in: memory/offsets.py
"""

import sys
import os
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymem
import pymem.process

from ..memory.offsets import PLAYER1_PTR_OFFSET


def scan_for_pattern(pm, module_base, module_size, pattern: bytes, mask: str) -> list[int]:
    """Scan memory for a byte pattern with mask (x = match, ? = wildcard)."""
    results = []
    try:
        data = pm.read_bytes(module_base, module_size)
        for i in range(len(data) - len(pattern)):
            match = True
            for j, (b, m) in enumerate(zip(pattern, mask)):
                if m == 'x' and data[i + j] != b:
                    match = False
                    break
            if match:
                results.append(module_base + i)
    except:
        pass
    return results


def main():
    print("=" * 70)
    print("  Map/World Offset Finder for Line-of-Sight")
    print("=" * 70)

    try:
        pm = pymem.Pymem("ac_client.exe")
        module = pymem.process.module_from_name(pm.process_handle, "ac_client.exe")
        module_base = module.lpBaseOfDll
        module_size = module.SizeOfImage
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"[+] Attached, module base: 0x{module_base:X}, size: 0x{module_size:X}")

    # Known offsets to check (from various AC versions)
    # These are common in Cube engine games
    known_offsets = {
        "worldroot": [0x17E0E8, 0x17E0EC, 0x18B0E8, 0x18B0EC],
        "worldsize": [0x17E0F0, 0x17E0F4, 0x18B0F0, 0x18B0F4],
        "worldscale": [0x17E0F8, 0x17E0FC],
    }

    print("\n[*] Checking known offsets for valid pointers...")

    for name, offsets in known_offsets.items():
        print(f"\n{name}:")
        for offset in offsets:
            try:
                val = pm.read_int(module_base + offset)
                # Check if it looks like a valid pointer or reasonable value
                if name == "worldroot" and 0x10000 < val < 0x7FFFFFFF:
                    print(f"  0x{offset:X}: 0x{val:X} (looks like valid pointer)")
                elif name == "worldsize" and 0 < val < 10000:
                    print(f"  0x{offset:X}: {val} (reasonable map size)")
                elif name == "worldscale" and 0 < val < 20:
                    print(f"  0x{offset:X}: {val} (reasonable scale)")
                else:
                    print(f"  0x{offset:X}: {val}")
            except:
                print(f"  0x{offset:X}: read error")

    # Scan for strings related to map/world
    print("\n[*] Scanning for world-related strings...")

    strings_to_find = [b"worldsize", b"worldroot", b"raycube", b"traceline", b"intersect"]

    for search_str in strings_to_find:
        try:
            # Simple string search in .rdata section (usually around offset 0x100000+)
            data = pm.read_bytes(module_base, module_size)
            idx = data.find(search_str)
            if idx != -1:
                print(f"  Found '{search_str.decode()}' at offset 0x{idx:X}")
        except:
            pass

    # Try to find the cube/octree structure by looking for specific patterns
    print("\n[*] Scanning memory for potential octree pointers...")

    # In Cube engine, worldroot is typically a pointer to a 'cube' struct
    # The cube struct has children pointers at offset 0
    # Let's scan the data segment for pointers that look like octree roots

    data_start = module_base + 0x150000  # Typical data segment start
    data_size = 0x50000  # 320KB to scan

    candidates = []
    try:
        for offset in range(0, data_size, 4):
            addr = data_start + offset
            ptr = pm.read_int(addr)

            # Check if it looks like a valid heap pointer
            if 0x00400000 < ptr < 0x7FFFFFFF:
                # Try to read what it points to - octree node has 8 child pointers
                try:
                    children = []
                    valid_children = 0
                    for i in range(8):
                        child = pm.read_int(ptr + i * 4)
                        children.append(child)
                        if child == 0 or (0x00400000 < child < 0x7FFFFFFF):
                            valid_children += 1

                    # If most children look valid, this might be octree root
                    if valid_children >= 6:
                        candidates.append((addr - module_base, ptr, children))
                except:
                    pass
    except Exception as e:
        print(f"  Scan error: {e}")

    print(f"\n  Found {len(candidates)} potential octree candidates")
    for offset, ptr, children in candidates[:10]:  # Show first 10
        non_null = sum(1 for c in children if c != 0)
        print(f"    0x{offset:X} -> 0x{ptr:X} ({non_null}/8 non-null children)")

    # Look for player1 and see what's near it (worldroot is often near player pointers)
    print("\n[*] Checking area around player1 pointer...")

    for delta in range(-0x100, 0x100, 4):
        offset = PLAYER1_PTR_OFFSET + delta
        try:
            val = pm.read_int(module_base + offset)
            if 0x00100000 < val < 0x7FFFFFFF and val != 0:
                # Try to identify what this pointer points to
                try:
                    first_bytes = pm.read_bytes(val, 32)
                    # Check if it looks like geometry data
                    if delta != 0:  # Skip player1 itself
                        print(f"  0x{offset:X}: 0x{val:X}")
                except:
                    pass
        except:
            pass

    pm.close_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
