"""
Interactive tool to find correct memory offsets for AssaultCube.

Run this while AssaultCube is running to discover correct offsets
for health, position, and other player state values.

Usage:
    python -m assaultcube_agent.memory.find_offsets
"""

import struct
import time
from typing import Optional

import pymem
import pymem.process


class OffsetFinder:
    """Interactive offset discovery tool."""

    PROCESS_NAME = "ac_client.exe"

    # Common player1 pointer offsets for different AC versions (32-bit)
    KNOWN_PLAYER_PTRS = [
        0x18AC00,  # AC 1.3.0.2 (most likely)
        0x17E0A8,  # AC 1.3.x alternative
        0x10F4F4,  # AC 1.2.0.2
        0x109B74,  # Older versions
        0x50F4F4,  # Some versions
        0x58AC00,  # Alternative
        0x18A8A8,  # AC 1.3.x alternative 2
    ]

    def __init__(self):
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self.player_ptr: int = 0

    def attach(self) -> bool:
        """Attach to AC process."""
        try:
            self.pm = pymem.Pymem(self.PROCESS_NAME)
            self.module_base = pymem.process.module_from_name(
                self.pm.process_handle,
                self.PROCESS_NAME
            ).lpBaseOfDll
            print(f"Attached to {self.PROCESS_NAME}")
            print(f"Module base: 0x{self.module_base:X}")
            return True
        except pymem.exception.ProcessNotFound:
            print(f"ERROR: {self.PROCESS_NAME} not found!")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def find_player_ptr_offset(self) -> int:
        """
        Try known player pointer offsets.

        Returns:
            Working offset, or 0 if none found.
        """
        print("\nSearching for player1 pointer offset...")

        for offset in self.KNOWN_PLAYER_PTRS:
            try:
                ptr_addr = self.module_base + offset
                ptr = self.pm.read_int(ptr_addr)

                # Check if pointer looks valid (points to reasonable memory)
                if ptr > 0x10000 and ptr < 0x7FFFFFFF:
                    # Try reading something from this pointer
                    test_val = self.pm.read_int(ptr)

                    print(f"  0x{offset:X} -> 0x{ptr:X} (test read: {test_val})")

                    # If we can read and the first few values look like floats (position)
                    # then this might be valid
                    try:
                        pos_x = self.pm.read_float(ptr + 0x04)
                        pos_y = self.pm.read_float(ptr + 0x08)
                        pos_z = self.pm.read_float(ptr + 0x0C)

                        # Position values should be reasonable (-10000 to 10000 ish)
                        if -10000 < pos_x < 10000 and -10000 < pos_y < 10000:
                            print(f"    -> Position looks valid: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f})")
                            self.player_ptr = ptr
                            return offset
                    except:
                        pass

            except Exception as e:
                print(f"  0x{offset:X} -> Error: {e}")

        print("\nNo known offset worked. Need to scan...")
        return 0

    def scan_for_player_ptr(self) -> int:
        """
        Scan for player pointer in common regions.

        This is slower but more thorough.
        """
        print("\nScanning for player1 pointer in static data section...")

        # Scan common static data offsets
        for offset in range(0x100000, 0x200000, 4):
            try:
                ptr_addr = self.module_base + offset
                ptr = self.pm.read_int(ptr_addr)

                if ptr > 0x10000 and ptr < 0x7FFFFFFF:
                    # Check for position-like floats
                    try:
                        pos_x = self.pm.read_float(ptr + 0x04)
                        if -10000 < pos_x < 10000:
                            pos_y = self.pm.read_float(ptr + 0x08)
                            pos_z = self.pm.read_float(ptr + 0x0C)

                            if -10000 < pos_y < 10000 and 0 < pos_z < 1000:
                                print(f"Possible player ptr at offset 0x{offset:X}")
                                print(f"  Position: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f})")

                                # Prompt user to verify
                                response = input("Does this look right? Move in-game and I'll check again (y/n): ")
                                if response.lower() == 'y':
                                    self.player_ptr = ptr
                                    return offset
                    except:
                        pass

            except:
                pass

        return 0

    def find_health_offset(self) -> int:
        """
        Find the health offset by having user take/heal damage.
        """
        if not self.player_ptr:
            print("No player pointer found!")
            return 0

        print("\n" + "=" * 50)
        print("HEALTH OFFSET FINDER")
        print("=" * 50)

        input("Make sure you have FULL HEALTH (100). Press Enter when ready...")

        # Scan for value 100
        candidates = []
        for offset in range(0, 0x300, 4):
            try:
                val = self.pm.read_int(self.player_ptr + offset)
                if val == 100:
                    candidates.append(offset)
            except:
                pass

        print(f"Found {len(candidates)} values that are 100:")
        for off in candidates[:20]:
            print(f"  0x{off:X}")

        if not candidates:
            print("No candidates found!")
            return 0

        input("\nNow TAKE SOME DAMAGE (get hit). Press Enter when your health has changed...")

        # See which candidates changed
        print("\nChecking which values changed:")
        health_offset = 0
        for off in candidates:
            try:
                val = self.pm.read_int(self.player_ptr + off)
                if val != 100 and 0 < val < 100:
                    print(f"  0x{off:X}: now {val} - THIS IS LIKELY HEALTH!")
                    health_offset = off
            except:
                pass

        return health_offset

    def find_position_offsets(self) -> tuple:
        """
        Find position offsets by having user move.
        """
        if not self.player_ptr:
            print("No player pointer found!")
            return (0, 0, 0)

        print("\n" + "=" * 50)
        print("POSITION OFFSET FINDER")
        print("=" * 50)

        input("Stand still. Press Enter to record initial position...")

        # Record initial float values
        initial = {}
        for offset in range(0, 0x100, 4):
            try:
                val = self.pm.read_float(self.player_ptr + offset)
                if -10000 < val < 10000:  # Reasonable position range
                    initial[offset] = val
            except:
                pass

        input("\nNow MOVE FORWARD a bit. Press Enter when done...")

        # Check which values changed
        print("\nValues that changed:")
        changed = []
        for offset, old_val in initial.items():
            try:
                new_val = self.pm.read_float(self.player_ptr + offset)
                diff = abs(new_val - old_val)
                if diff > 0.5:  # Significant change
                    print(f"  0x{offset:X}: {old_val:.2f} -> {new_val:.2f} (diff: {diff:.2f})")
                    changed.append((offset, old_val, new_val, diff))
            except:
                pass

        # Position vectors are usually 3 consecutive floats (12 bytes)
        # Look for groups of 3
        print("\nLooking for position vector (3 consecutive changing floats)...")

        for i, (off, _, _, _) in enumerate(changed):
            # Check if next 2 offsets are also in changed list
            next_offs = [off + 4, off + 8]
            if all(any(c[0] == no for c in changed) for no in next_offs):
                print(f"Found potential position vector starting at 0x{off:X}")
                return (off, off + 4, off + 8)

        return (0, 0, 0)

    def run_interactive(self):
        """Run interactive offset discovery."""
        if not self.attach():
            return

        print("\n" + "=" * 60)
        print("AssaultCube Offset Finder")
        print("=" * 60)
        print("\nThis tool will help you find the correct memory offsets.")
        print("Make sure you're in-game (not in menu) before starting.\n")

        # Step 1: Find player pointer
        ptr_offset = self.find_player_ptr_offset()
        if not ptr_offset:
            ptr_offset = self.scan_for_player_ptr()

        if not ptr_offset:
            print("\nCould not find player pointer. Try different AC version?")
            return

        print(f"\n>>> Player1 pointer offset: 0x{ptr_offset:X}")

        # Step 2: Find health offset
        health_off = self.find_health_offset()
        if health_off:
            print(f"\n>>> Health offset: 0x{health_off:X}")

        # Step 3: Find position
        pos_offs = self.find_position_offsets()
        if pos_offs[0]:
            print(f"\n>>> Position offsets: X=0x{pos_offs[0]:X}, Y=0x{pos_offs[1]:X}, Z=0x{pos_offs[2]:X}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY - Update these in reader.py:")
        print("=" * 60)
        print(f"PLAYER1_PTR_OFFSET = 0x{ptr_offset:X}")
        if health_off:
            print(f'OFFSETS["health"] = 0x{health_off:X}')
        if pos_offs[0]:
            print(f'OFFSETS["pos_x"] = 0x{pos_offs[0]:X}')
            print(f'OFFSETS["pos_y"] = 0x{pos_offs[1]:X}')
            print(f'OFFSETS["pos_z"] = 0x{pos_offs[2]:X}')


def main():
    finder = OffsetFinder()
    finder.run_interactive()


if __name__ == "__main__":
    main()
