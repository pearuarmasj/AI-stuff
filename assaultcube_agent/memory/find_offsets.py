"""
Interactive tool to find correct memory offsets for AssaultCube.

Run this while AssaultCube is running to discover correct offsets
for health, position, and other player state values.

Usage:
    python -m assaultcube_agent.memory.find_offsets

After finding offsets, update them in: memory/offsets.py
"""

import math
from typing import Optional

import pymem
import pymem.process

from .offsets import KNOWN_PLAYER_PTRS, HEALTH, POS_X, POS_Y, POS_Z


class OffsetFinder:
    """Interactive offset discovery tool."""

    PROCESS_NAME = "ac_client.exe"

    # Import known player ptr offsets from central config
    KNOWN_PLAYER_PTRS = KNOWN_PLAYER_PTRS

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

    def validate_player_ptr(self, ptr: int, require_full_health: bool = True, verbose: bool = False) -> tuple[bool, str]:
        """
        Validate if a pointer looks like a valid playerent.

        STRICT validation:
        - Position must be reasonable (not near origin, not garbage)
        - Z must be at least 1.0 (player is above ground)
        - Health must be exactly 100 if require_full_health is True

        Returns (is_valid, reason_string)
        """
        try:
            # Read position
            pos_x = self.pm.read_float(ptr + POS_X)
            pos_y = self.pm.read_float(ptr + POS_Y)
            pos_z = self.pm.read_float(ptr + POS_Z)

            # Read health (at offset 0xEC)
            health = self.pm.read_int(ptr + HEALTH)

            # Validation checks
            issues = []

            # Check for NaN or infinity (garbage floats)
            if math.isnan(pos_x) or math.isinf(pos_x):
                issues.append(f"x={pos_x} is NaN/inf")
            if math.isnan(pos_y) or math.isinf(pos_y):
                issues.append(f"y={pos_y} is NaN/inf")
            if math.isnan(pos_z) or math.isinf(pos_z):
                issues.append(f"z={pos_z} is NaN/inf")

            # If any NaN/inf, fail immediately
            if issues:
                if verbose:
                    print(f"    Issues: {', '.join(issues)}")
                return False, ', '.join(issues)

            # Position must be in reasonable range (not garbage huge numbers)
            if not (-5000 < pos_x < 5000):
                issues.append(f"x={pos_x:.1f} out of range")
            if not (-5000 < pos_y < 5000):
                issues.append(f"y={pos_y:.1f} out of range")

            # Position must NOT be near origin - use tolerance, not exact comparison
            # Player position is NEVER near (0,0,0) in any real map
            if abs(pos_x) < 1.0 and abs(pos_y) < 1.0 and abs(pos_z) < 1.0:
                issues.append("position near origin")

            # Z must be at least 1.0 (player feet are above ground)
            # Most maps have floor at z=0-10, player eye height adds ~4-5 units
            if pos_z < 1.0:
                issues.append(f"z={pos_z:.1f} too low")
            elif pos_z > 500:
                issues.append(f"z={pos_z:.1f} too high")

            # Health validation
            if require_full_health:
                # User should have EXACTLY 100 health when scanning
                if health != 100:
                    issues.append(f"health={health} (need 100)")
            else:
                # Just check it's in valid range
                if not (1 <= health <= 100):
                    issues.append(f"health={health} invalid")

            if verbose and issues:
                print(f"    Pos: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f}), Health: {health}")
                print(f"    Issues: {', '.join(issues)}")

            if not issues:
                return True, f"Pos: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f}), Health: {health}"
            return False, ', '.join(issues)

        except Exception as e:
            return False, f"read error: {e}"

    def find_player_ptr_offset(self) -> int:
        """
        Try known player pointer offsets.

        Returns:
            Working offset, or 0 if none found.
        """
        print("\nSearching for player1 pointer offset...")
        print("(Make sure you're IN-GAME with FULL HEALTH 100, not in menu!)\n")

        for offset in self.KNOWN_PLAYER_PTRS:
            try:
                ptr_addr = self.module_base + offset
                ptr = self.pm.read_int(ptr_addr)

                # Check if pointer looks valid (points to reasonable memory)
                if ptr > 0x10000 and ptr < 0x7FFFFFFF:
                    print(f"  0x{offset:X} -> 0x{ptr:X}")
                    is_valid, info = self.validate_player_ptr(ptr, require_full_health=True, verbose=True)

                    if is_valid:
                        print(f"    -> VALID: {info}")
                        self.player_ptr = ptr
                        return offset
                    else:
                        print(f"    -> INVALID: {info}")

            except Exception as e:
                print(f"  0x{offset:X} -> Error: {e}")

        print("\nNo known offset worked. Running full scan...")
        return 0

    def scan_for_player_ptr(self) -> int:
        """
        Scan for player pointer in data sections.

        This is slower but more thorough.
        """
        print("\n" + "=" * 60)
        print("SCANNING DATA SEGMENT FOR PLAYER POINTER")
        print("=" * 60)
        print("\nREQUIREMENTS:")
        print("  1. You MUST be IN-GAME (not in menu)")
        print("  2. You MUST have EXACTLY 100 health (full health)")
        print("  3. You MUST be standing on a map (not spectating)")
        print("  4. Stand somewhere NOT near map origin\n")

        input("Press ENTER when ready to scan...")

        # Scan the data segment
        scan_ranges = [
            (0x100000, 0x200000),  # Primary data segment
            (0x080000, 0x100000),  # Earlier sections
        ]

        candidates = []

        for start, end in scan_ranges:
            print(f"Scanning 0x{start:X} - 0x{end:X}...")

            for offset in range(start, end, 4):
                try:
                    ptr_addr = self.module_base + offset
                    ptr = self.pm.read_int(ptr_addr)

                    # Must look like a heap pointer
                    if not (0x01000000 < ptr < 0x7FFFFFFF):
                        continue

                    is_valid, info = self.validate_player_ptr(ptr, require_full_health=True, verbose=False)
                    if is_valid:
                        candidates.append((offset, ptr, info))

                except:
                    pass

        if not candidates:
            print("\nNo valid candidates found!")
            print("\nTROUBLESHOOTING:")
            print("  - Are you IN-GAME (not in menu)?")
            print("  - Do you have EXACTLY 100 health?")
            print("  - Try healing to full health first")
            return 0

        print(f"\nFound {len(candidates)} candidate(s):\n")

        for i, (offset, ptr, info) in enumerate(candidates):
            print(f"  [{i+1}] Offset 0x{offset:X} -> 0x{ptr:X}")
            print(f"      {info}")

        if len(candidates) == 1:
            print("\nOnly one candidate found - using it.")
            offset, ptr, _ = candidates[0]
            self.player_ptr = ptr
            return offset

        # Multiple candidates - verify by movement
        print("\n" + "-" * 40)
        print("Multiple candidates. Verifying by movement...")
        print("-" * 40)

        # Record initial positions
        initial_positions = {}
        for offset, ptr, info in candidates:
            try:
                pos_x = self.pm.read_float(ptr + POS_X)
                pos_y = self.pm.read_float(ptr + POS_Y)
                pos_z = self.pm.read_float(ptr + POS_Z)
                initial_positions[offset] = (pos_x, pos_y, pos_z, ptr)
            except:
                pass

        input("\nMove around in-game (WASD) for a few seconds, then press ENTER...")

        # Check which position changed REASONABLY (not garbage)
        verified = []
        for offset, (old_x, old_y, old_z, ptr) in initial_positions.items():
            try:
                new_x = self.pm.read_float(ptr + POS_X)
                new_y = self.pm.read_float(ptr + POS_Y)
                new_z = self.pm.read_float(ptr + POS_Z)

                # Check for garbage values
                if math.isnan(new_x) or math.isinf(new_x):
                    continue
                if math.isnan(new_y) or math.isinf(new_y):
                    continue
                if math.isnan(new_z) or math.isinf(new_z):
                    continue

                # New position must still be reasonable
                if not (-5000 < new_x < 5000 and -5000 < new_y < 5000 and 1 < new_z < 500):
                    continue

                # Calculate movement
                diff = abs(new_x - old_x) + abs(new_y - old_y) + abs(new_z - old_z)

                # Movement should be reasonable (1-100 units, not garbage like 1e30)
                if 1.0 < diff < 100.0:
                    print(f"\n>>> Offset 0x{offset:X} - Position changed reasonably!")
                    print(f"    Old: ({old_x:.1f}, {old_y:.1f}, {old_z:.1f})")
                    print(f"    New: ({new_x:.1f}, {new_y:.1f}, {new_z:.1f})")
                    print(f"    Movement: {diff:.1f} units")
                    verified.append((offset, ptr, diff))

            except:
                pass

        if verified:
            # Pick the one with most reasonable movement
            best = min(verified, key=lambda x: abs(x[2] - 10))  # Prefer ~10 unit movement
            offset, ptr, _ = best
            self.player_ptr = ptr
            return offset

        print("\nCouldn't verify any candidate by movement.")
        print("Using first candidate (may be wrong).")
        offset, ptr, _ = candidates[0]
        self.player_ptr = ptr
        return offset

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
                # Must be a reasonable coordinate value
                if not math.isnan(val) and not math.isinf(val) and -5000 < val < 5000:
                    initial[offset] = val
            except:
                pass

        input("\nNow MOVE FORWARD a bit. Press Enter when done...")

        # Check which values changed REASONABLY
        print("\nValues that changed reasonably:")
        changed = []
        for offset, old_val in initial.items():
            try:
                new_val = self.pm.read_float(self.player_ptr + offset)

                # Skip garbage
                if math.isnan(new_val) or math.isinf(new_val):
                    continue
                if not (-5000 < new_val < 5000):
                    continue

                diff = abs(new_val - old_val)
                # Reasonable movement is 0.5 - 50 units
                if 0.5 < diff < 50:
                    print(f"  0x{offset:X}: {old_val:.2f} -> {new_val:.2f} (diff: {diff:.2f})")
                    changed.append((offset, old_val, new_val, diff))
            except:
                pass

        # Position vectors are usually 3 consecutive floats (12 bytes)
        print("\nLooking for position vector (3 consecutive changing floats)...")

        for off, _, _, _ in changed:
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
        print("\nBEFORE STARTING:")
        print("  1. Start a game (bot match or online)")
        print("  2. Make sure you have EXACTLY 100 health")
        print("  3. Stand somewhere on the map (not at origin)\n")

        # Step 1: Find player pointer
        ptr_offset = self.find_player_ptr_offset()
        if not ptr_offset:
            ptr_offset = self.scan_for_player_ptr()

        if not ptr_offset:
            print("\nCould not find player pointer.")
            print("\nMake sure you're IN-GAME with 100 health!")
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
        print("SUMMARY - Update these in memory/offsets.py:")
        print("=" * 60)
        print(f"PLAYER1_PTR_OFFSET = 0x{ptr_offset:X}")
        if health_off:
            print(f'HEALTH = 0x{health_off:X}')
        if pos_offs[0]:
            print(f'POS_X = 0x{pos_offs[0]:X}')
            print(f'POS_Y = 0x{pos_offs[1]:X}')
            print(f'POS_Z = 0x{pos_offs[2]:X}')


def main():
    finder = OffsetFinder()
    finder.run_interactive()


if __name__ == "__main__":
    main()
