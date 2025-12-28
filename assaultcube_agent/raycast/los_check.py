"""
Line-of-Sight checking for AssaultCube.

Uses the world map data to check if there's a clear line between two points.
Based on raycubelos() from AC source.

Run test: python -m assaultcube_agent.raycast.los_check
"""

import sys
import os
import math
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymem
import pymem.process

from ..memory.offsets import (
    SFACTOR_OFFSET,
    SSIZE_OFFSET,
    WORLD_PTR_OFFSET,
    SQR_SIZE,
    SQR_SOLID as SOLID,
    SQR_CORNER as CORNER,
    SQR_FHF as FHF,
    SQR_CHF as CHF,
    SQR_SPACE as SPACE,
    SQR_SEMISOLID as SEMISOLID,
    PLAYER1_PTR_OFFSET,
    POS_X, POS_Y, POS_Z, YAW,
)


class LOSChecker:
    """
    Check line-of-sight between two 3D positions using map geometry.

    Based on raycube/raycubelos from AC physics.cpp.
    """

    PROCESS_NAME = "ac_client.exe"

    # Offsets imported from centralized config (memory/offsets.py)
    SFACTOR_OFFSET = SFACTOR_OFFSET
    SSIZE_OFFSET = SSIZE_OFFSET
    WORLD_PTR_OFFSET = WORLD_PTR_OFFSET

    def __init__(self):
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self._attached = False
        self._world_ptr: int = 0
        self._ssize: int = 0

    def attach(self) -> bool:
        """Attach to AC process and read world data."""
        try:
            self.pm = pymem.Pymem(self.PROCESS_NAME)
            self.module_base = pymem.process.module_from_name(
                self.pm.process_handle,
                self.PROCESS_NAME
            ).lpBaseOfDll

            # Read world data
            self._ssize = self.pm.read_int(self.module_base + self.SSIZE_OFFSET)
            self._world_ptr = self.pm.read_int(self.module_base + self.WORLD_PTR_OFFSET)

            # Validate
            if self._ssize < 64 or self._ssize > 8192:
                print(f"[LOSChecker] Invalid ssize: {self._ssize}")
                return False

            if self._world_ptr < 0x00400000:
                print(f"[LOSChecker] Invalid world ptr: 0x{self._world_ptr:X}")
                return False

            self._attached = True
            print(f"[LOSChecker] Attached: world=0x{self._world_ptr:X}, ssize={self._ssize}")
            return True

        except pymem.exception.ProcessNotFound:
            print(f"[LOSChecker] ERROR: {self.PROCESS_NAME} not found")
            return False
        except Exception as e:
            print(f"[LOSChecker] ERROR: {e}")
            return False

    def detach(self):
        """Detach from process."""
        if self.pm:
            self.pm.close_process()
            self.pm = None
        self._attached = False

    @property
    def attached(self) -> bool:
        """Check if still attached."""
        if not self._attached or not self.pm:
            return False
        try:
            self.pm.read_int(self.module_base)
            return True
        except:
            self._attached = False
            return False

    def _refresh_world(self):
        """Refresh world pointer (in case map changed)."""
        if not self.pm:
            return
        try:
            self._ssize = self.pm.read_int(self.module_base + self.SSIZE_OFFSET)
            self._world_ptr = self.pm.read_int(self.module_base + self.WORLD_PTR_OFFSET)
        except:
            pass

    def _get_sqr(self, x: int, y: int) -> tuple[int, int, int]:
        """
        Get sqr data at position (x, y).

        Returns:
            (type, floor, ceil)
        """
        if x < 0 or y < 0 or x >= self._ssize or y >= self._ssize:
            return (SOLID, 0, 0)  # Out of bounds = solid

        idx = y * self._ssize + x
        sqr_addr = self._world_ptr + idx * SQR_SIZE

        try:
            # Read sqr struct bytes: type(1), floor(1 signed), ceil(1 signed), ...
            data = self.pm.read_bytes(sqr_addr, 3)
            sqr_type = data[0]
            # Convert to signed bytes
            floor = data[1] if data[1] < 128 else data[1] - 256
            ceil = data[2] if data[2] < 128 else data[2] - 256
            return (sqr_type, floor, ceil)
        except:
            return (SOLID, 0, 0)

    def raycube(
        self,
        ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        max_dist: float = 512.0
    ) -> float:
        """
        Trace a ray through the world.

        Args:
            ox, oy, oz: Ray origin
            dx, dy, dz: Ray direction (normalized)
            max_dist: Maximum distance to trace

        Returns:
            Distance to first solid hit, or -1 if no hit
        """
        if not self._attached:
            return -1

        # Normalize direction
        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if mag < 0.0001:
            return -1
        dx, dy, dz = dx/mag, dy/mag, dz/mag

        vx, vy, vz = ox, oy, oz
        dist = 0.0

        # Step through grid (simplified raycube)
        for _ in range(512):  # Max iterations
            x, y = int(vx), int(vy)

            if x < 0 or y < 0 or x >= self._ssize or y >= self._ssize:
                return -1  # Out of bounds

            sqr_type, floor, ceil = self._get_sqr(x, y)

            # Check for solid or out of vertical bounds
            if sqr_type == SOLID:
                return max(dist - 0.1, 0.0)

            # floor/ceil are already in world-compatible units for this implementation
            if vz < floor or vz > ceil:
                return max(dist - 0.1, 0.0)

            # Calculate distance to next grid cell
            if dx != 0:
                t_x = (x + (1 if dx > 0 else 0) - vx) / dx
            else:
                t_x = 1e16

            if dy != 0:
                t_y = (y + (1 if dy > 0 else 0) - vy) / dy
            else:
                t_y = 1e16

            if dz != 0:
                t_z = ((ceil if dz > 0 else floor) - vz) / dz
            else:
                t_z = 1e16

            # Check vertical hit
            if t_z < t_x and t_z < t_y:
                dist += t_z
                return max(dist - 0.1, 0.0)

            # Step to next cell
            step = 0.1 + min(t_x, t_y)
            vx += dx * step
            vy += dy * step
            vz += dz * step
            dist += step

            if dist > max_dist:
                return -1

        return -1

    def check_los(
        self,
        from_x: float, from_y: float, from_z: float,
        to_x: float, to_y: float, to_z: float,
        margin: float = 0.5
    ) -> bool:
        """
        Check line-of-sight between two points.

        Based on raycubelos() from AC physics.cpp.

        Args:
            from_x, from_y, from_z: Starting point
            to_x, to_y, to_z: Target point
            margin: Safety margin for the check

        Returns:
            True if line-of-sight is clear, False if blocked
        """
        if not self._attached:
            return False

        # Calculate direction and distance
        dx = to_x - from_x
        dy = to_y - from_y
        dz = to_z - from_z
        target_dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if target_dist < 0.01:
            return True  # Same point

        # Trace ray
        hit_dist = self.raycube(from_x, from_y, from_z, dx, dy, dz, target_dist + 10)

        # LOS is clear if hit distance is greater than target distance (minus margin)
        if hit_dist < 0:
            return True  # No hit = clear

        return hit_dist > max(target_dist - margin, 0.0)


def test_los_checker():
    """Test the LOS checker."""
    print("=" * 70)
    print("  LOS Checker Test")
    print("=" * 70)

    checker = LOSChecker()
    if not checker.attach():
        return 1

    # Read player position for testing
    try:
        player1_ptr = checker.pm.read_int(checker.module_base + PLAYER1_PTR_OFFSET)
        pos_x = checker.pm.read_float(player1_ptr + POS_X)
        pos_y = checker.pm.read_float(player1_ptr + POS_Y)
        pos_z = checker.pm.read_float(player1_ptr + POS_Z)
        print(f"\n[+] Player position: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f})")

        # Test sqr at player position
        x, y = int(pos_x), int(pos_y)
        sqr_type, floor, ceil = checker._get_sqr(x, y)
        type_names = {SOLID: "SOLID", CORNER: "CORNER", FHF: "FHF", CHF: "CHF", SPACE: "SPACE", SEMISOLID: "SEMISOLID"}
        print(f"[+] Sqr at ({x},{y}): type={type_names.get(sqr_type, sqr_type)}, floor={floor}, ceil={ceil}")

        # Test ray in front of player
        yaw = checker.pm.read_float(player1_ptr + YAW)
        print(f"[+] Player yaw: {yaw:.1f}")

        # Calculate direction from yaw
        yaw_rad = math.radians(yaw)
        dir_x = math.sin(yaw_rad)
        dir_y = -math.cos(yaw_rad)

        # Trace ray forward
        hit_dist = checker.raycube(pos_x, pos_y, pos_z, dir_x, dir_y, 0, 200)
        print(f"[+] Ray forward: hit at dist={hit_dist:.1f}")

        # Test LOS to some points
        print("\n[*] Testing LOS to nearby points...")

        test_points = [
            (pos_x + 10, pos_y, pos_z),
            (pos_x - 10, pos_y, pos_z),
            (pos_x, pos_y + 10, pos_z),
            (pos_x, pos_y - 10, pos_z),
            (pos_x + 50 * dir_x, pos_y + 50 * dir_y, pos_z),
        ]

        for tx, ty, tz in test_points:
            los = checker.check_los(pos_x, pos_y, pos_z, tx, ty, tz)
            dist = math.sqrt((tx-pos_x)**2 + (ty-pos_y)**2 + (tz-pos_z)**2)
            print(f"  ({tx:.0f},{ty:.0f},{tz:.0f}) dist={dist:.0f}: {'CLEAR' if los else 'BLOCKED'}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    checker.detach()
    return 0


if __name__ == "__main__":
    sys.exit(test_los_checker())
