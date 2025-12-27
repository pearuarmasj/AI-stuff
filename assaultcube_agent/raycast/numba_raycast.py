"""
Numba-accelerated raycast observation system.

Uses JIT compilation for C-like speed in Python.
Target: 200+ fps with dense raycasting.
"""

import math
from typing import Optional

import numpy as np
from numba import njit, prange
import pymem
import pymem.process

from ..memory.offsets import (
    PLAYER1_PTR_OFFSET,
    PLAYERS_ARRAY_OFFSET,
    PLAYERS_COUNT_OFFSET,
    POS_X, POS_Y, POS_Z, YAW, PITCH,
    HEALTH, TEAM,
    SSIZE_OFFSET,
    WORLD_PTR_OFFSET,
    SQR_SIZE,
    SQR_SOLID as SOLID,
)


# Hit types
HIT_NOTHING = 0
HIT_WALL = 1
HIT_ENEMY = 2


@njit(cache=True)
def trace_ray_numba(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    world_types: np.ndarray,
    ssize: int,
    enemies: np.ndarray,
    num_enemies: int,
    max_dist: float,
    step_size: float,
    enemy_radius_sq: float,
) -> tuple:
    """
    Trace a single ray through the world (numba-compiled).

    Returns (normalized_distance, hit_type)
    """
    dist = 0.0
    vx, vy, vz = ox, oy, oz

    while dist < max_dist:
        # Check bounds and wall
        ix, iy = int(vx), int(vy)
        if ix < 0 or iy < 0 or ix >= ssize or iy >= ssize:
            return dist / max_dist, HIT_WALL

        if world_types[iy * ssize + ix] == SOLID:
            return dist / max_dist, HIT_WALL

        # Check enemies
        for i in range(num_enemies):
            ex = enemies[i, 0]
            ey = enemies[i, 1]
            ez = enemies[i, 2]
            dist_sq = (vx - ex)**2 + (vy - ey)**2 + (vz - ez)**2
            if dist_sq < enemy_radius_sq:
                return dist / max_dist, HIT_ENEMY

        # Step forward
        vx += dx * step_size
        vy += dy * step_size
        vz += dz * step_size
        dist += step_size

    return 1.0, HIT_NOTHING


@njit(parallel=True, cache=True)
def trace_all_rays_numba(
    origin: np.ndarray,
    directions: np.ndarray,
    world_types: np.ndarray,
    ssize: int,
    enemies: np.ndarray,
    num_enemies: int,
    max_dist: float,
    step_size: float,
    enemy_radius_sq: float,
    output: np.ndarray,
):
    """
    Trace all rays in parallel (numba-compiled with threading).
    """
    ox, oy, oz = origin[0], origin[1], origin[2]
    num_rays = directions.shape[0]

    for i in prange(num_rays):
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]

        dist, hit = trace_ray_numba(
            ox, oy, oz, dx, dy, dz,
            world_types, ssize, enemies, num_enemies,
            max_dist, step_size, enemy_radius_sq
        )

        output[i, 0] = dist
        output[i, 1] = hit / 2.0  # Normalize hit type to 0-1


@njit(cache=True)
def angles_to_direction(yaw: float, pitch: float) -> tuple:
    """Convert angles to direction vector."""
    yaw_rad = yaw * 0.017453292519943295  # math.pi / 180
    pitch_rad = pitch * 0.017453292519943295
    cos_pitch = math.cos(pitch_rad)
    dx = math.sin(yaw_rad) * cos_pitch
    dy = -math.cos(yaw_rad) * cos_pitch
    dz = math.sin(pitch_rad)
    return dx, dy, dz


@njit(cache=True)
def compute_ray_directions(
    base_yaw: float,
    base_pitch: float,
    ray_offsets: np.ndarray,
    output: np.ndarray,
):
    """Compute world-space ray directions from angle offsets."""
    num_rays = ray_offsets.shape[0]
    for i in range(num_rays):
        yaw = base_yaw + ray_offsets[i, 0]
        pitch = base_pitch + ray_offsets[i, 1]
        dx, dy, dz = angles_to_direction(yaw, pitch)
        output[i, 0] = dx
        output[i, 1] = dy
        output[i, 2] = dz


class NumbaRaycastObserver:
    """
    Numba-accelerated raycast observation.

    Configuration:
        - FOV rays: 41x17 = 697 rays (dense combat coverage)
        - Omni rays: 24 rays at ground level (360° navigation)
        - Total: 721 rays

    This should achieve 150+ fps on modern hardware.
    """

    PROCESS_NAME = "ac_client.exe"

    def __init__(
        self,
        # FOV configuration
        fov_rays_h: int = 41,
        fov_rays_v: int = 17,
        fov_h: float = 110.0,
        fov_v: float = 80.0,
        fov_max_dist: float = 300.0,

        # Omni configuration
        omni_rays: int = 24,
        omni_max_dist: float = 100.0,

        # Tracing config
        step_size: float = 2.0,
        enemy_hitbox_radius: float = 4.0,
    ):
        self.fov_rays_h = fov_rays_h
        self.fov_rays_v = fov_rays_v
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.fov_max_dist = fov_max_dist
        self.omni_rays = omni_rays
        self.omni_max_dist = omni_max_dist
        self.step_size = step_size
        self.enemy_radius_sq = enemy_hitbox_radius ** 2

        # Memory
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self._attached = False

        # World data
        self._ssize: int = 0
        self._world_ptr: int = 0
        self._world_types: Optional[np.ndarray] = None

        # Counts
        self.total_fov = fov_rays_h * fov_rays_v
        self.total_omni = omni_rays
        self.total_rays = self.total_fov + self.total_omni

        # Precompute ray angle offsets
        self._fov_offsets = self._compute_fov_offsets()
        self._omni_offsets = self._compute_omni_offsets()

        # Preallocate arrays
        self._fov_directions = np.zeros((self.total_fov, 3), dtype=np.float64)
        self._omni_directions = np.zeros((self.total_omni, 3), dtype=np.float64)
        self._fov_output = np.zeros((self.total_fov, 2), dtype=np.float64)
        self._omni_output = np.zeros((self.total_omni, 2), dtype=np.float64)
        self._origin = np.zeros(3, dtype=np.float64)
        self._enemies = np.zeros((32, 3), dtype=np.float64)

    def _compute_fov_offsets(self) -> np.ndarray:
        """Precompute FOV ray angle offsets."""
        offsets = []
        half_h = self.fov_h / 2
        half_v = self.fov_v / 2

        for v in range(self.fov_rays_v):
            pitch_off = half_v - (v / max(self.fov_rays_v - 1, 1)) * self.fov_v
            for h in range(self.fov_rays_h):
                yaw_off = -half_h + (h / max(self.fov_rays_h - 1, 1)) * self.fov_h
                offsets.append((yaw_off, pitch_off))

        return np.array(offsets, dtype=np.float64)

    def _compute_omni_offsets(self) -> np.ndarray:
        """Precompute omnidirectional ray angle offsets."""
        offsets = []
        for i in range(self.omni_rays):
            yaw_off = i * (360.0 / self.omni_rays)
            offsets.append((yaw_off, 0.0))  # Ground level
        return np.array(offsets, dtype=np.float64)

    def attach(self) -> bool:
        """Attach to game process."""
        try:
            self.pm = pymem.Pymem(self.PROCESS_NAME)
            self.module_base = pymem.process.module_from_name(
                self.pm.process_handle,
                self.PROCESS_NAME
            ).lpBaseOfDll

            self._refresh_world()
            self._attached = True

            print(f"[NumbaRaycast] Attached!")
            print(f"[NumbaRaycast] FOV: {self.fov_rays_h}x{self.fov_rays_v} = {self.total_fov} rays")
            print(f"[NumbaRaycast] Omni: {self.total_omni} rays")
            print(f"[NumbaRaycast] Total: {self.total_rays} rays")
            print(f"[NumbaRaycast] World size: {self._ssize}x{self._ssize}")

            # Warm up numba (first call compiles)
            print("[NumbaRaycast] Warming up JIT compilation...")
            self.get_observation()
            print("[NumbaRaycast] Ready!")

            return True

        except pymem.exception.ProcessNotFound:
            print(f"[NumbaRaycast] ERROR: {self.PROCESS_NAME} not found")
            return False
        except Exception as e:
            print(f"[NumbaRaycast] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detach(self):
        """Detach from process."""
        if self.pm:
            self.pm.close_process()
            self.pm = None
        self._attached = False

    def _refresh_world(self):
        """Read world geometry into numpy array."""
        if not self.pm:
            return

        try:
            self._ssize = self.pm.read_int(self.module_base + SSIZE_OFFSET)
            self._world_ptr = self.pm.read_int(self.module_base + WORLD_PTR_OFFSET)

            # Read all sqr types into array (first byte of each sqr is type)
            # This is the expensive operation - we cache it
            total_sqrs = self._ssize * self._ssize
            world_bytes = self.pm.read_bytes(self._world_ptr, total_sqrs * SQR_SIZE)

            # Extract just the type byte from each sqr
            self._world_types = np.array([
                world_bytes[i * SQR_SIZE] for i in range(total_sqrs)
            ], dtype=np.uint8)

        except Exception as e:
            print(f"[NumbaRaycast] World refresh failed: {e}")
            self._world_types = np.zeros(1, dtype=np.uint8)

    def _read_player_data(self) -> tuple:
        """Read player position and angles."""
        try:
            ptr = self.pm.read_int(self.module_base + PLAYER1_PTR_OFFSET)
            pos = (
                self.pm.read_float(ptr + POS_X),
                self.pm.read_float(ptr + POS_Y),
                self.pm.read_float(ptr + POS_Z),
            )
            yaw = self.pm.read_float(ptr + YAW)
            pitch = self.pm.read_float(ptr + PITCH)
            team = self.pm.read_int(ptr + TEAM)
            return pos, yaw, pitch, team
        except:
            return (0, 0, 0), 0, 0, 0

    def _read_enemies(self, own_team: int) -> int:
        """Read enemy positions into preallocated array. Returns count."""
        count = 0
        try:
            player1_ptr = self.pm.read_int(self.module_base + PLAYER1_PTR_OFFSET)
            array_ptr = self.pm.read_int(self.module_base + PLAYERS_ARRAY_OFFSET)
            player_count = self.pm.read_int(self.module_base + PLAYERS_COUNT_OFFSET)

            for i in range(min(player_count, 32)):
                ptr = self.pm.read_int(array_ptr + i * 4)
                if ptr == 0 or ptr == player1_ptr:
                    continue

                team = self.pm.read_int(ptr + TEAM)
                if team == own_team:
                    continue

                health = self.pm.read_int(ptr + HEALTH)
                if health <= 0:
                    continue

                self._enemies[count, 0] = self.pm.read_float(ptr + POS_X)
                self._enemies[count, 1] = self.pm.read_float(ptr + POS_Y)
                self._enemies[count, 2] = self.pm.read_float(ptr + POS_Z)
                count += 1

                if count >= 32:
                    break
        except:
            pass

        return count

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get raycast observations.

        Returns:
            fov_rays: (total_fov, 2) - [distance, hit_type] per ray
            omni_rays: (total_omni, 2) - [distance, hit_type] per ray
        """
        if not self._attached or self._world_types is None:
            return (
                np.zeros((self.total_fov, 2), dtype=np.float32),
                np.zeros((self.total_omni, 2), dtype=np.float32),
            )

        # Get player data
        pos, yaw, pitch, own_team = self._read_player_data()
        self._origin[0], self._origin[1], self._origin[2] = pos

        # Get enemies
        num_enemies = self._read_enemies(own_team)

        # Compute FOV ray directions
        compute_ray_directions(yaw, pitch, self._fov_offsets, self._fov_directions)

        # Trace FOV rays (parallel)
        trace_all_rays_numba(
            self._origin, self._fov_directions,
            self._world_types, self._ssize,
            self._enemies, num_enemies,
            self.fov_max_dist, self.step_size, self.enemy_radius_sq,
            self._fov_output
        )

        # Compute omni ray directions
        compute_ray_directions(yaw, 0, self._omni_offsets, self._omni_directions)

        # Trace omni rays (parallel)
        trace_all_rays_numba(
            self._origin, self._omni_directions,
            self._world_types, self._ssize,
            self._enemies, num_enemies,
            self.omni_max_dist, self.step_size, self.enemy_radius_sq,
            self._omni_output
        )

        return (
            self._fov_output.astype(np.float32),
            self._omni_output.astype(np.float32),
        )

    def get_flat_observation(self) -> np.ndarray:
        """Get all rays as flat array."""
        fov, omni = self.get_observation()
        return np.concatenate([fov.flatten(), omni.flatten()])

    def visualize_fov(self, fov_rays: np.ndarray) -> str:
        """ASCII visualization of FOV."""
        lines = []
        idx = 0
        for v in range(self.fov_rays_v):
            row = ""
            for h in range(self.fov_rays_h):
                dist, hit = fov_rays[idx]
                hit_type = int(hit * 2 + 0.1)

                if hit_type == HIT_ENEMY:
                    char = "E"
                elif hit_type == HIT_WALL:
                    if dist < 0.15:
                        char = "#"
                    elif dist < 0.3:
                        char = "+"
                    elif dist < 0.5:
                        char = "."
                    else:
                        char = ","
                else:
                    char = " "
                row += char
                idx += 1
            lines.append(row)
        return "\n".join(lines)


def benchmark():
    """Benchmark the numba raycast observer."""
    import time

    print("=" * 60)
    print("  NUMBA RAYCAST BENCHMARK")
    print("=" * 60)

    observer = NumbaRaycastObserver()

    if not observer.attach():
        return 1

    print("\n[*] Running benchmark (1000 iterations)...")

    times = []
    for _ in range(1000):
        start = time.perf_counter()
        fov, omni = observer.get_observation()
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000
    fps = 1000 / avg_ms

    print(f"\n" + "=" * 40)
    print(f"  RESULTS")
    print(f"=" * 40)
    print(f"  Total rays: {observer.total_rays}")
    print(f"  Average: {avg_ms:.2f}ms per observation")
    print(f"  Min: {min_ms:.2f}ms, Max: {max_ms:.2f}ms")
    print(f"  Achievable FPS: {fps:.0f}")
    print(f"=" * 40)

    # Show sample
    fov, omni = observer.get_observation()
    print(f"\n[*] Sample FOV visualization:")
    print(observer.visualize_fov(fov))

    # Count hits
    fov_walls = np.sum((fov[:, 1] > 0.4) & (fov[:, 1] < 0.6))
    fov_enemies = np.sum(fov[:, 1] > 0.9)

    print(f"\n[*] FOV hits: {fov_walls} walls, {fov_enemies} enemies")

    observer.detach()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(benchmark())
