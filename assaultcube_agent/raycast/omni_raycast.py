"""
Omnidirectional dense raycasting - 360° spatial awareness.

Rays blast in ALL directions at all times:
- Full 360° horizontal coverage
- Multiple vertical layers (-60° to +60°)
- No distinction between "FOV" and "behind" - agent sees everything

Configuration:
    horizontal_rays: 72 (every 5°)
    vertical_layers: 9 (-60° to +60° in 15° steps)
    Total: 648 rays covering full sphere

This gives true omnidirectional awareness like a LiDAR sensor.
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
)

# Cube types from AC's world.h
SOLID = 0       # Fully solid wall
CORNER = 1      # Half-solid corner
FHF = 2         # Floor heightfield
CHF = 3         # Ceiling heightfield
SPACE = 4       # Open space (but has floor/ceil)
SEMISOLID = 5   # Mipped solid

# In AC: 1 cube = 4 world units for floor/ceil height
CUBE_HEIGHT = 4.0

# Hit types
HIT_NOTHING = 0
HIT_WALL = 1
HIT_ENEMY = 2
HIT_SKY = 3  # Ray escaped into open sky (upward direction)


@njit(cache=True)
def trace_ray(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    world_data: np.ndarray,  # Now contains (type, floor, ceil) per sqr
    ssize: int,
    enemies: np.ndarray,
    num_enemies: int,
    max_dist: float,
    step_size: float,
    enemy_radius_sq: float,
) -> tuple:
    """Trace a single ray with proper floor/ceil collision. Returns (distance, hit_type)."""
    dist = 0.0
    vx, vy, vz = ox, oy, oz

    # Height tolerance for archways/transitions (in world units)
    # This prevents false collisions at room boundaries
    FLOOR_MARGIN = 2.0  # Allow ray to be slightly below floor
    CEIL_MARGIN = 2.0   # Allow ray to be slightly above ceiling

    while dist < max_dist:
        ix, iy = int(vx), int(vy)
        if ix < 0 or iy < 0 or ix >= ssize or iy >= ssize:
            # Escaped map bounds - sky if going up, wall otherwise
            if dz > 0.1:
                return dist, HIT_SKY
            return dist, HIT_WALL

        idx = iy * ssize + ix
        sqr_type = world_data[idx, 0]
        sqr_floor = world_data[idx, 1]
        sqr_ceil = world_data[idx, 2]

        # Solid types always block
        if sqr_type == SOLID or sqr_type == CORNER or sqr_type == SEMISOLID:
            return dist, HIT_WALL

        # For space/heightfield cubes, check floor/ceil heights
        # floor/ceil are in cubes, convert to world units
        floor_z = sqr_floor * CUBE_HEIGHT
        ceil_z = sqr_ceil * CUBE_HEIGHT

        # Ray blocked if significantly below floor or above ceiling
        # Use margin to handle archway transitions gracefully
        if vz < (floor_z - FLOOR_MARGIN):
            return dist, HIT_WALL
        if vz > (ceil_z + CEIL_MARGIN):
            # Above ceiling - sky if going up, wall otherwise
            if dz > 0.1:
                return dist, HIT_SKY
            return dist, HIT_WALL

        # Check enemy collision
        for i in range(num_enemies):
            dist_sq = (vx - enemies[i, 0])**2 + (vy - enemies[i, 1])**2 + (vz - enemies[i, 2])**2
            if dist_sq < enemy_radius_sq:
                return dist, HIT_ENEMY

        vx += dx * step_size
        vy += dy * step_size
        vz += dz * step_size
        dist += step_size

    # Reached max distance without hitting - sky if looking up
    if dz > 0.1:
        return max_dist, HIT_SKY
    return max_dist, HIT_NOTHING


@njit(parallel=True, cache=True)
def trace_all_rays(
    origin: np.ndarray,
    directions: np.ndarray,
    world_data: np.ndarray,  # (type, floor, ceil) per sqr
    ssize: int,
    enemies: np.ndarray,
    num_enemies: int,
    max_dist: float,
    step_size: float,
    enemy_radius_sq: float,
    output: np.ndarray,
):
    """Trace all rays in parallel."""
    ox, oy, oz = origin[0], origin[1], origin[2]
    num_rays = directions.shape[0]

    for i in prange(num_rays):
        dist, hit = trace_ray(
            ox, oy, oz,
            directions[i, 0], directions[i, 1], directions[i, 2],
            world_data, ssize, enemies, num_enemies,
            max_dist, step_size, enemy_radius_sq
        )
        # dist is raw distance in world units (for depth perception)
        # hit is normalized: 0=nothing, 0.33=wall, 0.67=enemy, 1.0=sky
        output[i, 0] = dist
        output[i, 1] = hit / 3.0


@njit(cache=True)
def compute_directions(
    base_yaw: float,
    ray_angles: np.ndarray,
    output: np.ndarray,
):
    """Compute world-space ray directions from angle offsets."""
    deg_to_rad = 0.017453292519943295
    num_rays = ray_angles.shape[0]

    for i in range(num_rays):
        yaw = (base_yaw + ray_angles[i, 0]) * deg_to_rad
        pitch = ray_angles[i, 1] * deg_to_rad
        cos_pitch = math.cos(pitch)
        output[i, 0] = math.sin(yaw) * cos_pitch
        output[i, 1] = -math.cos(yaw) * cos_pitch
        output[i, 2] = math.sin(pitch)


class OmniRaycastObserver:
    """
    Full 360° omnidirectional raycasting.

    Default configuration:
        - 72 horizontal rays (every 5°, full 360°)
        - 9 vertical layers (-60° to +60°, every 15°)
        - Total: 648 rays

    The observation is organized as:
        [v_layer_0 (72 rays), v_layer_1 (72 rays), ..., v_layer_8 (72 rays)]

    Each ray: [distance (0-1), hit_type (0=nothing, 0.5=wall, 1=enemy)]
    """

    PROCESS_NAME = "ac_client.exe"

    def __init__(
        self,
        horizontal_rays: int = 72,      # Every 5° = 360°
        vertical_layers: int = 9,       # -60° to +60° in 15° steps
        vertical_min: float = -60.0,
        vertical_max: float = 60.0,
        max_distance: float = 250.0,
        step_size: float = 2.0,
        enemy_hitbox_radius: float = 4.0,
    ):
        self.h_rays = horizontal_rays
        self.v_layers = vertical_layers
        self.v_min = vertical_min
        self.v_max = vertical_max
        self.max_dist = max_distance
        self.step_size = step_size
        self.enemy_radius_sq = enemy_hitbox_radius ** 2

        self.total_rays = horizontal_rays * vertical_layers

        # Memory
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self._attached = False
        self._ssize: int = 0
        self._world_ptr: int = 0
        self._world_data: Optional[np.ndarray] = None  # (type, floor, ceil) per sqr

        # Precompute ray angle offsets (relative to player yaw, absolute pitch)
        self._ray_angles = self._compute_ray_angles()

        # Preallocate arrays
        self._directions = np.zeros((self.total_rays, 3), dtype=np.float64)
        self._output = np.zeros((self.total_rays, 2), dtype=np.float64)
        self._origin = np.zeros(3, dtype=np.float64)
        self._enemies = np.zeros((32, 3), dtype=np.float64)

    def _compute_ray_angles(self) -> np.ndarray:
        """Compute all ray angle offsets (yaw_offset, pitch)."""
        angles = []

        # Vertical layers
        if self.v_layers > 1:
            pitches = np.linspace(self.v_min, self.v_max, self.v_layers)
        else:
            pitches = [0.0]

        # Horizontal rays (full 360°)
        yaw_step = 360.0 / self.h_rays

        for pitch in pitches:
            for h in range(self.h_rays):
                yaw_offset = h * yaw_step  # 0° to 360°
                angles.append((yaw_offset, pitch))

        return np.array(angles, dtype=np.float64)

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

            print(f"[OmniRaycast] Attached!")
            print(f"[OmniRaycast] Horizontal: {self.h_rays} rays (every {360/self.h_rays:.1f}°)")
            print(f"[OmniRaycast] Vertical: {self.v_layers} layers ({self.v_min}° to {self.v_max}°)")
            print(f"[OmniRaycast] Total: {self.total_rays} rays (full 360° coverage)")
            print(f"[OmniRaycast] World: {self._ssize}x{self._ssize}")

            # Warm up JIT
            print("[OmniRaycast] Warming up JIT...")
            self.get_observation()
            print("[OmniRaycast] Ready!")

            return True

        except pymem.exception.ProcessNotFound:
            print(f"[OmniRaycast] ERROR: {self.PROCESS_NAME} not found")
            return False
        except Exception as e:
            print(f"[OmniRaycast] ERROR: {e}")
            return False

    def detach(self):
        """Detach from process."""
        if self.pm:
            self.pm.close_process()
            self.pm = None
        self._attached = False

    def _refresh_world(self):
        """Read world geometry with floor/ceil for proper collision."""
        if not self.pm:
            return
        try:
            self._ssize = self.pm.read_int(self.module_base + SSIZE_OFFSET)
            self._world_ptr = self.pm.read_int(self.module_base + WORLD_PTR_OFFSET)

            total_sqrs = self._ssize * self._ssize
            world_bytes = self.pm.read_bytes(self._world_ptr, total_sqrs * SQR_SIZE)

            # Read type (byte 0), floor (byte 1, signed), ceil (byte 2, signed) for each sqr
            self._world_data = np.zeros((total_sqrs, 3), dtype=np.float64)
            for i in range(total_sqrs):
                base = i * SQR_SIZE
                self._world_data[i, 0] = world_bytes[base]  # type (uchar)
                # floor and ceil are signed chars (-128 to 127)
                floor_val = world_bytes[base + 1]
                ceil_val = world_bytes[base + 2]
                # Convert to signed
                self._world_data[i, 1] = floor_val if floor_val < 128 else floor_val - 256
                self._world_data[i, 2] = ceil_val if ceil_val < 128 else ceil_val - 256
        except Exception as e:
            print(f"[OmniRaycast] World refresh error: {e}")
            self._world_data = np.zeros((1, 3), dtype=np.float64)

    def _read_player(self) -> tuple:
        """Read player position and yaw."""
        try:
            ptr = self.pm.read_int(self.module_base + PLAYER1_PTR_OFFSET)
            pos = (
                self.pm.read_float(ptr + POS_X),
                self.pm.read_float(ptr + POS_Y),
                self.pm.read_float(ptr + POS_Z),
            )
            yaw = self.pm.read_float(ptr + YAW)
            team = self.pm.read_int(ptr + TEAM)
            return pos, yaw, team
        except:
            return (0, 0, 0), 0, 0

    def _read_enemies(self, own_team: int) -> int:
        """Read enemy positions. Returns count."""
        count = 0
        try:
            player1_ptr = self.pm.read_int(self.module_base + PLAYER1_PTR_OFFSET)
            array_ptr = self.pm.read_int(self.module_base + PLAYERS_ARRAY_OFFSET)
            player_count = self.pm.read_int(self.module_base + PLAYERS_COUNT_OFFSET)

            for i in range(min(player_count, 32)):
                ptr = self.pm.read_int(array_ptr + i * 4)
                if ptr == 0 or ptr == player1_ptr:
                    continue
                if self.pm.read_int(ptr + TEAM) == own_team:
                    continue
                if self.pm.read_int(ptr + HEALTH) <= 0:
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

    def get_observation(self) -> np.ndarray:
        """
        Get omnidirectional raycast observation.

        Returns:
            (total_rays, 2) array of [distance, hit_type]
        """
        if not self._attached or self._world_data is None:
            return np.zeros((self.total_rays, 2), dtype=np.float32)

        pos, yaw, own_team = self._read_player()
        self._origin[0], self._origin[1], self._origin[2] = pos

        num_enemies = self._read_enemies(own_team)

        compute_directions(yaw, self._ray_angles, self._directions)

        trace_all_rays(
            self._origin, self._directions,
            self._world_data, self._ssize,
            self._enemies, num_enemies,
            self.max_dist, self.step_size, self.enemy_radius_sq,
            self._output
        )

        return self._output.astype(np.float32)

    def get_layered_observation(self) -> np.ndarray:
        """
        Get observation reshaped as (v_layers, h_rays, 2).

        Useful for visualization or CNN processing.
        """
        obs = self.get_observation()
        return obs.reshape(self.v_layers, self.h_rays, 2)

    def visualize_layer(self, layer_idx: int, obs: np.ndarray) -> str:
        """Visualize a single horizontal layer as a ring."""
        if obs.ndim == 2:
            obs = obs.reshape(self.v_layers, self.h_rays, 2)

        layer = obs[layer_idx]
        chars = []
        for i in range(self.h_rays):
            dist, hit = layer[i]
            hit_type = int(hit * 2 + 0.1)

            if hit_type == HIT_ENEMY:
                char = "E"
            elif hit_type == HIT_WALL:
                if dist < 0.2:
                    char = "#"
                elif dist < 0.4:
                    char = "+"
                elif dist < 0.6:
                    char = "."
                else:
                    char = ","
            else:
                char = " "
            chars.append(char)

        return "".join(chars)

    def visualize_radar(self, obs: np.ndarray) -> str:
        """Create a proper radar visualization showing walls in all directions."""
        if obs.ndim == 2:
            obs = obs.reshape(self.v_layers, self.h_rays, 2)

        mid_layer = self.v_layers // 2
        layer = obs[mid_layer]

        # Radar dimensions
        size = 31
        center = size // 2
        max_radius = center - 1

        # Create grid
        grid = [[' ' for _ in range(size)] for _ in range(size)]
        grid[center][center] = '@'  # Player

        # Draw rays as lines from center outward
        for i in range(self.h_rays):
            dist, hit = layer[i]
            hit_type = int(hit * 2 + 0.1)

            # Angle for this ray (0 = forward/up in viz)
            angle = (i / self.h_rays) * 2 * math.pi

            # Distance determines how far to draw
            # Invert: close walls = short line, far walls = long line
            line_length = int((1.0 - dist) * max_radius) + 1  # At least 1

            # Draw line from center outward
            for r in range(1, min(line_length + 1, max_radius + 1)):
                x = int(center + r * math.sin(angle))
                y = int(center - r * math.cos(angle))

                if 0 <= x < size and 0 <= y < size and grid[y][x] == ' ':
                    if hit_type == HIT_ENEMY:
                        grid[y][x] = 'E'
                    elif hit_type == HIT_WALL:
                        if r == line_length:  # End of line = wall
                            grid[y][x] = '#'
                        else:  # Path to wall
                            grid[y][x] = '.'

        lines = [''.join(row) for row in grid]
        return '\n'.join(lines)


def benchmark():
    """Benchmark omnidirectional raycasting."""
    import time

    print("=" * 60)
    print("  OMNIDIRECTIONAL RAYCAST BENCHMARK")
    print("=" * 60)

    observer = OmniRaycastObserver()

    if not observer.attach():
        return 1

    print("\n[*] Running benchmark (1000 iterations)...")

    times = []
    for _ in range(1000):
        start = time.perf_counter()
        obs = observer.get_observation()
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    fps = 1000 / avg_ms

    print(f"\n{'=' * 40}")
    print(f"  RESULTS")
    print(f"{'=' * 40}")
    print(f"  Total rays: {observer.total_rays}")
    print(f"  Average: {avg_ms:.2f}ms")
    print(f"  FPS: {fps:.0f}")
    print(f"{'=' * 40}")

    # Visualize
    obs = observer.get_observation()
    print(f"\n[*] Middle layer (horizon):")
    print(observer.visualize_layer(observer.v_layers // 2, obs))

    print(f"\n[*] Radar view:")
    print(observer.visualize_radar(obs))

    observer.detach()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(benchmark())
