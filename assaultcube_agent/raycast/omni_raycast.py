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
    DEFAULT_SKY_TEXTURE,
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

# Player body height offset - positions read from memory are at foot level,
# but we want to check collision at body center for proper detection
PLAYER_BODY_CENTER_OFFSET = 4.0  # Half player height (~8 units total)

# Hit types
HIT_NOTHING = 0
HIT_WALL = 1
HIT_ENEMY = 2
HIT_SKY = 3  # Ray escaped into open sky (upward direction)


@njit(cache=True)
def trace_ray(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    world_data: np.ndarray,  # (type, floor, ceil, ctex, vdelta) per sqr
    ssize: int,
    enemies: np.ndarray,
    num_enemies: int,
    max_dist: float,
    step_size: float,
    enemy_radius_sq: float,
    sky_texture: int,
) -> tuple:
    """
    Trace a single ray with proper floor/ceil collision.
    Based on AC's raycube() from physics.cpp.
    Returns (distance, hit_type).
    """
    dist = 0.0
    vx, vy, vz = ox, oy, oz

    while dist < max_dist:
        ix, iy = int(vx), int(vy)
        if ix < 0 or iy < 0 or ix >= ssize or iy >= ssize:
            # Escaped map bounds
            return dist, HIT_SKY if dz > 0.0 else HIT_NOTHING

        idx = iy * ssize + ix
        sqr_type = world_data[idx, 0]
        sqr_floor = world_data[idx, 1]
        sqr_ceil = world_data[idx, 2]
        sqr_ctex = int(world_data[idx, 3])
        sqr_vdelta = world_data[idx, 4]

        # Apply vdelta adjustment for heightfield cubes (from AC physics.cpp)
        floor_z = sqr_floor
        ceil_z = sqr_ceil
        if sqr_type == FHF:
            floor_z -= sqr_vdelta / 4.0
        if sqr_type == CHF:
            ceil_z += sqr_vdelta / 4.0

        # Check collision (matching AC's raycube logic)
        if sqr_type == SOLID or sqr_type == CORNER or sqr_type == SEMISOLID:
            return dist, HIT_WALL

        if vz < floor_z or vz > ceil_z:
            # Hit floor or ceiling
            # Check for sky: ceiling texture is sky and we're above ceiling going up
            if vz > ceil_z and sqr_ctex == sky_texture and dz > 0:
                return dist, HIT_SKY
            return dist, HIT_WALL

        # Check enemy collision (before stepping further)
        for i in range(num_enemies):
            dist_sq = (vx - enemies[i, 0])**2 + (vy - enemies[i, 1])**2 + (vz - enemies[i, 2])**2
            if dist_sq < enemy_radius_sq:
                return dist, HIT_ENEMY

        # Step forward
        vx += dx * step_size
        vy += dy * step_size
        vz += dz * step_size
        dist += step_size

    # Reached max distance without hitting anything solid
    if dz > 0.0:
        return max_dist, HIT_SKY
    return max_dist, HIT_NOTHING


@njit(parallel=True, cache=True)
def trace_all_rays(
    origin: np.ndarray,
    directions: np.ndarray,
    world_data: np.ndarray,  # (type, floor, ceil, ctex) per sqr
    ssize: int,
    enemies: np.ndarray,
    num_enemies: int,
    max_dist: float,
    step_size: float,
    enemy_radius_sq: float,
    sky_texture: int,
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
            max_dist, step_size, enemy_radius_sq, sky_texture
        )
        # dist is raw distance in world units (for depth perception)
        # hit is normalized: 0=nothing, 0.33=wall, 0.67=enemy, 1.0=sky
        output[i, 0] = dist
        output[i, 1] = hit / 3.0


@njit(cache=True)
def compute_directions(
    base_yaw: float,
    base_pitch: float,
    ray_angles: np.ndarray,
    output: np.ndarray,
):
    """Compute world-space ray directions from angle offsets + player view angles."""
    deg_to_rad = 0.017453292519943295
    num_rays = ray_angles.shape[0]

    for i in range(num_rays):
        yaw = (base_yaw + ray_angles[i, 0]) * deg_to_rad
        # Add player's pitch to the ray's vertical offset
        pitch = (base_pitch + ray_angles[i, 1]) * deg_to_rad
        cos_pitch = math.cos(pitch)
        output[i, 0] = math.sin(yaw) * cos_pitch
        output[i, 1] = -math.cos(yaw) * cos_pitch
        output[i, 2] = math.sin(pitch)


class OmniRaycastObserver:
    """
    FOV-focused raycasting matching numba_raycast.py configuration.

    Default configuration (matches numba_raycast.py):
        - 41 horizontal rays within 120° FOV
        - 17 vertical layers within 80° vertical span
        - Total: 697 rays (matches fov_rays_h × fov_rays_v)
        - Max distance: 350 units

    Rays are densely packed within the FOV for proper depth visualization.

    The observation is organized as:
        [v_layer_0 (41 rays), v_layer_1 (41 rays), ..., v_layer_16 (41 rays)]

    Each ray: [distance (world units), hit_type (0=nothing, 0.33=wall, 0.67=enemy, 1.0=sky)]
    """

    PROCESS_NAME = "ac_client.exe"

    def __init__(
        self,
        # Match numba_raycast.py FOV config for consistency
        horizontal_rays: int = 41,      # Match fov_rays_h
        vertical_layers: int = 17,      # Match fov_rays_v
        fov_h: float = 120.0,           # Horizontal FOV in degrees
        fov_v: float = 80.0,            # Vertical FOV in degrees
        max_distance: float = 350.0,    # Match fov_max_dist
        step_size: float = 0.5,         # Small enough to not skip 1-unit walls
        enemy_hitbox_radius: float = 5.0,  # Increased from 4.0 for reliable body detection
    ):
        self.h_rays = horizontal_rays
        self.v_layers = vertical_layers
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.max_dist = max_distance
        self.step_size = step_size
        self.enemy_radius_sq = enemy_hitbox_radius ** 2

        self.total_rays = horizontal_rays * vertical_layers
        self.sky_texture = DEFAULT_SKY_TEXTURE

        # Memory
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self._attached = False
        self._ssize: int = 0
        self._world_ptr: int = 0
        self._world_data: Optional[np.ndarray] = None  # (type, floor, ceil, ctex) per sqr

        # Precompute ray angle offsets (relative to player yaw, absolute pitch)
        self._ray_angles = self._compute_ray_angles()

        # Preallocate arrays
        self._directions = np.zeros((self.total_rays, 3), dtype=np.float64)
        self._output = np.zeros((self.total_rays, 2), dtype=np.float64)
        self._origin = np.zeros(3, dtype=np.float64)
        self._enemies = np.zeros((32, 3), dtype=np.float64)

    def _compute_ray_angles(self) -> np.ndarray:
        """Compute all ray angle offsets (yaw_offset, pitch_offset).

        Rays are packed within the FOV (matching numba_raycast.py layout):
        - Horizontal: -fov_h/2 to +fov_h/2
        - Vertical: +fov_v/2 to -fov_v/2 (top to bottom)
        """
        angles = []
        half_h = self.fov_h / 2
        half_v = self.fov_v / 2

        for v in range(self.v_layers):
            # Pitch: top (+half_v) to bottom (-half_v)
            pitch_off = half_v - (v / max(self.v_layers - 1, 1)) * self.fov_v
            for h in range(self.h_rays):
                # Yaw: left (-half_h) to right (+half_h)
                yaw_off = -half_h + (h / max(self.h_rays - 1, 1)) * self.fov_h
                angles.append((yaw_off, pitch_off))

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
            print(f"[OmniRaycast] Horizontal: {self.h_rays} rays within {self.fov_h}° FOV")
            print(f"[OmniRaycast] Vertical: {self.v_layers} layers within {self.fov_v}° FOV")
            print(f"[OmniRaycast] Total: {self.total_rays} rays (FOV-focused, matching numba_raycast)")
            print(f"[OmniRaycast] World: {self._ssize}x{self._ssize}")

            # Debug ceiling textures
            self.debug_ceiling_textures()

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
        """Read world geometry with floor/ceil/vdelta/ctex for proper collision and sky detection."""
        if not self.pm:
            return
        try:
            self._ssize = self.pm.read_int(self.module_base + SSIZE_OFFSET)
            self._world_ptr = self.pm.read_int(self.module_base + WORLD_PTR_OFFSET)

            total_sqrs = self._ssize * self._ssize
            world_bytes = self.pm.read_bytes(self._world_ptr, total_sqrs * SQR_SIZE)

            # FAST: Convert to numpy array and reshape for vectorized extraction
            # sqr struct layout (from world.h):
            # 0: type, 1: floor, 2: ceil, 3: wtex, 4: ftex, 5: ctex, 6-8: rgb, 9: vdelta, ...
            raw = np.frombuffer(world_bytes, dtype=np.uint8).reshape(total_sqrs, SQR_SIZE)

            self._world_data = np.zeros((total_sqrs, 5), dtype=np.float64)
            self._world_data[:, 0] = raw[:, 0]  # type (uchar)

            # floor and ceil are signed chars (-128 to 127) - vectorized conversion
            floor_u8 = raw[:, 1].astype(np.int16)
            ceil_u8 = raw[:, 2].astype(np.int16)
            self._world_data[:, 1] = np.where(floor_u8 < 128, floor_u8, floor_u8 - 256)
            self._world_data[:, 2] = np.where(ceil_u8 < 128, ceil_u8, ceil_u8 - 256)

            self._world_data[:, 3] = raw[:, 5]  # ctex (ceiling texture) - byte 5!
            self._world_data[:, 4] = raw[:, 9]  # vdelta for heightfields
        except Exception as e:
            print(f"[OmniRaycast] World refresh error: {e}")
            self._world_data = np.zeros((1, 5), dtype=np.float64)

    def _read_player(self) -> tuple:
        """Read player position, yaw, pitch, and team."""
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
        """Read enemy positions. Returns count.

        Note: Enemy Z position is offset by PLAYER_BODY_CENTER_OFFSET to check
        collision at body center rather than feet. This ensures rays at chest/head
        height properly detect enemies.
        """
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
                # Add offset to check at body center, not feet
                self._enemies[count, 2] = self.pm.read_float(ptr + POS_Z) + PLAYER_BODY_CENTER_OFFSET
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

        pos, yaw, pitch, own_team = self._read_player()
        self._origin[0], self._origin[1], self._origin[2] = pos
        # Offset Z to eye level (same as enemy body center offset)
        self._origin[2] += PLAYER_BODY_CENTER_OFFSET

        num_enemies = self._read_enemies(own_team)

        compute_directions(yaw, pitch, self._ray_angles, self._directions)

        trace_all_rays(
            self._origin, self._directions,
            self._world_data, self._ssize,
            self._enemies, num_enemies,
            self.max_dist, self.step_size, self.enemy_radius_sq,
            self.sky_texture, self._output
        )

        return self._output.astype(np.float32)

    def get_layered_observation(self) -> np.ndarray:
        """
        Get observation reshaped as (v_layers, h_rays, 2).

        Useful for visualization or CNN processing.
        """
        obs = self.get_observation()
        return obs.reshape(self.v_layers, self.h_rays, 2)

    def debug_ceiling_textures(self):
        """Debug: Print ceiling texture distribution in current map."""
        if self._world_data is None:
            print("[DEBUG] No world data loaded")
            return

        from collections import Counter
        ctex_counts = Counter()
        ceil_heights = Counter()

        for i in range(len(self._world_data)):
            ctex = int(self._world_data[i, 3])
            ceil = int(self._world_data[i, 2])
            ctex_counts[ctex] += 1
            ceil_heights[ceil] += 1

        print(f"\n[DEBUG] Ceiling Texture Distribution (top 10):")
        for tex, count in ctex_counts.most_common(10):
            print(f"  Texture {tex}: {count} sqrs")

        print(f"\n[DEBUG] Ceiling Height Distribution (top 10):")
        for height, count in ceil_heights.most_common(10):
            print(f"  Ceiling {height} cubes ({height*4} units): {count} sqrs")

    def visualize_layer(self, layer_idx: int, obs: np.ndarray) -> str:
        """Visualize a single horizontal layer as a ring."""
        if obs.ndim == 2:
            obs = obs.reshape(self.v_layers, self.h_rays, 2)

        layer = obs[layer_idx]
        chars = []
        for i in range(self.h_rays):
            raw_dist, hit = layer[i]
            hit_type = int(hit * 3 + 0.5)  # Decode: 0=nothing, 1=wall, 2=enemy, 3=sky
            # Normalize distance (raw is 0-max_dist world units)
            dist = min(1.0, raw_dist / self.max_dist)

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
            elif hit_type == HIT_SKY:
                char = "~"
            else:
                char = " "
            chars.append(char)

        return "".join(chars)

    def visualize_radar(self, obs: np.ndarray) -> str:
        """Create a proper radar visualization showing walls in all directions.

        Uses middle layer for walls, but checks ALL layers for enemies so they
        show up even if at different vertical angles.
        """
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

        # First pass: Check ALL layers for enemies at each horizontal angle
        enemy_info = {}  # horizontal_idx -> (distance, layer_idx)
        for v_idx in range(self.v_layers):
            for h_idx in range(self.h_rays):
                raw_dist, hit = obs[v_idx, h_idx]
                hit_type = int(hit * 3 + 0.5)
                if hit_type == HIT_ENEMY:
                    # Track closest enemy at this horizontal angle
                    if h_idx not in enemy_info or raw_dist < enemy_info[h_idx][0]:
                        enemy_info[h_idx] = (raw_dist, v_idx)

        # Draw rays as lines from center outward
        for i in range(self.h_rays):
            raw_dist, hit = layer[i]
            hit_type = int(hit * 3 + 0.5)  # Decode: 0=nothing, 1=wall, 2=enemy, 3=sky
            # Normalize distance (raw is 0-max_dist world units)
            dist = min(1.0, raw_dist / self.max_dist)

            # Check if enemy was found at this angle (any layer)
            has_enemy = i in enemy_info
            enemy_dist = enemy_info[i][0] / self.max_dist if has_enemy else 0

            # Angle for this ray (0 = forward/up in viz)
            angle = (i / self.h_rays) * 2 * math.pi

            # Distance determines how far to draw (closer = longer line from center)
            line_length = int(dist * max_radius) + 1  # At least 1

            # Draw line from center outward
            for r in range(1, min(line_length + 1, max_radius + 1)):
                x = int(center + r * math.sin(angle))
                y = int(center - r * math.cos(angle))

                if 0 <= x < size and 0 <= y < size and grid[y][x] == ' ':
                    # Check if this is where enemy is (from any layer)
                    if has_enemy and abs(r / max_radius - enemy_dist) < 0.1:
                        grid[y][x] = 'E'
                    elif hit_type == HIT_ENEMY:
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

    # Debug: Show player position and floor/ceil at their location
    pos, yaw, pitch, _ = observer._read_player()
    ix, iy = int(pos[0]), int(pos[1])
    if 0 <= ix < observer._ssize and 0 <= iy < observer._ssize:
        idx = iy * observer._ssize + ix
        sqr_floor = observer._world_data[idx, 1]
        sqr_ceil = observer._world_data[idx, 2]
        print(f"\n[*] Player position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print(f"[*] Player yaw: {yaw:.1f}°, pitch: {pitch:.1f}°")
        print(f"[*] Floor at player: {sqr_floor:.0f}, Ceiling: {sqr_ceil:.0f}")
        print(f"[*] Height above floor: {pos[2] - sqr_floor:.1f}, below ceiling: {sqr_ceil - pos[2]:.1f}")

    # Debug: Show distance for rays pointing up vs down
    obs_3d = obs.reshape(observer.v_layers, observer.h_rays, 2)
    middle = observer.v_layers // 2
    bottom = 0  # Most downward layer (-60°)
    top = observer.v_layers - 1  # Most upward layer (+60°)

    # Average distances for forward ray (index 0)
    down_dist = obs_3d[bottom, 0, 0]
    mid_dist = obs_3d[middle, 0, 0]
    up_dist = obs_3d[top, 0, 0]
    print(f"\n[*] Forward ray distances by vertical layer:")
    print(f"    DOWN (-60°): {down_dist:.1f} units")
    print(f"    MID  (  0°): {mid_dist:.1f} units")
    print(f"    UP   (+60°): {up_dist:.1f} units")

    print(f"\n[*] Middle layer (horizon):")
    print(observer.visualize_layer(observer.v_layers // 2, obs))

    print(f"\n[*] Radar view:")
    print(observer.visualize_radar(obs))

    observer.detach()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(benchmark())
