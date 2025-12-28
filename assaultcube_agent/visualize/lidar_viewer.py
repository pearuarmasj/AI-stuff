"""
Real-time Depth Perception Visualization for AssaultCube Agent.

Two view modes (press V to toggle):
1. RADAR VIEW: Top-down 360° view showing all rays
2. FOV VIEW: First-person depth viewport

Color coding:
- Sky: Blue gradient
- Wall: Gray gradient (brighter = closer)
- Enemy: Red (brighter = closer)
- Nothing: Dark void

Runs in a separate process so it doesn't affect training speed.

Usage:
    python -m assaultcube_agent.visualize.lidar_viewer
"""

import math
import time
import multiprocessing as mp
from typing import Optional

import numpy as np
from numba import njit, prange

try:
    import pygame
    import pygame.surfarray
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not installed. Install with: pip install pygame")

# Import EnemyDetector for standalone test
try:
    from ..raycast.enemy_detector import EnemyDetector
    ENEMY_DETECTOR_AVAILABLE = True
except ImportError:
    ENEMY_DETECTOR_AVAILABLE = False


# =============================================================================
# NUMBA-ACCELERATED RADAR RENDERING
# =============================================================================

@njit(cache=True)
def _get_color_for_ray(norm_distance: float, hit_type: float, max_dist: float) -> tuple:
    """Get RGB color for a ray (numba-compiled).

    Args:
        norm_distance: Already normalized distance (0-1) from raycast
        hit_type: Normalized hit type (0=nothing, 0.33=wall, 0.67=enemy, 1.0=sky)
        max_dist: Unused, kept for API compatibility
    """
    # Distance is already normalized 0-1, don't divide again!
    depth = min(1.0, max(0.0, norm_distance))
    inv_depth = 1.0 - depth  # Closer = higher value = brighter

    if hit_type > 0.9:  # Sky
        base = 60
        intensity = int(base + inv_depth * 140)
        return (intensity // 2, intensity // 2 + 30, min(255, intensity + 60))
    elif hit_type > 0.6:  # Enemy
        base = 100
        intensity = int(base + inv_depth * 155)
        return (min(255, intensity), 30, 30)
    elif hit_type > 0.25:  # Wall
        base = 30
        intensity = int(base + inv_depth * 200)
        return (intensity, intensity, min(255, intensity + 10))
    else:  # Nothing (max distance, no hit)
        return (20, 20, 25)


@njit(cache=True)
def _draw_line_to_array(
    pixels: np.ndarray,
    cx: int, cy: int,
    end_x: int, end_y: int,
    r: int, g: int, b: int,
    thickness: int = 1
):
    """Draw a line on pixel array using Bresenham's algorithm (numba-compiled)."""
    height, width = pixels.shape[0], pixels.shape[1]

    dx = abs(end_x - cx)
    dy = abs(end_y - cy)
    sx = 1 if cx < end_x else -1
    sy = 1 if cy < end_y else -1
    err = dx - dy

    x, y = cx, cy

    while True:
        # Draw pixel with thickness
        for tx in range(-thickness//2, thickness//2 + 1):
            for ty in range(-thickness//2, thickness//2 + 1):
                px, py = x + tx, y + ty
                if 0 <= px < width and 0 <= py < height:
                    pixels[py, px, 0] = r
                    pixels[py, px, 1] = g
                    pixels[py, px, 2] = b

        if x == end_x and y == end_y:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


@njit(parallel=True, cache=True)
def _render_radar_fast(
    pixels: np.ndarray,
    omni_rays: np.ndarray,
    num_omni_rays: int,
    center_x: int,
    center_y: int,
    max_radius: int,
    max_dist: float,
    fov_degrees: float,
):
    """
    Render radar view to pixel array using parallel numba (GPU-like speed).

    Uses 360° omni rays (separate from FOV rays).
    Ray 0 = forward (0° offset from player yaw) = UP on screen
    Ray N = N * (360/num_rays)° clockwise from forward

    This replaces thousands of pygame.draw.line() calls with direct pixel manipulation.
    """
    # Draw all omni rays in parallel
    for i in prange(num_omni_rays):
        # Ray angle offset from forward (player yaw)
        # Ray 0 = 0° (forward), ray 9 = 90° (right), etc.
        angle_offset_deg = i * (360.0 / num_omni_rays)

        # Convert to screen angle:
        # Forward (0°) should be UP (-90° in screen coords)
        # Right (90°) should be RIGHT (0° in screen coords)
        screen_angle_rad = (angle_offset_deg - 90.0) * 0.017453292519943295

        distance = omni_rays[i, 0]
        hit_type = omni_rays[i, 1]

        # Calculate end point (distance is already normalized 0-1 from raycast)
        ray_length = distance * max_radius
        end_x = int(center_x + math.cos(screen_angle_rad) * ray_length)
        end_y = int(center_y + math.sin(screen_angle_rad) * ray_length)

        # Get color - unnormalize distance for color calc
        actual_dist = distance * max_dist
        r, g, b = _get_color_for_ray(actual_dist, hit_type, max_dist)

        # Determine line thickness based on hit type
        thickness = 2
        if hit_type > 0.9:  # Sky
            thickness = 2
        elif hit_type > 0.6:  # Enemy
            thickness = 4

        # Draw line to pixel array
        _draw_line_to_array(pixels, center_x, center_y, end_x, end_y, r, g, b, thickness)


@njit(parallel=True, cache=True)
def _render_fov_fast(
    pixels: np.ndarray,
    rays: np.ndarray,
    h_rays: int,
    v_layers: int,
    vp_x: int, vp_y: int,
    vp_width: int, vp_height: int,
    fov_rays: int,
    half_fov_rays: int,
    max_dist: float,
):
    """Render FOV viewport to pixel array using parallel numba.

    Rays are now FOV-focused (packed within FOV, matching numba_raycast layout):
    - All h_rays are within the horizontal FOV
    - All v_layers are within the vertical FOV
    - Ray order: left-to-right for each row, top-to-bottom rows
    """
    cell_width = vp_width / h_rays
    cell_height = vp_height / v_layers

    # Process each layer in parallel
    for layer in prange(v_layers):
        layer_start = layer * h_rays
        # Layer 0 = top of FOV (highest pitch), render at top of viewport
        screen_row = layer
        y_start = int(vp_y + screen_row * cell_height)
        y_end = int(vp_y + (screen_row + 1) * cell_height)

        # All rays in this layer are within FOV, render left-to-right
        for h in range(h_rays):
            ray_idx = layer_start + h
            distance = rays[ray_idx, 0]
            hit_type = rays[ray_idx, 1]

            x_start = int(vp_x + h * cell_width)
            x_end = int(vp_x + (h + 1) * cell_width)
            r, g, b = _get_color_for_ray(distance, hit_type, max_dist)

            for py in range(y_start, min(y_end + 1, pixels.shape[0])):
                for px in range(x_start, min(x_end + 1, pixels.shape[1])):
                    if 0 <= px < pixels.shape[1] and 0 <= py < pixels.shape[0]:
                        pixels[py, px, 0] = r
                        pixels[py, px, 1] = g
                        pixels[py, px, 2] = b


# View modes
VIEW_RADAR = 0
VIEW_FOV = 1

# Colors (RGB)
COLOR_BG = (10, 10, 15)
COLOR_VOID = (15, 15, 20)  # Dark void for max distance/nothing
COLOR_TEXT = (200, 200, 200)
COLOR_CROSSHAIR = (50, 255, 50)
COLOR_BORDER = (40, 40, 50)
COLOR_GRID = (30, 30, 40)
COLOR_PLAYER = (50, 255, 50)
COLOR_FOV_LINE = (60, 60, 80)
COLOR_ENEMY = (255, 50, 50)
COLOR_ENEMY_GLOW = (255, 100, 100)


class DepthViewer:
    """
    Real-time depth perception visualization using pygame.

    Two view modes (press V to toggle):
    - RADAR: Top-down 360° view showing all rays
    - FOV: First-person depth viewport

    Color coding:
    - Sky: Blue gradient
    - Wall: Gray gradient (brighter = closer)
    - Enemy: Red (brighter = closer)
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        # FOV config (for viewport)
        horizontal_rays: int = 41,      # Match fov_rays_h
        vertical_layers: int = 17,      # Match fov_rays_v
        fov_degrees: float = 120.0,     # Match fov_h
        max_distance: float = 350.0,    # Match fov_max_dist
        # Omni config (for radar) - 360° coverage
        omni_rays: int = 72,            # 5° per ray over 360°
        omni_max_dist: float = 150.0,   # Radar max distance
    ):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for visualization")

        self.width = width
        self.height = height
        # FOV (viewport)
        self.h_rays = horizontal_rays
        self.v_layers = vertical_layers
        self.fov_degrees = fov_degrees
        self.max_dist = max_distance
        self.total_fov_rays = horizontal_rays * vertical_layers
        # Omni (radar)
        self.omni_rays = omni_rays
        self.omni_max_dist = omni_max_dist

        # Legacy compat
        self.total_rays = self.total_fov_rays
        self.fov_rays = horizontal_rays  # All h_rays are within FOV now
        self.half_fov_rays = self.fov_rays // 2

        # View mode (toggle with V key)
        self.view_mode = VIEW_RADAR  # Start with radar view

        # State
        self.running = False
        self.screen = None
        self.font = None
        self.font_large = None
        self.clock = None

        # Stats
        self.fps = 0
        self.last_update = time.time()
        self.frame_count = 0

    def init_display(self):
        """Initialize pygame display with hardware acceleration."""
        pygame.init()
        pygame.display.set_caption("Depth View - AssaultCube Agent")
        # Use hardware surface + double buffering for GPU acceleration
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_large = pygame.font.SysFont("consolas", 18, bold=True)
        self.clock = pygame.time.Clock()
        self.running = True

        # Pre-allocate pixel buffer for fast rendering
        self._pixel_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._render_surface = pygame.Surface((self.width, self.height))

        # Warm up numba JIT compilation
        print("[DepthViewer] Warming up Numba JIT...")
        dummy_omni = np.zeros((self.omni_rays, 2), dtype=np.float64)
        _render_radar_fast(
            self._pixel_buffer, dummy_omni,
            self.omni_rays,
            self.width // 2, self.height // 2,
            min(self.width, self.height) // 2 - 80,
            self.omni_max_dist, self.fov_degrees
        )
        print("[DepthViewer] JIT ready!")

    def _get_depth_color(self, norm_distance: float, hit_type: float) -> tuple:
        """
        Get color based on distance and hit type.

        Distance is already normalized (0-1) from raycast output.
        Hit type is normalized: 0=nothing, 0.33=wall, 0.67=enemy, 1.0=sky
        """
        # Distance is already normalized 0-1, don't divide again!
        depth = min(1.0, max(0.0, norm_distance))
        inv_depth = 1.0 - depth  # Closer = higher value = brighter

        if hit_type > 0.9:  # Sky (HIT_SKY = 3, normalized to 1.0)
            # Blue sky gradient - lighter when closer (atmosphere effect)
            base = 60
            intensity = int(base + inv_depth * 140)
            return (intensity // 2, intensity // 2 + 30, min(255, intensity + 60))

        elif hit_type > 0.6:  # Enemy (HIT_ENEMY = 2, normalized to 0.67)
            # Red - brighter when closer
            base = 100
            intensity = int(base + inv_depth * 155)
            return (min(255, intensity), 30, 30)

        elif hit_type > 0.25:  # Wall (HIT_WALL = 1, normalized to 0.33)
            # Gray gradient - brighter when closer = more detail
            base = 30
            intensity = int(base + inv_depth * 200)
            return (intensity, intensity, min(255, intensity + 10))

        else:  # Nothing (HIT_NOTHING = 0)
            # Dark void - no depth info
            return COLOR_VOID

    def draw_fov_viewport(self, rays: np.ndarray):
        """
        Draw full-screen FOV viewport with depth coloring.

        Uses Numba-accelerated pixel rendering for high ray counts.
        Only displays rays within the configured FOV.
        """
        # Viewport layout - full screen with margins
        margin_top = 60
        margin_bottom = 80
        margin_sides = 30

        vp_x = margin_sides
        vp_y = margin_top
        vp_width = self.width - (margin_sides * 2)
        vp_height = self.height - margin_top - margin_bottom

        # Clear pixel buffer to background color
        self._pixel_buffer[:, :, 0] = COLOR_BG[0]
        self._pixel_buffer[:, :, 1] = COLOR_BG[1]
        self._pixel_buffer[:, :, 2] = COLOR_BG[2]

        # Fast Numba rendering of FOV rays to pixel buffer
        rays_f64 = rays.astype(np.float64) if rays.dtype != np.float64 else rays
        _render_fov_fast(
            self._pixel_buffer, rays_f64,
            self.h_rays, self.v_layers,
            vp_x, vp_y, vp_width, vp_height,
            self.fov_rays, self.half_fov_rays,
            self.max_dist
        )

        # Blit pixel buffer to screen using surfarray
        pygame.surfarray.blit_array(self._render_surface, self._pixel_buffer.swapaxes(0, 1))
        self.screen.blit(self._render_surface, (0, 0))

        # Draw border
        pygame.draw.rect(self.screen, COLOR_BORDER,
                        (vp_x - 2, vp_y - 2, vp_width + 4, vp_height + 4), 2)

        # Draw crosshair at center
        cx = vp_x + vp_width // 2
        cy = vp_y + vp_height // 2
        crosshair_size = 15
        pygame.draw.line(self.screen, COLOR_CROSSHAIR,
                        (cx - crosshair_size, cy), (cx + crosshair_size, cy), 2)
        pygame.draw.line(self.screen, COLOR_CROSSHAIR,
                        (cx, cy - crosshair_size), (cx, cy + crosshair_size), 2)

    def draw_radar(self, omni_rays: np.ndarray, enemies: list = None):
        """
        Draw top-down radar view showing all 360° omni rays.

        Uses Numba-accelerated pixel rendering.

        Args:
            omni_rays: Omni ray data (36 rays over 360°) from NumbaRaycast
            enemies: Optional list of enemy info dicts with 'angle_h' and 'distance' keys
                     (from EnemyDetector, shows enemies even if rays are blocked)
        """
        # Center the radar in the window
        margin = 80
        radar_size = min(self.width, self.height) - margin * 2
        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = radar_size // 2

        # Clear pixel buffer to background color
        self._pixel_buffer[:, :, 0] = COLOR_BG[0]
        self._pixel_buffer[:, :, 1] = COLOR_BG[1]
        self._pixel_buffer[:, :, 2] = COLOR_BG[2]

        # Fast Numba rendering of omni rays to pixel buffer
        rays_f64 = omni_rays.astype(np.float64) if omni_rays.dtype != np.float64 else omni_rays
        _render_radar_fast(
            self._pixel_buffer, rays_f64,
            self.omni_rays,
            center_x, center_y, max_radius,
            self.omni_max_dist, self.fov_degrees
        )

        # Blit pixel buffer to screen using surfarray
        pygame.surfarray.blit_array(self._render_surface, self._pixel_buffer.swapaxes(0, 1))
        self.screen.blit(self._render_surface, (0, 0))

        # Draw overlays with pygame (these are few items, so fast)
        # Grid circles for distance reference
        for r in range(50, max_radius, 50):
            pygame.draw.circle(self.screen, COLOR_GRID, (center_x, center_y), r, 1)

        # FOV indicator lines (shows the forward FOV cone)
        half_fov = self.fov_degrees / 2
        for angle in [-half_fov, half_fov]:
            # angle is offset from forward, screen -90° = forward/up
            rad = math.radians(angle - 90)
            end_x = int(center_x + math.cos(rad) * max_radius)
            end_y = int(center_y + math.sin(rad) * max_radius)
            pygame.draw.line(self.screen, COLOR_FOV_LINE,
                           (center_x, center_y), (end_x, end_y), 1)

        # Draw player at center (triangle pointing UP = forward)
        pygame.draw.circle(self.screen, COLOR_PLAYER, (center_x, center_y), 6)
        pygame.draw.line(self.screen, COLOR_PLAYER,
                        (center_x, center_y), (center_x, center_y - 15), 2)

        # Overlay enemies from memory (EnemyDetector) - shows even if rays blocked
        if enemies:
            for enemy in enemies:
                angle_h = enemy.get('angle_h', 0)
                dist = enemy.get('distance', 0)
                has_los = enemy.get('has_los', True)

                # angle_h is relative to player view (0 = directly ahead)
                # Screen: -90° = up = forward
                angle_rad = math.radians(angle_h - 90)
                norm_dist = min(1.0, dist / self.omni_max_dist)
                enemy_radius = norm_dist * max_radius

                enemy_x = int(center_x + math.cos(angle_rad) * enemy_radius)
                enemy_y = int(center_y + math.sin(angle_rad) * enemy_radius)

                if has_los:
                    pygame.draw.line(self.screen, COLOR_ENEMY,
                                   (center_x, center_y), (enemy_x, enemy_y), 2)
                    pygame.draw.circle(self.screen, COLOR_ENEMY, (enemy_x, enemy_y), 8)
                    pygame.draw.circle(self.screen, COLOR_ENEMY_GLOW, (enemy_x, enemy_y), 12, 2)
                else:
                    dim_color = (150, 50, 50)
                    pygame.draw.circle(self.screen, dim_color, (enemy_x, enemy_y), 6)

    def draw_stats(self, player_state: Optional[dict] = None):
        """Draw stats overlay."""
        # Title bar - show current view mode
        view_name = "RADAR VIEW" if self.view_mode == VIEW_RADAR else "FOV VIEWPORT"
        title = self.font_large.render(view_name, True, COLOR_TEXT)
        self.screen.blit(title, (10, 10))

        # Toggle hint
        hint = self.font.render("[V] Toggle view", True, (120, 120, 120))
        self.screen.blit(hint, (self.width - 110, 10))

        # FPS
        fps_text = self.font.render(f"FPS: {self.fps:.0f}", True, COLOR_TEXT)
        self.screen.blit(fps_text, (10, 35))

        # Ray info - show FOV and omni ray counts
        if self.view_mode == VIEW_RADAR:
            deg_per_ray = 360.0 / self.omni_rays
            ray_text = self.font.render(
                f"Omni: {self.omni_rays} rays (360°, {deg_per_ray:.0f}°/ray)",
                True, COLOR_TEXT
            )
        else:
            ray_text = self.font.render(
                f"FOV: {self.h_rays}×{self.v_layers}={self.total_fov_rays} rays ({self.fov_degrees}°×{80}°)",
                True, COLOR_TEXT
            )
        self.screen.blit(ray_text, (100, 35))

        # Player stats (bottom bar)
        bottom_y = self.height - 60

        if player_state:
            hp = player_state.get('health', 0)
            frags = player_state.get('frags', 0)
            damage = player_state.get('damage', 0)

            hp_color = (100, 255, 100) if hp > 50 else (255, 100, 100)
            hp_text = self.font_large.render(f"HP: {hp}", True, hp_color)
            self.screen.blit(hp_text, (30, bottom_y))

            stats_text = self.font.render(f"Frags: {frags}  |  Damage: {damage}", True, COLOR_TEXT)
            self.screen.blit(stats_text, (120, bottom_y + 2))

        # Legend
        legend_y = bottom_y + 25
        legend_items = [
            ((100, 100, 180), "Sky"),
            ((180, 180, 180), "Wall"),
            ((255, 50, 50), "Enemy"),
            (COLOR_VOID, "Void"),
        ]

        x = 30
        for color, label in legend_items:
            pygame.draw.rect(self.screen, color, (x, legend_y, 12, 12))
            text = self.font.render(label, True, COLOR_TEXT)
            self.screen.blit(text, (x + 16, legend_y - 1))
            x += 80

    def update(self, fov_rays: np.ndarray, player_state: Optional[dict] = None,
               enemies: list = None, omni_rays: np.ndarray = None):
        """
        Update the visualization with new ray data.

        Args:
            fov_rays: Shape (total_fov, 2) - FOV rays for viewport [distance, hit_type]
            player_state: Optional dict with health, frags, damage, enemies, etc.
            enemies: Optional list of enemy dicts with 'angle_h', 'distance', 'has_los'
                     (from EnemyDetector - shows on radar even if rays blocked)
            omni_rays: Shape (omni_rays, 2) - 360° rays for radar [distance, hit_type]
        """
        if not self.running:
            return False

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_v:
                    # Toggle view mode
                    self.view_mode = VIEW_FOV if self.view_mode == VIEW_RADAR else VIEW_RADAR

        # Extract enemies from player_state if not passed separately
        if enemies is None and player_state:
            enemies = player_state.get('enemies', None)

        # Use omni_rays for radar if provided, otherwise fall back to fov_rays
        # (This handles backwards compat and standalone vs aim_trainer usage)
        radar_rays = omni_rays if omni_rays is not None else fov_rays

        # Clear screen
        self.screen.fill(COLOR_BG)

        # Draw based on view mode
        if self.view_mode == VIEW_RADAR:
            self.draw_radar(radar_rays, enemies)
        else:
            self.draw_fov_viewport(fov_rays)
        self.draw_stats(player_state)

        # Update display
        pygame.display.flip()

        # FPS tracking
        self.frame_count += 1
        now = time.time()
        if now - self.last_update >= 1.0:
            self.fps = self.frame_count / (now - self.last_update)
            self.frame_count = 0
            self.last_update = now

        self.clock.tick(60)  # Cap at 60 FPS
        return True

    def close(self):
        """Clean up."""
        self.running = False
        pygame.quit()


def run_viewer_process(
    ray_queue: mp.Queue,
    state_queue: mp.Queue,
    omni_queue: mp.Queue,
    h_rays: int,
    v_layers: int,
    fov_degrees: float,
    max_dist: float,
    omni_rays_count: int = 72,
):
    """
    Run the viewer in a separate process.
    """
    viewer = DepthViewer(
        horizontal_rays=h_rays,
        vertical_layers=v_layers,
        fov_degrees=fov_degrees,
        max_distance=max_dist,
        omni_rays=omni_rays_count,
    )
    viewer.init_display()

    last_rays = np.zeros((h_rays * v_layers, 2), dtype=np.float32)
    last_omni = np.zeros((omni_rays_count, 2), dtype=np.float32)
    last_state = {}

    while viewer.running:
        # Get latest data (non-blocking)
        try:
            while not ray_queue.empty():
                last_rays = ray_queue.get_nowait()
        except:
            pass

        try:
            while not omni_queue.empty():
                last_omni = omni_queue.get_nowait()
        except:
            pass

        try:
            while not state_queue.empty():
                last_state = state_queue.get_nowait()
        except:
            pass

        if not viewer.update(last_rays, last_state, omni_rays=last_omni):
            break

    viewer.close()


# Alias for backwards compatibility
LidarViewer = DepthViewer


class LidarVisualizer:
    """
    Wrapper to run depth visualization in a separate process.

    Usage:
        viz = LidarVisualizer(h_rays=72, v_layers=9, fov_degrees=120)
        viz.start()

        # In your training loop:
        viz.update(rays, {'health': 100, 'frags': 0, 'damage': 0})

        # When done:
        viz.stop()
    """

    def __init__(
        self,
        # Match numba_raycast.py FOV config
        h_rays: int = 41,
        v_layers: int = 17,
        fov_degrees: float = 120.0,
        max_dist: float = 350.0,
        omni_rays: int = 72,
    ):
        self.h_rays = h_rays
        self.v_layers = v_layers
        self.fov_degrees = fov_degrees
        self.max_dist = max_dist
        self.omni_rays = omni_rays
        self.ray_queue: Optional[mp.Queue] = None
        self.state_queue: Optional[mp.Queue] = None
        self.omni_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None
        self.running = False

    def start(self):
        """Start the visualization process."""
        if self.running:
            return

        self.ray_queue = mp.Queue(maxsize=2)
        self.state_queue = mp.Queue(maxsize=2)
        self.omni_queue = mp.Queue(maxsize=2)

        self.process = mp.Process(
            target=run_viewer_process,
            args=(
                self.ray_queue,
                self.state_queue,
                self.omni_queue,
                self.h_rays,
                self.v_layers,
                self.fov_degrees,
                self.max_dist,
                self.omni_rays,
            ),
            daemon=True
        )
        self.process.start()
        self.running = True
        print(f"[DepthViz] Started (FOV: {self.fov_degrees}°, omni_rays: {self.omni_rays})")

    def update(self, rays: np.ndarray, player_state: Optional[dict] = None,
               omni_rays: np.ndarray = None):
        """
        Send new data to the visualization.

        Non-blocking - if the queue is full, drops the update.

        Args:
            rays: FOV rays for viewport
            player_state: dict with health, frags, damage, enemies
            omni_rays: 360° rays for radar
        """
        if not self.running:
            return

        # Send FOV rays (drop if queue full)
        try:
            if self.ray_queue.full():
                try:
                    self.ray_queue.get_nowait()
                except:
                    pass
            self.ray_queue.put_nowait(rays.copy())
        except:
            pass

        # Send omni rays for radar
        if omni_rays is not None and self.omni_queue:
            try:
                if self.omni_queue.full():
                    try:
                        self.omni_queue.get_nowait()
                    except:
                        pass
                self.omni_queue.put_nowait(omni_rays.copy())
            except:
                pass

        # Send state
        if player_state:
            try:
                if self.state_queue.full():
                    try:
                        self.state_queue.get_nowait()
                    except:
                        pass
                self.state_queue.put_nowait(player_state.copy())
            except:
                pass

    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        print("[DepthViz] Stopped")

    def is_running(self) -> bool:
        """Check if visualization is still running."""
        if self.process is None:
            return False
        return self.process.is_alive()


def test_standalone():
    """Test the viewer with live raycast data.

    Uses NumbaRaycastObserver which provides BOTH:
    - FOV rays (41x17 = 697 rays within 120°x80° FOV) for viewport
    - Omni rays (36 rays over 360°) for radar

    Now includes EnemyDetector for proper enemy overlay on radar view.
    """
    from ..raycast.numba_raycast import NumbaRaycastObserver
    from ..memory import ACMemoryReader

    print("=" * 60)
    print("  DEPTH VISUALIZATION TEST")
    print("=" * 60)
    print("\nPress ESC or close window to exit\n")

    # Initialize
    reader = ACMemoryReader()
    if not reader.attach():
        print("Failed to attach to AssaultCube!")
        return

    # Use NumbaRaycastObserver which provides both FOV and omni rays
    raycast = NumbaRaycastObserver(
        fov_rays_h=41,      # 41 rays horizontal within FOV
        fov_rays_v=17,      # 17 layers vertical within FOV
        fov_h=120.0,        # 120° horizontal FOV
        fov_v=80.0,         # 80° vertical FOV
        fov_max_dist=350.0, # FOV max distance
        omni_rays=72,       # 72 rays over 360° for radar (5° per ray)
        omni_max_dist=150.0,# Radar max distance
    )
    if not raycast.attach():
        print("Failed to attach raycast observer!")
        reader.detach()
        return

    # Initialize EnemyDetector for radar overlay
    detector = None
    if ENEMY_DETECTOR_AVAILABLE:
        detector = EnemyDetector(fov_h=120.0, fov_v=90.0, use_los=True)
        if not detector.attach():
            print("[!] Failed to attach enemy detector - radar overlay disabled")
            detector = None
        else:
            print("[+] EnemyDetector attached - radar overlay enabled")

    # Create viewer with FOV ray config for viewport
    # Radar will use separate omni rays
    viewer = DepthViewer(
        horizontal_rays=41,   # For FOV viewport
        vertical_layers=17,
        fov_degrees=120.0,
        max_distance=350.0,
        omni_rays=72,         # For radar view (360°, 5° per ray)
        omni_max_dist=150.0,
    )
    viewer.init_display()

    print(f"[+] FOV: 41×17=697 rays (120°×80°)")
    print(f"[+] Radar: 72 rays (360°, 5° per ray)")
    print("[+] Press ESC to exit, V to toggle view")

    try:
        while viewer.running:
            # Get both FOV and omni rays
            fov_rays, omni_rays = raycast.get_observation()

            # Get player state
            state = reader.read_state()

            # Detect enemies for radar overlay
            enemy_data = []
            if detector:
                enemies = detector.detect_enemies(
                    state.position, state.yaw, state.pitch,
                    max_distance=350.0, filter_team=True, filter_los=True
                )
                enemy_data = [
                    {'angle_h': e.angle_h, 'distance': e.distance, 'has_los': e.has_los}
                    for e in enemies
                ]

            player_state = {
                'health': state.health,
                'frags': state.frags,
                'damage': state.damage_dealt,
                'enemies': enemy_data,
            }

            # Update viewer with both ray types
            if not viewer.update(fov_rays, player_state, omni_rays=omni_rays):
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted")

    viewer.close()
    raycast.detach()
    if detector:
        detector.detach()
    reader.detach()
    print("[+] Done!")


if __name__ == "__main__":
    test_standalone()
