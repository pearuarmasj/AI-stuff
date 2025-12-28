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

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not installed. Install with: pip install pygame")


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
        horizontal_rays: int = 72,
        vertical_layers: int = 9,
        fov_degrees: float = 120.0,
        max_distance: float = 250.0,
    ):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for visualization")

        self.width = width
        self.height = height
        self.h_rays = horizontal_rays
        self.v_layers = vertical_layers
        self.fov_degrees = fov_degrees
        self.max_dist = max_distance
        self.total_rays = horizontal_rays * vertical_layers

        # Calculate FOV ray indices
        rays_per_degree = horizontal_rays / 360.0
        self.fov_rays = int(fov_degrees * rays_per_degree)
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

    def _get_depth_color(self, distance: float, hit_type: float) -> tuple:
        """
        Get color based on distance and hit type.

        Distance is in raw world units (0 to max_dist).
        Hit type is normalized: 0=nothing, 0.33=wall, 0.67=enemy, 1.0=sky
        """
        # Normalize distance for coloring (0 = close, 1 = far)
        depth = min(1.0, distance / self.max_dist)
        inv_depth = 1.0 - depth  # Closer = higher value

        if hit_type > 0.9:  # Sky (HIT_SKY = 3, normalized to 1.0)
            # Blue sky gradient - lighter when closer (atmosphere effect)
            base = 80
            intensity = int(base + inv_depth * 120)
            return (intensity // 2, intensity // 2 + 20, intensity + 40)

        elif hit_type > 0.6:  # Enemy (HIT_ENEMY = 2, normalized to 0.67)
            # Red - brighter when closer
            base = 80
            intensity = int(base + inv_depth * 175)
            return (intensity, 25, 25)

        elif hit_type > 0.25:  # Wall (HIT_WALL = 1, normalized to 0.33)
            # Gray gradient - brighter when closer = more detail
            base = 25
            intensity = int(base + inv_depth * 200)
            return (intensity, intensity, intensity + 5)

        else:  # Nothing (HIT_NOTHING = 0)
            # Dark void - no depth info
            return COLOR_VOID

    def draw_fov_viewport(self, rays: np.ndarray):
        """
        Draw full-screen FOV viewport with depth coloring.

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

        # Draw border
        pygame.draw.rect(self.screen, COLOR_BORDER,
                        (vp_x - 2, vp_y - 2, vp_width + 4, vp_height + 4), 2)

        # Calculate cell sizes
        total_fov_cols = self.half_fov_rays * 2 + 1  # Left + center + right
        cell_width = vp_width / total_fov_cols
        cell_height = vp_height / self.v_layers

        # Draw rays layer by layer
        for layer in range(self.v_layers):
            layer_start = layer * self.h_rays
            # Flip vertical so top layers are rendered at top of screen
            screen_row = self.v_layers - 1 - layer
            y = vp_y + screen_row * cell_height

            col = 0

            # Left side of FOV (rays h_rays-half_fov_rays to h_rays-1)
            for i in range(self.h_rays - self.half_fov_rays, self.h_rays):
                ray_idx = layer_start + i
                distance = rays[ray_idx, 0]  # Raw distance in world units
                hit_type = rays[ray_idx, 1]  # Normalized hit type

                x = vp_x + col * cell_width
                color = self._get_depth_color(distance, hit_type)
                pygame.draw.rect(self.screen, color,
                               (int(x), int(y), int(cell_width) + 1, int(cell_height) + 1))
                col += 1

            # Right side of FOV (rays 0 to half_fov_rays)
            for i in range(self.half_fov_rays + 1):
                ray_idx = layer_start + i
                distance = rays[ray_idx, 0]
                hit_type = rays[ray_idx, 1]

                x = vp_x + col * cell_width
                color = self._get_depth_color(distance, hit_type)
                pygame.draw.rect(self.screen, color,
                               (int(x), int(y), int(cell_width) + 1, int(cell_height) + 1))
                col += 1

        # Draw crosshair at center
        cx = vp_x + vp_width // 2
        cy = vp_y + vp_height // 2
        crosshair_size = 15
        pygame.draw.line(self.screen, COLOR_CROSSHAIR,
                        (cx - crosshair_size, cy), (cx + crosshair_size, cy), 2)
        pygame.draw.line(self.screen, COLOR_CROSSHAIR,
                        (cx, cy - crosshair_size), (cx, cy + crosshair_size), 2)

    def draw_radar(self, rays: np.ndarray, enemies: list = None):
        """
        Draw top-down radar view showing all 360° rays.

        This is the classic LiDAR visualization - shows spatial awareness
        in all directions with depth-based coloring.

        Args:
            rays: Ray data from OmniRaycast
            enemies: Optional list of enemy info dicts with 'angle_h' and 'distance' keys
                     (from EnemyDetector, shows enemies even if rays are blocked)
        """
        # Center the radar in the window
        margin = 80
        radar_size = min(self.width, self.height) - margin * 2
        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = radar_size // 2

        # Draw grid circles for distance reference
        for r in range(50, max_radius, 50):
            pygame.draw.circle(self.screen, COLOR_GRID, (center_x, center_y), r, 1)

        # Draw FOV indicator lines
        half_fov = self.fov_degrees / 2
        for angle in [-half_fov, half_fov]:
            rad = math.radians(angle - 90)  # -90 because forward is up
            end_x = int(center_x + math.cos(rad) * max_radius)
            end_y = int(center_y + math.sin(rad) * max_radius)
            pygame.draw.line(self.screen, COLOR_FOV_LINE,
                           (center_x, center_y), (end_x, end_y), 1)

        # Draw rays for middle vertical layer (eye level)
        middle_layer = self.v_layers // 2
        layer_start = middle_layer * self.h_rays

        # Draw all rays as lines from center
        for i in range(self.h_rays):
            ray_idx = layer_start + i

            # Angle: 0 = forward (up on screen), increases clockwise
            angle_deg = (i / self.h_rays) * 360
            angle_rad = math.radians(angle_deg - 90)  # -90 so 0° is up

            distance = rays[ray_idx, 0]  # Raw distance in world units
            hit_type = rays[ray_idx, 1]

            # Normalize distance for display
            norm_dist = min(1.0, distance / self.max_dist)

            # Calculate end point
            ray_length = norm_dist * max_radius
            end_x = int(center_x + math.cos(angle_rad) * ray_length)
            end_y = int(center_y + math.sin(angle_rad) * ray_length)

            # Get color based on hit type and depth
            color = self._get_depth_color(distance, hit_type)

            # Draw ray - check sky first (1.0) before enemy (0.67)
            if hit_type > 0.9:  # Sky
                pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 2)
            elif hit_type > 0.6:  # Enemy
                pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 3)
                pygame.draw.circle(self.screen, COLOR_ENEMY_GLOW, (end_x, end_y), 5)
            else:  # Wall or nothing
                pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 1)

        # Draw player at center
        pygame.draw.circle(self.screen, COLOR_PLAYER, (center_x, center_y), 6)
        # Direction indicator (forward arrow)
        pygame.draw.line(self.screen, COLOR_PLAYER,
                        (center_x, center_y), (center_x, center_y - 15), 2)

        # Overlay enemies from memory (EnemyDetector) - shows even if rays blocked
        if enemies:
            for enemy in enemies:
                angle_h = enemy.get('angle_h', 0)  # Horizontal angle relative to view
                dist = enemy.get('distance', 0)
                has_los = enemy.get('has_los', True)

                # Convert to radar coordinates
                # angle_h: 0 = forward, positive = right
                # Radar: 0° = up (forward), clockwise
                angle_rad = math.radians(angle_h - 90)  # -90 so 0° is up
                norm_dist = min(1.0, dist / self.max_dist)
                enemy_radius = norm_dist * max_radius

                enemy_x = int(center_x + math.cos(angle_rad) * enemy_radius)
                enemy_y = int(center_y + math.sin(angle_rad) * enemy_radius)

                # Draw enemy marker (red dot with glow)
                if has_los:
                    # Clear LOS - bright red with line
                    pygame.draw.line(self.screen, COLOR_ENEMY,
                                   (center_x, center_y), (enemy_x, enemy_y), 2)
                    pygame.draw.circle(self.screen, COLOR_ENEMY, (enemy_x, enemy_y), 8)
                    pygame.draw.circle(self.screen, COLOR_ENEMY_GLOW, (enemy_x, enemy_y), 12, 2)
                else:
                    # No LOS - dimmer marker (behind wall)
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

        # FOV info
        fov_text = self.font.render(f"FOV: {self.fov_degrees:.0f}° ({self.fov_rays} rays)", True, COLOR_TEXT)
        self.screen.blit(fov_text, (100, 35))

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

    def update(self, rays: np.ndarray, player_state: Optional[dict] = None, enemies: list = None):
        """
        Update the visualization with new ray data.

        Args:
            rays: Shape (total_rays, 2) - [raw_distance, hit_type]
            player_state: Optional dict with health, frags, damage, enemies, etc.
            enemies: Optional list of enemy dicts with 'angle_h', 'distance', 'has_los'
                     (from EnemyDetector - shows on radar even if rays blocked)
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

        # Clear screen
        self.screen.fill(COLOR_BG)

        # Draw based on view mode
        if self.view_mode == VIEW_RADAR:
            self.draw_radar(rays, enemies)
        else:
            self.draw_fov_viewport(rays)
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
    h_rays: int,
    v_layers: int,
    fov_degrees: float,
    max_dist: float,
):
    """
    Run the viewer in a separate process.
    """
    viewer = DepthViewer(
        horizontal_rays=h_rays,
        vertical_layers=v_layers,
        fov_degrees=fov_degrees,
        max_distance=max_dist,
    )
    viewer.init_display()

    last_rays = np.zeros((h_rays * v_layers, 2), dtype=np.float32)
    last_state = {}

    while viewer.running:
        # Get latest data (non-blocking)
        try:
            while not ray_queue.empty():
                last_rays = ray_queue.get_nowait()
        except:
            pass

        try:
            while not state_queue.empty():
                last_state = state_queue.get_nowait()
        except:
            pass

        if not viewer.update(last_rays, last_state):
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
        h_rays: int = 72,
        v_layers: int = 9,
        fov_degrees: float = 120.0,
        max_dist: float = 250.0,
    ):
        self.h_rays = h_rays
        self.v_layers = v_layers
        self.fov_degrees = fov_degrees
        self.max_dist = max_dist
        self.ray_queue: Optional[mp.Queue] = None
        self.state_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None
        self.running = False

    def start(self):
        """Start the visualization process."""
        if self.running:
            return

        self.ray_queue = mp.Queue(maxsize=2)
        self.state_queue = mp.Queue(maxsize=2)

        self.process = mp.Process(
            target=run_viewer_process,
            args=(
                self.ray_queue,
                self.state_queue,
                self.h_rays,
                self.v_layers,
                self.fov_degrees,
                self.max_dist,
            ),
            daemon=True
        )
        self.process.start()
        self.running = True
        print(f"[DepthViz] Started (FOV: {self.fov_degrees}°, max_dist: {self.max_dist})")

    def update(self, rays: np.ndarray, player_state: Optional[dict] = None):
        """
        Send new data to the visualization.

        Non-blocking - if the queue is full, drops the update.
        """
        if not self.running:
            return

        # Send rays (drop if queue full)
        try:
            if self.ray_queue.full():
                try:
                    self.ray_queue.get_nowait()
                except:
                    pass
            self.ray_queue.put_nowait(rays.copy())
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
    """Test the viewer with live raycast data."""
    from ..raycast import OmniRaycastObserver
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

    max_dist = 250.0
    raycast = OmniRaycastObserver(
        horizontal_rays=72,
        vertical_layers=9,
        max_distance=max_dist
    )
    if not raycast.attach():
        print("Failed to attach raycast observer!")
        reader.detach()
        return

    # Create viewer (in main process for standalone test)
    viewer = DepthViewer(
        horizontal_rays=72,
        vertical_layers=9,
        fov_degrees=120.0,
        max_distance=max_dist,
    )
    viewer.init_display()

    print("[+] Running... Press ESC to exit")

    try:
        while viewer.running:
            # Get raycast data
            rays = raycast.get_observation()

            # Get player state
            state = reader.read_state()
            player_state = {
                'health': state.health,
                'frags': state.frags,
                'damage': state.damage_dealt,
            }

            # Update viewer
            if not viewer.update(rays, player_state):
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted")

    viewer.close()
    raycast.detach()
    reader.detach()
    print("[+] Done!")


if __name__ == "__main__":
    test_standalone()
