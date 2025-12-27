"""
Gymnasium environment wrapper for AssaultCube.

This wraps the game as a reinforcement learning environment
compatible with Stable-Baselines3.
"""

import time
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..vision import ScreenCapture
from ..depth import DepthEstimator
from ..memory import ACMemoryReader
from ..control import ActionMapper, check_stop
from ..raycast import EnemyDetector


# Max enemies to track in observation
MAX_ENEMIES = 8


class AssaultCubeEnv(gym.Env):
    """
    AssaultCube as a Gymnasium environment.

    Observation space:
        - depth: Depth map as heat vision (H, W, 1)
        - state: [health, armor, ammo] normalized
        - enemies: Array of [in_fov, distance, angle_h, angle_v] per enemy slot

    Action space:
        - [0-5]: movement (WASD + jump + crouch)
        - [6-7]: aim delta (yaw, pitch)
        - [8-9]: combat (shoot, reload)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        obs_width: int = 160,
        obs_height: int = 120,
        depth_model_size: str = "small",
        frame_skip: int = 4,
        max_episode_steps: int = 10000,
        render_mode: str | None = "human",
        fov_h: float = 90.0,
        fov_v: float = 60.0,
    ):
        super().__init__()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.fov_h = fov_h
        self.fov_v = fov_v

        # Initialize components
        print("=" * 60)
        print("Initializing AssaultCube Environment")
        print("=" * 60)

        self.capture = ScreenCapture()
        print("[+] Screen capture ready")

        self.depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            output_size=(obs_width, obs_height),
        )
        print("[+] Depth estimator ready")

        self.memory_reader = ACMemoryReader()
        if not self.memory_reader.attach():
            raise RuntimeError("Failed to attach to AssaultCube process!")
        print("[+] Memory reader ready")

        self.enemy_detector = EnemyDetector(fov_h=fov_h, fov_v=fov_v, use_los=True)
        if not self.enemy_detector.attach():
            raise RuntimeError("Failed to attach enemy detector!")
        print("[+] Enemy detector ready (with LOS checking)")

        self.action_mapper = ActionMapper()
        print("[+] Action mapper ready")

        # Define observation space
        self.observation_space = spaces.Dict({
            # Depth map as single channel (heat vision)
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(obs_height, obs_width, 1),
                dtype=np.uint8,
            ),
            # Player state: [health, armor, ammo]
            'state': spaces.Box(
                low=0,
                high=1,
                shape=(3,),
                dtype=np.float32,
            ),
            # Enemy slots: [in_fov, has_los, distance_norm, angle_h_norm, angle_v_norm] x MAX_ENEMIES
            'enemies': spaces.Box(
                low=-1,
                high=1,
                shape=(MAX_ENEMIES, 5),
                dtype=np.float32,
            ),
        })

        # Action space: movement + aim + combat
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0
        self._last_game_state = None
        self._last_frame = None
        self._last_depth = None
        self._last_enemies = []

        # Reward tracking
        self._last_health = 100
        self._last_armor = 0
        self._last_frags = 0
        self._last_deaths = 0
        self._kills = 0
        self._deaths = 0
        self._is_dead = False  # Track dead state to detect transitions

        print("=" * 60)
        print("Environment ready!")
        print("=" * 60)

    def _get_observation(self) -> dict:
        """Capture and process observation from the game."""
        # Capture screen
        frame = self.capture.capture()
        self._last_frame = frame

        # Get depth (heat vision)
        depth = self.depth_estimator.estimate(frame)
        self._last_depth = depth

        # Convert depth to uint8 for observation
        depth_uint8 = (depth * 255).astype(np.uint8)
        depth_obs = depth_uint8[:, :, np.newaxis]

        # Get game state from memory
        game_state = self.memory_reader.read_state()
        self._last_game_state = game_state

        state = np.array([
            game_state.health / 100.0,
            game_state.armor / 100.0,
            min(game_state.ammo_mag, 100) / 100.0,
        ], dtype=np.float32)

        # Detect enemies (with LOS filtering - only returns visible enemies)
        own_pos = (game_state.pos_x, game_state.pos_y, game_state.pos_z)
        enemies = self.enemy_detector.detect_enemies(
            own_pos, game_state.yaw, game_state.pitch,
            max_distance=500.0,
            filter_team=True,
            filter_los=True,  # Only return enemies with clear line-of-sight
        )
        self._last_enemies = enemies

        # Build enemy observation array
        enemy_obs = np.zeros((MAX_ENEMIES, 5), dtype=np.float32)
        for i, e in enumerate(enemies[:MAX_ENEMIES]):
            enemy_obs[i, 0] = 1.0 if e.in_fov else 0.0
            enemy_obs[i, 1] = 1.0 if e.has_los else 0.0  # Line-of-sight flag
            enemy_obs[i, 2] = min(e.distance / 500.0, 1.0)  # Normalize distance
            enemy_obs[i, 3] = e.angle_h / 180.0  # Normalize angle
            enemy_obs[i, 4] = e.angle_v / 90.0

        return {
            'depth': depth_obs,
            'state': state,
            'enemies': enemy_obs,
        }

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state and enemy tracking."""
        if self._last_game_state is None:
            return 0.0

        reward = 0.0
        current_health = self._last_game_state.health
        is_currently_dead = current_health <= 0

        # Detect death transition (alive -> dead) - only count ONCE
        if is_currently_dead and not self._is_dead:
            # Just died this frame
            reward -= 50.0
            self._deaths += 1
            self._is_dead = True
            print(f"[REWARD] DEATH! -50.0 (Total deaths: {self._deaths})")
        elif not is_currently_dead and self._is_dead:
            # Just respawned (dead -> alive)
            self._is_dead = False
            self._last_health = current_health  # Reset health tracking
            print(f"[INFO] Respawned with {current_health} health")
        elif not is_currently_dead:
            # Alive - normal reward calculation

            # Survival bonus (only when alive)
            reward += 0.01

            # Health change penalty (only when alive, ignore respawn heal)
            health_delta = current_health - self._last_health
            if health_delta < 0:
                reward += health_delta * 0.5  # Penalty for damage taken

            # Reward for aiming at visible enemies (in FOV with line-of-sight)
            for e in self._last_enemies:
                if e.in_fov and e.has_los:
                    # Small reward for having visible enemy in view
                    reward += 0.1
                    # Bonus for being close to center of view
                    aim_accuracy = 1.0 - (abs(e.angle_h) / 45.0)
                    if aim_accuracy > 0:
                        reward += aim_accuracy * 0.2

        # Track health for next step (only update when alive to avoid respawn delta)
        if not is_currently_dead:
            self._last_health = current_health
            self._last_armor = self._last_game_state.armor

        # Track frags from memory (proper kill detection) - works even when dead
        current_frags = self._last_game_state.frags
        if current_frags > self._last_frags:
            new_kills = current_frags - self._last_frags
            reward += new_kills * 100.0
            self._kills += new_kills
            print(f"[REWARD] KILL! +{new_kills * 100.0:.0f} (Total: {self._kills})")
        self._last_frags = current_frags

        return reward

    def _is_done(self) -> bool:
        """Check if episode should end."""
        if self._last_game_state and self._last_game_state.health <= 0:
            return True
        if self._step_count >= self.max_episode_steps:
            return True
        return False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_reward = 0

        self.action_mapper.reset()

        # Check if we're dead and need to respawn
        obs = self._get_observation()
        if self._last_game_state and self._last_game_state.health <= 0:
            print("[INFO] Dead - initiating respawn cycle...")

            # Wait 3 seconds on death screen
            time.sleep(3.0)

            # Click mouse1 to respawn
            self.action_mapper.click_to_respawn()

            # Wait for respawn (health back to 100)
            for _ in range(50):  # 5 second timeout
                time.sleep(0.1)
                obs = self._get_observation()
                if self._last_game_state and self._last_game_state.health > 0:
                    break

            print(f"[INFO] Respawned with {self._last_game_state.health if self._last_game_state else '?'} health")

        # Initialize tracking from ACTUAL current game state
        if self._last_game_state:
            self._last_frags = self._last_game_state.frags
            self._last_deaths = self._last_game_state.deaths
            self._is_dead = False  # We waited for respawn, so we're alive
            self._last_health = self._last_game_state.health
            self._last_armor = self._last_game_state.armor
        else:
            self._last_health = 100
            self._last_armor = 0
            self._is_dead = False

        info = {
            'episode_reward': self._episode_reward,
            'kills': self._kills,
            'deaths': self._deaths,
        }

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute action and get next state."""
        # Check for emergency stop
        if check_stop():
            self.action_mapper.reset()  # Release all keys
            # Return terminal state
            obs = self._get_observation()
            return obs, 0.0, True, False, {'emergency_stop': True}

        # Execute action
        for _ in range(self.frame_skip):
            self.action_mapper.execute(action)
            time.sleep(1/60)

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()
        self._episode_reward += reward

        # Check done
        terminated = self._is_done()
        truncated = self._step_count >= self.max_episode_steps

        self._step_count += 1

        info = {
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'health': self._last_game_state.health if self._last_game_state else 0,
            'armor': self._last_game_state.armor if self._last_game_state else 0,
            'enemies_detected': len(self._last_enemies),
            'kills': self._kills,
            'deaths': self._deaths,
        }

        # Render if in human mode
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment with visualization overlay."""
        if self._last_frame is None or self._last_depth is None:
            return None

        if self.render_mode == "rgb_array":
            return self._last_frame

        elif self.render_mode == "human":
            # Create visualization
            vis = self._create_visualization()
            cv2.imshow("AssaultCube Agent", vis)
            cv2.waitKey(1)
            return vis

    def _create_visualization(self) -> np.ndarray:
        """Create a visualization frame with all debug info."""
        h, w = self.screen_height // 2, self.screen_width // 2

        # Create depth heat map (purple=far, yellow/white=close)
        depth_colored = self._depth_to_heatmap(self._last_depth)
        depth_large = cv2.resize(depth_colored, (w, h))

        # Resize original frame
        frame_small = cv2.resize(self._last_frame, (w, h))

        # Stack side by side: Original | Depth
        top_row = np.hstack([frame_small, depth_large])

        # Draw overlays on both
        self._draw_enemy_overlay(top_row, w, h)
        self._draw_hud_overlay(top_row, w, h)

        return top_row

    def _depth_to_heatmap(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth to heat vision colormap."""
        # Depth Anything outputs close=high, far=low - no inversion needed
        # INFERNO colormap: low=purple/black (far), high=yellow/white (close)
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Apply colormap (COLORMAP_INFERNO: black->purple->red->yellow->white)
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return colored

    def _draw_enemy_overlay(self, frame: np.ndarray, w: int, h: int):
        """Draw enemy indicators on the frame."""
        if not self._last_enemies:
            return

        center_x = w + w // 2  # Center of depth view
        center_y = h // 2

        for e in self._last_enemies:
            # Calculate screen position based on angle
            # Map angle to screen position
            screen_x = center_x + int(e.angle_h * (w / self.fov_h))
            screen_y = center_y - int(e.angle_v * (h / self.fov_v))

            # Clamp to screen
            screen_x = max(w, min(2*w - 1, screen_x))
            screen_y = max(0, min(h - 1, screen_y))

            # Color based on distance (close=red, far=blue)
            dist_norm = min(e.distance / 200.0, 1.0)
            color = (
                int(255 * (1 - dist_norm)),  # B
                0,                            # G
                int(255 * dist_norm),         # R -> actually inverted for close=red
            )
            color = (0, 0, 255) if e.in_fov else (255, 100, 0)

            # Draw marker
            if e.in_fov:
                # In FOV: filled circle + crosshair
                cv2.circle(frame, (screen_x, screen_y), 15, color, 2)
                cv2.line(frame, (screen_x - 20, screen_y), (screen_x + 20, screen_y), color, 2)
                cv2.line(frame, (screen_x, screen_y - 20), (screen_x, screen_y + 20), color, 2)
            else:
                # Out of FOV: arrow pointing to enemy
                cv2.circle(frame, (screen_x, screen_y), 10, color, 2)

            # Distance text
            cv2.putText(
                frame, f"{e.distance:.0f}m",
                (screen_x + 20, screen_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Draw FOV box on depth view
        fov_left = w + int(w * 0.5 * (1 - 1))  # Full width = FOV
        fov_right = w + w
        cv2.rectangle(frame, (w, 0), (2*w, h), (50, 50, 50), 2)

    def _draw_hud_overlay(self, frame: np.ndarray, w: int, h: int):
        """Draw HUD information overlay."""
        if not self._last_game_state:
            return

        gs = self._last_game_state
        team = self.enemy_detector._get_own_team()
        team_name = "CLA" if team == 0 else "RVSF" if team == 1 else "?"

        # HUD background
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (100, 100, 100), 2)

        # Health bar
        hp_width = int(2.5 * gs.health)
        hp_color = (0, 255, 0) if gs.health > 50 else (0, 255, 255) if gs.health > 25 else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (20 + hp_width, 40), hp_color, -1)
        cv2.putText(frame, f"HP: {gs.health}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Armor bar
        ar_width = int(2.5 * gs.armor)
        cv2.rectangle(frame, (20, 60), (20 + ar_width, 75), (255, 200, 0), -1)
        cv2.putText(frame, f"Armor: {gs.armor}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Team and stats
        cv2.putText(frame, f"Team: {team_name}", (150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"K/D: {self._kills}/{self._deaths}", (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Enemies: {len(self._last_enemies)}", (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Step counter
        cv2.putText(frame, f"Step: {self._step_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Reward: {self._episode_reward:.1f}", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Enemy list on right side of original frame
        y_offset = 20
        cv2.putText(frame, "ENEMIES:", (w - 290, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25

        for e in self._last_enemies[:5]:
            fov_str = "FOV" if e.in_fov else "   "
            text = f"[{fov_str}] {e.distance:5.1f}m  ({e.angle_h:+5.1f}h)"
            color = (0, 255, 0) if e.in_fov else (150, 150, 150)
            cv2.putText(frame, text, (w - 290, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18

    def close(self):
        """Clean up resources."""
        self.action_mapper.reset()
        self.memory_reader.detach()
        self.enemy_detector.detach()
        try:
            cv2.destroyAllWindows()
        except:
            pass


def register_env():
    """Register the environment with Gymnasium."""
    gym.register(
        id='AssaultCube-v0',
        entry_point='assaultcube_agent.env:AssaultCubeEnv',
    )
