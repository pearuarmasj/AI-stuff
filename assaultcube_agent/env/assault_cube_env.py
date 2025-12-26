"""
Gymnasium environment wrapper for AssaultCube.

This wraps the game as a reinforcement learning environment
compatible with Stable-Baselines3.
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..vision import ScreenCapture
from ..depth import DepthEstimator
from ..hud import HUDReader
from ..control import ActionMapper


class AssaultCubeEnv(gym.Env):
    """
    AssaultCube as a Gymnasium environment.

    Observation space:
        - rgb: Downscaled RGB image (H, W, 3)
        - depth: Depth map (H, W, 1)
        - hud: [health, armor, ammo] normalized

    Action space:
        - Hybrid: MultiDiscrete for movement + Box for aim + MultiDiscrete for combat
        - Or flattened to Box for simplicity

    The environment captures the game screen, processes it through
    depth estimation, and executes actions via keyboard/mouse.
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
        render_mode: str | None = None,
    ):
        """
        Args:
            screen_width: Game screen width
            screen_height: Game screen height
            obs_width: Observation image width
            obs_height: Observation image height
            depth_model_size: "small", "base", or "large"
            frame_skip: Number of frames to repeat each action
            max_episode_steps: Max steps per episode
            render_mode: "human" or "rgb_array"
        """
        super().__init__()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Initialize components
        print("Initializing AssaultCube environment...")
        self.capture = ScreenCapture()
        print("  Screen capture ready")

        self.depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            output_size=(obs_width, obs_height),
        )
        print("  Depth estimator ready")

        self.hud_reader = HUDReader(
            screen_width=screen_width,
            screen_height=screen_height,
        )
        print("  HUD reader ready")

        self.action_mapper = ActionMapper()
        print("  Action mapper ready")

        # Define observation space
        # RGB + Depth stacked as 4 channels
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(obs_height, obs_width, 4),  # RGB + Depth
                dtype=np.uint8,
            ),
            'hud': spaces.Box(
                low=0,
                high=1,
                shape=(3,),  # health, armor, ammo (normalized)
                dtype=np.float32,
            ),
        })

        # Define action space
        # Using Box for everything to keep it simple for SB3
        # [0-5]: movement (binary, but represented as continuous)
        # [6-7]: aim (continuous)
        # [8-9]: combat (binary)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0
        self._last_hud_state = None
        self._last_frame = None

        # Reward tracking for shaping
        self._last_health = 100
        self._last_armor = 0
        self._kills = 0
        self._deaths = 0

        print("AssaultCube environment initialized!")

    def _get_observation(self) -> dict:
        """Capture and process observation from the game."""
        # Capture screen
        frame = self.capture.capture()
        self._last_frame = frame

        # Get depth
        depth = self.depth_estimator.estimate(frame)

        # Downscale RGB
        import cv2
        rgb = cv2.resize(frame, (self.obs_width, self.obs_height))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Stack RGB + depth (depth as 4th channel)
        depth_uint8 = (depth * 255).astype(np.uint8)
        depth_uint8 = depth_uint8[:, :, np.newaxis]
        image = np.concatenate([rgb, depth_uint8], axis=2)

        # Get HUD state
        hud_state = self.hud_reader.read(frame)
        self._last_hud_state = hud_state

        hud = np.array([
            hud_state.health / 100.0,
            hud_state.armor / 100.0,
            min(hud_state.ammo_mag, 100) / 100.0,  # Normalize ammo
        ], dtype=np.float32)

        return {
            'image': image,
            'hud': hud,
        }

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state changes."""
        if self._last_hud_state is None:
            return 0.0

        reward = 0.0

        # Survival bonus
        reward += 0.01

        # Health change penalty
        health_delta = self._last_hud_state.health - self._last_health
        if health_delta < 0:
            reward += health_delta * 0.5  # Penalty for taking damage

        # Track for next step
        self._last_health = self._last_hud_state.health
        self._last_armor = self._last_hud_state.armor

        # Death detection (health dropped to 0)
        if self._last_hud_state.health <= 0:
            reward -= 50.0
            self._deaths += 1

        # TODO: Kill detection requires more sophisticated tracking
        # Could use screen text detection or audio cues

        return reward

    def _is_done(self) -> bool:
        """Check if episode should end."""
        # End on death
        if self._last_hud_state and self._last_hud_state.health <= 0:
            return True

        # End on max steps
        if self._step_count >= self.max_episode_steps:
            return True

        return False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment.

        Note: This doesn't actually reset the game - you need to
        handle respawning in-game or via console commands.
        """
        super().reset(seed=seed)

        # Reset tracking
        self._step_count = 0
        self._episode_reward = 0
        self._last_health = 100
        self._last_armor = 0

        # Release all controls
        self.action_mapper.reset()

        # Wait for respawn (if dead)
        time.sleep(0.5)

        # Get initial observation
        obs = self._get_observation()

        info = {
            'episode_reward': self._episode_reward,
            'kills': self._kills,
            'deaths': self._deaths,
        }

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute action and get next state.

        Args:
            action: Action vector of shape (10,)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute action for frame_skip frames
        for _ in range(self.frame_skip):
            self.action_mapper.execute(action)
            time.sleep(1/60)  # ~60 FPS

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()
        self._episode_reward += reward

        # Check done conditions
        terminated = self._is_done()
        truncated = self._step_count >= self.max_episode_steps

        self._step_count += 1

        info = {
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'health': self._last_hud_state.health if self._last_hud_state else 0,
            'armor': self._last_hud_state.armor if self._last_hud_state else 0,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self._last_frame is None:
            return None

        if self.render_mode == "rgb_array":
            return self._last_frame

        elif self.render_mode == "human":
            import cv2
            cv2.imshow("AssaultCube Env", self._last_frame)
            cv2.waitKey(1)

    def close(self):
        """Clean up resources."""
        self.action_mapper.reset()
        import cv2
        cv2.destroyAllWindows()


# Register the environment
def register_env():
    """Register the environment with Gymnasium."""
    gym.register(
        id='AssaultCube-v0',
        entry_point='assaultcube_agent.env:AssaultCubeEnv',
    )
