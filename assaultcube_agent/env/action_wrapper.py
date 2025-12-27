"""
Action space wrapper for SB3 compatibility.

Converts Dict action space to flat Box for SB3, then converts back to Dict.
This properly handles the hybrid discrete + continuous action space.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenActionWrapper(gym.ActionWrapper):
    """
    Flattens a Dict action space to a Box for SB3 compatibility.

    Original Dict space:
        movement: MultiDiscrete([3, 3, 2, 2]) - 4 discrete values
        aim: Box([-1, 1], (2,)) - 2 continuous values
        combat: MultiDiscrete([2, 2]) - 2 discrete values

    Flattened Box space:
        [movement_0, movement_1, movement_2, movement_3, aim_0, aim_1, combat_0, combat_1]
        All in range [-1, 1]

    Discrete actions are decoded using thresholds:
        - For MultiDiscrete([3]): value < -0.33 → 0, value > 0.33 → 2, else → 1
        - For MultiDiscrete([2]): value > 0 → 1, else → 0
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Verify original action space
        assert isinstance(env.action_space, spaces.Dict), "Expected Dict action space"
        assert 'movement' in env.action_space.spaces
        assert 'aim' in env.action_space.spaces
        assert 'combat' in env.action_space.spaces

        self._original_space = env.action_space

        # Calculate flat size: 4 (movement) + 2 (aim) + 2 (combat) = 8
        self._flat_size = 4 + 2 + 2

        # New flat action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._flat_size,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict:
        """Convert flat Box action to Dict action."""
        # Movement: 4 values
        # [0]: forward/none/backward → MultiDiscrete([3])
        # [1]: left/none/right → MultiDiscrete([3])
        # [2]: jump → MultiDiscrete([2])
        # [3]: crouch → MultiDiscrete([2])
        movement = np.array([
            self._continuous_to_3(action[0]),  # fwd/none/back
            self._continuous_to_3(action[1]),  # left/none/right
            self._continuous_to_2(action[2]),  # jump
            self._continuous_to_2(action[3]),  # crouch
        ], dtype=np.int64)

        # Aim: 2 continuous values (pass through)
        aim = np.array([action[4], action[5]], dtype=np.float32)

        # Combat: 2 values
        combat = np.array([
            self._continuous_to_2(action[6]),  # shoot
            self._continuous_to_2(action[7]),  # reload
        ], dtype=np.int64)

        return {
            'movement': movement,
            'aim': aim,
            'combat': combat,
        }

    def _continuous_to_3(self, value: float) -> int:
        """Convert continuous [-1, 1] to discrete [0, 1, 2]."""
        if value < -0.33:
            return 0  # First option (e.g., forward)
        elif value > 0.33:
            return 2  # Third option (e.g., backward)
        else:
            return 1  # Middle option (e.g., none)

    def _continuous_to_2(self, value: float) -> int:
        """Convert continuous [-1, 1] to discrete [0, 1]."""
        return 1 if value > 0 else 0

    def reverse_action(self, action: dict) -> np.ndarray:
        """Convert Dict action back to flat Box (for logging/debugging)."""
        flat = np.zeros(self._flat_size, dtype=np.float32)

        # Movement
        flat[0] = self._3_to_continuous(action['movement'][0])
        flat[1] = self._3_to_continuous(action['movement'][1])
        flat[2] = self._2_to_continuous(action['movement'][2])
        flat[3] = self._2_to_continuous(action['movement'][3])

        # Aim
        flat[4] = action['aim'][0]
        flat[5] = action['aim'][1]

        # Combat
        flat[6] = self._2_to_continuous(action['combat'][0])
        flat[7] = self._2_to_continuous(action['combat'][1])

        return flat

    def _3_to_continuous(self, value: int) -> float:
        """Convert discrete [0, 1, 2] to continuous [-1, 0, 1]."""
        return float(value - 1)

    def _2_to_continuous(self, value: int) -> float:
        """Convert discrete [0, 1] to continuous [-1, 1]."""
        return 1.0 if value == 1 else -1.0


def wrap_env_for_sb3(env: gym.Env) -> gym.Env:
    """Wrap environment with action flattening for SB3 compatibility."""
    if isinstance(env.action_space, spaces.Dict):
        return FlattenActionWrapper(env)
    return env
