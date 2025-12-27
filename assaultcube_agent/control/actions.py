"""
Action mapping from RL action space to game inputs.

Converts abstract action vectors to keyboard/mouse commands.
"""

import numpy as np
from dataclasses import dataclass

from .mouse import MouseController
from .keyboard import KeyboardController


@dataclass
class GameAction:
    """
    Decoded game action.

    movement: [forward, backward, left, right, jump, crouch]
    aim: (delta_yaw, delta_pitch) - mouse movement
    shoot: bool
    reload: bool
    """
    movement: list[int]
    aim: tuple[float, float]
    shoot: bool
    reload: bool


class ActionMapper:
    """
    Maps RL action vectors to game inputs.

    Action space:
        - movement: 6 binary values [forward, backward, left, right, jump, crouch]
        - aim: 2 continuous values [delta_yaw, delta_pitch] in range [-1, 1]
        - shoot: 1 binary value
        - reload: 1 binary value

    Total action vector: 10 elements
        [0-5]: movement (6 binary)
        [6-7]: aim (2 continuous, normalized to [-1, 1])
        [8]: shoot (binary)
        [9]: reload (binary)
    """

    def __init__(
        self,
        mouse_sensitivity: float = 5.0,
        aim_scale: float = 10.0,
    ):
        """
        Args:
            mouse_sensitivity: Multiplier for mouse movement
            aim_scale: Scale factor for aim deltas (pixels per unit)
        """
        self.mouse = MouseController(sensitivity=mouse_sensitivity)
        self.keyboard = KeyboardController()
        self.aim_scale = aim_scale

        self._shooting = False

    def decode_action(self, action: np.ndarray) -> GameAction:
        """
        Decode action vector into game action.

        Args:
            action: Action vector of shape (10,) or dict

        Returns:
            GameAction with decoded values
        """
        if isinstance(action, dict):
            return self._decode_dict_action(action)

        # Flat array format
        movement = [int(action[i] > 0.5) for i in range(6)]
        aim = (float(action[6]), float(action[7]))
        shoot = action[8] > 0.5
        reload = action[9] > 0.5

        return GameAction(
            movement=movement,
            aim=aim,
            shoot=shoot,
            reload=reload,
        )

    def _decode_dict_action(self, action: dict) -> GameAction:
        """Decode dict-format action (from Dict action space)."""
        movement = list(action.get('movement', [0]*6))
        aim = tuple(action.get('aim', (0, 0)))
        shoot = action.get('shoot', 0) > 0.5
        reload = action.get('reload', 0) > 0.5

        return GameAction(
            movement=movement,
            aim=aim,
            shoot=shoot,
            reload=reload,
        )

    def execute(self, action: np.ndarray | dict):
        """
        Execute an action in the game.

        Args:
            action: Action vector or dict
        """
        game_action = self.decode_action(action)

        # Apply movement
        self.keyboard.set_movement_from_action(game_action.movement)

        # Apply aim (convert normalized [-1, 1] to pixel deltas)
        dx = int(game_action.aim[0] * self.aim_scale)
        dy = int(game_action.aim[1] * self.aim_scale)
        if dx != 0 or dy != 0:
            self.mouse.move(dx, dy)

        # Apply shooting
        if game_action.shoot and not self._shooting:
            self.mouse.press('left')
            self._shooting = True
        elif not game_action.shoot and self._shooting:
            self.mouse.release('left')
            self._shooting = False

        # Apply reload
        if game_action.reload:
            self.keyboard.reload()

    def reset(self):
        """Reset controller state (release all inputs)."""
        self.keyboard.release_all()
        if self._shooting:
            self.mouse.release('left')
            self._shooting = False

    def click_to_respawn(self):
        """Click mouse1 to respawn after death."""
        import time
        self.mouse.press('left')
        time.sleep(0.05)
        self.mouse.release('left')

    def __del__(self):
        """Cleanup on destruction."""
        self.reset()


def get_action_space_info():
    """
    Get information about the action space.

    Returns dict with:
        - shape: Action vector shape
        - movement_indices: Indices for movement actions
        - aim_indices: Indices for aim actions
        - combat_indices: Indices for combat actions
    """
    return {
        'shape': (10,),
        'movement_indices': (0, 6),      # [0:6] = movement
        'aim_indices': (6, 8),           # [6:8] = aim
        'combat_indices': (8, 10),       # [8:10] = shoot, reload

        'movement_labels': ['forward', 'backward', 'left', 'right', 'jump', 'crouch'],
        'aim_labels': ['delta_yaw', 'delta_pitch'],
        'combat_labels': ['shoot', 'reload'],
    }
