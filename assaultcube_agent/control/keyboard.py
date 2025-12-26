"""
Keyboard control module using pydirectinput.

Handles movement and action inputs for the game.
"""

import time
from typing import Set

try:
    import pydirectinput
    DIRECT_INPUT_AVAILABLE = True
except ImportError:
    DIRECT_INPUT_AVAILABLE = False
    print("WARNING: pydirectinput not installed. Run: pip install pydirectinput")


class KeyboardController:
    """
    Control keyboard for game input.

    Uses pydirectinput which sends DirectInput events that games
    actually respond to.
    """

    # Key mappings for AssaultCube
    KEYS = {
        'forward': 'w',
        'backward': 's',
        'strafe_left': 'a',
        'strafe_right': 'd',
        'jump': 'space',
        'crouch': 'ctrl',
        'reload': 'r',
        'use': 'e',
        'weapon_1': '1',
        'weapon_2': '2',
        'weapon_3': '3',
        'weapon_4': '4',
        'weapon_5': '5',
        'weapon_6': '6',
        'chat': 't',
        'console': '/',
    }

    def __init__(self):
        """Initialize the keyboard controller."""
        if not DIRECT_INPUT_AVAILABLE:
            raise RuntimeError("pydirectinput is required. Install with: pip install pydirectinput")

        # Disable pydirectinput's built-in pause
        pydirectinput.PAUSE = 0

        # Track currently held keys
        self._held_keys: Set[str] = set()

    def press(self, key: str):
        """Press and release a key."""
        actual_key = self.KEYS.get(key, key)
        pydirectinput.press(actual_key)

    def key_down(self, key: str):
        """
        Hold a key down.

        Args:
            key: Key name from KEYS dict or raw key
        """
        actual_key = self.KEYS.get(key, key)
        if actual_key not in self._held_keys:
            pydirectinput.keyDown(actual_key)
            self._held_keys.add(actual_key)

    def key_up(self, key: str):
        """
        Release a held key.

        Args:
            key: Key name from KEYS dict or raw key
        """
        actual_key = self.KEYS.get(key, key)
        if actual_key in self._held_keys:
            pydirectinput.keyUp(actual_key)
            self._held_keys.discard(actual_key)

    def release_all(self):
        """Release all currently held keys."""
        for key in list(self._held_keys):
            pydirectinput.keyUp(key)
        self._held_keys.clear()

    def set_movement(
        self,
        forward: bool = False,
        backward: bool = False,
        left: bool = False,
        right: bool = False,
        jump: bool = False,
        crouch: bool = False,
    ):
        """
        Set movement state. Call each frame to update.

        Args:
            forward: W key
            backward: S key
            left: A key
            right: D key
            jump: Space
            crouch: Ctrl
        """
        # Forward/backward
        if forward:
            self.key_down('forward')
        else:
            self.key_up('forward')

        if backward:
            self.key_down('backward')
        else:
            self.key_up('backward')

        # Strafe
        if left:
            self.key_down('strafe_left')
        else:
            self.key_up('strafe_left')

        if right:
            self.key_down('strafe_right')
        else:
            self.key_up('strafe_right')

        # Jump
        if jump:
            self.key_down('jump')
        else:
            self.key_up('jump')

        # Crouch
        if crouch:
            self.key_down('crouch')
        else:
            self.key_up('crouch')

    def set_movement_from_action(self, action: list[int]):
        """
        Set movement from action array.

        Args:
            action: [forward, backward, left, right, jump, crouch]
                    Each element is 0 or 1
        """
        if len(action) != 6:
            raise ValueError("Action must have 6 elements: [forward, backward, left, right, jump, crouch]")

        self.set_movement(
            forward=bool(action[0]),
            backward=bool(action[1]),
            left=bool(action[2]),
            right=bool(action[3]),
            jump=bool(action[4]),
            crouch=bool(action[5]),
        )

    def reload(self):
        """Press reload key."""
        self.press('reload')

    def switch_weapon(self, weapon_num: int):
        """
        Switch to a weapon.

        Args:
            weapon_num: 1-6 for weapon slots
        """
        if 1 <= weapon_num <= 6:
            self.press(f'weapon_{weapon_num}')

    def send_command(self, command: str):
        """
        Send a console command to the game.

        Args:
            command: Command without the leading /
        """
        # Open console
        self.press('/')
        time.sleep(0.05)

        # Type command
        pydirectinput.typewrite(command)
        time.sleep(0.05)

        # Press enter
        pydirectinput.press('enter')

    def __del__(self):
        """Ensure all keys are released on cleanup."""
        self.release_all()
