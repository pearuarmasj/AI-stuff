"""
Mouse control module.

Uses pydirectinput for DirectInput compatibility with games.
Falls back to standard methods if needed.
"""

import time

try:
    import pydirectinput
    DIRECT_INPUT_AVAILABLE = True
except ImportError:
    DIRECT_INPUT_AVAILABLE = False
    print("WARNING: pydirectinput not installed. Run: pip install pydirectinput")


class MouseController:
    """
    Control mouse for game input.

    Uses pydirectinput which sends DirectInput events that games
    actually respond to (unlike regular win32 SendInput in many cases).
    """

    def __init__(self, sensitivity: float = 1.0):
        """
        Args:
            sensitivity: Multiplier for mouse movement. Adjust based on
                         in-game sensitivity settings.
        """
        if not DIRECT_INPUT_AVAILABLE:
            raise RuntimeError("pydirectinput is required. Install with: pip install pydirectinput")

        self.sensitivity = sensitivity

        # Disable pydirectinput's built-in pause
        pydirectinput.PAUSE = 0

    def move(self, dx: int, dy: int):
        """
        Move mouse by relative offset.

        Args:
            dx: Horizontal movement (positive = right)
            dy: Vertical movement (positive = down)
        """
        # Apply sensitivity scaling
        dx = int(dx * self.sensitivity)
        dy = int(dy * self.sensitivity)

        pydirectinput.moveRel(dx, dy, relative=True)

    def move_smooth(self, dx: int, dy: int, steps: int = 5, duration: float = 0.05):
        """
        Move mouse smoothly over multiple steps.

        Looks more human and may be needed for some games.
        """
        step_dx = dx / steps
        step_dy = dy / steps
        step_delay = duration / steps

        for _ in range(steps):
            self.move(int(step_dx), int(step_dy))
            time.sleep(step_delay)

    def click(self, button: str = "left"):
        """Click mouse button."""
        pydirectinput.click(button=button)

    def press(self, button: str = "left"):
        """Press and hold mouse button."""
        pydirectinput.mouseDown(button=button)

    def release(self, button: str = "left"):
        """Release mouse button."""
        pydirectinput.mouseUp(button=button)
