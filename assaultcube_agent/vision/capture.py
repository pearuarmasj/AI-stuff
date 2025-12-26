"""
Screen capture module using MSS for fast frame grabbing.
"""

import numpy as np
import mss
import mss.tools


class ScreenCapture:
    """
    Fast screen capture for game frames.

    Uses MSS which is significantly faster than PIL or pyautogui
    for continuous frame capture.
    """

    def __init__(self, monitor: int = 1, region: dict | None = None):
        """
        Args:
            monitor: Monitor index (1 = primary)
            region: Optional dict with top, left, width, height to capture subset
        """
        self.sct = mss.mss()
        self.monitor = monitor
        self.region = region

    def get_monitor_info(self) -> dict:
        """Get dimensions of the target monitor."""
        return self.sct.monitors[self.monitor]

    def capture(self) -> np.ndarray:
        """
        Capture a single frame.

        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        target = self.region if self.region else self.sct.monitors[self.monitor]
        screenshot = self.sct.grab(target)

        # Convert to numpy array (BGRA format)
        frame = np.array(screenshot)

        # Drop alpha channel -> BGR
        return frame[:, :, :3]

    def capture_center(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Capture a centered region of the screen.

        Useful for focusing on the crosshair area.
        """
        mon = self.sct.monitors[self.monitor]
        center_x = mon["left"] + mon["width"] // 2
        center_y = mon["top"] + mon["height"] // 2

        region = {
            "left": center_x - width // 2,
            "top": center_y - height // 2,
            "width": width,
            "height": height,
        }

        screenshot = self.sct.grab(region)
        frame = np.array(screenshot)
        return frame[:, :, :3]
