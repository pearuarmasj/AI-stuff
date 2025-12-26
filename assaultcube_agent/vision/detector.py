"""
Enemy detection module.

Phase 1: Color-based detection (AssaultCube has distinct team colors)
Phase 2: Train a proper CNN/YOLO model for robust detection
"""

import numpy as np
import cv2


class EnemyDetector:
    """
    Detect enemies in game frames.

    Initial implementation uses color-based detection since AssaultCube
    has distinct red/blue team colors. Can be upgraded to a trained
    model later for more robust detection.
    """

    # AssaultCube enemy colors (HSV ranges) - will need tuning
    # Red team (approximate, adjust based on actual game)
    RED_LOWER = np.array([0, 120, 70])
    RED_UPPER = np.array([10, 255, 255])
    RED_LOWER2 = np.array([170, 120, 70])
    RED_UPPER2 = np.array([180, 255, 255])

    # Blue team (approximate)
    BLUE_LOWER = np.array([100, 120, 70])
    BLUE_UPPER = np.array([130, 255, 255])

    def __init__(self, target_color: str = "red", min_area: int = 100):
        """
        Args:
            target_color: "red" or "blue" depending on enemy team
            min_area: Minimum contour area to consider as a valid detection
        """
        self.target_color = target_color
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect enemies in a frame.

        Args:
            frame: BGR image from screen capture

        Returns:
            List of detections, each with:
                - bbox: (x, y, w, h) bounding box
                - center: (cx, cy) center point
                - area: contour area
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.target_color == "red":
            mask1 = cv2.inRange(hsv, self.RED_LOWER, self.RED_UPPER)
            mask2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
            mask = mask1 | mask2
        else:
            mask = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            detections.append({
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "area": area,
            })

        # Sort by area (largest first - likely closest enemy)
        detections.sort(key=lambda d: d["area"], reverse=True)

        return detections

    def get_target(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Get the primary target to aim at.

        Returns:
            (x, y) pixel coordinates of target, or None if no enemy found
        """
        detections = self.detect(frame)
        if not detections:
            return None
        return detections[0]["center"]

    def get_offset_from_center(
        self, frame: np.ndarray, frame_center: tuple[int, int] | None = None
    ) -> tuple[int, int] | None:
        """
        Get the offset from crosshair to target.

        This is what you feed to the mouse controller.

        Args:
            frame: BGR image
            frame_center: (x, y) of crosshair. If None, uses frame center.

        Returns:
            (dx, dy) offset to move, or None if no target
        """
        target = self.get_target(frame)
        if target is None:
            return None

        if frame_center is None:
            h, w = frame.shape[:2]
            frame_center = (w // 2, h // 2)

        dx = target[0] - frame_center[0]
        dy = target[1] - frame_center[1]

        return (dx, dy)
