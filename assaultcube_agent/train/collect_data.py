"""
Data collection utilities for training detection models.

Use this to capture screenshots and label them for training
a proper object detection model later.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..vision import ScreenCapture


class DataCollector:
    """
    Collect labeled training data from gameplay.

    Workflow:
    1. Run this while playing the game
    2. Press hotkey to save current frame
    3. Later, use a labeling tool (LabelImg, CVAT, etc.) to annotate
    4. Train detection model on labeled data
    """

    def __init__(self, output_dir: str = "data/assaultcube"):
        """
        Args:
            output_dir: Where to save captured frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.capture = ScreenCapture()
        self.frame_count = 0

        # Create session folder
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / session_name
        self.session_dir.mkdir(exist_ok=True)

        print(f"Saving frames to: {self.session_dir}")

    def save_frame(self, frame: np.ndarray | None = None, label: str = ""):
        """
        Save a frame to disk.

        Args:
            frame: BGR image, or None to capture now
            label: Optional label suffix for the filename
        """
        if frame is None:
            frame = self.capture.capture()

        filename = f"frame_{self.frame_count:05d}"
        if label:
            filename += f"_{label}"
        filename += ".png"

        filepath = self.session_dir / filename
        cv2.imwrite(str(filepath), frame)

        self.frame_count += 1
        print(f"Saved: {filepath.name}")

        return filepath

    def continuous_capture(self, interval: float = 1.0, max_frames: int = 100):
        """
        Automatically capture frames at regular intervals.

        Good for building a dataset quickly while playing.

        Args:
            interval: Seconds between captures
            max_frames: Stop after this many frames
        """
        print(f"Continuous capture: {interval}s interval, max {max_frames} frames")
        print("Press Ctrl+C to stop")

        try:
            for i in range(max_frames):
                self.save_frame()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nCapture stopped")

        print(f"Total frames captured: {self.frame_count}")


def create_annotation_template(image_dir: str, output_file: str = "annotations.json"):
    """
    Create empty annotation file for labeling.

    Generates a JSON file with all images listed, ready for
    manual annotation of bounding boxes.
    """
    image_dir = Path(image_dir)
    images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    annotations = {
        "images": [],
        "categories": [
            {"id": 1, "name": "enemy"},
            {"id": 2, "name": "teammate"},
        ],
        "annotations": [],
    }

    for idx, img_path in enumerate(sorted(images)):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        annotations["images"].append({
            "id": idx,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

    output_path = image_dir / output_file
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Created annotation template: {output_path}")
    print(f"Contains {len(images)} images ready for labeling")
