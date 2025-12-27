"""
Depth estimation module using Depth Anything V2.

Provides spatial awareness for the agent by estimating depth from RGB frames.
"""

import numpy as np
import torch
import cv2
from PIL import Image


class DepthEstimator:
    """
    Depth estimation using Depth Anything V2, or fast grayscale mode.

    Converts RGB frames to depth maps for spatial awareness.
    The agent uses this to understand distances to walls/enemies.
    """

    def __init__(
        self,
        model_size: str = "none",
        device: str | None = None,
        output_size: tuple[int, int] = (160, 120),
    ):
        """
        Args:
            model_size: "small", "base", "large", or "none" (grayscale, no model)
            device: "cuda", "cpu", or None for auto-detect
            output_size: (width, height) of output depth map
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.output_size = output_size
        self.model_size = model_size

        # "none" = no depth model, just use grayscale (FAST)
        if model_size == "none":
            print("[DepthEstimator] Using grayscale mode (no depth model) - FAST")
            self.pipe = None
            return

        # Model selection - using transformers pipeline for simplicity
        model_ids = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }

        if model_size not in model_ids:
            raise ValueError(f"model_size must be one of {list(model_ids.keys())} or 'none'")

        self.model_id = model_ids[model_size]
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Load the depth estimation model."""
        from transformers import pipeline

        print(f"Loading depth model: {self.model_id}")
        print(f"Device: {self.device}")

        self.pipe = pipeline(
            task="depth-estimation",
            model=self.model_id,
            device=self.device,
        )

        print("Depth model loaded.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from an RGB frame.

        Args:
            frame: BGR image from screen capture (H, W, 3)

        Returns:
            Depth map as float32 array (output_size[1], output_size[0])
            Values normalized to 0-1 range
        """
        # FAST MODE: Just use grayscale (no depth model)
        if self.pipe is None:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self.output_size, interpolation=cv2.INTER_LINEAR)
            # Normalize to 0-1
            return gray.astype(np.float32) / 255.0

        # SLOW MODE: Full depth model inference
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image (required by pipeline)
        pil_image = Image.fromarray(rgb)

        # Run depth estimation
        result = self.pipe(pil_image)

        # Get depth map
        depth = np.array(result["depth"])

        # Resize to output size
        depth = cv2.resize(
            depth,
            self.output_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize to 0-1
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def estimate_with_rays(
        self,
        frame: np.ndarray,
        num_horizontal_rays: int = 30,
        num_vertical_rays: int = 5,
    ) -> np.ndarray:
        """
        Estimate depth and sample as pseudo-rays.

        Args:
            frame: BGR image from screen capture
            num_horizontal_rays: Number of horizontal samples
            num_vertical_rays: Number of vertical samples

        Returns:
            Ray depths as (num_vertical_rays, num_horizontal_rays) array
            Values in 0-1 range (0 = close, 1 = far)
        """
        depth = self.estimate(frame)

        # Sample rays from depth map
        h, w = depth.shape
        v_indices = np.linspace(0, h - 1, num_vertical_rays, dtype=int)
        h_indices = np.linspace(0, w - 1, num_horizontal_rays, dtype=int)

        rays = depth[np.ix_(v_indices, h_indices)]

        return rays

    def visualize(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth map to colorized visualization.

        Args:
            depth: Depth map (H, W) in 0-1 range

        Returns:
            Colorized depth map (H, W, 3) as BGR
        """
        # Convert to uint8
        depth_uint8 = (depth * 255).astype(np.uint8)

        # Apply colormap
        colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        return colorized
