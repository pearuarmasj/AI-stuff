"""
Seven-segment state extraction pipeline.

Progression:
  1. Single digit → digit + confidence
  2. Multiple digits → string (e.g., "123")
  3. Mixed state → structured dict (digits + booleans)
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from sevenseg_train import load_model, TinyCNN


@dataclass
class DigitResult:
    """Result from a single digit prediction."""
    char: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)


@dataclass
class DisplayState:
    """Structured state extracted from a display image."""
    digits: List[DigitResult]
    text: str  # Combined string, e.g., "123"
    confidence: float  # Overall confidence (product or min)
    booleans: Dict[str, bool] = None  # Named boolean states


class SevenSegReader:
    """
    Reads seven-segment displays from images.

    Usage:
        reader = SevenSegReader()

        # Single digit
        result = reader.read_digit(image_path)

        # Multiple digits with known layout
        result = reader.read_display(image_path, digit_regions=[
            (0, 0, 50, 100),    # First digit bbox
            (50, 0, 100, 100),  # Second digit bbox
            (100, 0, 150, 100), # Third digit bbox
        ])

        # Auto-detect digits (TODO: implement segmentation)
        result = reader.read_display_auto(image_path)
    """

    def __init__(self, model_path: str = "sevenseg_cnn.pth", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.idx_to_char = load_model(model_path, self.device)

    def preprocess_region(self, img: Image.Image, bbox: Tuple[int, int, int, int] = None) -> torch.Tensor:
        """Preprocess an image region for the model."""
        if bbox:
            img = img.crop(bbox)

        # Handle transparency
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")

        # Convert to segment mask
        arr = np.array(img, dtype=np.float32)
        white = np.array([255.0, 255.0, 255.0])
        dist_from_white = np.linalg.norm(arr - white, axis=2) / np.sqrt(3 * 255**2)
        binary = (dist_from_white > 0.3).astype(np.float32)

        # Resize with nearest neighbor
        binary_img = Image.fromarray((binary * 255).astype(np.uint8), mode='L')
        binary_img = binary_img.resize((16, 24), Image.NEAREST)

        gray = np.array(binary_img, dtype=np.float32) / 255.0
        return torch.tensor(gray).unsqueeze(0).unsqueeze(0)  # (1, 1, 24, 16)

    def predict_single(self, img_tensor: torch.Tensor) -> DigitResult:
        """Predict a single digit from preprocessed tensor."""
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
            char = self.idx_to_char[idx]

        return DigitResult(char=char, confidence=conf)

    def read_digit(self, image_path: str, bbox: Tuple[int, int, int, int] = None) -> DigitResult:
        """Read a single digit from an image."""
        img = Image.open(image_path)
        tensor = self.preprocess_region(img, bbox)
        return self.predict_single(tensor)

    def read_display(
        self,
        image_path: str,
        digit_regions: List[Tuple[int, int, int, int]],
        boolean_regions: Dict[str, Tuple[int, int, int, int]] = None,
        boolean_threshold: float = 0.5
    ) -> DisplayState:
        """
        Read multiple digits from known regions.

        Args:
            image_path: Path to the display image
            digit_regions: List of (left, top, right, bottom) bboxes for each digit
            boolean_regions: Dict of name -> bbox for boolean indicators
            boolean_threshold: Brightness threshold for boolean detection

        Returns:
            DisplayState with all extracted information
        """
        img = Image.open(image_path)

        # Handle transparency once
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")

        # Read each digit region
        digits = []
        for bbox in digit_regions:
            tensor = self.preprocess_region(img, bbox)
            result = self.predict_single(tensor)
            result.bbox = bbox
            digits.append(result)

        # Combine into text
        text = "".join(d.char for d in digits)
        overall_conf = min(d.confidence for d in digits) if digits else 0.0

        # Read boolean regions if provided
        booleans = {}
        if boolean_regions:
            for name, bbox in boolean_regions.items():
                region = img.crop(bbox)
                arr = np.array(region.convert("L"), dtype=np.float32) / 255.0
                # Check if region is "lit" (has significant non-white pixels)
                brightness = 1.0 - arr.mean()  # Invert: dark = not lit, bright = lit
                booleans[name] = brightness > boolean_threshold

        return DisplayState(
            digits=digits,
            text=text,
            confidence=overall_conf,
            booleans=booleans if booleans else None
        )

    def find_character_regions(self, img: Image.Image, min_width: int = 5, min_height: int = 10) -> List[Tuple[int, int, int, int]]:
        """
        Auto-detect character bounding boxes using column/row projection.
        Handles multiple rows of characters.
        Returns list of (left, top, right, bottom) tuples sorted top-to-bottom, left-to-right.
        """
        # Handle transparency
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")

        # Convert to binary mask (1 = segment, 0 = background)
        arr = np.array(img, dtype=np.float32)
        white = np.array([255.0, 255.0, 255.0])
        dist_from_white = np.linalg.norm(arr - white, axis=2) / np.sqrt(3 * 255**2)
        binary = (dist_from_white > 0.2).astype(np.uint8)

        # First, find row bands (horizontal slices with content)
        row_sums = binary.sum(axis=1)

        # Find runs of rows with content (these are text rows)
        row_bands = []
        in_row = False
        row_start = 0

        for i, val in enumerate(row_sums):
            if val > 0 and not in_row:
                in_row = True
                row_start = i
            elif val == 0 and in_row:
                in_row = False
                if i - row_start >= min_height:
                    row_bands.append((row_start, i))

        if in_row and len(row_sums) - row_start >= min_height:
            row_bands.append((row_start, len(row_sums)))

        if not row_bands:
            return []

        # For each row band, find character columns
        all_regions = []

        for row_top, row_bottom in row_bands:
            # Get column projection for this row only
            col_sums = binary[row_top:row_bottom, :].sum(axis=0)

            # Find runs of non-zero columns (characters)
            in_char = False
            char_start = 0

            for i, val in enumerate(col_sums):
                if val > 0 and not in_char:
                    in_char = True
                    char_start = i
                elif val == 0 and in_char:
                    in_char = False
                    if i - char_start >= min_width:
                        all_regions.append((char_start, row_top, i, row_bottom))

            if in_char and len(col_sums) - char_start >= min_width:
                all_regions.append((char_start, row_top, len(col_sums), row_bottom))

        return all_regions

    def read_auto(self, image_path: str, min_char_width: int = 5) -> DisplayState:
        """
        Automatically detect and read all characters in an image.
        No need to specify regions - finds them automatically.
        """
        img = Image.open(image_path)
        regions = self.find_character_regions(img, min_width=min_char_width)

        if not regions:
            return DisplayState(digits=[], text="", confidence=0.0)

        return self.read_display(image_path, digit_regions=regions)

    def read_display_grid(
        self,
        image_path: str,
        rows: int,
        cols: int,
        margin: Tuple[int, int, int, int] = (0, 0, 0, 0),  # left, top, right, bottom
        spacing: Tuple[int, int] = (0, 0)  # horizontal, vertical gap between digits
    ) -> DisplayState:
        """
        Read a grid of digits with uniform spacing.

        Useful for displays like "12:34" or multi-line readouts.
        """
        img = Image.open(image_path)
        w, h = img.size

        # Calculate digit dimensions
        content_w = w - margin[0] - margin[2]
        content_h = h - margin[1] - margin[3]

        digit_w = (content_w - (cols - 1) * spacing[0]) / cols
        digit_h = (content_h - (rows - 1) * spacing[1]) / rows

        # Generate bboxes
        regions = []
        for row in range(rows):
            for col in range(cols):
                left = margin[0] + col * (digit_w + spacing[0])
                top = margin[1] + row * (digit_h + spacing[1])
                right = left + digit_w
                bottom = top + digit_h
                regions.append((int(left), int(top), int(right), int(bottom)))

        return self.read_display(image_path, digit_regions=regions)


# Convenience functions
def read_single(image_path: str, bbox: Tuple[int, int, int, int] = None) -> DigitResult:
    """Quick single digit read."""
    reader = SevenSegReader()
    return reader.read_digit(image_path, bbox)


def read_multi(image_path: str, regions: List[Tuple[int, int, int, int]]) -> DisplayState:
    """Quick multi-digit read."""
    reader = SevenSegReader()
    return reader.read_display(image_path, digit_regions=regions)


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  AUTO DETECT:   python sevenseg_state.py image.png auto")
        print("  Single digit:  python sevenseg_state.py image.png")
        print("  Grid mode:     python sevenseg_state.py image.png 3  (for 3 horizontal digits)")
        print("  With regions:  python sevenseg_state.py image.png l1,t1,r1,b1 l2,t2,r2,b2 ...")
        sys.exit(1)

    image_path = sys.argv[1]
    reader = SevenSegReader()

    if len(sys.argv) >= 3 and sys.argv[2].lower() == 'auto':
        # AUTO MODE - detect all characters automatically
        print(f"Auto-detecting characters in {image_path}...")
        img = Image.open(image_path)
        regions = reader.find_character_regions(img)
        print(f"Found {len(regions)} character regions:")
        for i, r in enumerate(regions):
            print(f"  [{i}] bbox: {r}")

        if regions:
            result = reader.read_auto(image_path)
            print(f"\nResult: '{result.text}' (confidence: {result.confidence:.1%})")
            for i, d in enumerate(result.digits):
                print(f"  [{i}] '{d.char}' @ {d.bbox} ({d.confidence:.1%})")
        else:
            print("No characters found!")

    elif len(sys.argv) == 2:
        # Single digit mode
        result = reader.read_digit(image_path)
        print(f"Digit: '{result.char}' (confidence: {result.confidence:.1%})")

    elif len(sys.argv) == 3 and sys.argv[2].isdigit():
        # Grid mode: N horizontal digits
        n_digits = int(sys.argv[2])
        result = reader.read_display_grid(image_path, rows=1, cols=n_digits)
        print(f"Text: '{result.text}' (confidence: {result.confidence:.1%})")
        for i, d in enumerate(result.digits):
            print(f"  [{i}] '{d.char}' @ {d.bbox} ({d.confidence:.1%})")

    else:
        # Explicit regions mode
        regions = []
        for arg in sys.argv[2:]:
            try:
                parts = [int(x) for x in arg.split(",")]
                if len(parts) == 4:
                    regions.append(tuple(parts))
            except ValueError:
                print(f"Skipping invalid arg: {arg}")
                continue

        if regions:
            result = reader.read_display(image_path, digit_regions=regions)
            print(f"Text: '{result.text}' (confidence: {result.confidence:.1%})")
            for i, d in enumerate(result.digits):
                print(f"  [{i}] '{d.char}' @ {d.bbox} ({d.confidence:.1%})")
        else:
            print("No valid regions found.")
            print("Usage: python sevenseg_state.py image.png auto")
            print("   or: python sevenseg_state.py image.png left,top,right,bottom ...")
