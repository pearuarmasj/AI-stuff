"""
HUD reading module for extracting game state from screen.

Reads health, armor, ammo from the AssaultCube HUD using OCR.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class HUDState:
    """Current game state read from HUD."""
    health: int = 100
    armor: int = 0
    ammo_mag: int = 0      # Current magazine
    ammo_reserve: int = 0  # Reserve ammo
    grenades: int = 0


class HUDReader:
    """
    Read game state from AssaultCube HUD.

    Uses OCR to extract health, armor, and ammo values from
    calibrated screen regions. Run calibrate.py first to define regions.
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        use_easyocr: bool = True,
        config_path: str | Path | None = None,
    ):
        """
        Args:
            screen_width: Game screen width
            screen_height: Game screen height
            use_easyocr: Use EasyOCR (more accurate) vs pytesseract
            config_path: Path to HUD config JSON (from calibration)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.use_easyocr = use_easyocr
        self.ocr = None
        self._init_ocr()

        # Load calibrated regions or use defaults
        self._load_regions(config_path)

    def _init_ocr(self):
        """Initialize OCR engine."""
        if self.use_easyocr:
            try:
                import easyocr
                self.ocr = easyocr.Reader(['en'], gpu=True)
                print("EasyOCR initialized with GPU")
            except ImportError:
                print("EasyOCR not available, falling back to pytesseract")
                self.use_easyocr = False

        if not self.use_easyocr:
            try:
                import pytesseract
                self.ocr = pytesseract
                print("Using pytesseract for OCR")
            except ImportError:
                print("WARNING: No OCR engine available!")
                self.ocr = None

    def _load_regions(self, config_path: str | Path | None):
        """Load HUD regions from config file or use defaults."""
        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent / "hud_config.json"
        else:
            config_path = Path(config_path)

        # Try to load calibrated regions
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                regions = config.get("regions", {})

                if "health" in regions:
                    r = regions["health"]
                    self.health_region = (r["x"], r["y"], r["width"], r["height"])
                    print(f"Loaded health region: {self.health_region}")
                else:
                    self._set_default_health_region()

                if "ammo_mag" in regions:
                    r = regions["ammo_mag"]
                    self.ammo_mag_region = (r["x"], r["y"], r["width"], r["height"])
                    print(f"Loaded ammo_mag region: {self.ammo_mag_region}")
                else:
                    self.ammo_mag_region = None

                if "ammo_reserve" in regions:
                    r = regions["ammo_reserve"]
                    self.ammo_reserve_region = (r["x"], r["y"], r["width"], r["height"])
                    print(f"Loaded ammo_reserve region: {self.ammo_reserve_region}")
                else:
                    self.ammo_reserve_region = None

                # Armor is optional (user said to ignore for now)
                if "armor" in regions:
                    r = regions["armor"]
                    self.armor_region = (r["x"], r["y"], r["width"], r["height"])
                else:
                    self.armor_region = None

                print(f"Loaded HUD config from: {config_path}")
                return

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading HUD config: {e}")

        # No config found - use defaults (probably wrong, run calibration!)
        print("WARNING: No HUD calibration found! Run calibrate.py first.")
        self._set_default_health_region()
        self.ammo_mag_region = None
        self.ammo_reserve_region = None
        self.armor_region = None

    def _set_default_health_region(self):
        """Set default health region (probably wrong)."""
        h = self.screen_height
        w = self.screen_width
        self.health_region = (
            int(w * 0.02),
            int(h * 0.85),
            int(w * 0.08),
            int(h * 0.12),
        )

    def _extract_region(self, frame: np.ndarray, region: tuple) -> np.ndarray:
        """Extract a region from the frame."""
        x, y, w, h = region
        return frame[y:y+h, x:x+w].copy()

    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Preprocess region for AC's stylized HUD font (white with black outline)."""
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # The AC HUD has white text with black outlines
        # Use adaptive threshold or high threshold to get just the white parts
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Dilate slightly to fill in the stylized font gaps
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Upscale for better OCR
        scale = 3
        thresh = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return thresh

    def _ocr_number(self, region: np.ndarray) -> int:
        """Extract a single number from an image region using OCR."""
        if self.ocr is None:
            return 0

        processed = self._preprocess_for_ocr(region)

        try:
            if self.use_easyocr:
                # Get detailed results with bounding boxes
                results = self.ocr.readtext(processed, detail=1, allowlist='0123456789')
                if results:
                    # Take the text with highest confidence
                    best = max(results, key=lambda x: x[2])
                    text = best[1]
                else:
                    text = ''
            else:
                text = self.ocr.image_to_string(
                    processed,
                    config='--psm 7 -c tessedit_char_whitelist=0123456789'
                )

            # Extract digits
            digits = ''.join(c for c in text if c.isdigit())
            return int(digits) if digits else 0

        except Exception:
            return 0

    def read(self, frame: np.ndarray) -> HUDState:
        """
        Read HUD state from a frame.

        Args:
            frame: BGR image of the game screen

        Returns:
            HUDState with extracted values
        """
        state = HUDState()

        # Extract and OCR each region
        if self.health_region:
            health_img = self._extract_region(frame, self.health_region)
            state.health = self._ocr_number(health_img)
            state.health = max(0, min(100, state.health))

        if self.armor_region:
            armor_img = self._extract_region(frame, self.armor_region)
            state.armor = self._ocr_number(armor_img)
            state.armor = max(0, min(100, state.armor))

        if self.ammo_mag_region:
            ammo_mag_img = self._extract_region(frame, self.ammo_mag_region)
            state.ammo_mag = self._ocr_number(ammo_mag_img)

        if self.ammo_reserve_region:
            ammo_reserve_img = self._extract_region(frame, self.ammo_reserve_region)
            state.ammo_reserve = self._ocr_number(ammo_reserve_img)

        return state

    def read_fast(self, frame: np.ndarray) -> HUDState:
        """
        Fast HUD reading using color detection instead of OCR.

        Less accurate but much faster. Estimates values based on
        HUD bar lengths or icon states.

        For training, accuracy matters less than speed.
        """
        # Simplified: just return defaults for now
        # TODO: Implement bar-length detection
        return HUDState()

    def visualize_regions(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw HUD regions on frame for debugging.

        Args:
            frame: BGR image

        Returns:
            Frame with regions drawn
        """
        vis = np.ascontiguousarray(frame.copy())

        # Draw health region (green)
        if self.health_region:
            x, y, w, h = self.health_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, "HEALTH", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw armor region (blue) - optional
        if self.armor_region:
            x, y, w, h = self.armor_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis, "ARMOR", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw ammo mag region (orange)
        if self.ammo_mag_region:
            x, y, w, h = self.ammo_mag_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(vis, "AMMO_MAG", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Draw ammo reserve region (cyan)
        if self.ammo_reserve_region:
            x, y, w, h = self.ammo_reserve_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(vis, "AMMO_RES", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return vis

    def calibrate(self, frame: np.ndarray):
        """
        Interactive calibration mode to adjust HUD regions.

        Opens a window where you can see the regions and adjust them.
        """
        print("Calibration mode - showing detected regions")
        print("Press 'q' to quit")

        vis = self.visualize_regions(frame)
        cv2.imshow("HUD Calibration", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
