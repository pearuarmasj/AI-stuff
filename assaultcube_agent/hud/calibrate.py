"""
Interactive HUD region calibration tool using tkinter.

Click and drag to define regions for health and ammo.
"""

import json
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

import mss
import numpy as np


class HUDCalibrator:
    """Interactive tool to calibrate HUD regions by clicking."""

    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor

        self.regions = {}
        self.current_region = None
        self.start_point = None
        self.end_point = None
        self.drawing = False

        # Regions to calibrate (in order)
        self.regions_to_calibrate = ["health", "ammo"]
        self.current_index = 0

        self.frame = None
        self.tk_image = None
        self.canvas = None
        self.root = None
        self.rect_id = None

    def capture_frame(self) -> np.ndarray:
        """Capture current screen."""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        # Convert BGRA to RGB
        frame = frame[:, :, [2, 1, 0]]
        return frame

    def update_instructions(self):
        """Update the instruction label."""
        if self.current_index < len(self.regions_to_calibrate):
            current = self.regions_to_calibrate[self.current_index]
            text = f"Draw {current.upper()} region - Click and drag"
        else:
            text = "Done! Press ENTER to save, ESC to cancel"
        self.instruction_label.config(text=text)

    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if self.current_index >= len(self.regions_to_calibrate):
            return

        self.drawing = True
        self.start_point = (event.x, event.y)
        self.end_point = (event.x, event.y)

        # Create rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='yellow', width=2
        )

    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if self.drawing and self.rect_id:
            self.end_point = (event.x, event.y)
            self.canvas.coords(
                self.rect_id,
                self.start_point[0], self.start_point[1],
                event.x, event.y
            )

    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if not self.drawing:
            return

        self.drawing = False
        self.end_point = (event.x, event.y)

        # Calculate region
        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])

        # Scale back to original screen coordinates
        scale_x = self.monitor["width"] / self.display_width
        scale_y = self.monitor["height"] / self.display_height

        orig_x = int(x1 * scale_x)
        orig_y = int(y1 * scale_y)
        orig_w = int((x2 - x1) * scale_x)
        orig_h = int((y2 - y1) * scale_y)

        if orig_w > 5 and orig_h > 5:  # Minimum size
            region_name = self.regions_to_calibrate[self.current_index]
            self.regions[region_name] = {
                "x": orig_x,
                "y": orig_y,
                "width": orig_w,
                "height": orig_h
            }
            print(f"  Saved {region_name} region: x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")

            # Color the saved rectangle
            colors = {"health": "green", "ammo": "orange"}
            color = colors.get(region_name, "white")
            self.canvas.itemconfig(self.rect_id, outline=color)

            # Add label
            self.canvas.create_text(
                x1, y1 - 10,
                text=region_name.upper(),
                fill=color,
                anchor='sw',
                font=('Arial', 12, 'bold')
            )

            self.rect_id = None

            # Move to next region
            self.current_index += 1
            self.update_instructions()

    def on_key(self, event):
        """Handle key presses."""
        if event.keysym == 'Escape':
            print("\nCalibration cancelled.")
            self.regions = {}
            self.root.destroy()

        elif event.keysym == 'Return':
            if len(self.regions) >= 2:
                self.save_config()
                self.root.destroy()
            else:
                print("Please define all regions before saving.")

        elif event.keysym.lower() == 'r':
            print("Refreshing screenshot...")
            self.refresh_screenshot()

    def refresh_screenshot(self):
        """Capture a new screenshot and update display."""
        self.frame = self.capture_frame()
        pil_image = Image.fromarray(self.frame)
        pil_image = pil_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

        # Redraw saved regions
        self.redraw_regions()

    def redraw_regions(self):
        """Redraw all saved regions."""
        colors = {"health": "green", "ammo": "orange"}
        scale_x = self.display_width / self.monitor["width"]
        scale_y = self.display_height / self.monitor["height"]

        for name, region in self.regions.items():
            x = int(region["x"] * scale_x)
            y = int(region["y"] * scale_y)
            w = int(region["width"] * scale_x)
            h = int(region["height"] * scale_y)
            color = colors.get(name, "white")

            self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=2)
            self.canvas.create_text(x, y - 10, text=name.upper(), fill=color, anchor='sw', font=('Arial', 12, 'bold'))

    def save_config(self):
        """Save regions to JSON config file."""
        path = Path(__file__).parent / "hud_config.json"

        config = {
            "screen_width": self.monitor["width"],
            "screen_height": self.monitor["height"],
            "regions": self.regions
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved config to: {path}")
        print("Calibration complete!")

    def run(self):
        """Run the calibration tool."""
        print("=" * 60)
        print("HUD Region Calibration")
        print("=" * 60)
        print("\nInstructions:")
        print("  1. Make sure AssaultCube is visible on screen")
        print("  2. Click and drag to draw a box around each HUD element")
        print("  3. Press ENTER when done to save")
        print("  4. Press ESC to cancel")
        print("  5. Press R to refresh the screenshot")
        print("\n" + "-" * 60)

        # Capture initial frame
        self.frame = self.capture_frame()

        # Create tkinter window
        self.root = tk.Tk()
        self.root.title("HUD Calibration - Draw regions")

        # Calculate display size (80% of screen)
        self.display_width = int(self.monitor["width"] * 0.7)
        self.display_height = int(self.monitor["height"] * 0.7)

        # Create instruction label
        self.instruction_label = tk.Label(
            self.root,
            text="Draw HEALTH region first...",
            font=('Arial', 14),
            pady=10
        )
        self.instruction_label.pack()

        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.display_width,
            height=self.display_height
        )
        self.canvas.pack()

        # Convert frame to tkinter image
        pil_image = Image.fromarray(self.frame)
        pil_image = pil_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # Display image on canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

        # Bind mouse events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)

        # Bind keyboard events
        self.root.bind('<Key>', self.on_key)

        # Help text at bottom
        help_label = tk.Label(
            self.root,
            text="ENTER = Save | ESC = Cancel | R = Refresh screenshot",
            font=('Arial', 10),
            fg='gray'
        )
        help_label.pack(pady=5)

        print(f"\nDraw the HEALTH region first...")

        self.root.mainloop()

        return self.regions


def main():
    """Run calibration tool."""
    calibrator = HUDCalibrator()
    regions = calibrator.run()

    if regions:
        print("\nFinal regions:")
        for name, region in regions.items():
            print(f"  {name}: {region}")


if __name__ == "__main__":
    main()
