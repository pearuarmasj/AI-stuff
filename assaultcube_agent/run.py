"""
Quick run script for the aim trainer.

Usage:
    python -m assaultcube_agent.run

Make sure AssaultCube is running and visible before starting.
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="AssaultCube Aim Trainer")
    parser.add_argument(
        "--color",
        choices=["red", "blue"],
        default="red",
        help="Enemy team color (default: red)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Mouse sensitivity multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--aim-speed",
        type=float,
        default=0.5,
        help="Aim correction speed 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--shoot",
        action="store_true",
        help="Auto-fire when on target",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Seconds to wait before starting (default: 3)",
    )
    parser.add_argument(
        "--test-capture",
        action="store_true",
        help="Just test screen capture, don't aim",
    )

    args = parser.parse_args()

    if args.test_capture:
        test_capture()
        return

    from .agent import AimTrainer

    print(f"Starting aim trainer in {args.delay} seconds...")
    print("Switch to AssaultCube window NOW!")

    for i in range(args.delay, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    trainer = AimTrainer(
        target_color=args.color,
        sensitivity=args.sensitivity,
        aim_speed=args.aim_speed,
        shoot=args.shoot,
    )

    trainer.run()


def test_capture():
    """Test screen capture and detection without mouse control."""
    import cv2
    import numpy as np
    from .vision import ScreenCapture, EnemyDetector

    print("Testing screen capture...")
    print("Click the OpenCV window first, then press 'q' to quit, 's' to save frame")
    print("Or just close the window with X button")

    capture = ScreenCapture()
    detector = EnemyDetector(target_color="red")

    window_name = "Capture Test - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = capture.capture()
            frame = np.ascontiguousarray(frame)  # Make writable for OpenCV drawing
            detections = detector.detect(frame)

            # Draw detections
            for det in detections:
                x, y, w, h = det["bbox"]
                cx, cy = det["center"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw crosshair
            h, w = frame.shape[:2]
            cv2.drawMarker(
                frame,
                (w // 2, h // 2),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            # Draw detection count on frame
            cv2.putText(
                frame,
                f"Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Resize for display
            display = cv2.resize(frame, (960, 540))
            cv2.imshow(window_name, display)

            # Check if window was closed with X button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("s"):
                cv2.imwrite("test_capture.png", frame)
                print("Saved test_capture.png")

    except KeyboardInterrupt:
        print("\nStopping...")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
