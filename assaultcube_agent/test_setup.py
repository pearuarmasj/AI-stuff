"""
Test script to verify the AssaultCube agent setup.

Run this to check all components are working before training.
"""

import sys
import time

import cv2
import numpy as np


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    errors = []

    try:
        import torch
        print(f"  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import gymnasium
        print(f"  gymnasium: {gymnasium.__version__}")
    except ImportError as e:
        errors.append(f"gymnasium: {e}")

    try:
        import stable_baselines3
        print(f"  stable-baselines3: {stable_baselines3.__version__}")
    except ImportError as e:
        errors.append(f"stable-baselines3: {e}")

    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError as e:
        errors.append(f"transformers: {e}")

    try:
        import mss
        print(f"  mss: OK")
    except ImportError as e:
        errors.append(f"mss: {e}")

    try:
        import pydirectinput
        print(f"  pydirectinput: OK")
    except ImportError as e:
        errors.append(f"pydirectinput: {e}")

    try:
        import easyocr
        print(f"  easyocr: OK")
    except ImportError as e:
        errors.append(f"easyocr: {e}")

    if errors:
        print("\nMissing packages:")
        for e in errors:
            print(f"  {e}")
        return False

    print("\nAll imports OK!")
    return True


def test_screen_capture():
    """Test screen capture."""
    print("\nTesting screen capture...")

    from assaultcube_agent.vision import ScreenCapture

    capture = ScreenCapture()
    frame = capture.capture()

    print(f"  Captured frame: {frame.shape}")
    print(f"  dtype: {frame.dtype}")

    # Save test frame
    cv2.imwrite("test_capture.png", frame)
    print("  Saved to test_capture.png")

    return True


def test_depth_estimation():
    """Test depth estimation."""
    print("\nTesting depth estimation...")

    from assaultcube_agent.vision import ScreenCapture
    from assaultcube_agent.depth import DepthEstimator

    print("  Loading depth model (this may take a moment)...")
    depth_est = DepthEstimator(model_size="small")

    capture = ScreenCapture()
    frame = capture.capture()

    print("  Running depth estimation...")
    start = time.time()
    depth = depth_est.estimate(frame)
    elapsed = time.time() - start

    print(f"  Depth map shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"  Inference time: {elapsed*1000:.1f}ms")

    # Save visualization
    depth_vis = depth_est.visualize(depth)
    cv2.imwrite("test_depth.png", depth_vis)
    print("  Saved to test_depth.png")

    return True


def test_hud_calibration():
    """Run HUD calibration tool."""
    print("\nRunning HUD calibration...")
    print("  This lets you click and drag to define HUD regions.")

    from assaultcube_agent.hud import HUDCalibrator

    calibrator = HUDCalibrator()
    regions = calibrator.run()

    if regions:
        print("  Calibration saved!")
        return True
    else:
        print("  Calibration cancelled.")
        return False


def test_hud_reading():
    """Test HUD reading."""
    print("\nTesting HUD reading...")

    from assaultcube_agent.vision import ScreenCapture
    from assaultcube_agent.hud import HUDReader

    capture = ScreenCapture()
    hud_reader = HUDReader()

    frame = capture.capture()

    # Show HUD regions
    vis = hud_reader.visualize_regions(frame)
    cv2.imwrite("test_hud_regions.png", vis)
    print("  Saved HUD regions to test_hud_regions.png")

    # Try to read HUD
    print("  Reading HUD values...")
    state = hud_reader.read(frame)
    print(f"  Health: {state.health}")
    print(f"  Armor: {state.armor}")
    print(f"  Ammo: {state.ammo_mag}")

    return True


def test_keyboard():
    """Test keyboard input (careful - this will send keystrokes!)."""
    print("\nTesting keyboard (will send test keystrokes in 3 seconds)...")
    print("  Switch to a safe window!")

    time.sleep(3)

    from assaultcube_agent.control import KeyboardController

    kb = KeyboardController()

    print("  Pressing W for 0.5 seconds...")
    kb.key_down('forward')
    time.sleep(0.5)
    kb.key_up('forward')

    print("  Keyboard test complete")
    kb.release_all()

    return True


def test_environment():
    """Test the Gymnasium environment (without full training)."""
    print("\nTesting Gymnasium environment...")
    print("  Note: This will interact with the game!")

    from assaultcube_agent.env import AssaultCubeEnv

    print("  Creating environment...")
    env = AssaultCubeEnv(
        obs_width=84,
        obs_height=84,
        depth_model_size="small",
    )

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    print("  Resetting environment...")
    obs, info = env.reset()

    print(f"  Observation shapes:")
    print(f"    image: {obs['image'].shape}")
    print(f"    hud: {obs['hud'].shape}")

    print("  Taking 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"    Step {i+1}: reward={reward:.3f}, health={info.get('health', '?')}")

    env.close()
    print("  Environment test complete")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("AssaultCube Agent Setup Test")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Screen Capture", test_screen_capture),
        ("Depth Estimation", test_depth_estimation),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    # HUD Calibration (interactive)
    print("\n" + "-" * 60)
    print("HUD Calibration (required for HUD reading):")
    print("-" * 60)

    response = input("Run HUD calibration? (y/n): ").strip().lower()
    if response == 'y':
        test_hud_calibration()

    # Test HUD reading
    response = input("Test HUD reading? (y/n): ").strip().lower()
    if response == 'y':
        test_hud_reading()

    # Optional tests (require game running)
    print("\n" + "-" * 60)
    print("Optional tests (require AssaultCube running):")
    print("-" * 60)

    response = input("Run keyboard test? (y/n): ").strip().lower()
    if response == 'y':
        test_keyboard()

    response = input("Run full environment test? (y/n): ").strip().lower()
    if response == 'y':
        test_environment()

    print("\nSetup test complete!")


if __name__ == "__main__":
    main()
