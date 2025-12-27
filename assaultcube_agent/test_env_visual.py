"""
Test the environment with visualization.

Run: python -m assaultcube_agent.test_env_visual

This shows the OpenCV visualization window with:
- Left: Original game view
- Right: Depth heat map with enemy markers
- HUD overlay with health, armor, K/D, enemies
"""

import numpy as np
from assaultcube_agent.env import AssaultCubeEnv


def main():
    print("Starting environment visualization test...")
    print("Press Ctrl+C to stop.\n")

    env = AssaultCubeEnv(
        render_mode="human",
        depth_model_size="small",
        frame_skip=1,  # No frame skip for visualization
    )

    obs, info = env.reset()
    print(f"Initial observation shapes:")
    print(f"  depth: {obs['depth'].shape}")
    print(f"  state: {obs['state'].shape}")
    print(f"  enemies: {obs['enemies'].shape}")
    print()

    step = 0
    try:
        while True:
            # Random action (or no action - just observe)
            action = np.zeros(10, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            step += 1
            if step % 30 == 0:  # Print every ~0.5 seconds
                print(
                    f"Step {step:5d} | "
                    f"HP: {info['health']:3d} | "
                    f"Enemies: {info['enemies_detected']} | "
                    f"Reward: {info['episode_reward']:7.2f} | "
                    f"K/D: {info['kills']}/{info['deaths']}"
                )

            if terminated or truncated:
                print("\nEpisode ended. Resetting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n\nStopped.")

    env.close()


if __name__ == "__main__":
    main()
