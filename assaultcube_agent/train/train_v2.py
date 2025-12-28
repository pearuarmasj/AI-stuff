"""
Training script for AssaultCube agent v2.

Uses OmniAssaultCubeEnv with 360° omnidirectional raycasting.
Full spatial awareness in all directions.

EMERGENCY CONTROLS:
    F10 = STOP (immediately halt everything, release all keys)
    F9  = PAUSE/RESUME (toggle training pause)

MONITORING:
    TensorBoard: tensorboard --logdir logs/assaultcube_v2
    Live monitor: python -m assaultcube_agent.debug.monitor

Usage:
    python -m assaultcube_agent.train.train_v2 --timesteps 1000000
    python -m assaultcube_agent.train.train_v2 --eval logs/assaultcube_v2/PPO_xxx/final_model.zip
"""

import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Literal, Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..env import OmniAssaultCubeEnv
from ..env.action_wrapper import wrap_env_for_sb3
from ..control import emergency_stop, check_stop, wait_if_paused


@dataclass
class TrainingConfigV2:
    """Training configuration for omnidirectional environment."""

    # Environment - 360° omnidirectional raycasting
    horizontal_rays: int = 72       # Every 5° = full 360°
    vertical_layers: int = 9        # -60° to +60°
    vertical_min: float = -60.0
    vertical_max: float = 60.0
    max_ray_distance: float = 250.0
    max_episode_steps: int = 10000

    # Training
    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # Network
    policy: str = "MultiInputPolicy"
    net_arch: list = field(default_factory=lambda: [256, 256])

    # Logging
    log_dir: str = "logs/assaultcube_v2"
    checkpoint_freq: int = 50000  # ~3 min at 280 fps
    eval_freq: int = 25000
    n_eval_episodes: int = 3

    # Hardware
    device: str = "auto"


class EmergencyStopCallback(BaseCallback):
    """Callback for emergency stop and pause functionality."""

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        if check_stop():
            print("\n[!] EMERGENCY STOP - Halting training")
            self._release_all_keys()
            return False

        if not wait_if_paused():
            return False

        return True

    def _release_all_keys(self):
        """Release all keys in the environment."""
        try:
            if hasattr(self.env, 'envs'):
                for e in self.env.envs:
                    unwrapped = e
                    while hasattr(unwrapped, 'env'):
                        unwrapped = unwrapped.env
                    if hasattr(unwrapped, 'action_mapper'):
                        unwrapped.action_mapper.reset()
        except:
            pass


class NeuralNetworkVisualizerCallback(BaseCallback):
    """
    Callback to visualize neural network weights and biases during training.

    Logs to TensorBoard:
    - Weight/bias statistics (mean, std, min, max) for each layer
    - Weight distributions as histograms
    - Network architecture summary
    """

    def __init__(self, log_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._printed_architecture = False

    def _on_training_start(self) -> None:
        """Print network architecture at start of training."""
        if self._printed_architecture:
            return

        print("\n" + "=" * 70)
        print("  NEURAL NETWORK ARCHITECTURE")
        print("=" * 70)

        policy = self.model.policy
        total_params = 0

        print("\n  [LAYERS]")
        for name, param in policy.named_parameters():
            num_params = param.numel()
            total_params += num_params
            shape_str = "x".join(str(s) for s in param.shape)
            print(f"    {name}: {shape_str} = {num_params:,} params")

        print(f"\n  [TOTAL: {total_params:,} learnable parameters]")
        print("=" * 70 + "\n")
        self._printed_architecture = True

    def _on_step(self) -> bool:
        """Log weight statistics periodically."""
        if self.n_calls % self.log_freq != 0:
            return True

        policy = self.model.policy

        # Log weight statistics for each layer
        for name, param in policy.named_parameters():
            if param.requires_grad:
                data = param.data.cpu().numpy()

                # Log scalar statistics
                self.logger.record(f"nn/{name}/mean", float(np.mean(data)))
                self.logger.record(f"nn/{name}/std", float(np.std(data)))
                self.logger.record(f"nn/{name}/min", float(np.min(data)))
                self.logger.record(f"nn/{name}/max", float(np.max(data)))
                self.logger.record(f"nn/{name}/abs_mean", float(np.mean(np.abs(data))))

                # Log gradient statistics if available
                if param.grad is not None:
                    grad = param.grad.cpu().numpy()
                    self.logger.record(f"nn_grad/{name}/mean", float(np.mean(grad)))
                    self.logger.record(f"nn_grad/{name}/std", float(np.std(grad)))

        return True

    def _on_rollout_end(self) -> None:
        """Print weight summary at end of each rollout (optional verbose)."""
        if self.verbose < 2:
            return

        print("\n  [NN Weights Summary]")
        for name, param in self.model.policy.named_parameters():
            if "weight" in name and param.requires_grad:
                data = param.data.cpu().numpy()
                print(f"    {name}: mean={np.mean(data):.4f}, std={np.std(data):.4f}")


def make_env(config: TrainingConfigV2, render_mode: Optional[str] = None):
    """Create wrapped environment."""
    env = OmniAssaultCubeEnv(
        horizontal_rays=config.horizontal_rays,
        vertical_layers=config.vertical_layers,
        vertical_min=config.vertical_min,
        vertical_max=config.vertical_max,
        max_ray_distance=config.max_ray_distance,
        max_episode_steps=config.max_episode_steps,
        render_mode=render_mode,
    )

    # Wrap for SB3 (flatten action space)
    env = wrap_env_for_sb3(env)

    # Monitor wrapper
    env = Monitor(env)

    return env


def train(config: TrainingConfigV2, resume_from: Optional[str] = None):
    """Train the agent."""

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.algorithm}_{timestamp}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Log directory: {log_dir}")
    total_rays = config.horizontal_rays * config.vertical_layers
    print(f"  Rays: {config.horizontal_rays}h x {config.vertical_layers}v = {total_rays} (360 omni)")
    print("=" * 70)

    # Save config
    config_path = log_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Create environment (no render during training for speed)
    print("\n[*] Creating environment...")
    env = make_env(config, render_mode=None)

    # Create model
    if resume_from:
        print(f"\n[*] Loading model from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print("\n[*] Creating new model...")
        model = PPO(
            config.policy,
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            policy_kwargs={"net_arch": config.net_arch},
            device=config.device,
            verbose=1,
            tensorboard_log=str(log_dir),
        )

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=str(log_dir / "checkpoints"),
            name_prefix="model",
        ),
        EmergencyStopCallback(env),
        NeuralNetworkVisualizerCallback(log_freq=config.n_steps, verbose=1),
    ]

    # Start emergency stop listener
    def on_stop():
        try:
            unwrapped = env.envs[0] if hasattr(env, 'envs') else env
            while hasattr(unwrapped, 'env'):
                unwrapped = unwrapped.env
            if hasattr(unwrapped, 'action_mapper'):
                unwrapped.action_mapper.reset()
        except:
            pass

    emergency_stop.start(on_stop=on_stop)

    # Train
    print("\n" + "=" * 70)
    print("  STARTING TRAINING")
    print("=" * 70)
    print("  Make sure AssaultCube is running!")
    print("  Press F10 to STOP | F9 to PAUSE/RESUME")
    print("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
    finally:
        on_stop()
        emergency_stop.stop_listener()

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved: {final_path}")
    print(f"  Logs: {log_dir}")
    print("=" * 70)

    env.close()
    return model


def evaluate(model_path: str, n_episodes: int = 5, config: Optional[TrainingConfigV2] = None):
    """Evaluate a trained model."""
    if config is None:
        config = TrainingConfigV2()

    print(f"\n[*] Loading model: {model_path}")
    model = PPO.load(model_path)

    print("[*] Creating environment with rendering...")
    env = make_env(config, render_mode="human")

    print(f"\n[*] Evaluating for {n_episodes} episodes...")

    episode_rewards = []
    episode_kills = []
    episode_deaths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        episode_rewards.append(ep_reward)
        episode_kills.append(info.get('kills', 0))
        episode_deaths.append(info.get('deaths', 0))

        print(f"Episode {ep+1}: reward={ep_reward:.1f}, K/D={info.get('kills', 0)}/{info.get('deaths', 0)}")

    env.close()

    print(f"\n" + "=" * 40)
    print(f"  EVALUATION RESULTS ({n_episodes} episodes)")
    print(f"=" * 40)
    print(f"  Mean reward: {sum(episode_rewards)/len(episode_rewards):.1f}")
    print(f"  Total K/D: {sum(episode_kills)}/{sum(episode_deaths)}")
    print(f"=" * 40)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train AssaultCube agent v2")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training timesteps")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--eval", type=str, help="Evaluate model instead of training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        config = TrainingConfigV2(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
        train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
