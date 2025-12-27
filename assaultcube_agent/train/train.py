"""
Training script for the AssaultCube agent.

Uses Stable-Baselines3 for RL training.

EMERGENCY CONTROLS:
    F10 = STOP (immediately halt everything, release all keys)
    F9  = PAUSE/RESUME (toggle training pause)
"""

import os
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from ..env import AssaultCubeEnv
from ..control import emergency_stop, check_stop, wait_if_paused
from .config import TrainingConfig


class EmergencyStopCallback(BaseCallback):
    """Callback to check for emergency stop and pause during training."""

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Check for emergency stop
        if check_stop():
            print("\n[!] Emergency stop detected - halting training")
            # Make sure keys are released
            if hasattr(self.env, 'envs'):
                for e in self.env.envs:
                    if hasattr(e, 'action_mapper'):
                        e.action_mapper.reset()
            elif hasattr(self.env, 'action_mapper'):
                self.env.action_mapper.reset()
            return False  # Stop training

        # Handle pause (blocks until unpaused or stopped)
        if not wait_if_paused():
            return False  # Stopped while paused

        return True


def make_env(config: TrainingConfig, render_mode: str | None = "human") -> AssaultCubeEnv:
    """Create and wrap the environment."""
    env = AssaultCubeEnv(
        screen_width=config.screen_width,
        screen_height=config.screen_height,
        obs_width=config.obs_width,
        obs_height=config.obs_height,
        depth_model_size=config.depth_model_size,
        frame_skip=config.frame_skip,
        render_mode=render_mode,
    )
    env = Monitor(env)
    return env


def train(config: TrainingConfig | None = None, resume_from: str | None = None):
    """
    Train the AssaultCube agent.

    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Trained model
    """
    if config is None:
        config = TrainingConfig()

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.algorithm}_{timestamp}"
    log_dir = Path(config.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training config:")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Logging to: {log_dir}")

    # Create environment
    print("\nCreating environment...")
    env = make_env(config)

    # Create or load model
    if resume_from:
        print(f"Resuming from: {resume_from}")
        model_class = {"PPO": PPO, "SAC": SAC, "A2C": A2C}[config.algorithm]
        model = model_class.load(resume_from, env=env)
    else:
        print("Creating new model...")
        if config.algorithm == "PPO":
            model = PPO(
                config.policy,
                env,
                **config.to_ppo_kwargs(),
            )
        elif config.algorithm == "SAC":
            model = SAC(
                config.policy,
                env,
                **config.to_sac_kwargs(),
            )
        elif config.algorithm == "A2C":
            model = A2C(
                config.policy,
                env,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                device=config.device,
                verbose=1,
                tensorboard_log=config.log_dir,
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ac_agent",
    )
    callbacks.append(checkpoint_callback)

    # Eval callback (uses same env for now - ideally separate eval env)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_callback)

    # Add emergency stop callback
    emergency_callback = EmergencyStopCallback(env)
    callbacks.append(emergency_callback)

    callback_list = CallbackList(callbacks)

    # Start emergency stop listener
    def on_emergency_stop():
        """Called when F10 is pressed."""
        # Release all keys in the environment
        if hasattr(env, 'envs'):
            for e in env.envs:
                if hasattr(e, 'action_mapper'):
                    e.action_mapper.reset()
        elif hasattr(env, 'env') and hasattr(env.env, 'action_mapper'):
            env.env.action_mapper.reset()

    emergency_stop.start(on_stop=on_emergency_stop)

    # Train
    print(f"\nStarting training for {config.total_timesteps:,} timesteps...")
    print("Make sure AssaultCube is running!")
    print("Press F10 to STOP | F9 to PAUSE/RESUME\n")

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Always release keys on exit
        on_emergency_stop()
        emergency_stop.stop_listener()

    # Save final model
    final_path = log_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")

    env.close()

    return model


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    config: TrainingConfig | None = None,
):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        render: Whether to render
        config: Environment config
    """
    if config is None:
        config = TrainingConfig()

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = make_env(config)

    print(f"\nEvaluating for {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            if render:
                env.render()

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, length={ep_length}")

    env.close()

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"  Mean length: {sum(episode_lengths)/len(episode_lengths):.2f}")

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AssaultCube agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--algorithm",
        choices=["PPO", "SAC", "A2C"],
        default="PPO",
        help="RL algorithm",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Path to model to evaluate (instead of training)",
    )

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        config = TrainingConfig(
            total_timesteps=args.timesteps,
            algorithm=args.algorithm,
        )
        train(config, resume_from=args.resume)
