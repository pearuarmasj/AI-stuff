"""
Training script for the AssaultCube agent.

Uses Stable-Baselines3 for RL training.

EMERGENCY CONTROLS:
    F10 = STOP (immediately halt everything, release all keys)
    F9  = PAUSE/RESUME (toggle training pause)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

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


def find_existing_runs(log_dir: str = "logs/assaultcube") -> list[dict]:
    """
    Find existing training runs with checkpoints.

    Returns:
        List of dicts with run info: name, path, checkpoints, config, timesteps
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    runs = []
    for run_dir in sorted(log_path.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        run_info = {
            'name': run_dir.name,
            'path': str(run_dir),
            'checkpoints': [],
            'final_model': None,
            'config': None,
            'last_timestep': 0,
        }

        # Check for final model
        final_model = run_dir / "final_model.zip"
        if final_model.exists():
            run_info['final_model'] = str(final_model)

        # Check for checkpoints
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            for cp in sorted(checkpoint_dir.glob("*.zip"), reverse=True):
                # Extract step number from checkpoint name (e.g., ac_agent_10000_steps.zip)
                match = re.search(r'_(\d+)_steps\.zip$', cp.name)
                if match:
                    step = int(match.group(1))
                    run_info['checkpoints'].append({
                        'path': str(cp),
                        'steps': step,
                    })
                    if step > run_info['last_timestep']:
                        run_info['last_timestep'] = step

        # Also count final model steps if higher
        if run_info['final_model'] and run_info['last_timestep'] == 0:
            # Check for training config to get total timesteps
            config_file = run_dir / "training_config.json"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        run_info['config'] = json.load(f)
                        run_info['last_timestep'] = run_info['config'].get('total_timesteps', 0)
                except:
                    pass

        # Load config if exists
        config_file = run_dir / "training_config.json"
        if config_file.exists() and run_info['config'] is None:
            try:
                with open(config_file) as f:
                    run_info['config'] = json.load(f)
            except:
                pass

        # Only add runs that have something to resume from
        if run_info['checkpoints'] or run_info['final_model']:
            runs.append(run_info)

    return runs


def prompt_resume_or_new(runs: list[dict]) -> tuple[str | None, dict | None]:
    """
    Prompt user to select a run to resume or start new.

    Returns:
        (checkpoint_path, run_config) or (None, None) for new training
    """
    print("\n" + "=" * 60)
    print("  EXISTING TRAINING RUNS DETECTED")
    print("=" * 60)

    for i, run in enumerate(runs[:5]):  # Show top 5 most recent
        timesteps = run['last_timestep']
        cp_count = len(run['checkpoints'])
        has_final = "✓" if run['final_model'] else " "

        print(f"\n  [{i+1}] {run['name']}")
        print(f"      Timesteps: {timesteps:,}")
        print(f"      Checkpoints: {cp_count} | Final model: {has_final}")

        if run['config']:
            algo = run['config'].get('algorithm', 'unknown')
            print(f"      Algorithm: {algo}")

    print(f"\n  [N] Start NEW training run")
    print("\n" + "-" * 60)

    while True:
        choice = input("Select option (1-5 or N): ").strip().upper()

        if choice == 'N':
            return None, None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs[:5]):
                run = runs[idx]

                # Determine best checkpoint to resume from
                if run['checkpoints']:
                    latest_cp = run['checkpoints'][0]  # Already sorted descending
                    return latest_cp['path'], run['config']
                elif run['final_model']:
                    return run['final_model'], run['config']
        except ValueError:
            pass

        print("Invalid choice. Enter 1-5 or N.")


def save_run_config(log_dir: Path, config: TrainingConfig, resumed_from: str | None = None,
                    timesteps_done: int = 0):
    """Save training configuration for later resumption."""
    config_data = asdict(config)
    config_data['resumed_from'] = resumed_from
    config_data['timesteps_done'] = timesteps_done
    config_data['started_at'] = datetime.now().isoformat()

    config_file = log_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)


def get_checkpoint_steps(checkpoint_path: str) -> int:
    """Extract step count from checkpoint filename."""
    match = re.search(r'_(\d+)_steps\.zip$', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


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

    # Calculate timesteps already done if resuming
    timesteps_done = 0
    if resume_from:
        timesteps_done = get_checkpoint_steps(resume_from)

    print(f"Training config:")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    if resume_from:
        print(f"  Already completed: {timesteps_done:,}")
        print(f"  Remaining: {config.total_timesteps - timesteps_done:,}")
    print(f"  Logging to: {log_dir}")

    # Save config for resumption
    save_run_config(log_dir, config, resumed_from=resume_from, timesteps_done=timesteps_done)

    # Create environment
    print("\nCreating environment...")
    env = make_env(config)

    # Create or load model
    if resume_from:
        print(f"\nResuming from: {resume_from}")
        model_class = {"PPO": PPO, "SAC": SAC, "A2C": A2C}[config.algorithm]
        model = model_class.load(resume_from, env=env)
        print(f"  Model loaded successfully!")
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

    # Calculate remaining timesteps
    remaining_timesteps = config.total_timesteps - timesteps_done
    if remaining_timesteps <= 0:
        print(f"\nTraining already complete! ({timesteps_done:,} timesteps done)")
        print("Use --timesteps to specify more timesteps if you want to continue.")
        env.close()
        return None

    # Train
    if resume_from:
        print(f"\nContinuing training for {remaining_timesteps:,} more timesteps...")
        print(f"  (Previously completed: {timesteps_done:,})")
    else:
        print(f"\nStarting training for {config.total_timesteps:,} timesteps...")
    print("Make sure AssaultCube is running!")
    print("Press F10 to STOP | F9 to PAUSE/RESUME\n")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback_list,
            progress_bar=True,
            reset_num_timesteps=False if resume_from else True,
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
    print(f"\n" + "=" * 60)
    print(f"TRAINING SAVED")
    print(f"=" * 60)
    print(f"  Model saved to: {final_path}")
    print(f"  Run directory: {log_dir}")
    print(f"\nTo resume training later, just run the training script again")
    print(f"and select this run from the list, or use:")
    print(f"  --resume \"{final_path}.zip\"")
    print(f"=" * 60)

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
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start new training (skip resume prompt)",
    )

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        resume_from = args.resume
        saved_config = None

        # Check for existing runs to resume (unless --new or --resume specified)
        if not args.new and not args.resume:
            existing_runs = find_existing_runs()
            if existing_runs:
                resume_from, saved_config = prompt_resume_or_new(existing_runs)

        # Build config - use saved config values if resuming
        if saved_config:
            # Restore config from saved run, but allow overriding timesteps
            config = TrainingConfig(
                total_timesteps=args.timesteps,
                algorithm=saved_config.get('algorithm', args.algorithm),
                screen_width=saved_config.get('screen_width', 1920),
                screen_height=saved_config.get('screen_height', 1080),
                obs_width=saved_config.get('obs_width', 160),
                obs_height=saved_config.get('obs_height', 120),
                depth_model_size=saved_config.get('depth_model_size', 'small'),
                frame_skip=saved_config.get('frame_skip', 4),
                learning_rate=saved_config.get('learning_rate', 3e-4),
                batch_size=saved_config.get('batch_size', 64),
                n_steps=saved_config.get('n_steps', 2048),
                gamma=saved_config.get('gamma', 0.99),
            )
        else:
            config = TrainingConfig(
                total_timesteps=args.timesteps,
                algorithm=args.algorithm,
            )

        train(config, resume_from=resume_from)
