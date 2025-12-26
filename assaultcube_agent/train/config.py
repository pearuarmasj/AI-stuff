"""
Training configuration for the AssaultCube agent.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainingConfig:
    """Configuration for training runs."""

    # Environment settings
    screen_width: int = 1920
    screen_height: int = 1080
    obs_width: int = 160
    obs_height: int = 120
    depth_model_size: Literal["small", "base", "large"] = "small"
    frame_skip: int = 4

    # Training settings
    algorithm: Literal["PPO", "SAC", "A2C"] = "PPO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # PPO rollout buffer size
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clip range
    ent_coef: float = 0.01  # Entropy coefficient

    # Network architecture
    policy: str = "MultiInputPolicy"  # For Dict observation space
    net_arch: list = field(default_factory=lambda: [256, 256])

    # Logging and checkpoints
    log_dir: str = "logs/assaultcube"
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 5

    # Hardware
    device: str = "auto"  # "auto", "cuda", or "cpu"
    n_envs: int = 1  # Number of parallel environments

    def to_ppo_kwargs(self) -> dict:
        """Convert to PPO constructor kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "policy_kwargs": {
                "net_arch": self.net_arch,
            },
            "device": self.device,
            "verbose": 1,
            "tensorboard_log": self.log_dir,
        }

    def to_sac_kwargs(self) -> dict:
        """Convert to SAC constructor kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "policy_kwargs": {
                "net_arch": self.net_arch,
            },
            "device": self.device,
            "verbose": 1,
            "tensorboard_log": self.log_dir,
        }


# Preset configurations
FAST_TEST_CONFIG = TrainingConfig(
    total_timesteps=10_000,
    checkpoint_freq=1000,
    eval_freq=2000,
    n_steps=512,
    batch_size=32,
)

SHORT_TRAINING_CONFIG = TrainingConfig(
    total_timesteps=100_000,
    checkpoint_freq=5000,
    eval_freq=10000,
)

FULL_TRAINING_CONFIG = TrainingConfig(
    total_timesteps=10_000_000,
    depth_model_size="base",
    checkpoint_freq=50000,
    eval_freq=25000,
)
