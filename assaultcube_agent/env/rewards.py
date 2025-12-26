"""
Reward calculation for the AssaultCube environment.

Implements dense reward shaping to help the agent learn faster.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward weights."""
    # Combat rewards
    kill_reward: float = 100.0
    damage_dealt_multiplier: float = 1.0
    headshot_bonus: float = 50.0

    # Penalties
    death_penalty: float = -50.0
    damage_taken_multiplier: float = -0.5

    # Continuous rewards
    survival_per_step: float = 0.01
    aiming_at_enemy: float = 0.1

    # Exploration
    new_area_bonus: float = 0.05

    # Ammo management
    reload_when_needed: float = 0.1


class RewardCalculator:
    """
    Calculate rewards based on game state.

    Tracks state changes to compute dense rewards.
    """

    def __init__(self, config: RewardConfig | None = None):
        """
        Args:
            config: Reward weight configuration
        """
        self.config = config or RewardConfig()

        # State tracking
        self._last_health = 100
        self._last_armor = 0
        self._last_ammo = 30
        self._last_position = None
        self._visited_areas = set()
        self._kills = 0
        self._deaths = 0

    def reset(self):
        """Reset state tracking for new episode."""
        self._last_health = 100
        self._last_armor = 0
        self._last_ammo = 30
        self._last_position = None
        self._visited_areas.clear()
        self._kills = 0
        self._deaths = 0

    def calculate(
        self,
        health: int,
        armor: int,
        ammo: int,
        depth_map: np.ndarray | None = None,
        has_enemy_in_view: bool = False,
    ) -> tuple[float, dict]:
        """
        Calculate reward based on state.

        Args:
            health: Current health
            armor: Current armor
            ammo: Current ammo
            depth_map: Depth map for exploration tracking
            has_enemy_in_view: Whether crosshair is near enemy

        Returns:
            (total_reward, reward_breakdown_dict)
        """
        rewards = {}

        # Survival reward
        rewards['survival'] = self.config.survival_per_step

        # Health change (damage taken)
        health_delta = health - self._last_health
        if health_delta < 0:
            rewards['damage_taken'] = health_delta * self.config.damage_taken_multiplier
        else:
            rewards['damage_taken'] = 0

        # Death detection
        if health <= 0 and self._last_health > 0:
            rewards['death'] = self.config.death_penalty
            self._deaths += 1
        else:
            rewards['death'] = 0

        # Aiming at enemy
        if has_enemy_in_view:
            rewards['aim'] = self.config.aiming_at_enemy
        else:
            rewards['aim'] = 0

        # TODO: Kill detection - would need to track enemy health
        # or detect kill messages on screen
        rewards['kill'] = 0

        # Update state
        self._last_health = health
        self._last_armor = armor
        self._last_ammo = ammo

        # Total reward
        total = sum(rewards.values())

        return total, rewards

    def get_stats(self) -> dict:
        """Get episode statistics."""
        return {
            'kills': self._kills,
            'deaths': self._deaths,
            'kd_ratio': self._kills / max(1, self._deaths),
        }


class EnemyDetectionReward:
    """
    Reward shaping based on enemy visibility in depth/RGB.

    Gives small continuous reward when looking toward enemies.
    """

    def __init__(
        self,
        center_weight: float = 2.0,
        edge_weight: float = 0.5,
    ):
        """
        Args:
            center_weight: Higher reward for enemies in crosshair
            edge_weight: Lower reward for enemies at screen edges
        """
        self.center_weight = center_weight
        self.edge_weight = edge_weight

    def calculate(
        self,
        depth_map: np.ndarray,
        enemy_mask: np.ndarray | None = None,
    ) -> float:
        """
        Calculate aiming reward.

        Args:
            depth_map: Depth map (H, W)
            enemy_mask: Binary mask of enemy locations (H, W) if available

        Returns:
            Aiming reward (0 to 1)
        """
        if enemy_mask is None:
            return 0.0

        h, w = depth_map.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Create distance-from-center weights
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(
            (y_coords - center_y)**2 + (x_coords - center_x)**2
        )
        max_dist = np.sqrt(center_x**2 + center_y**2)
        weights = 1.0 - (dist_from_center / max_dist)

        # Weight higher for center
        weights = weights * (self.center_weight - self.edge_weight) + self.edge_weight

        # Calculate weighted enemy presence
        if enemy_mask.sum() > 0:
            weighted_enemy = (enemy_mask * weights).sum() / enemy_mask.sum()
            return float(weighted_enemy)

        return 0.0
