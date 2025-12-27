"""
Raycast module for enemy awareness.

Phase 1: Read player positions, calculate direction/distance
Phase 2: Full map raycasting for wall occlusion (later)
"""

from .enemy_detector import EnemyDetector, EnemyInfo

__all__ = ["EnemyDetector", "EnemyInfo"]
