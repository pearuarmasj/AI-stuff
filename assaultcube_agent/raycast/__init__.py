"""
Raycast module for spatial awareness and enemy detection.

Features:
- Memory-based enemy detection with LOS checking
- Dense raycast observation (numba-accelerated)
- Full 360° omnidirectional raycasting with depth perception
- Sky detection (rays escaping upward)
"""

from .enemy_detector import EnemyDetector, EnemyInfo
from .los_check import LOSChecker
from .numba_raycast import NumbaRaycastObserver
from .omni_raycast import (
    OmniRaycastObserver,
    HIT_NOTHING, HIT_WALL, HIT_ENEMY, HIT_SKY
)

__all__ = [
    "EnemyDetector", "EnemyInfo", "LOSChecker",
    "NumbaRaycastObserver", "OmniRaycastObserver",
    "HIT_NOTHING", "HIT_WALL", "HIT_ENEMY", "HIT_SKY"
]
