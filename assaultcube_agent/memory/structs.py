"""
AssaultCube memory structure definitions.

Game constants and dataclasses for player state.
Memory offsets are centralized in offsets.py - update there for version changes.

Player entity inheritance chain:
  physent -> dynent -> playerent
  playerstate -> playerent (multiple inheritance)
"""

from dataclasses import dataclass

# NUMGUNS from entity.h (line 59)
NUMGUNS = 9

# Weapon IDs (entity.h line 59)
GUN_KNIFE = 0
GUN_PISTOL = 1
GUN_CARBINE = 2
GUN_SHOTGUN = 3
GUN_SUBGUN = 4
GUN_SNIPER = 5
GUN_ASSAULT = 6
GUN_GRENADE = 7
GUN_AKIMBO = 8


# Player states (entity.h line 117)
CS_ALIVE = 0
CS_DEAD = 1
CS_SPAWNING = 2
CS_LAGGED = 3
CS_EDITING = 4
CS_SPECTATE = 5


@dataclass
class GameState:
    """Represents current player state read from memory."""
    health: int = 100
    armor: int = 0
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    current_weapon: int = GUN_PISTOL
    ammo_mag: int = 0      # Current magazine
    ammo_reserve: int = 0  # Reserve ammo
    frags: int = 0         # Kill count
    deaths: int = 0        # Death count

    @property
    def position(self) -> tuple:
        return (self.pos_x, self.pos_y, self.pos_z)

    @property
    def velocity(self) -> tuple:
        return (self.vel_x, self.vel_y, self.vel_z)

    @property
    def view_angles(self) -> tuple:
        return (self.yaw, self.pitch)
