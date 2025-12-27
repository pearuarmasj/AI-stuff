"""
AssaultCube memory structure definitions.

Offsets derived from AC source code (entity.h, geom.h).
These may need adjustment based on AC version and compiler.

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
class PlayerOffsets:
    """
    Offsets within playerent structure.

    playerent inherits from dynent (which inherits from physent) and playerstate.
    In MSVC with multiple inheritance, base classes are laid out in declaration order.

    Layout (approximate - verify with testing):
    0x00: physent/dynent vtable pointer (4/8 bytes)
    0x04: vec o (position) - 12 bytes (3 floats)
    0x10: vec vel (velocity) - 12 bytes
    0x1C: vec deltapos - 12 bytes
    0x28: vec newpos - 12 bytes
    0x34: float yaw, pitch, roll - 12 bytes
    ... more physent/dynent fields ...

    After dynent, playerstate fields start.
    After playerstate, playerent's own fields.

    These are PLACEHOLDER offsets - need to verify with CheatEngine or testing.
    Common AC offsets from various sources online:
    """
    # Static pointer offset from module base (needs verification)
    # Common values for different AC versions: 0x10F4F4, 0x109B74, etc.
    PLAYER1_PTR = 0x10F4F4

    # Offsets from player pointer (need verification)
    # These are common values but may differ by version
    HEALTH = 0xF8
    ARMOR = 0xFC

    # Position (vec o from physent)
    POS_X = 0x04
    POS_Y = 0x08
    POS_Z = 0x0C

    # Velocity
    VEL_X = 0x10
    VEL_Y = 0x14
    VEL_Z = 0x18

    # View angles
    YAW = 0x40
    PITCH = 0x44

    # Weapons
    CURRENT_WEAPON = 0x0  # gunselect - needs offset discovery

    # Ammo arrays (NUMGUNS = 9 ints each)
    AMMO_ARRAY = 0x0  # ammo[NUMGUNS] - reserve ammo
    MAG_ARRAY = 0x0   # mag[NUMGUNS] - magazine ammo


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

    @property
    def position(self) -> tuple:
        return (self.pos_x, self.pos_y, self.pos_z)

    @property
    def velocity(self) -> tuple:
        return (self.vel_x, self.vel_y, self.vel_z)

    @property
    def view_angles(self) -> tuple:
        return (self.yaw, self.pitch)


# Known offsets for specific AC versions (32-bit game)
# Format: {version_string: offset dict}
KNOWN_VERSIONS = {
    # AC 1.2.0.2 (older, well-documented)
    "1.2.0.2": {
        "player1_ptr": 0x10F4F4,
        "health": 0xF8,
        "armor": 0xFC,
        "pos_x": 0x04,
        "pos_y": 0x08,
        "pos_z": 0x0C,
        "yaw": 0x40,
        "pitch": 0x44,
    },
    # AC 1.3.0.2 (verified)
    "1.3.0.2": {
        "player1_ptr": 0x18AC00,
        "health": 0xEC,   # Verified
        "armor": 0xF0,    # Verified
        "pos_x": 0x04,    # Verified
        "pos_y": 0x08,    # Verified
        "pos_z": 0x0C,    # Verified
        "yaw": 0x34,      # Fixed: vtable(4)+o(12)+vel(12)+deltapos(12)+newpos(12)=0x34
        "pitch": 0x38,
    },
}
