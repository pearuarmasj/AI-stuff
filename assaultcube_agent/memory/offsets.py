"""
Centralized memory offsets for AssaultCube 1.3.0.2 (32-bit).

ALL OFFSETS IN ONE PLACE - Update here when recompiling AC or switching versions.

Run finder scripts to rediscover offsets:
    python -m assaultcube_agent.memory.find_offsets      # Player pointer, health, position
    python -m assaultcube_agent.memory.find_frags_offset # Frags/kills
    python -m assaultcube_agent.raycast.find_team_offset # Team
    python -m assaultcube_agent.raycast.find_players_offset  # Players array
    python -m assaultcube_agent.raycast.find_world_offset2   # World/map geometry
"""

# =============================================================================
# AC Version this config is for
# =============================================================================
AC_VERSION = "1.3.0.2-custom (deps/AC build)"


# =============================================================================
# STATIC POINTERS (from module base)
# =============================================================================

# Player1 pointer - points to local player's playerent struct
# From vcpp/Release/assaultcube.map: ?player1@@3PAVplayerent@@A = 0x0058A8C0
PLAYER1_PTR_OFFSET = 0x18A8C0

# Players array - std::vector<playerent*> containing all players
# From vcpp/Release/assaultcube.map: ?players@@3U?$vector@PAVplayerent@@@@A = 0x0058A8C4
PLAYERS_ARRAY_OFFSET = 0x18A8C4  # data pointer
PLAYERS_COUNT_OFFSET = 0x18A8CC  # (end - data) / 4 = count

# World/map geometry (for LOS checking)
# From vcpp/Release/assaultcube.map
SFACTOR_OFFSET = 0x1825F0      # log2(ssize) - ?sfactor@@3HA = 0x005825F0
SSIZE_OFFSET = 0x1825F4        # Map size (e.g., 256) - ?ssize@@3HA = 0x005825F4
WORLD_PTR_OFFSET = 0x1825F8    # Pointer to sqr array - ?world@@3PAUsqr@@A = 0x005825F8


# =============================================================================
# PLAYERENT STRUCT OFFSETS (from player pointer)
# =============================================================================

# Inheritance: physent -> dynent -> playerent + playerstate

# Position (vec o from physent, after vtable)
POS_X = 0x04
POS_Y = 0x08
POS_Z = 0x0C

# Velocity (vec vel)
VEL_X = 0x10
VEL_Y = 0x14
VEL_Z = 0x18

# View angles
# vtable(4) + o(12) + vel(12) + deltapos(12) + newpos(12) = 0x34
YAW = 0x34
PITCH = 0x38

# Health and armor (from playerstate)
HEALTH = 0xEC
ARMOR = 0xF0

# Frags and deaths (playerent fields)
FRAGS = 0x1DC
DEATHS = 0x1E4  # frags + flagscore(4) + deaths = 0x1DC + 8

# Team (0=CLA, 1=RVSF)
TEAM = 0x30C

# Weapon (needs discovery if changing)
GUNSELECT = 0x0  # TODO: Find this offset

# Ammo arrays (each NUMGUNS=9 ints)
AMMO_BASE = 0x104   # ammo[9] array
MAG_BASE = 0x128    # mag[9] array

# Damage stats (for reward tracking)
# pstatshots[9] at 0x170, pstatdamage[9] at 0x194
PSTAT_DAMAGE_BASE = 0x194  # int[9] - damage dealt per weapon


# =============================================================================
# WORLD GEOMETRY
# =============================================================================

# sqr struct size (from world.h)
SQR_SIZE = 16

# sqr types
SQR_SOLID = 0
SQR_CORNER = 1
SQR_FHF = 2      # floor heightfield
SQR_CHF = 3      # ceiling heightfield
SQR_SPACE = 4
SQR_SEMISOLID = 5


# =============================================================================
# KNOWN PLAYER PTR OFFSETS FOR DIFFERENT VERSIONS
# =============================================================================

# Used by find_offsets.py to try different versions
KNOWN_PLAYER_PTRS = [
    0x18A8C0,  # AC custom build from deps/AC (current)
    0x18AC00,  # AC 1.3.0.2 (official)
    0x17E0A8,  # AC 1.3.x alternative
    0x10F4F4,  # AC 1.2.0.2
    0x109B74,  # Older versions
    0x50F4F4,  # Some versions
    0x58AC00,  # Alternative
    0x18A8A8,  # AC 1.3.x alternative 2
]


# =============================================================================
# DICT FORMAT (for backwards compatibility with existing code)
# =============================================================================

OFFSETS = {
    "pos_x": POS_X,
    "pos_y": POS_Y,
    "pos_z": POS_Z,
    "vel_x": VEL_X,
    "vel_y": VEL_Y,
    "vel_z": VEL_Z,
    "yaw": YAW,
    "pitch": PITCH,
    "health": HEALTH,
    "armor": ARMOR,
    "frags": FRAGS,
    "deaths": DEATHS,
    "team": TEAM,
    "gunselect": GUNSELECT,
    "ammo_base": AMMO_BASE,
    "mag_base": MAG_BASE,
}


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def print_all_offsets():
    """Print all offsets for verification."""
    print(f"AssaultCube Memory Offsets (version {AC_VERSION})")
    print("=" * 50)
    print("\nStatic Pointers (from module base):")
    print(f"  PLAYER1_PTR_OFFSET  = 0x{PLAYER1_PTR_OFFSET:X}")
    print(f"  PLAYERS_ARRAY_OFFSET = 0x{PLAYERS_ARRAY_OFFSET:X}")
    print(f"  PLAYERS_COUNT_OFFSET = 0x{PLAYERS_COUNT_OFFSET:X}")
    print(f"  WORLD_PTR_OFFSET    = 0x{WORLD_PTR_OFFSET:X}")
    print(f"  SSIZE_OFFSET        = 0x{SSIZE_OFFSET:X}")
    print(f"  SFACTOR_OFFSET      = 0x{SFACTOR_OFFSET:X}")

    print("\nPlayerent Offsets (from player pointer):")
    print(f"  Position:  X=0x{POS_X:02X}, Y=0x{POS_Y:02X}, Z=0x{POS_Z:02X}")
    print(f"  Velocity:  X=0x{VEL_X:02X}, Y=0x{VEL_Y:02X}, Z=0x{VEL_Z:02X}")
    print(f"  View:      YAW=0x{YAW:02X}, PITCH=0x{PITCH:02X}")
    print(f"  Stats:     HEALTH=0x{HEALTH:02X}, ARMOR=0x{ARMOR:02X}")
    print(f"  Scores:    FRAGS=0x{FRAGS:03X}, DEATHS=0x{DEATHS:03X}")
    print(f"  TEAM=0x{TEAM:03X}")


if __name__ == "__main__":
    print_all_offsets()
