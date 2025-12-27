"""
Automatic offset finder for AssaultCube.

Finds all memory offsets by pattern matching after game recompilation.
Works by scanning for known value patterns in memory.

Usage:
    1. Start AssaultCube and join a bot match
    2. Make sure you're alive with full health (100)
    3. Run: python -m assaultcube_agent.memory.auto_offset_finder
    4. Follow prompts to verify and save offsets
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional

import pymem
import pymem.process

PROCESS_NAME = "ac_client.exe"
NUMGUNS = 9


@dataclass
class FoundOffsets:
    """Container for discovered offsets."""
    player1_ptr: int = 0
    pos_x: int = 0
    pos_y: int = 0
    pos_z: int = 0
    vel_x: int = 0
    vel_y: int = 0
    vel_z: int = 0
    yaw: int = 0
    pitch: int = 0
    health: int = 0
    armor: int = 0
    frags: int = 0
    deaths: int = 0
    team: int = 0
    ammo_base: int = 0
    mag_base: int = 0
    pstat_damage_base: int = 0


def find_player_pointer(pm: pymem.Pymem, module_base: int) -> Optional[int]:
    """
    Find the player1 pointer by scanning static memory for valid player struct pointers.

    Strategy: Scan the .data section for pointers that point to valid player structs.
    A valid player struct has:
    - Reasonable position values (floats between -10000 and 10000)
    - Health in range 0-100 at some offset
    """
    print("\n[*] Scanning for player1 pointer...")

    # Scan range in static memory (typical .data section range)
    scan_start = 0x100000
    scan_end = 0x300000

    candidates = []

    for offset in range(scan_start, scan_end, 4):
        try:
            ptr = pm.read_int(module_base + offset)

            # Valid pointer check (reasonable heap/stack address)
            if ptr < 0x10000 or ptr > 0x7FFFFFFF:
                continue

            # Try to read position (offset 0x04, 0x08, 0x0C from player pointer)
            try:
                x = pm.read_float(ptr + 0x04)
                y = pm.read_float(ptr + 0x08)
                z = pm.read_float(ptr + 0x0C)

                # Position should be reasonable
                if not (-10000 < x < 10000 and -10000 < y < 10000 and -1000 < z < 1000):
                    continue

                # Try to find health (scan nearby for value 100 or 0-100)
                for health_off in range(0x80, 0x200, 4):
                    health = pm.read_int(ptr + health_off)
                    if health == 100:
                        # Check armor next to it
                        armor = pm.read_int(ptr + health_off + 4)
                        if 0 <= armor <= 100:
                            candidates.append((offset, ptr, health_off, x, y, z))
                            break
            except:
                continue
        except:
            continue

    if not candidates:
        print("  [!] No candidates found")
        return None

    print(f"  [+] Found {len(candidates)} candidate(s)")
    for i, (off, ptr, health_off, x, y, z) in enumerate(candidates[:5]):
        print(f"      {i+1}. Offset 0x{off:X} -> ptr 0x{ptr:X}, health@0x{health_off:X}, pos=({x:.1f}, {y:.1f}, {z:.1f})")

    # Return most likely (first one with health=100)
    return candidates[0][0]


def find_struct_offsets(pm: pymem.Pymem, player_ptr: int) -> FoundOffsets:
    """
    Find all struct offsets within the player struct.
    """
    offsets = FoundOffsets()

    print("\n[*] Scanning player struct for field offsets...")

    # Read a chunk of the player struct
    struct_data = pm.read_bytes(player_ptr, 0x400)

    # Position: 3 consecutive floats at early offset (usually 0x04, 0x08, 0x0C)
    for off in range(0, 0x40, 4):
        x, y, z = struct.unpack_from('<fff', struct_data, off)
        if -10000 < x < 10000 and -10000 < y < 10000 and -1000 < z < 1000:
            # Verify these look like position (not velocity which is smaller)
            if abs(x) > 0.1 or abs(y) > 0.1:  # Position usually non-zero
                offsets.pos_x = off
                offsets.pos_y = off + 4
                offsets.pos_z = off + 8
                print(f"  [+] Position: 0x{off:02X} = ({x:.1f}, {y:.1f}, {z:.1f})")

                # Velocity follows position
                offsets.vel_x = off + 12
                offsets.vel_y = off + 16
                offsets.vel_z = off + 20
                vx, vy, vz = struct.unpack_from('<fff', struct_data, off + 12)
                print(f"  [+] Velocity: 0x{off+12:02X} = ({vx:.1f}, {vy:.1f}, {vz:.1f})")
                break

    # Yaw/Pitch: floats after velocity, deltapos, newpos
    # Search for angle-like values (0-360 for yaw, -90 to 90 for pitch)
    for off in range(0x30, 0x60, 4):
        yaw, pitch = struct.unpack_from('<ff', struct_data, off)
        if 0 <= yaw <= 360 and -90 <= pitch <= 90:
            offsets.yaw = off
            offsets.pitch = off + 4
            print(f"  [+] View angles: YAW=0x{off:02X}={yaw:.1f}, PITCH=0x{off+4:02X}={pitch:.1f}")
            break

    # Health/Armor: int=100 followed by int 0-100
    for off in range(0x80, 0x200, 4):
        health, armor = struct.unpack_from('<ii', struct_data, off)
        if health == 100 and 0 <= armor <= 100:
            offsets.health = off
            offsets.armor = off + 4
            print(f"  [+] Health/Armor: 0x{off:02X}={health}, 0x{off+4:02X}={armor}")
            break

    # From health, calculate other playerstate offsets
    # playerstate layout: health(4), armour(4), primary(4), nextprimary(4), gunselect(4), akimbo(4 padded)
    # then: ammo[9](36), mag[9](36), gunwait[9](36), pstatshots[9](36), pstatdamage[9](36)
    if offsets.health:
        playerstate_base = offsets.health
        offsets.ammo_base = playerstate_base + 24  # health + 6 ints
        offsets.mag_base = offsets.ammo_base + 36
        pstatshots_base = offsets.mag_base + 36 + 36  # after gunwait
        offsets.pstat_damage_base = pstatshots_base + 36

        print(f"  [+] Ammo base: 0x{offsets.ammo_base:02X}")
        print(f"  [+] Mag base: 0x{offsets.mag_base:02X}")
        print(f"  [+] pstatdamage base: 0x{offsets.pstat_damage_base:02X}")

    # Frags: search in playerent area (after playerstate) for small int (0 or small positive)
    # Usually around 0x1DC based on struct layout
    for off in range(0x1A0, 0x250, 4):
        val = struct.unpack_from('<i', struct_data, off)[0]
        if val == 0:  # Frags usually 0 at start
            # Check if followed by another 0 or small int (flagscore, deaths)
            next_vals = struct.unpack_from('<iii', struct_data, off)
            if all(-100 < v < 1000 for v in next_vals):
                offsets.frags = off
                offsets.deaths = off + 8  # frags + flagscore + deaths
                print(f"  [+] Frags: 0x{off:03X}={next_vals[0]}, Deaths: 0x{off+8:03X}={next_vals[2]}")
                break

    # Team: usually in the 0x300+ range
    for off in range(0x2F0, 0x350, 4):
        val = struct.unpack_from('<i', struct_data, off)[0]
        if val in [0, 1]:  # TEAM_CLA=0, TEAM_RVSF=1
            offsets.team = off
            print(f"  [+] Team: 0x{off:03X}={val}")
            break

    return offsets


def verify_offsets(pm: pymem.Pymem, player_ptr: int, offsets: FoundOffsets) -> bool:
    """Verify offsets by reading values."""
    print("\n[*] Verifying offsets...")

    try:
        health = pm.read_int(player_ptr + offsets.health)
        armor = pm.read_int(player_ptr + offsets.armor)
        x = pm.read_float(player_ptr + offsets.pos_x)
        y = pm.read_float(player_ptr + offsets.pos_y)
        z = pm.read_float(player_ptr + offsets.pos_z)
        yaw = pm.read_float(player_ptr + offsets.yaw)
        pitch = pm.read_float(player_ptr + offsets.pitch)
        frags = pm.read_int(player_ptr + offsets.frags)

        print(f"  Health: {health}")
        print(f"  Armor: {armor}")
        print(f"  Position: ({x:.1f}, {y:.1f}, {z:.1f})")
        print(f"  View: yaw={yaw:.1f}, pitch={pitch:.1f}")
        print(f"  Frags: {frags}")

        # Read pstatdamage
        total_dmg = 0
        for i in range(NUMGUNS):
            dmg = pm.read_int(player_ptr + offsets.pstat_damage_base + i * 4)
            if dmg > 0:
                total_dmg += dmg
        print(f"  Damage dealt: {total_dmg}")

        return True
    except Exception as e:
        print(f"  [!] Verification failed: {e}")
        return False


def generate_offsets_file(offsets: FoundOffsets, player1_offset: int) -> str:
    """Generate new offsets.py content."""
    return f'''"""
Centralized memory offsets for AssaultCube (auto-generated).

Generated by auto_offset_finder.py
Update by running: python -m assaultcube_agent.memory.auto_offset_finder
"""

# AC Version
AC_VERSION = "custom-build (auto-detected)"

# =============================================================================
# STATIC POINTERS (from module base)
# =============================================================================

PLAYER1_PTR_OFFSET = 0x{player1_offset:X}

# Players array (typically 4 bytes after player1)
PLAYERS_ARRAY_OFFSET = 0x{player1_offset + 4:X}
PLAYERS_COUNT_OFFSET = 0x{player1_offset + 12:X}

# World geometry (may need manual verification)
SFACTOR_OFFSET = 0x1825F0
SSIZE_OFFSET = 0x1825F4
WORLD_PTR_OFFSET = 0x1825F8

# =============================================================================
# PLAYERENT STRUCT OFFSETS (from player pointer)
# =============================================================================

# Position
POS_X = 0x{offsets.pos_x:02X}
POS_Y = 0x{offsets.pos_y:02X}
POS_Z = 0x{offsets.pos_z:02X}

# Velocity
VEL_X = 0x{offsets.vel_x:02X}
VEL_Y = 0x{offsets.vel_y:02X}
VEL_Z = 0x{offsets.vel_z:02X}

# View angles
YAW = 0x{offsets.yaw:02X}
PITCH = 0x{offsets.pitch:02X}

# Health and armor
HEALTH = 0x{offsets.health:02X}
ARMOR = 0x{offsets.armor:02X}

# Frags and deaths
FRAGS = 0x{offsets.frags:03X}
DEATHS = 0x{offsets.deaths:03X}

# Team
TEAM = 0x{offsets.team:03X}

# Weapon
GUNSELECT = 0x0  # TODO: Find this offset

# Ammo arrays
AMMO_BASE = 0x{offsets.ammo_base:03X}
MAG_BASE = 0x{offsets.mag_base:03X}

# Damage stats
PSTAT_DAMAGE_BASE = 0x{offsets.pstat_damage_base:03X}

# =============================================================================
# WORLD GEOMETRY
# =============================================================================

SQR_SIZE = 16
SQR_SOLID = 0
SQR_CORNER = 1
SQR_FHF = 2
SQR_CHF = 3
SQR_SPACE = 4
SQR_SEMISOLID = 5

# =============================================================================
# KNOWN PLAYER PTR OFFSETS
# =============================================================================

KNOWN_PLAYER_PTRS = [
    0x{player1_offset:X},  # Current build
    0x18AC00,
    0x17E0A8,
    0x10F4F4,
]

# =============================================================================
# DICT FORMAT
# =============================================================================

OFFSETS = {{
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
}}


def print_all_offsets():
    """Print all offsets for verification."""
    print(f"AssaultCube Memory Offsets (version {{AC_VERSION}})")
    print("=" * 50)
    print(f"\\nPlayer pointer offset: 0x{{PLAYER1_PTR_OFFSET:X}}")
    print(f"Position: X=0x{{POS_X:02X}}, Y=0x{{POS_Y:02X}}, Z=0x{{POS_Z:02X}}")
    print(f"Velocity: X=0x{{VEL_X:02X}}, Y=0x{{VEL_Y:02X}}, Z=0x{{VEL_Z:02X}}")
    print(f"View: YAW=0x{{YAW:02X}}, PITCH=0x{{PITCH:02X}}")
    print(f"Stats: HEALTH=0x{{HEALTH:02X}}, ARMOR=0x{{ARMOR:02X}}")
    print(f"Scores: FRAGS=0x{{FRAGS:03X}}, DEATHS=0x{{DEATHS:03X}}")
    print(f"TEAM=0x{{TEAM:03X}}")
    print(f"PSTAT_DAMAGE_BASE=0x{{PSTAT_DAMAGE_BASE:03X}}")


if __name__ == "__main__":
    print_all_offsets()
'''


def main():
    print("=" * 60)
    print("  ASSAULTCUBE AUTOMATIC OFFSET FINDER")
    print("=" * 60)
    print("\nRequirements:")
    print("  1. AssaultCube is running")
    print("  2. You're in a game (bot match or multiplayer)")
    print("  3. You're ALIVE with full health (100)")
    print()

    try:
        pm = pymem.Pymem(PROCESS_NAME)
        module_base = pymem.process.module_from_name(
            pm.process_handle, PROCESS_NAME
        ).lpBaseOfDll
        print(f"[+] Attached to {PROCESS_NAME}")
        print(f"[+] Module base: 0x{module_base:X}")
    except Exception as e:
        print(f"[!] Failed to attach: {e}")
        print("[!] Make sure AssaultCube is running!")
        return

    # Find player pointer
    player1_offset = find_player_pointer(pm, module_base)
    if not player1_offset:
        print("[!] Could not find player pointer. Make sure you're in a game!")
        pm.close_process()
        return

    print(f"\n[+] Player1 pointer offset: 0x{player1_offset:X}")

    # Read the actual player pointer
    player_ptr = pm.read_int(module_base + player1_offset)
    print(f"[+] Player struct at: 0x{player_ptr:X}")

    # Find struct offsets
    offsets = find_struct_offsets(pm, player_ptr)
    offsets.player1_ptr = player1_offset

    # Verify
    if not verify_offsets(pm, player_ptr, offsets):
        print("\n[!] Offset verification failed!")
        pm.close_process()
        return

    # Generate new offsets file
    print("\n" + "=" * 60)
    new_content = generate_offsets_file(offsets, player1_offset)

    # Ask to save
    print("\nGenerated offsets.py content:")
    print("-" * 40)
    for line in new_content.split('\n')[:30]:
        print(line)
    print("... (truncated)")
    print("-" * 40)

    save = input("\nSave to offsets.py? (y/n): ").strip().lower()
    if save == 'y':
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        offsets_path = os.path.join(script_dir, "offsets.py")

        with open(offsets_path, 'w') as f:
            f.write(new_content)
        print(f"[+] Saved to {offsets_path}")
    else:
        print("[*] Not saved. Copy the content manually if needed.")

    pm.close_process()
    print("\n[+] Done!")


if __name__ == "__main__":
    main()
