"""
Memory reading module for AssaultCube.

Reads player state directly from game memory using pymem.
Only reads own player state - no enemy positions (use raycasting for that).
"""

import struct
from dataclasses import dataclass
from typing import Optional

import pymem
import pymem.process

from .structs import GameState, NUMGUNS


class ACMemoryReader:
    """
    Read AssaultCube player state from memory.

    Usage:
        reader = ACMemoryReader()
        if reader.attach():
            state = reader.read_state()
            print(f"Health: {state.health}")
    """

    # Process name (32-bit game)
    PROCESS_NAME = "ac_client.exe"

    # Offsets for AC 1.3.0.2 (32-bit)
    # If these don't work, use find_offsets.py to discover correct values
    # Common player1 offsets to try: 0x10F4F4 (1.2.x), 0x18AC00 (1.3.x)
    PLAYER1_PTR_OFFSET = 0x18AC00  # AC 1.3.0.2

    # Offsets from player pointer
    # Based on AC source structure analysis
    OFFSETS = {
        # Position (physent::o - vec at start of physent after vtable)
        "pos_x": 0x04,
        "pos_y": 0x08,
        "pos_z": 0x0C,

        # Velocity (physent::vel)
        "vel_x": 0x10,
        "vel_y": 0x14,
        "vel_z": 0x18,

        # View angles (after deltapos and newpos)
        # vtable(4) + o(12) + vel(12) + deltapos(12) + newpos(12) = 0x34
        "yaw": 0x34,
        "pitch": 0x38,

        # Health and armor (from playerstate, after dynent)
        # Verified for AC 1.3.0.2
        "health": 0xEC,
        "armor": 0xF0,

        # Weapon selection
        "gunselect": 0x0,  # Needs discovery

        # Ammo arrays (each NUMGUNS=9 ints)
        "ammo_base": 0x0,  # Needs discovery
        "mag_base": 0x0,   # Needs discovery
    }

    def __init__(self):
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self.player_ptr: int = 0
        self._attached = False

    def attach(self) -> bool:
        """
        Attach to AssaultCube process.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.pm = pymem.Pymem(self.PROCESS_NAME)
            self.module_base = pymem.process.module_from_name(
                self.pm.process_handle,
                self.PROCESS_NAME
            ).lpBaseOfDll
            self._attached = True
            print(f"Attached to {self.PROCESS_NAME}")
            print(f"Module base: 0x{self.module_base:X}")
            return True

        except pymem.exception.ProcessNotFound:
            print(f"ERROR: {self.PROCESS_NAME} not found. Is AssaultCube running?")
            return False

        except Exception as e:
            print(f"ERROR attaching to process: {e}")
            return False

    def detach(self):
        """Detach from the process."""
        if self.pm:
            self.pm.close_process()
            self.pm = None
        self._attached = False

    @property
    def attached(self) -> bool:
        """Check if still attached."""
        if not self._attached or not self.pm:
            return False
        try:
            # Try reading something to verify connection
            self.pm.read_int(self.module_base)
            return True
        except:
            self._attached = False
            return False

    def _read_player_ptr(self) -> int:
        """Read the player1 pointer."""
        if not self.pm:
            return 0

        try:
            ptr_addr = self.module_base + self.PLAYER1_PTR_OFFSET
            self.player_ptr = self.pm.read_int(ptr_addr)
            return self.player_ptr
        except Exception as e:
            print(f"Error reading player pointer: {e}")
            return 0

    def _read_int(self, offset: int) -> int:
        """Read int at player pointer + offset."""
        try:
            return self.pm.read_int(self.player_ptr + offset)
        except:
            return 0

    def _read_float(self, offset: int) -> float:
        """Read float at player pointer + offset."""
        try:
            return self.pm.read_float(self.player_ptr + offset)
        except:
            return 0.0

    def read_state(self) -> GameState:
        """
        Read current player state from memory.

        Returns:
            GameState with current values, or defaults if read fails.
        """
        state = GameState()

        if not self.attached:
            return state

        # Get current player pointer
        if not self._read_player_ptr():
            return state

        # Read position
        state.pos_x = self._read_float(self.OFFSETS["pos_x"])
        state.pos_y = self._read_float(self.OFFSETS["pos_y"])
        state.pos_z = self._read_float(self.OFFSETS["pos_z"])

        # Read velocity
        state.vel_x = self._read_float(self.OFFSETS["vel_x"])
        state.vel_y = self._read_float(self.OFFSETS["vel_y"])
        state.vel_z = self._read_float(self.OFFSETS["vel_z"])

        # Read view angles
        state.yaw = self._read_float(self.OFFSETS["yaw"])
        state.pitch = self._read_float(self.OFFSETS["pitch"])

        # Read health/armor
        state.health = self._read_int(self.OFFSETS["health"])
        state.armor = self._read_int(self.OFFSETS["armor"])

        # Clamp health/armor to valid ranges
        state.health = max(0, min(100, state.health))
        state.armor = max(0, min(100, state.armor))

        return state

    def read_health(self) -> int:
        """Quick read of just health value."""
        if not self.attached:
            return 0
        if not self._read_player_ptr():
            return 0
        return max(0, min(100, self._read_int(self.OFFSETS["health"])))

    def read_position(self) -> tuple:
        """Quick read of just position."""
        if not self.attached:
            return (0.0, 0.0, 0.0)
        if not self._read_player_ptr():
            return (0.0, 0.0, 0.0)

        x = self._read_float(self.OFFSETS["pos_x"])
        y = self._read_float(self.OFFSETS["pos_y"])
        z = self._read_float(self.OFFSETS["pos_z"])
        return (x, y, z)

    def scan_for_health_offset(self, expected_health: int = 100) -> list:
        """
        Scan memory near player pointer for health value.

        Use this to find the correct health offset if the default doesn't work.
        Run with full health (100) for best results.

        Args:
            expected_health: Your current health value

        Returns:
            List of (offset, value) tuples where value matches expected_health
        """
        if not self.attached:
            return []

        if not self._read_player_ptr():
            return []

        matches = []
        # Scan 0x200 bytes from player pointer
        for offset in range(0, 0x200, 4):
            try:
                val = self.pm.read_int(self.player_ptr + offset)
                if val == expected_health:
                    matches.append((offset, val))
            except:
                pass

        return matches

    def debug_dump(self, num_bytes: int = 256) -> None:
        """
        Dump raw bytes from player pointer for debugging.

        Args:
            num_bytes: Number of bytes to dump
        """
        if not self.attached:
            print("Not attached")
            return

        if not self._read_player_ptr():
            print("Could not read player pointer")
            return

        print(f"Player pointer: 0x{self.player_ptr:X}")
        print(f"Dumping {num_bytes} bytes:\n")

        try:
            data = self.pm.read_bytes(self.player_ptr, num_bytes)

            for i in range(0, len(data), 16):
                # Hex offset
                line = f"0x{i:04X}: "

                # Hex bytes
                hex_part = ""
                for j in range(16):
                    if i + j < len(data):
                        hex_part += f"{data[i+j]:02X} "
                    else:
                        hex_part += "   "
                line += hex_part

                # Int interpretation (every 4 bytes)
                line += " | "
                for j in range(0, 16, 4):
                    if i + j + 4 <= len(data):
                        val = struct.unpack('<i', data[i+j:i+j+4])[0]
                        line += f"{val:10d} "
                    else:
                        line += " " * 11

                print(line)

        except Exception as e:
            print(f"Error reading memory: {e}")


def main():
    """Test memory reading."""
    reader = ACMemoryReader()

    if not reader.attach():
        print("\nMake sure AssaultCube is running!")
        return

    print("\n" + "=" * 50)
    print("Reading player state...")
    print("=" * 50)

    state = reader.read_state()
    print(f"Health: {state.health}")
    print(f"Armor:  {state.armor}")
    print(f"Position: ({state.pos_x:.1f}, {state.pos_y:.1f}, {state.pos_z:.1f})")
    print(f"View: yaw={state.yaw:.1f}, pitch={state.pitch:.1f}")

    print("\n" + "=" * 50)
    print("Scanning for health offset (looking for 100)...")
    print("=" * 50)

    matches = reader.scan_for_health_offset(100)
    if matches:
        print(f"Found {len(matches)} potential health offsets:")
        for offset, val in matches[:10]:  # Show first 10
            print(f"  0x{offset:04X}: {val}")
    else:
        print("No matches found. Is your health 100?")

    print("\n" + "=" * 50)
    print("Memory dump:")
    print("=" * 50)
    reader.debug_dump(256)

    reader.detach()


if __name__ == "__main__":
    main()
