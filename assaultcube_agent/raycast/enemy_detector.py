"""
Enemy detection via memory reading (Phase 1).

Reads the players array from AC memory and calculates
direction/distance to each enemy relative to own player's view.

No wall occlusion yet - that's Phase 2.
"""

import math
from dataclasses import dataclass
from typing import Optional

import pymem
import pymem.process


@dataclass
class EnemyInfo:
    """Information about a detected enemy."""
    index: int              # Player index in players array
    distance: float         # Distance from own player
    angle_h: float          # Horizontal angle relative to view (-180 to 180)
    angle_v: float          # Vertical angle relative to view (-90 to 90)
    in_fov: bool            # Whether enemy is in field of view
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    health: int = 0         # Enemy health
    is_alive: bool = True


class EnemyDetector:
    """
    Detect enemies by reading players array from memory.

    Usage:
        detector = EnemyDetector()
        if detector.attach():
            enemies = detector.detect_enemies(own_pos, own_angles)
            for e in enemies:
                print(f"Enemy {e.index}: dist={e.distance:.1f}, angle={e.angle_h:.1f}")
    """

    PROCESS_NAME = "ac_client.exe"

    # AC 1.3.0.2 offsets (32-bit)
    PLAYER1_PTR_OFFSET = 0x18AC00      # player1 pointer
    PLAYERS_ARRAY_OFFSET = 0x18AC04    # pointer to playerent* array
    PLAYERS_COUNT_OFFSET = 0x18AC0C    # number of players

    # Playerent offsets
    OFFSETS = {
        "pos_x": 0x04,
        "pos_y": 0x08,
        "pos_z": 0x0C,
        "health": 0xEC,
        "team": 0x30C,  # 0=CLA, 1=RVSF
    }

    def __init__(self, fov_h: float = 90.0, fov_v: float = 60.0):
        self.pm: Optional[pymem.Pymem] = None
        self.module_base: int = 0
        self._attached = False
        self.fov_h = fov_h
        self.fov_v = fov_v
        self._player1_ptr: int = 0

    def attach(self) -> bool:
        """Attach to AssaultCube process."""
        try:
            self.pm = pymem.Pymem(self.PROCESS_NAME)
            self.module_base = pymem.process.module_from_name(
                self.pm.process_handle,
                self.PROCESS_NAME
            ).lpBaseOfDll
            self._attached = True
            print(f"[EnemyDetector] Attached to {self.PROCESS_NAME}")
            print(f"[EnemyDetector] Module base: 0x{self.module_base:X}")
            return True

        except pymem.exception.ProcessNotFound:
            print(f"[EnemyDetector] ERROR: {self.PROCESS_NAME} not found")
            return False

        except Exception as e:
            print(f"[EnemyDetector] ERROR: {e}")
            return False

    def detach(self):
        """Detach from process."""
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
            self.pm.read_int(self.module_base)
            return True
        except:
            self._attached = False
            return False

    def _read_players_array(self) -> tuple[int, int]:
        """
        Read the players array pointer and count.

        Returns:
            (array_ptr, count)
        """
        if not self.pm:
            return (0, 0)

        try:
            self._player1_ptr = self.pm.read_int(
                self.module_base + self.PLAYER1_PTR_OFFSET
            )
            array_ptr = self.pm.read_int(
                self.module_base + self.PLAYERS_ARRAY_OFFSET
            )
            count = self.pm.read_int(
                self.module_base + self.PLAYERS_COUNT_OFFSET
            )
            return (array_ptr, count)

        except Exception as e:
            print(f"[EnemyDetector] Error reading players: {e}")
            return (0, 0)

    def _read_player_position(self, player_ptr: int) -> tuple[float, float, float]:
        """Read position from a playerent pointer."""
        if not self.pm or player_ptr == 0:
            return (0.0, 0.0, 0.0)

        try:
            x = self.pm.read_float(player_ptr + self.OFFSETS["pos_x"])
            y = self.pm.read_float(player_ptr + self.OFFSETS["pos_y"])
            z = self.pm.read_float(player_ptr + self.OFFSETS["pos_z"])
            return (x, y, z)
        except:
            return (0.0, 0.0, 0.0)

    def _read_player_health(self, player_ptr: int) -> int:
        """Read health from a playerent pointer."""
        if not self.pm or player_ptr == 0:
            return 0

        try:
            return self.pm.read_int(player_ptr + self.OFFSETS["health"])
        except:
            return 0

    def _read_player_team(self, player_ptr: int) -> int:
        """Read team from a playerent pointer. 0=CLA, 1=RVSF."""
        if not self.pm or player_ptr == 0:
            return -1

        try:
            return self.pm.read_int(player_ptr + self.OFFSETS["team"])
        except:
            return -1

    def _get_own_team(self) -> int:
        """Get own player's team."""
        return self._read_player_team(self._player1_ptr)

    def _calc_angle_distance(
        self,
        own_pos: tuple[float, float, float],
        own_yaw: float,
        own_pitch: float,
        enemy_pos: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        Calculate distance and relative angles to enemy.

        Returns:
            (distance, angle_h, angle_v)
        """
        dx = enemy_pos[0] - own_pos[0]
        dy = enemy_pos[1] - own_pos[1]
        dz = enemy_pos[2] - own_pos[2]

        dist_2d = math.sqrt(dx*dx + dy*dy)
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance < 0.01:
            return (0.0, 0.0, 0.0)

        # AC coordinate system: atan2(dx, -dy) matches yaw
        enemy_yaw = math.degrees(math.atan2(dx, -dy))
        enemy_pitch = math.degrees(math.atan2(dz, dist_2d))

        angle_h = enemy_yaw - own_yaw
        angle_v = enemy_pitch - own_pitch

        # Normalize to -180..180
        while angle_h > 180:
            angle_h -= 360
        while angle_h < -180:
            angle_h += 360

        return (distance, angle_h, angle_v)

    def detect_enemies(
        self,
        own_pos: tuple[float, float, float],
        own_yaw: float,
        own_pitch: float,
        max_distance: float = 500.0,
        filter_team: bool = True,
        debug: bool = False
    ) -> list[EnemyInfo]:
        """
        Detect all enemies and calculate their relative positions.

        Args:
            filter_team: If True, only return enemies (different team). If False, return all players.
            debug: If True, print debug info for each player.

        Returns:
            List of EnemyInfo for all detected enemies
        """
        if not self.attached:
            return []

        enemies = []

        array_ptr, count = self._read_players_array()
        if array_ptr == 0 or count == 0:
            return []

        own_team = self._get_own_team() if filter_team else -1

        for i in range(count):
            try:
                player_ptr = self.pm.read_int(array_ptr + i * 4)

                # Skip null or self
                if player_ptr == 0 or player_ptr == self._player1_ptr:
                    if debug:
                        print(f"  [Player {i}] SKIP: null or self (ptr=0x{player_ptr:X}, self=0x{self._player1_ptr:X})")
                    continue

                # Skip teammates
                if filter_team and own_team >= 0:
                    player_team = self._read_player_team(player_ptr)
                    if player_team == own_team:
                        if debug:
                            print(f"  [Player {i}] SKIP: teammate (team={player_team}, own={own_team})")
                        continue

                pos = self._read_player_position(player_ptr)
                health = self._read_player_health(player_ptr)

                # Skip dead
                if health <= 0:
                    if debug:
                        print(f"  [Player {i}] SKIP: dead (health={health})")
                    continue

                distance, angle_h, angle_v = self._calc_angle_distance(
                    own_pos, own_yaw, own_pitch, pos
                )

                if distance > max_distance:
                    if debug:
                        print(f"  [Player {i}] SKIP: too far (dist={distance:.0f} > max={max_distance})")
                    continue

                in_fov = abs(angle_h) <= self.fov_h / 2 and abs(angle_v) <= self.fov_v / 2

                if debug:
                    player_team = self._read_player_team(player_ptr)
                    print(f"  [Player {i}] FOUND: team={player_team}, hp={health}, dist={distance:.0f}, pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")

                enemies.append(EnemyInfo(
                    index=i,
                    distance=distance,
                    angle_h=angle_h,
                    angle_v=angle_v,
                    in_fov=in_fov,
                    pos_x=pos[0],
                    pos_y=pos[1],
                    pos_z=pos[2],
                    health=health,
                    is_alive=True,
                ))

            except:
                continue

        return enemies

    def get_closest_enemy_in_fov(
        self,
        own_pos: tuple[float, float, float],
        own_yaw: float,
        own_pitch: float,
        max_distance: float = 500.0
    ) -> Optional[EnemyInfo]:
        """Get the closest enemy in FOV."""
        enemies = self.detect_enemies(own_pos, own_yaw, own_pitch, max_distance)
        in_fov = [e for e in enemies if e.in_fov]
        if not in_fov:
            return None
        return min(in_fov, key=lambda e: e.distance)
