"""
Omnidirectional AssaultCube Environment - FINAL VERSION

Full 360° spatial awareness via dense raycasting.
This is the production environment for confident training.

Observation:
    player: [health, armor, ammo, vel_x, vel_y, vel_z] (6,)
    rays: [distance, hit_type] x 648 (full 360°, 9 vertical layers)

Action (Hybrid):
    movement: MultiDiscrete([3, 3, 2, 2])
    aim: Box([-1, 1], (2,))
    combat: MultiDiscrete([2, 2])

Target: 300+ FPS
"""

import time
import math
from typing import Optional
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..memory import ACMemoryReader
from ..control import ActionMapper, check_stop
from ..raycast import OmniRaycastObserver


class OmniAssaultCubeEnv(gym.Env):
    """
    AssaultCube environment with full 360° omnidirectional raycasting.

    The agent can "see" in ALL directions at all times - front, back, sides.
    This eliminates the issue of not knowing what's behind you.
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        # Raycast configuration
        horizontal_rays: int = 72,      # Every 5°
        vertical_layers: int = 9,       # -60° to +60°
        vertical_min: float = -60.0,
        vertical_max: float = 60.0,
        max_ray_distance: float = 250.0,

        # Environment settings
        max_episode_steps: int = 10000,
        render_mode: Optional[str] = None,
        aim_scale: float = 15.0,

        # Anti-stuck
        stuck_threshold: float = 3.0,
        stuck_history: int = 60,
    ):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.aim_scale = aim_scale
        self.stuck_threshold = stuck_threshold
        self.stuck_history_len = stuck_history

        self.h_rays = horizontal_rays
        self.v_layers = vertical_layers
        self.total_rays = horizontal_rays * vertical_layers

        # Initialize
        print("=" * 70)
        print("  OMNIDIRECTIONAL ASSAULTCUBE ENVIRONMENT")
        print("=" * 70)

        self.memory_reader = ACMemoryReader()
        if not self.memory_reader.attach():
            raise RuntimeError("Failed to attach to AssaultCube!")
        print("[+] Memory reader attached")

        self.raycast = OmniRaycastObserver(
            horizontal_rays=horizontal_rays,
            vertical_layers=vertical_layers,
            vertical_min=vertical_min,
            vertical_max=vertical_max,
            max_distance=max_ray_distance,
        )
        if not self.raycast.attach():
            raise RuntimeError("Failed to attach raycast observer!")

        self.action_mapper = ActionMapper(aim_scale=aim_scale)
        print("[+] Action mapper ready")

        # Observation space
        self.observation_space = spaces.Dict({
            'player': spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
            'rays': spaces.Box(0.0, 1.0, shape=(self.total_rays, 2), dtype=np.float32),
        })

        # Action space
        self.action_space = spaces.Dict({
            'movement': spaces.MultiDiscrete([3, 3, 2, 2]),
            'aim': spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            'combat': spaces.MultiDiscrete([2, 2]),
        })

        # State
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_game_state = None
        self._last_rays = None

        # Rewards
        self._last_health = 100
        self._last_frags = 0
        self._last_damage_dealt = 0  # Track damage dealt to enemies
        self._is_dead = False
        self._kills = 0
        self._deaths = 0
        self._total_damage = 0  # Total damage in episode

        # Anti-stuck
        self._position_history: deque = deque(maxlen=stuck_history)

        # Performance
        self._step_times = []
        self._last_fps = 0.0

        print(f"[+] Observation: player(6) + rays({self.total_rays}x2) = {6 + self.total_rays*2} values")
        print("=" * 70)

    def _get_stuck_ratio(self) -> float:
        """Calculate stuck ratio."""
        if len(self._position_history) < 10:
            return 0.0

        current = self._position_history[-1]
        same_count = sum(
            1 for pos in list(self._position_history)[:-1]
            if math.sqrt(sum((a-b)**2 for a, b in zip(pos, current))) < self.stuck_threshold
        )
        return same_count / (len(self._position_history) - 1)

    def _get_observation(self) -> dict:
        """Get observation."""
        gs = self.memory_reader.read_state()
        self._last_game_state = gs

        pos = (gs.pos_x, gs.pos_y, gs.pos_z)
        self._position_history.append(pos)

        player_obs = np.array([
            gs.health / 100.0,
            gs.armor / 100.0,
            min(gs.ammo_mag, 100) / 100.0,
            np.clip(gs.vel_x / 50.0, -1, 1),
            np.clip(gs.vel_y / 50.0, -1, 1),
            np.clip(gs.vel_z / 50.0, -1, 1),
        ], dtype=np.float32)

        rays = self.raycast.get_observation()
        self._last_rays = rays

        return {'player': player_obs, 'rays': rays}

    def _calculate_reward(self) -> float:
        """Calculate reward."""
        if self._last_game_state is None:
            return 0.0

        reward = 0.0
        gs = self._last_game_state
        is_dead = gs.health <= 0

        # Death
        if is_dead and not self._is_dead:
            reward -= 50.0
            self._deaths += 1
            self._is_dead = True
            return reward
        elif not is_dead and self._is_dead:
            self._is_dead = False
            self._last_health = gs.health
            return reward
        elif is_dead:
            return reward

        # Alive
        reward += 0.01  # Survival

        # Damage taken
        if gs.health < self._last_health:
            reward -= (self._last_health - gs.health) * 0.3
        self._last_health = gs.health

        # Enemy detection - ONLY reward for enemies in FOV (where agent is looking)
        # Rays are relative to player yaw: ray 0 = forward, ray 36 = behind (for 72 rays)
        # FOV is ±45° = rays 0-9 (right) and 63-71 (left) for 72 horizontal rays
        if self._last_rays is not None:
            hit_types = self._last_rays[:, 1]

            # Count enemies ONLY in forward FOV (±45° = 90° total)
            fov_half = self.h_rays // 8  # 9 rays = 45° for 72 rays
            fov_enemy_rays = 0

            for layer in range(self.v_layers):
                layer_start = layer * self.h_rays
                # Front-right: rays 0 to fov_half
                for i in range(fov_half + 1):
                    if hit_types[layer_start + i] > 0.9:
                        fov_enemy_rays += 1
                # Front-left: rays (h_rays - fov_half) to (h_rays - 1)
                for i in range(self.h_rays - fov_half, self.h_rays):
                    if hit_types[layer_start + i] > 0.9:
                        fov_enemy_rays += 1

            # Strong reward for enemies in FOV - this is what we want!
            if fov_enemy_rays > 0:
                reward += 0.5 * min(fov_enemy_rays, 10)

            # Small awareness bonus for enemies detected anywhere (encourages not ignoring threats)
            # But much weaker than FOV reward
            total_enemy_rays = np.sum(hit_types > 0.9)
            if total_enemy_rays > 0 and fov_enemy_rays == 0:
                # Enemy nearby but not looking at them - tiny bonus just for awareness
                reward += 0.01

        # Damage dealt to enemies - MAJOR reward signal
        if gs.damage_dealt > self._last_damage_dealt:
            damage_this_step = gs.damage_dealt - self._last_damage_dealt
            reward += damage_this_step * 1.0  # +1 per damage point dealt
            self._total_damage += damage_this_step
        self._last_damage_dealt = gs.damage_dealt

        # Kill (on top of damage reward)
        if gs.frags > self._last_frags:
            new_kills = gs.frags - self._last_frags
            reward += 50.0 * new_kills  # Bonus for securing the kill
            self._kills += new_kills
        self._last_frags = gs.frags

        # Movement bonus
        velocity = math.sqrt(gs.vel_x**2 + gs.vel_y**2)
        if velocity > 10.0:
            reward += 0.005

        # Anti-stuck
        stuck = self._get_stuck_ratio()
        if stuck > 0.7:
            reward -= 0.05 * stuck

        return reward

    def _execute_action(self, action: dict):
        """Execute action."""
        movement = action['movement']
        aim = action['aim']
        combat = action['combat']

        forward = 1 if movement[0] == 0 else 0
        backward = 1 if movement[0] == 2 else 0
        left = 1 if movement[1] == 0 else 0
        right = 1 if movement[1] == 2 else 0
        jump = int(movement[2])
        crouch = int(movement[3])

        flat = np.array([
            forward, backward, left, right, jump, crouch,
            aim[0], aim[1], combat[0], combat[1]
        ], dtype=np.float32)

        self.action_mapper.execute(flat)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """Reset."""
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_reward = 0.0
        self._step_times = []
        self._position_history.clear()

        self.action_mapper.reset()

        obs = self._get_observation()

        if self._last_game_state and self._last_game_state.health <= 0:
            time.sleep(1.5)
            self.action_mapper.click_to_respawn()
            time.sleep(0.5)
            obs = self._get_observation()

        if self._last_game_state:
            self._last_frags = self._last_game_state.frags
            self._last_health = self._last_game_state.health
            self._last_damage_dealt = self._last_game_state.damage_dealt
            self._is_dead = False
            self._total_damage = 0

        return obs, {'kills': self._kills, 'deaths': self._deaths, 'damage': self._total_damage}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Step."""
        step_start = time.perf_counter()

        if check_stop():
            self.action_mapper.reset()
            return self._get_observation(), 0.0, True, False, {'emergency_stop': True}

        self._execute_action(action)

        obs = self._get_observation()
        reward = self._calculate_reward()
        self._episode_reward += reward

        terminated = self._last_game_state and self._last_game_state.health <= 0
        truncated = self._step_count >= self.max_episode_steps
        self._step_count += 1

        step_time = time.perf_counter() - step_start
        self._step_times.append(step_time)
        if len(self._step_times) > 100:
            self._step_times.pop(0)
        self._last_fps = len(self._step_times) / sum(self._step_times) if self._step_times else 0

        info = {
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'kills': self._kills,
            'deaths': self._deaths,
            'damage': self._total_damage,
            'fps': self._last_fps,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render."""
        if not self._last_game_state or self._last_rays is None:
            return

        gs = self._last_game_state

        print("\033[H\033[J", end="")  # Clear
        print("=" * 50)
        print(f"  HP: {gs.health:3d}  Armor: {gs.armor:3d}  K/D: {self._kills}/{self._deaths}")
        print(f"  Reward: {self._episode_reward:.1f}  FPS: {self._last_fps:.0f}")
        print("=" * 50)

        # Radar
        print(self.raycast.visualize_radar(self._last_rays))

        # Layer summary
        print("\n  Layers (360° view, front is left side):")
        for i in range(self.v_layers):
            layer_vis = self.raycast.visualize_layer(i, self._last_rays)
            print(f"  [{layer_vis}]")

        print("=" * 50)

    def close(self):
        """Cleanup."""
        self.action_mapper.reset()
        self.memory_reader.detach()
        self.raycast.detach()

        if self._step_times:
            print(f"\n[OmniEnv] Final FPS: {self._last_fps:.0f}")


def test():
    """Test environment."""
    print("\n[*] Testing OmniAssaultCubeEnv...")

    env = OmniAssaultCubeEnv(render_mode="human")

    print(f"\nObservation: {env.observation_space}")
    print(f"Action: {env.action_space}")

    print("\n[*] Running 300 steps...")
    obs, _ = env.reset()

    for _ in range(300):
        action = {
            'movement': env.action_space['movement'].sample(),
            'aim': env.action_space['aim'].sample(),
            'combat': env.action_space['combat'].sample(),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print("\n[*] Done!")


if __name__ == "__main__":
    test()
