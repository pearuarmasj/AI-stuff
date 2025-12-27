"""
Live damage monitor - shows damage dealt in real-time.

Run this, then shoot enemies. Watch the numbers change.
If they change when you hit enemies, the fix is working.

Usage:
    python -m assaultcube_agent.memory.damage_monitor
"""

import time
import pymem
import pymem.process

from .offsets import PLAYER1_PTR_OFFSET, PSTAT_DAMAGE_BASE, HEALTH, FRAGS
from .structs import NUMGUNS

PROCESS_NAME = "ac_client.exe"

# Weapon names for display
GUN_NAMES = [
    "Knife", "Pistol", "Carbine", "Shotgun", "SMG",
    "Sniper", "Assault", "Grenade", "Akimbo"
]


def main():
    print("=" * 60)
    print("  LIVE DAMAGE MONITOR")
    print("=" * 60)
    print("\nShoot enemies and watch the damage values update!")
    print("Press Ctrl+C to exit\n")

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

    print(f"[+] PSTAT_DAMAGE_BASE offset: 0x{PSTAT_DAMAGE_BASE:X}")
    print("\n" + "-" * 60)

    last_total = 0
    last_damages = [0] * NUMGUNS

    try:
        while True:
            # Read player pointer
            player_ptr = pm.read_int(module_base + PLAYER1_PTR_OFFSET)

            if player_ptr == 0:
                print("\r[!] Player pointer is null - are you in a game?", end="")
                time.sleep(0.5)
                continue

            # Read health and frags for context
            health = pm.read_int(player_ptr + HEALTH)
            frags = pm.read_int(player_ptr + FRAGS)

            # Read all pstatdamage values
            damages = []
            total = 0
            for i in range(NUMGUNS):
                dmg = pm.read_int(player_ptr + PSTAT_DAMAGE_BASE + i * 4)
                damages.append(dmg)
                if dmg > 0:
                    total += dmg

            # Check for changes
            changed = total != last_total

            # Clear and print status
            print("\033[H\033[J", end="")  # Clear screen
            print("=" * 60)
            print("  LIVE DAMAGE MONITOR")
            print("=" * 60)
            print(f"\n  Health: {health}    Frags: {frags}")
            print(f"\n  Player ptr: 0x{player_ptr:X}")
            print(f"  Reading from: 0x{player_ptr + PSTAT_DAMAGE_BASE:X}")
            print("\n" + "-" * 60)
            print("  Damage by weapon:")
            print("-" * 60)

            for i, (name, dmg) in enumerate(zip(GUN_NAMES, damages)):
                delta = dmg - last_damages[i]
                delta_str = f" (+{delta})" if delta > 0 else ""
                marker = " <-- NEW!" if delta > 0 else ""
                print(f"  [{i}] {name:10s}: {dmg:6d}{delta_str}{marker}")

            print("-" * 60)
            print(f"  TOTAL DAMAGE: {total}")

            if changed and total > last_total:
                delta = total - last_total
                print(f"\n  >>> DAMAGE DEALT: +{delta} <<<")
            elif total == 0:
                print(f"\n  [No damage recorded yet - shoot some enemies!]")

            print("\n  Press Ctrl+C to exit")

            last_total = total
            last_damages = damages.copy()

            time.sleep(0.1)  # 10 Hz update

    except KeyboardInterrupt:
        print("\n\n[*] Stopped")

    pm.close_process()
    print("[+] Done!")


if __name__ == "__main__":
    main()
