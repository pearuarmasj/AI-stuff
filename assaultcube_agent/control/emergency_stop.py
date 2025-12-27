"""
Emergency stop system for training.

Press F10 to immediately stop all agent actions and halt training.
Press F9 to pause/resume training.

Uses pynput for global hotkey detection that works even when game has focus.
"""

import threading
import time
from typing import Callable

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("WARNING: pynput not installed. Run: pip install pynput")


class EmergencyStop:
    """
    Global emergency stop system.

    F10 = STOP (kills everything, releases all keys)
    F9 = PAUSE/RESUME (toggles pause state)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._stopped = False
        self._paused = False
        self._listener = None
        self._on_stop_callback: Callable | None = None
        self._on_pause_callback: Callable | None = None
        self._lock = threading.Lock()

    def start(self, on_stop: Callable | None = None, on_pause: Callable | None = None):
        """
        Start listening for emergency hotkeys.

        Args:
            on_stop: Callback when F10 is pressed (stop everything)
            on_pause: Callback when F9 is pressed (toggle pause)
        """
        if not PYNPUT_AVAILABLE:
            print("[!] Emergency stop disabled - pynput not available")
            return

        self._on_stop_callback = on_stop
        self._on_pause_callback = on_pause

        print("=" * 50)
        print("EMERGENCY CONTROLS ACTIVE:")
        print("  F10 = STOP (immediately halt everything)")
        print("  F9  = PAUSE/RESUME (toggle pause state)")
        print("=" * 50)

        self._listener = keyboard.Listener(on_press=self._on_key_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.f10:
                self._handle_stop()
            elif key == keyboard.Key.f9:
                self._handle_pause()
        except Exception as e:
            print(f"[!] Hotkey error: {e}")

    def _handle_stop(self):
        """Handle F10 emergency stop."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        print("\n" + "!" * 50)
        print("!!! EMERGENCY STOP TRIGGERED (F10) !!!")
        print("!" * 50)

        if self._on_stop_callback:
            try:
                self._on_stop_callback()
            except Exception as e:
                print(f"[!] Stop callback error: {e}")

    def _handle_pause(self):
        """Handle F9 pause toggle."""
        with self._lock:
            self._paused = not self._paused
            state = "PAUSED" if self._paused else "RESUMED"

        print(f"\n[*] Training {state} (F9)")

        if self._on_pause_callback:
            try:
                self._on_pause_callback(self._paused)
            except Exception as e:
                print(f"[!] Pause callback error: {e}")

    @property
    def stopped(self) -> bool:
        """Check if emergency stop was triggered."""
        return self._stopped

    @property
    def paused(self) -> bool:
        """Check if training is paused."""
        return self._paused

    def wait_if_paused(self, check_interval: float = 0.1):
        """
        Block while paused. Call this in training loop.

        Returns False if stopped, True otherwise.
        """
        while self._paused and not self._stopped:
            time.sleep(check_interval)
        return not self._stopped

    def reset(self):
        """Reset stop/pause state for a new run."""
        with self._lock:
            self._stopped = False
            self._paused = False

    def stop_listener(self):
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None


# Global instance
emergency_stop = EmergencyStop()


def check_stop() -> bool:
    """Quick check if we should stop. Returns True if stopped."""
    return emergency_stop.stopped


def check_paused() -> bool:
    """Quick check if paused."""
    return emergency_stop.paused


def wait_if_paused() -> bool:
    """Block while paused. Returns False if stopped."""
    return emergency_stop.wait_if_paused()
