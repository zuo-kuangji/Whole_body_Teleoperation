from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Any


KEYBOARD_MODE_HELP = "Keyboard mode controls: p=toggle PLANNER, o=POSE, f=PLANNER_FROZEN_UPPER_BODY, v=toggle VR_3PT from POSE"
MOVEMENT_KEYS = {"w", "s", "a", "d", "q", "e"}


class VR3PTKeyboardMotion:
    def __init__(self, clock=time.monotonic, hold_seconds: float = 0.18) -> None:
        self._clock = clock
        self._hold_seconds = hold_seconds
        self._last_seen: dict[str, float] = {}

    def observe_key(self, key: str | None) -> None:
        if key in MOVEMENT_KEYS:
            self._last_seen[key] = self._clock()

    def _is_active(self, key: str) -> bool:
        last_seen = self._last_seen.get(key)
        if last_seen is None:
            return False
        return (self._clock() - last_seen) <= self._hold_seconds

    def virtual_axes(self) -> tuple[float, float, float, float]:
        ly = float(self._is_active("w")) - float(self._is_active("s"))
        lx = float(self._is_active("d")) - float(self._is_active("a"))
        rx = float(self._is_active("e")) - float(self._is_active("q"))
        return (lx, ly, rx, 0.0)


def resolve_vr3pt_locomotion_mode(current_mode, raw_mag: float, locomotion_mode_enum):
    if raw_mag <= 0.0:
        return current_mode
    if current_mode == locomotion_mode_enum.IDLE:
        return locomotion_mode_enum.WALK
    return current_mode


def consume_latest_keyboard_key(read_key_once) -> str | None:
    latest_key = None
    latest_movement_key = None

    while True:
        try:
            key = read_key_once()
        except StopIteration:
            break
        if key is None:
            break
        latest_key = key
        if key in MOVEMENT_KEYS:
            latest_movement_key = key

    return latest_movement_key if latest_movement_key is not None else latest_key


class ConsoleKeyMonitor:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._original_settings: list[Any] | None = None

    def __enter__(self) -> "ConsoleKeyMonitor":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._original_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._fd is not None and self._original_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)

    def read_key(self) -> str | None:
        if self._fd is None:
            return None

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        return sys.stdin.read(1).lower()


def next_mode_from_keyboard(current_mode, key: str | None, stream_mode_enum):
    if not key:
        return current_mode

    if key == "p":
        return stream_mode_enum.OFF if current_mode == stream_mode_enum.PLANNER else stream_mode_enum.PLANNER
    if key == "o":
        return stream_mode_enum.POSE
    if key == "f":
        return stream_mode_enum.PLANNER_FROZEN_UPPER_BODY
    if key == "v":
        if current_mode == stream_mode_enum.POSE:
            return stream_mode_enum.PLANNER_VR_3PT
        if current_mode == stream_mode_enum.PLANNER_VR_3PT:
            return stream_mode_enum.POSE
    return current_mode


def hand_control_enabled_for_mode(current_mode, stream_mode_enum) -> bool:
    return current_mode == stream_mode_enum.POSE
