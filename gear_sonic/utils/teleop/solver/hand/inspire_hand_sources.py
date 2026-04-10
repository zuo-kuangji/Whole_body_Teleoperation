"""Source-of-truth helpers for Inspire hand assets and upstream integration."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path


XR_TELEOP_ROOT = Path("/home/g1/zuo/xr_teleoperate")
XR_TELEOP_TELEOP_DIR = XR_TELEOP_ROOT / "teleop"
XR_TELEOP_ROBOT_CONTROL_DIR = XR_TELEOP_ROOT / "teleop" / "robot_control"
XR_TELEOP_INSPIRE_CONFIG = XR_TELEOP_ROOT / "assets" / "inspire_hand" / "inspire_hand.yml"


def get_local_inspire_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "data"
        / "robot_model"
        / "model_data"
        / "g1"
        / "inspire_hand"
        / "inspire_hand.yml"
    )


def choose_inspire_config_path(
    *,
    preferred_path: Path = XR_TELEOP_INSPIRE_CONFIG,
    fallback_path: Path | None = None,
) -> Path:
    fallback_path = fallback_path or get_local_inspire_config_path()
    if preferred_path.exists():
        return preferred_path
    return fallback_path


def prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@contextmanager
def temporary_workdir(path: Path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
