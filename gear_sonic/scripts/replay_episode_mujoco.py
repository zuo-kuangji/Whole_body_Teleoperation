"""Replay an xr_teleoperate-style episode (data.json) in MuJoCo.

Loads `<episode>/data.json`, drives the G1 + Inspire scene by writing per-frame
`states.body.qpos` (29 joints, mujoco order) into `qpos[7:36]` and the 6 inspire
driver joints per hand from `states.{left,right}_ee.qpos[:6]` (hardware order
`[pinky, ring, middle, index, thumb_pitch, thumb_yaw]`). Equality constraints
in the MJCF handle the coupled fingertip joints.

Usage:
    python gear_sonic/scripts/replay_episode_mujoco.py \
        --episode /home/g1/hsj/tmp/episode_0000 \
        [--source actions] [--scene scene_inspire.xml] [--speed 1.0] [--loop]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_inspire.xml"

# Inspire hardware -> mujoco hand driver joint order.
# Hardware publishes  : [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# MuJoCo driver order : [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
INSPIRE_HW_TO_MJ = [5, 4, 3, 2, 1, 0]

BODY_JOINTS_MUJOCO_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

LEFT_HAND_DRIVER_JOINTS = [
    "left_hand_thumb_yaw_joint",
    "left_hand_thumb_pitch_joint",
    "left_hand_index_joint",
    "left_hand_middle_joint",
    "left_hand_ring_joint",
    "left_hand_pinky_joint",
]
RIGHT_HAND_DRIVER_JOINTS = [j.replace("left_", "right_") for j in LEFT_HAND_DRIVER_JOINTS]


def _segment(frame_entry: dict, group: str, field: str) -> list[float]:
    return list(frame_entry.get(group, {}).get(field, {}).get("qpos", []) or [])


def _build_hand_qposadr(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    adr = []
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise RuntimeError(f"joint '{n}' not found in model")
        adr.append(int(model.jnt_qposadr[jid]))
    return np.asarray(adr, dtype=int)


def _apply_inspire_hw_qpos(data: mujoco.MjData, driver_adr: np.ndarray, hw_q: list[float]) -> None:
    if len(hw_q) < 6:
        return
    for mj_slot, hw_idx in enumerate(INSPIRE_HW_TO_MJ):
        data.qpos[driver_adr[mj_slot]] = float(hw_q[hw_idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, type=Path, help="episode directory containing data.json")
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--source", choices=["states", "actions"], default="states")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--base-height", type=float, default=0.793, help="floating base z when replaying")
    args = ap.parse_args()

    data_json = args.episode / "data.json"
    with data_json.open() as f:
        episode = json.load(f)

    frames = episode["data"]
    fps = float(episode.get("info", {}).get("image", {}).get("fps") or 50)
    dt = 1.0 / fps / max(args.speed, 1e-6)
    print(f"[replay] {len(frames)} frames @ {fps} Hz (speed x{args.speed}) source={args.source}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    left_adr = _build_hand_qposadr(model, LEFT_HAND_DRIVER_JOINTS)
    right_adr = _build_hand_qposadr(model, RIGHT_HAND_DRIVER_JOINTS)
    body_adr = _build_hand_qposadr(model, BODY_JOINTS_MUJOCO_ORDER)
    body_dof = len(body_adr)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_start = time.perf_counter()
            for i, frame in enumerate(frames):
                if not viewer.is_running():
                    return

                body = _segment(frame, args.source, "body")
                left_ee = _segment(frame, args.source, "left_ee")
                right_ee = _segment(frame, args.source, "right_ee")

                data.qpos[:3] = [0.0, 0.0, args.base_height]
                data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                data.qvel[:] = 0.0

                if len(body) == body_dof:
                    data.qpos[body_adr] = body

                _apply_inspire_hw_qpos(data, left_adr, left_ee)
                _apply_inspire_hw_qpos(data, right_adr, right_ee)

                mujoco.mj_forward(model, data)
                viewer.sync()

                sleep_for = t_start + (i + 1) * dt - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            if not args.loop:
                print("[replay] done")
                while viewer.is_running():
                    time.sleep(0.1)
                return


if __name__ == "__main__":
    main()
