"""Helpers for preparing Inspire hand tracking and command remapping."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


INSPIRE_NUM_MOTORS = 6
THUMB_PITCH_INDEX = 4
THUMB_PITCH_CLOSING_GAIN = 1.5

R_ROBOT_OPENXR = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Same fixed transform used by xr_teleoperate after converting to the wrist-local
# OpenXR hand frame. It rotates the points into the Unitree hand convention that the
# hand retargeter was tuned against.
T_TO_UNITREE_HAND = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _as_command_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (INSPIRE_NUM_MOTORS,):
        raise ValueError(f"{name} must have shape ({INSPIRE_NUM_MOTORS},), got {array.shape}")
    return array


def remap_normalized_hand_command(
    command,
) -> np.ndarray:
    """Keep the normalized Inspire command unchanged, except for thumb bend gain.

    The retargeted thumb bend signal is slightly too conservative for the MuJoCo
    Inspire hand. We therefore amplify only the thumb-pitch closing amount while
    leaving every other channel untouched.
    """

    command = _as_command_array(command, name="command").copy()

    thumb_pitch = command[THUMB_PITCH_INDEX]
    thumb_pitch_closing = 1.0 - thumb_pitch
    command[THUMB_PITCH_INDEX] = np.clip(
        1.0 - THUMB_PITCH_CLOSING_GAIN * thumb_pitch_closing,
        0.0,
        1.0,
    )

    return command


def xrt_hand_state_to_unitree_hand_positions(hand_state, arm_pose=None) -> np.ndarray:
    """Convert raw XRToolkit hand poses into Unitree-hand positions.

    XRoboToolkit exposes 26 hand poses shaped like ``[x, y, z, qx, qy, qz, qw]``.
    The PC Service hand layout is palm-first:
      0 palm, 1 wrist, 2-5 thumb, 6-10 index, 11-15 middle, 16-20 ring, 21-25 little.
    We therefore drop the palm and keep joints ``1:26`` so the downstream
    retargeter sees the wrist-first 25-joint layout expected by ``xr_teleoperate``.

    When an ``arm_pose`` is provided, we transform the hand joints from WORLD into
    that arm frame before rotating them into the Unitree hand convention. Without
    ``arm_pose`` we use the wrist pose from the hand-state itself.
    """

    state = np.asarray(hand_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] < 26 or state.shape[1] < 7:
        raise ValueError(f"hand_state must have shape (>=26, >=7), got {state.shape}")

    joint_poses = state[1:26, :7].copy()
    joint_positions = joint_poses[:, :3]
    joint_positions_robot = (R_ROBOT_OPENXR @ joint_positions.T).T

    if arm_pose is not None:
        arm_pose = np.asarray(arm_pose, dtype=np.float64)
        if arm_pose.shape != (7,):
            raise ValueError(f"arm_pose must have shape (7,), got {arm_pose.shape}")

        arm_translation = R_ROBOT_OPENXR @ arm_pose[:3]
        arm_quat = arm_pose[3:7]
        arm_quat_norm = np.linalg.norm(arm_quat)
        local_positions = joint_positions_robot - arm_translation
        if np.isfinite(arm_quat_norm) and arm_quat_norm > 1e-8:
            arm_rot_openxr = R.from_quat(arm_quat / arm_quat_norm)
            arm_rot = R.from_matrix(
                R_ROBOT_OPENXR @ arm_rot_openxr.as_matrix() @ R_ROBOT_OPENXR.T
            )
            local_positions = arm_rot.inv().apply(local_positions)
    else:
        wrist_pos = joint_positions_robot[0].copy()
        local_positions = joint_positions_robot - wrist_pos

        wrist_quat = joint_poses[0, 3:7].copy()
        wrist_quat_norm = np.linalg.norm(wrist_quat)
        if np.isfinite(wrist_quat_norm) and wrist_quat_norm > 1e-8:
            wrist_rot_openxr = R.from_quat(wrist_quat / wrist_quat_norm)
            wrist_rot = R.from_matrix(
                R_ROBOT_OPENXR @ wrist_rot_openxr.as_matrix() @ R_ROBOT_OPENXR.T
            )
            local_positions = wrist_rot.inv().apply(local_positions)

    return (T_TO_UNITREE_HAND @ local_positions.T).T
