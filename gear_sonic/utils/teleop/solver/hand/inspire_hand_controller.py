"""Inspire dexterous hand controller for PICO VR hand tracking.

Reads PICO hand tracking, converts the XRoboToolkit palm-first 26-joint hand
layout into the wrist-first 25-joint format expected by the upstream
retargeting pipeline, runs DexPilot retargeting to get 6-DOF joint angles, and
sends commands to Inspire hands via Unitree DDS.

Usage:
    controller = InspireHandController(mode="FTP")
    controller.start()  # starts background thread
    # In your loop:
    controller.update(left_hand_state, right_hand_state)
    # Cleanup:
    controller.stop()
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

from gear_sonic.utils.teleop.solver.hand.inspire_hand_mapping import (
    remap_normalized_hand_command,
    xrt_hand_state_to_unitree_hand_positions,
)
from gear_sonic.utils.teleop.solver.hand.inspire_hand_sources import (
    INSPIRE_HAND_SDK_ROOT,
    XR_TELEOP_ROOT,
    XR_TELEOP_TELEOP_DIR,
    choose_inspire_config_path,
    prepend_sys_path,
    temporary_workdir,
)

# Add dex-retargeting to path (from xr_teleoperate)
_DEX_RETARGET_PATH = "/home/g1/zuo/xr_teleoperate/teleop/robot_control/dex-retargeting/src"
prepend_sys_path(XR_TELEOP_ROOT)
prepend_sys_path(Path(_DEX_RETARGET_PATH))
prepend_sys_path(INSPIRE_HAND_SDK_ROOT)

from dex_retargeting.retargeting_config import RetargetingConfig

try:
    from teleop.robot_control.hand_retargeting import HandRetargeting as UpstreamHandRetargeting
    from teleop.robot_control.hand_retargeting import HandType as UpstreamHandType
except ImportError:
    UpstreamHandRetargeting = None
    UpstreamHandType = None

# Inspire hand has 6 actuated DOF per hand
INSPIRE_NUM_MOTORS = 6
FINGER_COMMAND_NAMES = [
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_pitch",
    "thumb_yaw",
]

# Joint normalization ranges (radians) — from xr_teleoperate
# Order after hardware reorder: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
JOINT_RANGES = [
    (0.0, 1.7),   # pinky
    (0.0, 1.7),   # ring
    (0.0, 1.7),   # middle
    (0.0, 1.7),   # index
    (0.0, 0.5),   # thumb pitch (bend)
    (-0.1, 1.3),  # thumb yaw (rotation)
]


def get_inspire_hand_transport_mode(*, hand_sim: bool) -> str:
    """Return the DDS transport mode used for Inspire hands.

    We intentionally keep simulation and real hardware on the same FTP-style DDS
    topics so the receive side can be tested end-to-end without changing
    transport conventions between environments.
    """
    _ = hand_sim
    return "FTP"


def _normalize(val, min_val, max_val):
    """Convert radians to [0, 1] where 1=open, 0=closed."""
    return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)


def _denormalize(val, min_val, max_val):
    """Convert [0, 1] back to radians. Inverse of _normalize."""
    return max_val - val * (max_val - min_val)


def pico_to_openxr_25(pico_state, arm_pose=None):
    """Convert XRoboToolkit hand state to 25x3 retargeting positions.

    XRoboToolkit hand tracking is palm-first (0=palm, 1=wrist, 2..25 fingers),
    so the conversion drops the palm and keeps joints 1..25 in wrist-first
    order.
    """
    return xrt_hand_state_to_unitree_hand_positions(pico_state, arm_pose=arm_pose)


class InspireHandRetargeting:
    """Wrapper around dex_retargeting for Inspire hands."""

    def __init__(self, config_path=None):
        self.source = "local"
        self.config_path = None
        self.upstream_error = None

        if config_path is None and UpstreamHandRetargeting is not None and UpstreamHandType is not None:
            try:
                with temporary_workdir(XR_TELEOP_TELEOP_DIR):
                    upstream = UpstreamHandRetargeting(UpstreamHandType.INSPIRE_HAND)

                self.left_retargeting = upstream.left_retargeting
                self.right_retargeting = upstream.right_retargeting
                self.left_joint_names = upstream.left_retargeting_joint_names
                self.right_joint_names = upstream.right_retargeting_joint_names
                self.left_indices = upstream.left_indices
                self.right_indices = upstream.right_indices
                self.left_reorder = upstream.left_dex_retargeting_to_hardware
                self.right_reorder = upstream.right_dex_retargeting_to_hardware
                self.config_path = choose_inspire_config_path()
                self.source = "xr_teleoperate"
                return
            except Exception as exc:
                self.upstream_error = exc

        if config_path is None:
            config_path = choose_inspire_config_path()
        else:
            config_path = Path(config_path)

        self.config_path = config_path

        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Set URDF search directory to parent of inspire_hand/
        urdf_dir = str(config_path.parent.parent)
        RetargetingConfig.set_default_urdf_dir(urdf_dir)

        left_config = RetargetingConfig.from_dict(cfg["left"])
        right_config = RetargetingConfig.from_dict(cfg["right"])
        self.left_retargeting = left_config.build()
        self.right_retargeting = right_config.build()

        # Get joint name ordering from retargeting
        self.left_joint_names = self.left_retargeting.joint_names
        self.right_joint_names = self.right_retargeting.joint_names

        # Human keypoint indices for vector computation
        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

        # Hardware joint order: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        left_hw_names = [
            "L_pinky_proximal_joint", "L_ring_proximal_joint",
            "L_middle_proximal_joint", "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint",
        ]
        right_hw_names = [
            "R_pinky_proximal_joint", "R_ring_proximal_joint",
            "R_middle_proximal_joint", "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
        ]
        self.left_reorder = [self.left_joint_names.index(n) for n in left_hw_names]
        self.right_reorder = [self.right_joint_names.index(n) for n in right_hw_names]

    def retarget(self, left_pos_25x3, right_pos_25x3):
        """Retarget 25-joint hand positions to 6-DOF normalized [0,1] commands.

        Args:
            left_pos_25x3: (25, 3) array, wrist-centered OpenXR positions
            right_pos_25x3: (25, 3) array, wrist-centered OpenXR positions

        Returns:
            left_cmd: (6,) array normalized [0,1]
            right_cmd: (6,) array normalized [0,1]
        """
        # Compute direction vectors for DexPilot
        ref_left = left_pos_25x3[self.left_indices[1, :]] - left_pos_25x3[self.left_indices[0, :]]
        ref_right = right_pos_25x3[self.right_indices[1, :]] - right_pos_25x3[self.right_indices[0, :]]

        # Run retargeting optimizer -> joint angles in radians
        left_q = self.left_retargeting.retarget(ref_left)[self.left_reorder]
        right_q = self.right_retargeting.retarget(ref_right)[self.right_reorder]

        # Normalize radians -> [0, 1]
        left_cmd = np.array([_normalize(left_q[i], *JOINT_RANGES[i]) for i in range(INSPIRE_NUM_MOTORS)])
        right_cmd = np.array([_normalize(right_q[i], *JOINT_RANGES[i]) for i in range(INSPIRE_NUM_MOTORS)])

        return left_cmd, right_cmd


class InspireHandController:
    """Controls Inspire hands via DDS, reading PICO hand tracking data.

    Supports two DDS modes:
      - "DFX": Single topic rt/inspire/cmd (MotorCmds_, 12 motors: R=0-5, L=6-11)
      - "FTP": Separate topics rt/inspire_hand/ctrl/l and rt/inspire_hand/ctrl/r
    """

    def __init__(self, mode="FTP", fps=50.0, sim=False):
        self.mode = mode
        self.fps = fps
        self.sim = sim
        self.running = False
        self._thread = None
        self.retargeting = InspireHandRetargeting()
        if self.retargeting.upstream_error is not None:
            print(
                "[InspireHand] Upstream HandRetargeting unavailable, using local fallback:"
                f" {self.retargeting.upstream_error}"
            )

        # Latest hand tracking data (set by update())
        self._left_hand_state = None
        self._right_hand_state = None
        self._left_arm_pose = None
        self._right_arm_pose = None
        self._lock = threading.Lock()

        # DDS publishers (initialized in start())
        self._publishers_initialized = False

    def _init_dds(self):
        """Initialize DDS publishers. Called once in the background thread."""
        if self._publishers_initialized:
            return

        from unitree_sdk2py.core.channel import ChannelPublisher

        if self.mode == "DFX":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
            self._pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
            self._pub.Init()
        else:  # FTP
            from inspire_dds import inspire_hand_ctrl
            self._left_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_hand_ctrl)
            self._right_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_hand_ctrl)
            self._left_pub.Init()
            self._right_pub.Init()

        self._publishers_initialized = True

    def update(self, left_pico_state, right_pico_state, left_arm_pose=None, right_arm_pose=None):
        """Feed new hand tracking data (26x7 arrays or None)."""
        with self._lock:
            self._left_hand_state = left_pico_state
            self._right_hand_state = right_pico_state
            self._left_arm_pose = left_arm_pose
            self._right_arm_pose = right_arm_pose

    def start(self):
        """Start the background control thread."""
        if self._thread is not None:
            return
        self.running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background control thread."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _control_loop(self):
        self._init_dds()

        left_cmd = np.ones(INSPIRE_NUM_MOTORS)  # 1.0 = fully open
        right_cmd = np.ones(INSPIRE_NUM_MOTORS)

        while self.running:
            t0 = time.time()

            with self._lock:
                left_state = self._left_hand_state
                right_state = self._right_hand_state
                left_arm_pose = self._left_arm_pose
                right_arm_pose = self._right_arm_pose

            has_data = left_state is not None and right_state is not None

            if has_data:
                left_25 = pico_to_openxr_25(left_state, arm_pose=left_arm_pose)
                right_25 = pico_to_openxr_25(right_state, arm_pose=right_arm_pose)

                # Skip if data looks uninitialized
                if not (np.all(left_25 == 0) or np.all(right_25 == 0)):
                    try:
                        left_cmd_raw, right_cmd_raw = self.retargeting.retarget(left_25, right_25)
                        left_cmd = remap_normalized_hand_command(left_cmd_raw)
                        right_cmd = remap_normalized_hand_command(right_cmd_raw)
                    except Exception as e:
                        print(f"[InspireHand] Retargeting error: {e}")

            self._send_command(left_cmd, right_cmd)

            elapsed = time.time() - t0
            sleep_time = max(0, (1.0 / self.fps) - elapsed)
            time.sleep(sleep_time)

    def _send_command(self, left_cmd, right_cmd):
        """Send normalized [0,1] commands to Inspire hands via DDS."""
        if self.mode == "DFX":
            self._send_dfx(left_cmd, right_cmd)
        else:
            self._send_ftp(left_cmd, right_cmd)

    def _send_dfx(self, left_cmd, right_cmd):
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(12)]

        for i in range(INSPIRE_NUM_MOTORS):
            # Right hand: indices 0-5, Left hand: indices 6-11
            if self.sim:
                # Sim needs radians + PD gains for torque computation
                r_rad = _denormalize(float(right_cmd[i]), *JOINT_RANGES[i])
                l_rad = _denormalize(float(left_cmd[i]), *JOINT_RANGES[i])
                msg.cmds[i].q = r_rad
                msg.cmds[i].kp = 4.0
                msg.cmds[i].kd = 0.2
                msg.cmds[6 + i].q = l_rad
                msg.cmds[6 + i].kp = 4.0
                msg.cmds[6 + i].kd = 0.2
            else:
                # Real robot: normalized [0,1] values
                msg.cmds[i].q = float(right_cmd[i])
                msg.cmds[6 + i].q = float(left_cmd[i])

        self._pub.Write(msg)

    def _send_ftp(self, left_cmd, right_cmd):
        from inspire_dds import inspire_hand_ctrl

        left_msg = inspire_hand_ctrl(
            pos_set=[0 for _ in range(INSPIRE_NUM_MOTORS)],
            angle_set=[int(np.clip(v * 1000, 0, 1000)) for v in left_cmd],
            force_set=[0 for _ in range(INSPIRE_NUM_MOTORS)],
            speed_set=[0 for _ in range(INSPIRE_NUM_MOTORS)],
            mode=0b0001,
        )
        self._left_pub.Write(left_msg)

        right_msg = inspire_hand_ctrl(
            pos_set=[0 for _ in range(INSPIRE_NUM_MOTORS)],
            angle_set=[int(np.clip(v * 1000, 0, 1000)) for v in right_cmd],
            force_set=[0 for _ in range(INSPIRE_NUM_MOTORS)],
            speed_set=[0 for _ in range(INSPIRE_NUM_MOTORS)],
            mode=0b0001,
        )
        self._right_pub.Write(right_msg)
