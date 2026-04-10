# Inspire Hand Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Inspire dexterous hand (6-DOF) support to GR00T-WholeBodyControl, selectable via `--hand-type inspire` flag, working in both MuJoCo simulation and real robot deployment.

**Architecture:** Two independent subsystems: (1) Python teleop side — PICO hand tracking data feeds into DexPilot retargeting, outputting 6-DOF Inspire joint commands sent directly via DDS (bypassing C++ deploy for hand control, since the whole-body policy doesn't control Inspire fingers). (2) MuJoCo sim side — new XML model with Inspire hand geometry replacing Dex3, new YAML config with 6-DOF hands, updated DDS bridge listening on Inspire topics.

**Tech Stack:** Python, MuJoCo XML, dex_retargeting (DexPilot), unitree_sdk2py (DDS), inspire_sdkpy, xrobotoolkit_sdk

---

## File Structure

### New Files
- `gear_sonic/utils/teleop/solver/hand/inspire_hand_controller.py` — Inspire hand retargeting + DDS control (self-contained module)
- `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_hand.xml` — MuJoCo XML for G1 body + Inspire hands
- `gear_sonic/data/robot_model/model_data/g1/scene_inspire.xml` — Scene file including the Inspire model
- `gear_sonic/data/robot_model/model_data/g1/inspire_hand/` — Symlink or copy of Inspire hand assets (URDF, meshes, config)
- `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire.yaml` — WBC config for Inspire hand variant

### Modified Files
- `gear_sonic/scripts/pico_manager_thread_server.py` — Add `--hand-type` arg, launch Inspire controller when selected
- `gear_sonic/utils/mujoco_sim/base_sim.py` — Handle variable hand DOF (6 for Inspire vs 7 for Dex3) based on joint name detection
- `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` — Add Inspire DDS topic support alongside Dex3

---

### Task 1: Copy Inspire Hand Assets Into Project

**Files:**
- Create: `gear_sonic/data/robot_model/model_data/g1/inspire_hand/` (directory with meshes, URDF, config)

- [ ] **Step 1: Copy Inspire hand assets**

```bash
# Copy URDF, meshes, and config from xr_teleoperate
cp -r /home/g1/zuo/xr_teleoperate/assets/inspire_hand \
      /home/g1/hsj/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/inspire_hand
```

- [ ] **Step 2: Verify files are in place**

```bash
ls gear_sonic/data/robot_model/model_data/g1/inspire_hand/
# Expected: inspire_hand.yml  inspire_hand_left.urdf  inspire_hand_right.urdf  meshes/
ls gear_sonic/data/robot_model/model_data/g1/inspire_hand/meshes/ | wc -l
# Expected: 26
```

- [ ] **Step 3: Commit**

```bash
git add gear_sonic/data/robot_model/model_data/g1/inspire_hand/
git commit -m "feat: add Inspire dexterous hand assets (URDF, meshes, retarget config)"
```

---

### Task 2: Create Inspire Hand Controller Module

**Files:**
- Create: `gear_sonic/utils/teleop/solver/hand/inspire_hand_controller.py`

This module reads PICO hand tracking, runs DexPilot retargeting, and sends DDS commands to Inspire hands. It runs as a daemon thread inside the pico_manager process.

- [ ] **Step 1: Create the controller module**

Create `gear_sonic/utils/teleop/solver/hand/inspire_hand_controller.py` with:

```python
"""Inspire dexterous hand controller for PICO VR hand tracking.

Reads PICO hand tracking (27 joints per hand), converts to 25-joint OpenXR
format, runs DexPilot retargeting to get 6-DOF joint angles, and sends
commands to Inspire hands via Unitree DDS.

Usage:
    controller = InspireHandController(mode="DFX")
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

# Add dex-retargeting to path (from xr_teleoperate)
_DEX_RETARGET_PATH = "/home/g1/zuo/xr_teleoperate/teleop/robot_control/dex-retargeting/src"
if _DEX_RETARGET_PATH not in sys.path:
    sys.path.insert(0, _DEX_RETARGET_PATH)

from dex_retargeting.retargeting_config import RetargetingConfig

# Inspire hand has 6 actuated DOF per hand
INSPIRE_NUM_MOTORS = 6

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


def _normalize(val, min_val, max_val):
    """Convert radians to [0, 1] where 1=open, 0=closed."""
    return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)


def pico_to_openxr_25(pico_state):
    """Convert PICO 27x7 hand state to OpenXR 25x3 positions.

    PICO joints 0=Palm, 1=Wrist, 2-25=finger joints, 26=extra.
    OpenXR 25 joints: 0=Wrist, 1-4=Thumb, 5-9=Index, 10-14=Middle,
    15-19=Ring, 20-24=Pinky.
    """
    # Take joints 1-25 (skip Palm at 0), positions only
    positions = pico_state[1:26, :3].copy()
    # Center at wrist (joint 0 in output)
    positions -= positions[0:1, :]
    return positions


class InspireHandRetargeting:
    """Wrapper around dex_retargeting for Inspire hands."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "data" / "robot_model" / "model_data" / "g1"
                / "inspire_hand" / "inspire_hand.yml"
            )

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

        # Run retargeting optimizer → joint angles in radians
        left_q = self.left_retargeting.retarget(ref_left)[self.left_reorder]
        right_q = self.right_retargeting.retarget(ref_right)[self.right_reorder]

        # Normalize radians → [0, 1]
        left_cmd = np.array([_normalize(left_q[i], *JOINT_RANGES[i]) for i in range(INSPIRE_NUM_MOTORS)])
        right_cmd = np.array([_normalize(right_q[i], *JOINT_RANGES[i]) for i in range(INSPIRE_NUM_MOTORS)])

        return left_cmd, right_cmd


class InspireHandController:
    """Controls Inspire hands via DDS, reading PICO hand tracking data.

    Supports two DDS modes:
      - "DFX": Single topic rt/inspire/cmd (MotorCmds_, 12 motors: R=0-5, L=6-11)
      - "FTP": Separate topics rt/inspire_hand/ctrl/l and rt/inspire_hand/ctrl/r
    """

    def __init__(self, mode="DFX", fps=50.0):
        self.mode = mode
        self.fps = fps
        self.running = False
        self._thread = None
        self.retargeting = InspireHandRetargeting()

        # Latest hand tracking data (set by update())
        self._left_hand_state = None
        self._right_hand_state = None
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
            from inspire_sdkpy import inspire_dds
            self._left_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
            self._right_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
            self._left_pub.Init()
            self._right_pub.Init()

        self._publishers_initialized = True

    def update(self, left_pico_state, right_pico_state):
        """Feed new PICO hand tracking data (27x7 arrays or None)."""
        with self._lock:
            self._left_hand_state = left_pico_state
            self._right_hand_state = right_pico_state

    def start(self):
        """Start the background control thread."""
        if self._thread is not None:
            return
        self.running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        print("[InspireHand] Control thread started")

    def stop(self):
        """Stop the background control thread."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[InspireHand] Control thread stopped")

    def _control_loop(self):
        self._init_dds()

        left_cmd = np.ones(INSPIRE_NUM_MOTORS)  # 1.0 = fully open
        right_cmd = np.ones(INSPIRE_NUM_MOTORS)

        while self.running:
            t0 = time.time()

            with self._lock:
                left_state = self._left_hand_state
                right_state = self._right_hand_state

            if left_state is not None and right_state is not None:
                left_25 = pico_to_openxr_25(left_state)
                right_25 = pico_to_openxr_25(right_state)

                # Skip if data looks uninitialized
                if not (np.all(left_25 == 0) or np.all(right_25 == 0)):
                    try:
                        left_cmd, right_cmd = self.retargeting.retarget(left_25, right_25)
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

        # Right hand: indices 0-5
        for i in range(INSPIRE_NUM_MOTORS):
            msg.cmds[i].q = float(right_cmd[i])

        # Left hand: indices 6-11
        for i in range(INSPIRE_NUM_MOTORS):
            msg.cmds[6 + i].q = float(left_cmd[i])

        self._pub.Write(msg)

    def _send_ftp(self, left_cmd, right_cmd):
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default

        left_msg = inspire_hand_default.get_inspire_hand_ctrl()
        left_msg.angle_set = [int(np.clip(v * 1000, 0, 1000)) for v in left_cmd]
        left_msg.mode = 0b0001
        self._left_pub.Write(left_msg)

        right_msg = inspire_hand_default.get_inspire_hand_ctrl()
        right_msg.angle_set = [int(np.clip(v * 1000, 0, 1000)) for v in right_cmd]
        right_msg.mode = 0b0001
        self._right_pub.Write(right_msg)
```

- [ ] **Step 2: Verify import works**

```bash
cd /home/g1/hsj/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python -c "from gear_sonic.utils.teleop.solver.hand.inspire_hand_controller import InspireHandController; print('OK')"
```

If import fails due to missing `dex_retargeting`, install it:
```bash
pip install -e /home/g1/zuo/xr_teleoperate/teleop/robot_control/dex-retargeting
```

- [ ] **Step 3: Commit**

```bash
git add gear_sonic/utils/teleop/solver/hand/inspire_hand_controller.py
git commit -m "feat: add Inspire hand controller with DexPilot retargeting"
```

---

### Task 3: Add --hand-type Argument to pico_manager

**Files:**
- Modify: `gear_sonic/scripts/pico_manager_thread_server.py`

Add `--hand-type` argument and integrate InspireHandController into the main manager loop.

- [ ] **Step 1: Add argument and import**

At the top of the file, after the existing `G1GripperInverseKinematicsSolver` import block (around line 80-85), add:

```python
try:
    from gear_sonic.utils.teleop.solver.hand.inspire_hand_controller import (
        InspireHandController,
        pico_to_openxr_25,
    )
except ImportError:
    print("Warning: InspireHandController not available.")
    InspireHandController = None
```

In the argparse section (after line 2137, before `args = parser.parse_args()`), add:

```python
parser.add_argument(
    "--hand-type",
    type=str,
    choices=["dex3", "inspire", "none"],
    default="dex3",
    help="Hand type: 'dex3' (default 3-finger gripper), 'inspire' (Inspire dexterous hand), 'none' (no hand control)",
)
```

- [ ] **Step 2: Pass hand_type into run_pico_manager**

Update `run_pico_manager()` function signature (line ~1803) to accept `hand_type`:

```python
def run_pico_manager(
    port: int = 5556,
    ...
    enable_smpl_vis: bool = False,
    hand_type: str = "dex3",
):
```

And the call site (line ~2165):

```python
run_pico_manager(
    ...
    enable_smpl_vis=args.vis_smpl,
    hand_type=args.hand_type,
)
```

- [ ] **Step 3: Launch InspireHandController in manager**

Inside `run_pico_manager()`, after the XRT init block (around line 1833), add:

```python
# Initialize Inspire hand controller if requested
inspire_controller = None
if hand_type == "inspire":
    if InspireHandController is None:
        raise ImportError("InspireHandController not available. Check dex_retargeting installation.")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0)  # domain_id=0 for real robot
    inspire_controller = InspireHandController(mode="DFX", fps=50.0)
    inspire_controller.start()
    print("[Manager] Inspire hand controller started")
```

- [ ] **Step 4: Feed PICO hand tracking data to Inspire controller**

Inside the main `while True` loop of `run_pico_manager()` (around line 1906), after the button polling, add hand tracking update:

```python
# Feed PICO hand tracking to Inspire controller
if inspire_controller is not None:
    try:
        left_ht = xrt.get_left_hand_tracking_state() if xrt.get_left_hand_is_active() else None
        right_ht = xrt.get_right_hand_tracking_state() if xrt.get_right_hand_is_active() else None
        if left_ht is not None and right_ht is not None:
            inspire_controller.update(left_ht, right_ht)
    except Exception:
        pass
```

- [ ] **Step 5: Skip Dex3 hand joints in ZMQ when using Inspire**

In `PlannerStreamer.run_once()` (around line 1763-1772) and `PoseStreamer.run_once()` (around line 1296-1303), the `compute_hand_joints_from_inputs()` call produces Dex3 7-DOF joints. When `hand_type == "inspire"`, we should send zeros (Inspire is controlled directly via DDS, not through ZMQ→C++).

Pass `hand_type` to PlannerStreamer and PoseStreamer constructors and skip hand joint computation when inspire:

In the PlannerStreamer `__init__` (add `hand_type` param):
```python
self.hand_type = hand_type
```

In PlannerStreamer.run_once, wrap the hand joint computation:
```python
if self.hand_type != "inspire":
    lh_joints, rh_joints = compute_hand_joints_from_inputs(...)
    left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
    right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()
# else: left_hand_position/right_hand_position remain None
```

Same pattern for PoseStreamer.

- [ ] **Step 6: Cleanup on exit**

In the `finally` block of `run_pico_manager()` (around line 2046):

```python
if inspire_controller is not None:
    inspire_controller.stop()
```

- [ ] **Step 7: Commit**

```bash
git add gear_sonic/scripts/pico_manager_thread_server.py
git commit -m "feat: add --hand-type argument for Inspire hand teleop via PICO"
```

---

### Task 4: Create MuJoCo XML for G1 + Inspire Hands

**Files:**
- Create: `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_hand.xml`
- Create: `gear_sonic/data/robot_model/model_data/g1/scene_inspire.xml`

The new XML replaces the Dex3 hand bodies/joints/actuators/sensors with Inspire hand equivalents. The Inspire hand has 6 actuated joints + 6 mimic joints per hand. MuJoCo handles URDF mimic joints via `<equality><joint>` constraints.

- [ ] **Step 1: Create G1 + Inspire hand XML**

Copy `g1_29dof_with_hand.xml` and replace the hand sections. The key differences:

**Dex3 (current):** 7 joints per hand (thumb_0/1/2, index_0/1, middle_0/1), joint names contain `left_hand_` / `right_hand_`

**Inspire:** 6 actuated + 6 mimic per hand. Actuated joints use names like `L_thumb_proximal_yaw_joint`. For MuJoCo compatibility, we rename to `left_hand_*` pattern so `base_sim.py` joint detection works.

The new XML must:
1. Replace hand mesh assets with Inspire STL files
2. Replace hand body hierarchy with Inspire kinematic chain
3. Replace hand actuators (6 per hand instead of 7)
4. Add equality constraints for mimic joints
5. Replace hand sensors (6 per hand)

Create the file by:
1. Taking the original `g1_29dof_with_hand.xml`
2. Removing all `left_hand_*` and `right_hand_*` body/joint/actuator/sensor entries
3. Adding Inspire hand bodies attached to `left_wrist_yaw_link` and `right_wrist_yaw_link`
4. The Inspire base link attaches at the same position as the Dex3 palm

The mesh file paths in the XML should be: `inspire_hand/meshes/Link11_L.STL` etc.

Joint naming convention for sim compatibility:
- `left_hand_thumb_yaw_joint` (L_thumb_proximal_yaw_joint)
- `left_hand_thumb_pitch_joint` (L_thumb_proximal_pitch_joint)
- `left_hand_index_joint` (L_index_proximal_joint)
- `left_hand_middle_joint` (L_middle_proximal_joint)
- `left_hand_ring_joint` (L_ring_proximal_joint)
- `left_hand_pinky_joint` (L_pinky_proximal_joint)
- (Same pattern for right_hand_*)

Mimic joints (not actuated, driven by equality constraints):
- `left_hand_thumb_intermediate_joint` mimics `left_hand_thumb_pitch_joint` × 1.6
- `left_hand_thumb_distal_joint` mimics `left_hand_thumb_pitch_joint` × 2.4
- `left_hand_index_intermediate_joint` mimics `left_hand_index_joint` × 1.0
- `left_hand_middle_intermediate_joint` mimics `left_hand_middle_joint` × 1.0
- `left_hand_ring_intermediate_joint` mimics `left_hand_ring_joint` × 1.0
- `left_hand_pinky_intermediate_joint` mimics `left_hand_pinky_joint` × 1.0

- [ ] **Step 2: Create scene file**

Create `scene_inspire.xml`:
```xml
<mujoco model="g1_inspire_hand scene">
  <include file="g1_29dof_with_inspire_hand.xml"/>

  <statistic center="0 0 0.5" extent="2.0"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <site name="com_marker" pos="0.1 0 0" size="0.05" rgba="1 0 0 1" type="sphere"/>
  </worldbody>
  <default>
    <geom friction="1.0"/>
  </default>
</mujoco>
```

- [ ] **Step 3: Verify MuJoCo can load the model**

```bash
cd /home/g1/hsj/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('gear_sonic/data/robot_model/model_data/g1/scene_inspire.xml')
print(f'nq={m.nq}, nv={m.nv}, nu={m.nu}')
for i in range(m.njnt):
    print(f'  joint {i}: {m.joint(i).name}')
"
```

Expected: 29 body joints + 6+6 actuated hand joints + 6+6 mimic hand joints = ~53 joints total, with 29+12=41 actuators.

- [ ] **Step 4: Commit**

```bash
git add gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_inspire_hand.xml
git add gear_sonic/data/robot_model/model_data/g1/scene_inspire.xml
git commit -m "feat: add MuJoCo model for G1 with Inspire dexterous hands"
```

---

### Task 5: Create WBC Config for Inspire Hand Sim

**Files:**
- Create: `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire.yaml`

- [ ] **Step 1: Create config**

Copy `g1_29dof_sonic_model12.yaml` and modify:

```yaml
# Key changes from dex3 config:
ROBOT_SCENE: "gear_sonic/data/robot_model/model_data/g1/scene_inspire.xml"
NUM_HAND_MOTORS: 6
NUM_HAND_JOINTS: 6

# Hand DDS topics (used by unitree_sdk2py_bridge)
HAND_TYPE: "inspire"  # "dex3" or "inspire"
```

All other values (JOINT_KP, MOTOR_KP, body config, etc.) stay the same since we're only changing the hand model, not the body.

- [ ] **Step 2: Commit**

```bash
git add gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire.yaml
git commit -m "feat: add WBC config for Inspire hand simulation"
```

---

### Task 6: Update base_sim.py for Variable Hand DOF

**Files:**
- Modify: `gear_sonic/utils/mujoco_sim/base_sim.py`

The current code (line 237-238) asserts exactly `NUM_HAND_JOINTS` hand joints. With Inspire hands, MuJoCo will detect both actuated AND mimic joints as `left_hand_*` joints. We need to filter to only count actuated joints (those that have motors).

- [ ] **Step 1: Update joint detection to handle mimic joints**

In `base_sim.py`, around line 220-239, the hand joint detection currently collects ALL joints with "left_hand" in the name. For Inspire, this would include mimic joints. Change to only count joints that have a corresponding actuator:

```python
# Build set of actuated joint names
actuated_joints = set()
for i in range(self.mj_model.nu):
    jnt_id = self.mj_model.actuator(i).trnid[0]
    if jnt_id >= 0:
        actuated_joints.add(self.mj_model.joint(jnt_id).name)

self.body_joint_index = []
self.left_hand_index = []
self.right_hand_index = []
for i in range(self.mj_model.njnt):
    name = self.mj_model.joint(i).name
    if any(
        part_name in name
        for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
    ):
        self.body_joint_index.append(i)
    elif "left_hand" in name and name in actuated_joints:
        self.left_hand_index.append(i)
    elif "right_hand" in name and name in actuated_joints:
        self.right_hand_index.append(i)
```

- [ ] **Step 2: Verify the sim loads with both configs**

```bash
# Test with dex3 config (should still work)
python -c "
from gear_sonic.utils.mujoco_sim.configs import SimConfig
# ... verify dex3 still works
"

# Test with inspire config
python -c "
from gear_sonic.utils.mujoco_sim.configs import SimConfig
# ... verify inspire works
"
```

- [ ] **Step 3: Commit**

```bash
git add gear_sonic/utils/mujoco_sim/base_sim.py
git commit -m "fix: handle variable hand DOF in sim (support both Dex3 and Inspire)"
```

---

### Task 7: Update DDS Bridge for Inspire Hand Topics

**Files:**
- Modify: `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`

The current bridge subscribes to `rt/dex3/left/cmd` and publishes `rt/dex3/left/state`. For Inspire hands, we need to subscribe to `rt/inspire/cmd` and publish `rt/inspire/state` (DFX mode).

- [ ] **Step 1: Add hand_type parameter to bridge**

Add `hand_type` to the bridge constructor and conditionally subscribe to the correct DDS topics.

For Inspire DFX mode:
- Subscribe: `rt/inspire/cmd` (MotorCmds_, 12 motors)
- Publish: `rt/inspire/state` (MotorStates_, 12 motors)

The handler maps motors 0-5 to right hand and 6-11 to left hand.

- [ ] **Step 2: Update state publishing**

For Inspire, publish hand state to `rt/inspire/state` instead of `rt/dex3/*/state`.

- [ ] **Step 3: Commit**

```bash
git add gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py
git commit -m "feat: add Inspire DDS topic support to sim bridge"
```

---

### Task 8: Integration Test — Sim with Inspire Hands

- [ ] **Step 1: Launch MuJoCo sim with Inspire config**

```bash
cd /home/g1/hsj/GR00T-WholeBodyControl
source .venv_sim/bin/activate
# Run sim with Inspire config (need to add config selection arg or modify configs.py)
python gear_sonic/scripts/run_sim_loop.py --wbc-config gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12_inspire.yaml
```

Verify:
- MuJoCo window opens with G1 robot + Inspire 5-finger hands
- No assertion errors about hand joint count
- Robot stands normally (body control unaffected)

- [ ] **Step 2: Test keyboard deploy with Inspire sim**

```bash
# Terminal 2:
cd gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim --input-type keyboard
```

Verify body control works normally. Hand joints won't move via keyboard (that's expected — Inspire hands are controlled via separate DDS).

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration fixes for Inspire hand simulation"
```
