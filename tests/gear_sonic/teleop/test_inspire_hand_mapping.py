import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.teleop.solver.hand.inspire_hand_mapping import (
    remap_normalized_hand_command,
    xrt_hand_state_to_unitree_hand_positions,
)


class InspireHandMappingTest(unittest.TestCase):
    def test_remap_keeps_upstream_normalized_command_unchanged(self):
        command = np.array([0.122, 0.23, 0.315, 0.378, 1.0, 0.187], dtype=np.float64)
        adjusted = remap_normalized_hand_command(command)

        np.testing.assert_allclose(
            adjusted,
            np.array([0.122, 0.23, 0.315, 0.378, 1.0, 0.187], dtype=np.float64),
        )

    def test_remap_only_boosts_thumb_pitch_closing_amount(self):
        command = np.array([0.122, 0.23, 0.315, 0.378, 0.8, 0.187], dtype=np.float64)
        adjusted = remap_normalized_hand_command(command)

        np.testing.assert_allclose(
            adjusted,
            np.array([0.122, 0.23, 0.315, 0.378, 0.7, 0.187], dtype=np.float64),
            atol=1e-6,
        )

    def test_xrobotoolkit_layout_skips_palm_and_uses_wrist_first_25_joints(self):
        wrist_first_positions = np.zeros((25, 3), dtype=np.float64)
        wrist_first_positions[0] = [0.01, -0.02, 0.03]
        wrist_first_positions[4] = [0.06, -0.02, 0.01]
        wrist_first_positions[9] = [0.08, -0.03, 0.02]
        wrist_first_positions[14] = [0.09, -0.01, 0.015]
        wrist_first_positions[19] = [0.075, 0.02, 0.012]
        wrist_first_positions[24] = [0.055, 0.035, 0.01]

        state = np.zeros((26, 7), dtype=np.float64)
        state[0, :3] = [9.0, 9.0, 9.0]  # Palm should be ignored
        state[1:26, :3] = wrist_first_positions
        state[1, 3:7] = [0.0, 0.0, 0.0, 1.0]

        output = xrt_hand_state_to_unitree_hand_positions(state)
        expected = wrist_first_positions - wrist_first_positions[0:1]
        openxr_to_robot = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        robot_to_unitree_hand = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected = (robot_to_unitree_hand @ (openxr_to_robot @ expected.T)).T

        np.testing.assert_allclose(output, expected, atol=1e-6)

    def test_arm_pose_frame_transform_preserves_local_hand_positions(self):
        local_positions = np.zeros((25, 3), dtype=np.float64)
        local_positions[4] = [0.06, -0.02, 0.01]
        local_positions[9] = [0.08, -0.03, 0.02]
        local_positions[14] = [0.09, -0.01, 0.015]
        local_positions[19] = [0.075, 0.02, 0.012]
        local_positions[24] = [0.055, 0.035, 0.01]

        identity_state = np.zeros((26, 7), dtype=np.float64)
        identity_state[1:26, :3] = local_positions
        identity_state[1, 3:7] = [0.0, 0.0, 0.0, 1.0]
        identity_arm_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        world_rot = R.from_euler("xyz", [25.0, -10.0, 40.0], degrees=True)
        translated_state = np.zeros((26, 7), dtype=np.float64)
        translated_state[1:26, :3] = world_rot.apply(local_positions) + np.array(
            [0.35, -0.18, 0.42], dtype=np.float64
        )
        translated_state[1, 3:7] = world_rot.as_quat()
        translated_arm_pose = np.array(
            [0.35, -0.18, 0.42, *world_rot.as_quat()],
            dtype=np.float64,
        )

        identity_output = xrt_hand_state_to_unitree_hand_positions(
            identity_state,
            arm_pose=identity_arm_pose,
        )
        transformed_output = xrt_hand_state_to_unitree_hand_positions(
            translated_state,
            arm_pose=translated_arm_pose,
        )

        np.testing.assert_allclose(transformed_output, identity_output, atol=1e-6)

    def test_rigid_wrist_rotation_does_not_change_local_hand_positions(self):
        local_positions = np.zeros((25, 3), dtype=np.float64)
        local_positions[4] = [0.06, -0.02, 0.01]
        local_positions[9] = [0.08, -0.03, 0.02]
        local_positions[14] = [0.09, -0.01, 0.015]
        local_positions[19] = [0.075, 0.02, 0.012]
        local_positions[24] = [0.055, 0.035, 0.01]

        identity_state = np.zeros((26, 7), dtype=np.float64)
        identity_state[1:26, :3] = local_positions
        identity_state[1, 3:7] = [0.0, 0.0, 0.0, 1.0]

        world_rot = R.from_euler("xyz", [25.0, -10.0, 40.0], degrees=True)
        translated_state = np.zeros((26, 7), dtype=np.float64)
        translated_state[1:26, :3] = world_rot.apply(local_positions) + np.array(
            [0.35, -0.18, 0.42], dtype=np.float64
        )
        translated_state[1, 3:7] = world_rot.as_quat()

        identity_output = xrt_hand_state_to_unitree_hand_positions(identity_state)
        rotated_output = xrt_hand_state_to_unitree_hand_positions(translated_state)

        np.testing.assert_allclose(rotated_output, identity_output, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
