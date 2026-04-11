import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.utils.teleop.vis.raw_hand_visualizer import (
    XRT_HAND_BONE_EDGES,
    extract_xrt_hand_joint_positions,
    extract_xrt_hand_joint_positions_wrist_local,
)


class RawHandVisualizerTest(unittest.TestCase):
    def test_extract_xrt_hand_joint_positions_uses_first_three_columns(self):
        state = np.zeros((26, 7), dtype=np.float64)
        positions = np.arange(26 * 3, dtype=np.float64).reshape(26, 3)
        state[:, :3] = positions
        state[:, 3:] = 123.0

        extracted = extract_xrt_hand_joint_positions(state)

        np.testing.assert_allclose(extracted, positions)

    def test_extract_xrt_hand_joint_positions_validates_shape(self):
        with self.assertRaises(ValueError):
            extract_xrt_hand_joint_positions(np.zeros((25, 7), dtype=np.float64))

        with self.assertRaises(ValueError):
            extract_xrt_hand_joint_positions(np.zeros((26, 6), dtype=np.float64))

    def test_extract_xrt_hand_joint_positions_wrist_local_centers_wrist(self):
        state = np.zeros((26, 7), dtype=np.float64)
        positions = np.zeros((26, 3), dtype=np.float64)
        positions[1] = [0.4, -0.3, 0.2]  # wrist
        positions[5] = [0.5, -0.28, 0.18]  # thumb tip
        positions[10] = [0.48, -0.24, 0.15]  # index tip
        state[:, :3] = positions
        state[1, 3:7] = [0.0, 0.0, 0.0, 1.0]

        local = extract_xrt_hand_joint_positions_wrist_local(state)

        np.testing.assert_allclose(local[1], np.zeros(3), atol=1e-8)
        np.testing.assert_allclose(local[5], positions[5] - positions[1], atol=1e-8)
        np.testing.assert_allclose(local[10], positions[10] - positions[1], atol=1e-8)

    def test_extract_xrt_hand_joint_positions_wrist_local_is_invariant_to_rigid_motion(self):
        base = np.zeros((26, 7), dtype=np.float64)
        base_positions = np.zeros((26, 3), dtype=np.float64)
        base_positions[1] = [0.0, 0.0, 0.0]  # wrist
        base_positions[5] = [0.08, 0.01, -0.02]
        base_positions[10] = [0.09, 0.04, -0.01]
        base_positions[15] = [0.07, 0.05, 0.0]
        base[:, :3] = base_positions
        base[1, 3:7] = [0.0, 0.0, 0.0, 1.0]

        world_rot = R.from_euler("xyz", [30.0, -20.0, 45.0], degrees=True)
        translated = np.zeros((26, 7), dtype=np.float64)
        translated[:, :3] = world_rot.apply(base_positions) + np.array([0.6, -0.4, 0.3])
        translated[1, 3:7] = world_rot.as_quat()

        base_local = extract_xrt_hand_joint_positions_wrist_local(base)
        transformed_local = extract_xrt_hand_joint_positions_wrist_local(translated)

        np.testing.assert_allclose(transformed_local, base_local, atol=1e-8)

    def test_xrt_hand_bone_edges_cover_palm_wrist_and_finger_chains(self):
        expected_edges = {
            (0, 1),
            (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
            (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
            (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
        }

        self.assertEqual(set(XRT_HAND_BONE_EDGES), expected_edges)


if __name__ == "__main__":
    unittest.main()
