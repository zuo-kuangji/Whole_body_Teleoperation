import unittest

import numpy as np

from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa


class RotationConversionTest(unittest.TestCase):
    def test_decompose_rotation_aa_handles_zero_rotation(self):
        rotation_aa = np.zeros((1, 3), dtype=np.float64)
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        q_twist, q_swing = decompose_rotation_aa(rotation_aa, axis)

        np.testing.assert_allclose(q_twist, np.array([[1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(q_swing, np.array([[1.0, 0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
