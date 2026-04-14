import unittest
from enum import Enum, IntEnum

from gear_sonic.utils.teleop.manager_mode_controls import (
    VR3PTKeyboardMotion,
    consume_latest_keyboard_key,
    hand_control_enabled_for_mode,
    resolve_vr3pt_locomotion_mode,
    next_mode_from_keyboard,
)


class _Mode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


class _Loco(IntEnum):
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3


class ManagerModeControlsTest(unittest.TestCase):
    def test_p_toggles_planner(self):
        self.assertEqual(next_mode_from_keyboard(_Mode.OFF, "p", _Mode), _Mode.PLANNER)
        self.assertEqual(next_mode_from_keyboard(_Mode.PLANNER, "p", _Mode), _Mode.OFF)

    def test_o_switches_to_pose_without_toggling(self):
        self.assertEqual(next_mode_from_keyboard(_Mode.OFF, "o", _Mode), _Mode.POSE)
        self.assertEqual(next_mode_from_keyboard(_Mode.PLANNER, "o", _Mode), _Mode.POSE)
        self.assertEqual(next_mode_from_keyboard(_Mode.POSE, "o", _Mode), _Mode.POSE)

    def test_f_switches_to_frozen_upper_body_without_toggling(self):
        self.assertEqual(
            next_mode_from_keyboard(_Mode.OFF, "f", _Mode),
            _Mode.PLANNER_FROZEN_UPPER_BODY,
        )
        self.assertEqual(
            next_mode_from_keyboard(_Mode.POSE, "f", _Mode),
            _Mode.PLANNER_FROZEN_UPPER_BODY,
        )
        self.assertEqual(
            next_mode_from_keyboard(_Mode.PLANNER_FROZEN_UPPER_BODY, "f", _Mode),
            _Mode.PLANNER_FROZEN_UPPER_BODY,
        )

    def test_v_only_toggles_vr3pt_from_pose(self):
        self.assertEqual(
            next_mode_from_keyboard(_Mode.POSE, "v", _Mode),
            _Mode.PLANNER_VR_3PT,
        )
        self.assertEqual(
            next_mode_from_keyboard(_Mode.PLANNER_VR_3PT, "v", _Mode),
            _Mode.POSE,
        )
        self.assertEqual(next_mode_from_keyboard(_Mode.OFF, "v", _Mode), _Mode.OFF)
        self.assertEqual(next_mode_from_keyboard(_Mode.PLANNER, "v", _Mode), _Mode.PLANNER)
        self.assertEqual(
            next_mode_from_keyboard(_Mode.PLANNER_FROZEN_UPPER_BODY, "v", _Mode),
            _Mode.PLANNER_FROZEN_UPPER_BODY,
        )

    def test_unmapped_key_keeps_current_mode(self):
        self.assertEqual(next_mode_from_keyboard(_Mode.POSE, None, _Mode), _Mode.POSE)
        self.assertEqual(next_mode_from_keyboard(_Mode.POSE, "x", _Mode), _Mode.POSE)

    def test_only_pose_mode_enables_hand_control(self):
        self.assertFalse(hand_control_enabled_for_mode(_Mode.OFF, _Mode))
        self.assertTrue(hand_control_enabled_for_mode(_Mode.POSE, _Mode))
        self.assertFalse(hand_control_enabled_for_mode(_Mode.PLANNER, _Mode))
        self.assertFalse(hand_control_enabled_for_mode(_Mode.PLANNER_FROZEN_UPPER_BODY, _Mode))
        self.assertFalse(hand_control_enabled_for_mode(_Mode.PLANNER_VR_3PT, _Mode))

    def test_vr3pt_keyboard_motion_maps_wasdqe(self):
        now = [100.0]

        motion = VR3PTKeyboardMotion(clock=lambda: now[0], hold_seconds=0.2)
        motion.observe_key("w")
        motion.observe_key("a")
        motion.observe_key("q")

        self.assertEqual(motion.virtual_axes(), (-1.0, 1.0, -1.0, 0.0))

        now[0] += 0.05
        motion.observe_key("d")
        motion.observe_key("e")
        self.assertEqual(motion.virtual_axes(), (0.0, 1.0, 0.0, 0.0))

    def test_vr3pt_keyboard_motion_expires_without_repeat(self):
        now = [10.0]

        motion = VR3PTKeyboardMotion(clock=lambda: now[0], hold_seconds=0.1)
        motion.observe_key("s")
        self.assertEqual(motion.virtual_axes(), (0.0, -1.0, 0.0, 0.0))

        now[0] += 0.11
        self.assertEqual(motion.virtual_axes(), (0.0, 0.0, 0.0, 0.0))

    def test_vr3pt_keyboard_motion_ignores_non_movement_keys(self):
        motion = VR3PTKeyboardMotion(clock=lambda: 0.0)
        motion.observe_key("v")
        motion.observe_key("p")
        self.assertEqual(motion.virtual_axes(), (0.0, 0.0, 0.0, 0.0))

    def test_vr3pt_uses_walk_when_keyboard_motion_is_active_from_idle(self):
        self.assertEqual(
            resolve_vr3pt_locomotion_mode(_Loco.IDLE, raw_mag=1.0, locomotion_mode_enum=_Loco),
            _Loco.WALK,
        )

    def test_vr3pt_preserves_non_idle_mode_when_keyboard_motion_is_active(self):
        self.assertEqual(
            resolve_vr3pt_locomotion_mode(_Loco.RUN, raw_mag=1.0, locomotion_mode_enum=_Loco),
            _Loco.RUN,
        )

    def test_vr3pt_preserves_idle_when_no_keyboard_motion(self):
        self.assertEqual(
            resolve_vr3pt_locomotion_mode(_Loco.IDLE, raw_mag=0.0, locomotion_mode_enum=_Loco),
            _Loco.IDLE,
        )

    def test_consume_latest_keyboard_key_prefers_last_movement_key(self):
        self.assertEqual(
            consume_latest_keyboard_key(iter(["w", "w", "w", "s"]).__next__),
            "s",
        )

    def test_consume_latest_keyboard_key_keeps_last_non_movement_when_no_movement(self):
        self.assertEqual(
            consume_latest_keyboard_key(iter(["o", "v"]).__next__),
            "v",
        )


if __name__ == "__main__":
    unittest.main()
