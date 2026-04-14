import unittest

from gear_sonic.utils.teleop.solver.hand.inspire_hand_controller import (
    get_inspire_hand_transport_mode,
)


class InspireHandTransportTest(unittest.TestCase):
    def test_inspire_hand_transport_defaults_to_ftp_for_sim(self):
        self.assertEqual(get_inspire_hand_transport_mode(hand_sim=True), "FTP")

    def test_inspire_hand_transport_defaults_to_ftp_for_real(self):
        self.assertEqual(get_inspire_hand_transport_mode(hand_sim=False), "FTP")

    def test_inspire_hand_transport_accepts_explicit_dfx_override(self):
        self.assertEqual(
            get_inspire_hand_transport_mode(hand_sim=True, transport="dfx"),
            "DFX",
        )

    def test_inspire_hand_transport_accepts_explicit_ftp_override(self):
        self.assertEqual(
            get_inspire_hand_transport_mode(hand_sim=False, transport="ftp"),
            "FTP",
        )

    def test_inspire_hand_transport_rejects_unknown_override(self):
        with self.assertRaises(ValueError):
            get_inspire_hand_transport_mode(hand_sim=True, transport="weird")


if __name__ == "__main__":
    unittest.main()
