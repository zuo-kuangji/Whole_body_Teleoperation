import unittest

from gear_sonic.utils.teleop.solver.hand.inspire_hand_controller import (
    get_inspire_hand_transport_mode,
)


class InspireHandTransportTest(unittest.TestCase):
    def test_inspire_hand_transport_is_ftp_for_sim(self):
        self.assertEqual(get_inspire_hand_transport_mode(hand_sim=True), "FTP")

    def test_inspire_hand_transport_is_ftp_for_real(self):
        self.assertEqual(get_inspire_hand_transport_mode(hand_sim=False), "FTP")


if __name__ == "__main__":
    unittest.main()
