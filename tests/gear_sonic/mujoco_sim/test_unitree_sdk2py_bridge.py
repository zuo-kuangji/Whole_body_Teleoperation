import unittest
from unittest import mock

import gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge as bridge_module


class _FakePublisher:
    instances = []

    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type
        self.messages = []
        _FakePublisher.instances.append(self)

    def Init(self, *args, **kwargs):
        return None

    def Write(self, msg):
        self.messages.append(msg)


class _FakeSubscriber:
    instances = []

    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type
        self.callback = None
        _FakeSubscriber.instances.append(self)

    def Init(self, callback, queue_size):
        self.callback = callback
        self.queue_size = queue_size


class UnitreeSdk2BridgeTransportTest(unittest.TestCase):
    def setUp(self):
        _FakePublisher.instances.clear()
        _FakeSubscriber.instances.clear()

    def test_inspire_bridge_subscribes_ftp_control_topics(self):
        config = {
            "ROBOT_TYPE": "g1",
            "NUM_MOTORS": 29,
            "NUM_HAND_MOTORS": 6,
            "USE_SENSOR": False,
            "HAND_TYPE": "inspire",
        }

        with mock.patch.object(bridge_module, "ChannelPublisher", _FakePublisher), mock.patch.object(
            bridge_module, "ChannelSubscriber", _FakeSubscriber
        ):
            bridge_module.UnitreeSdk2Bridge(config)

        subscriber_topics = {subscriber.topic for subscriber in _FakeSubscriber.instances}
        self.assertIn("rt/inspire_hand/ctrl/l", subscriber_topics)
        self.assertIn("rt/inspire_hand/ctrl/r", subscriber_topics)
        self.assertNotIn("rt/inspire/cmd", subscriber_topics)


if __name__ == "__main__":
    unittest.main()
