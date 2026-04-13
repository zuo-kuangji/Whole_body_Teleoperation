#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "output_interface/output_payload_utils.hpp"

TEST(OutputPayloadUtilsTest, AppendsRecorderFeedbackFieldsFromStateLoggerEntry) {
    StateLogger::Entry entry;
    entry.index = 42;
    entry.ros_timestamp = 1.25;

    entry.body_q.resize(29);
    std::iota(entry.body_q.begin(), entry.body_q.end(), 0.0);

    entry.body_dq.resize(29);
    std::iota(entry.body_dq.begin(), entry.body_dq.end(), 10.0);

    entry.last_action.assign(29, 0.0);
    entry.left_hand_q.assign(7, 1.0);
    entry.left_hand_dq.assign(7, 2.0);
    entry.right_hand_q.assign(7, 3.0);
    entry.right_hand_dq.assign(7, 4.0);
    entry.last_left_hand_action.assign(7, 5.0);
    entry.last_right_hand_action.assign(7, 6.0);

    std::map<std::string, std::vector<double>> output;
    append_logger_feedback_fields(output, entry);

    ASSERT_TRUE(output.contains("index"));
    ASSERT_TRUE(output.contains("ros_timestamp"));
    ASSERT_TRUE(output.contains("body_dq_measured"));
    ASSERT_TRUE(output.contains("last_action"));
    ASSERT_TRUE(output.contains("left_hand_dq_measured"));
    ASSERT_TRUE(output.contains("right_hand_dq_measured"));
    ASSERT_TRUE(output.contains("last_left_hand_action"));
    ASSERT_TRUE(output.contains("last_right_hand_action"));

    EXPECT_EQ(output["index"], std::vector<double>({42.0}));
    EXPECT_EQ(output["ros_timestamp"], std::vector<double>({1.25}));
    EXPECT_EQ(output["body_dq_measured"].size(), 29);
    EXPECT_EQ(output["left_hand_dq_measured"], std::vector<double>(7, 2.0));
    EXPECT_EQ(output["right_hand_dq_measured"], std::vector<double>(7, 4.0));
    EXPECT_EQ(output["last_left_hand_action"], std::vector<double>(7, 5.0));
    EXPECT_EQ(output["last_right_hand_action"], std::vector<double>(7, 6.0));

    std::vector<double> expected_body_dq(29);
    for (size_t i = 0; i < 29; ++i) {
        expected_body_dq[i] = entry.body_dq[isaaclab_to_mujoco[i]];
    }
    EXPECT_EQ(output["body_dq_measured"], expected_body_dq);

    ASSERT_EQ(output["last_action"].size(), 29);
    for (size_t i = 0; i < 29; ++i) {
        EXPECT_DOUBLE_EQ(output["last_action"][i], default_angles[i]);
    }
}
