#ifndef OUTPUT_PAYLOAD_UTILS_HPP
#define OUTPUT_PAYLOAD_UTILS_HPP

#include <map>
#include <span>
#include <string>
#include <vector>

#include "../policy_parameters.hpp"
#include "../state_logger.hpp"

inline std::vector<double> reorder_body_values_to_mujoco(
    const std::span<const double>& values,
    bool add_default_offset
) {
    if (values.size() != 29) {
        return std::vector<double>(values.begin(), values.end());
    }

    std::vector<double> reordered(29, 0.0);
    for (size_t i = 0; i < 29; ++i) {
        reordered[i] = values[isaaclab_to_mujoco[i]];
        if (add_default_offset) {
            reordered[i] += default_angles[i];
        }
    }
    return reordered;
}

inline std::vector<double> convert_actions_to_mujoco_targets(const std::span<const double>& actions) {
    if (actions.size() != 29) {
        return std::vector<double>(actions.begin(), actions.end());
    }

    std::vector<double> targets(29, 0.0);
    for (size_t i = 0; i < 29; ++i) {
        targets[i] = actions[isaaclab_to_mujoco[i]] * g1_action_scale[i] + default_angles[i];
    }
    return targets;
}

inline void append_logger_feedback_fields(
    std::map<std::string, std::vector<double>>& output,
    const StateLogger::Entry& state
) {
    output["index"] = {static_cast<double>(state.index)};
    output["ros_timestamp"] = {state.ros_timestamp};
    output["body_dq_measured"] = reorder_body_values_to_mujoco(state.body_dq, false);
    output["last_action"] = convert_actions_to_mujoco_targets(state.last_action);
    output["left_hand_dq_measured"] = state.left_hand_dq;
    output["right_hand_dq_measured"] = state.right_hand_dq;
    output["last_left_hand_action"] = state.last_left_hand_action;
    output["last_right_hand_action"] = state.last_right_hand_action;
}

#endif  // OUTPUT_PAYLOAD_UTILS_HPP
