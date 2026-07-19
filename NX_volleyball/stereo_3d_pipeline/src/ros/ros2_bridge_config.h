#ifndef STEREO3D_ROS2_BRIDGE_CONFIG_H
#define STEREO3D_ROS2_BRIDGE_CONFIG_H

#include <string>

namespace stereo3d {

struct Ros2BridgeConfig {
    bool enabled = false;
    std::string world_frame_id;
    std::string base_frame_id;
    std::string odom_topic;
    double odom_timeout_sec = 0.5;
    std::string topic_realtime = "/nx/debug/ball/realtime";
    std::string topic_landing = "/nx/debug/ball/landing";
    std::string topic_predicted_path = "/nx/debug/ball/predicted_path";
    std::string topic_actual_path = "/nx/debug/ball/actual_path";
    std::string topic_realtime_base = "/nx/debug/ball/realtime_base";
    std::string topic_landing_base = "/nx/debug/ball/landing_base";
    bool nx_observation_enabled = true;
    std::string nx_observation_topic = "/nx/ball/observation";
    std::string nx_observation_frame_id = "nx_left_rectified_optical_frame";
    std::string nx_observation_source_epoch_file =
        "/run/volleyball/nx_source_epoch";
    int nx_observation_class_id = 0;
    double nx_observation_min_depth_m = 0.8;
    double nx_observation_max_depth_m = 15.0;
    double nx_observation_max_speed_mps = 35.0;
    double nx_observation_reacquire_timeout_s = 3.0;
    double nx_observation_reacquire_base_gate_m = 0.75;
    bool nx_observation_allow_fallback = false;
    int nx_observation_timestamp_warmup_frames = 30;
    int nx_observation_timestamp_window_frames = 180;
    double nx_observation_timestamp_offset_us = 0.0;
    double nx_observation_max_timestamp_uncertainty_s = 0.002;
    bool swap_xy = false;
    bool invert_x = false;
    bool invert_y = false;
    double rotation_deg = 0.0;
    double translation_x = 0.0;
    double translation_y = 0.0;
    bool control_goal_enabled = false;
    std::string control_goal_topic = "/nx/debug/auto_goal_pose";
    double control_min_depth_m = 0.0;
    double control_max_depth_m = 14.0;
    double control_max_abs_x_m = 3.6;
    double control_min_confidence = 0.70;
    double control_min_time_to_land_s = 0.25;
    double control_max_time_to_land_s = 2.20;
    double control_min_speed_mps = 0.80;
    double control_min_student_w = 0.15;
    int control_stable_frames = 3;
    double control_max_stable_jump_m = 0.35;
    bool control_allow_polynomial = false;
    bool control_allow_fallback_observation = false;
};

}  // namespace stereo3d

#endif  // STEREO3D_ROS2_BRIDGE_CONFIG_H
