#ifndef STEREO3D_NX_OBSERVATION_ROS2_CONFIG_H
#define STEREO3D_NX_OBSERVATION_ROS2_CONFIG_H

#include <string>

namespace stereo3d {

// 旧 NX 入口只负责发布带来源证据的直接观测；不包含预测或控制配置。
struct NxObservationRos2Config {
    bool enabled = false;
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
};

}  // namespace stereo3d

#endif  // STEREO3D_NX_OBSERVATION_ROS2_CONFIG_H
