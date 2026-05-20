#pragma once

#include <Eigen/Core>

namespace stereo3d {

struct HybridDepthResult {
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f vel = Eigen::Vector3f::Zero();
    Eigen::Vector3f acc = Eigen::Vector3f::Zero();
    float confidence = 0.f;
    int   method     = 0;  // 0=mono, 1=stereo, 2=blend
    float z_mono     = 0.f;
    float z_stereo   = 0.f;
    bool  valid      = false;
};

} // namespace stereo3d
