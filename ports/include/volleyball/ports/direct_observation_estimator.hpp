#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/rectified_frame.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace volleyball::ports {

// 关联使用的检测框中心视差只是门控证据。这里单独保存真正用于直接 3D
// 观测的球心代理方法及其测量输入，避免最终测量反向改写原始关联事实。
struct DirectObservationEvidence {
    volleyball::core::StereoDepthMethodId method;
    double disparity_px{0.0};
    std::optional<double> disparity_stddev_px;
    std::optional<std::uint32_t> support_points;
    // 保存左校正坐标系中实际进入 UVD->XYZ 的有效锚点；它可以是左右
    // 观测融合后的代理量，不允许回放从输出 XYZ 反算来伪造输入证据。
    std::optional<core::Vec2> left_anchor_px;
};

[[nodiscard]] inline bool isValid(
    const DirectObservationEvidence& evidence) noexcept {
    return !evidence.method.empty() &&
           std::isfinite(evidence.disparity_px) &&
           evidence.disparity_px > 0.0 &&
           (!evidence.disparity_stddev_px ||
            (std::isfinite(*evidence.disparity_stddev_px) &&
             *evidence.disparity_stddev_px >= 0.0)) &&
           (!evidence.support_points || *evidence.support_points > 0) &&
           (!evidence.left_anchor_px || core::isFinite(*evidence.left_anchor_px));
}

struct EstimatedDirectObservation {
    volleyball::core::TriangulatedObservation measurement;
    DirectObservationEvidence evidence;
};

struct DirectObservationRequest {
    const RectifiedStereoFrame& rectified_frame;
    const volleyball::core::Detection2D& left_detection;
    const volleyball::core::Detection2D& right_detection;
    const volleyball::core::StereoAssociationAttempt& association;
    volleyball::core::ObservationId observation_id{0};
};

enum class DirectObservationStatus {
    observation,
    rejected,
    failed,
};

struct DirectObservationResult {
    DirectObservationStatus status{DirectObservationStatus::failed};
    std::optional<EstimatedDirectObservation> observation;
    // rejected/failed 没有观测记录，仍需指出发生拒绝或故障的方法。
    volleyball::core::StereoDepthMethodId attempted_method;
    std::string message;
};

// 实现可以使用 Gray8 ROI、CUDA 或纯几何参考算法。rejected 表示一次合法的
// 算法无测量；failed 表示后端/资源故障，调用方必须中止 trial。
class DirectObservationEstimator {
public:
    virtual ~DirectObservationEstimator() = default;
    [[nodiscard]] virtual DirectObservationResult estimate(
        const DirectObservationRequest& request) = 0;
};

}  // namespace volleyball::ports
