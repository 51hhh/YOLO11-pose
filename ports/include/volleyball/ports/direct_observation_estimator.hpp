#pragma once

#include "volleyball/core/observation.hpp"
#include "volleyball/ports/rectified_frame.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace volleyball::ports {

// 关联使用的检测框中心视差只是门控证据。这里单独保存真正用于直接 3D
// 观测的深度方法及其测量证据，避免 ROI 精化结果反向改写原始关联事实。
struct DirectObservationEvidence {
    volleyball::core::StereoDepthMethodId method;
    std::optional<double> disparity_px;
    std::optional<double> disparity_stddev_px;
    std::optional<std::uint32_t> support_points;
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
    std::optional<volleyball::core::TriangulatedObservation> observation;
    DirectObservationEvidence evidence;
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
