#pragma once

#include "volleyball/algorithms/stereo/triangulation.hpp"
#include "volleyball/ports/direct_observation_estimator.hpp"

namespace volleyball::algorithms::stereo {

// 仅供回放、测试和几何基线使用。实时 profile 必须显式注入与配置一致的
// ROI 深度实现，不能把该中心三角化基线当作 roi_multi_point。
class RectifiedCenterEstimator final
    : public ports::DirectObservationEstimator {
public:
    RectifiedCenterEstimator(
        core::RectifiedStereoCalibration calibration,
        TriangulationConfig config);

    [[nodiscard]] ports::DirectObservationResult estimate(
        const ports::DirectObservationRequest& request) override;

private:
    core::RectifiedStereoCalibration calibration_;
    TriangulationConfig config_;
};

}  // namespace volleyball::algorithms::stereo
