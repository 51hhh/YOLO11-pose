#include "volleyball/algorithms/stereo/rectified_center_estimator.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace volleyball::algorithms::stereo {

RectifiedCenterEstimator::RectifiedCenterEstimator(
    core::RectifiedStereoCalibration calibration,
    TriangulationConfig config)
    : calibration_(std::move(calibration)), config_(config) {
    if (!isValidTriangulationConfig(calibration_, config_)) {
        throw std::invalid_argument(
            "invalid rectified center estimator calibration/configuration");
    }
}

ports::DirectObservationResult RectifiedCenterEstimator::estimate(
    const ports::DirectObservationRequest& request) {
    ports::DirectObservationEvidence evidence;
    evidence.method = core::StereoDepthMethodId{"detection_center"};
    evidence.disparity_px =
        request.left_detection.center_px.x -
        request.right_detection.center_px.x;
    evidence.disparity_stddev_px =
        std::sqrt(2.0) * config_.center_sigma_px;
    evidence.support_points = 1;
    // 参考三角化的 x 使用左目中心，y 使用左右中心均值；证据必须保存
    // 真正进入 UVD->XYZ 的有效锚点，不能只复制左目检测中心。
    evidence.left_anchor_px = core::Vec2{
        request.left_detection.center_px.x,
        0.5 * (request.left_detection.center_px.y +
               request.right_detection.center_px.y),
    };

    auto result = triangulateRectifiedCenters(
        request.left_detection,
        request.right_detection,
        request.association,
        calibration_,
        config_,
        request.observation_id);
    if (!result) {
        return {
            ports::DirectObservationStatus::rejected,
            std::nullopt,
            evidence.method,
            std::move(result.message),
        };
    }
    return {
        ports::DirectObservationStatus::observation,
        ports::EstimatedDirectObservation{
            std::move(*result.observation),
            std::move(evidence),
        },
        core::StereoDepthMethodId{},
        {},
    };
}

}  // namespace volleyball::algorithms::stereo
