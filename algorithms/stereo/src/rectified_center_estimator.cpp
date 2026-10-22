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
