#include "volleyball/algorithms/stereo/triangulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace volleyball::algorithms::stereo {
namespace {

bool nearlyEqual(double left, double right) noexcept {
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <=
           8.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool validCalibration(
    const core::RectifiedStereoCalibration& calibration) noexcept {
    const auto& left = calibration.left;
    const auto& right = calibration.right;
    return std::isfinite(left.fx) && left.fx > 0.0 &&
           std::isfinite(left.fy) && left.fy > 0.0 &&
           std::isfinite(left.cx) && std::isfinite(left.cy) &&
           std::isfinite(right.fx) && right.fx > 0.0 &&
           std::isfinite(right.fy) && right.fy > 0.0 &&
           std::isfinite(right.cx) && std::isfinite(right.cy) &&
           nearlyEqual(left.fx, right.fx) &&
           nearlyEqual(left.fy, right.fy) &&
           nearlyEqual(left.cx, right.cx) &&
           nearlyEqual(left.cy, right.cy) &&
           std::isfinite(calibration.baseline_m) &&
           calibration.baseline_m > 0.0 && !calibration.rig_frame.empty();
}

}  // namespace

bool isValidTriangulationConfig(
    const core::RectifiedStereoCalibration& calibration,
    const TriangulationConfig& config) noexcept {
    return validCalibration(calibration) &&
           std::isfinite(config.center_sigma_px) &&
           config.center_sigma_px >= 0.0 &&
           std::isfinite(config.min_disparity_px) &&
           config.min_disparity_px > 0.0;
}

TriangulationResult triangulateRectifiedCenters(
    const core::Detection2D& left,
    const core::Detection2D& right,
    const core::StereoAssociationAttempt& association,
    const core::RectifiedStereoCalibration& calibration,
    const TriangulationConfig& config,
    core::ObservationId observation_id) {
    if (!core::validate(left) || !core::validate(right)) {
        return {std::nullopt, "left/right detection is invalid"};
    }
    if (!core::validate(association)) {
        return {std::nullopt, "association is invalid"};
    }
    if (association.decision != core::AssociationDecision::accepted) {
        return {std::nullopt, "association is not accepted"};
    }
    if (left.frame.role != core::CameraRole::left ||
        right.frame.role != core::CameraRole::right ||
        left.frame.camera_id == right.frame.camera_id ||
        left.frame.frame_index != association.left_frame_index ||
        right.frame.frame_index != association.right_frame_index ||
        left.candidate_id != association.left_candidate_id ||
        right.candidate_id != association.right_candidate_id) {
        return {
            std::nullopt,
            "association does not reference the supplied left/right candidates",
        };
    }
    if (!isValidTriangulationConfig(calibration, config)) {
        return {std::nullopt, "invalid rectified stereo calibration/config"};
    }
    const double disparity = left.center_px.x - right.center_px.x;
    const double epipolar_error =
        std::abs(left.center_px.y - right.center_px.y);
    if (!association.disparity_px || !association.epipolar_error_px ||
        !nearlyEqual(*association.disparity_px, disparity) ||
        !nearlyEqual(*association.epipolar_error_px, epipolar_error)) {
        return {
            std::nullopt,
            "association geometry disagrees with the supplied detections",
        };
    }
    if (!std::isfinite(disparity) || disparity < config.min_disparity_px) {
        return {std::nullopt, "disparity is too small for triangulation"};
    }

    const double fx = calibration.left.fx;
    const double fy = calibration.left.fy;
    const double baseline = calibration.baseline_m;
    const double z = fx * baseline / disparity;
    const double x = (left.center_px.x - calibration.left.cx) * z / fx;
    const double mean_v = 0.5 * (left.center_px.y + right.center_px.y);
    const double y = (mean_v - calibration.left.cy) * z / fy;

    // 独立同方差像素中心噪声经一阶 Jacobian 传播到 [x,y,z]。
    const double inv_disparity = 1.0 / disparity;
    const double dx_du_left = baseline * inv_disparity - x * inv_disparity;
    const double dx_du_right = x * inv_disparity;
    const double dy_du_left = -y * inv_disparity;
    const double dy_du_right = y * inv_disparity;
    const double dy_dv = z / (2.0 * fy);
    const double dz_du_left = -z * inv_disparity;
    const double dz_du_right = z * inv_disparity;
    const std::array<std::array<double, 4>, 3> jacobian{{
        {{dx_du_left, 0.0, dx_du_right, 0.0}},
        {{dy_du_left, dy_dv, dy_du_right, dy_dv}},
        {{dz_du_left, 0.0, dz_du_right, 0.0}},
    }};
    const double variance_px = config.center_sigma_px * config.center_sigma_px;
    auto covariance = [&](std::size_t row, std::size_t column) {
        double sum = 0.0;
        for (std::size_t index = 0; index < 4; ++index) {
            sum += jacobian[row][index] * jacobian[column][index];
        }
        return variance_px * sum;
    };

    core::TriangulatedObservation observation;
    observation.observation_id = observation_id;
    observation.source_pair_id = association.pair_id;
    observation.capture_reference = {
        association.left_capture.nanoseconds / 2 +
            association.right_capture.nanoseconds / 2 +
            (association.left_capture.nanoseconds % 2 +
             association.right_capture.nanoseconds % 2) / 2,
        association.left_capture.clock_domain,
    };
    observation.coordinate_frame = calibration.rig_frame;
    observation.position_m = {x, y, z};
    observation.covariance_m2 = core::Covariance3D{{
        covariance(0, 0),
        covariance(0, 1),
        covariance(0, 2),
        covariance(1, 1),
        covariance(1, 2),
        covariance(2, 2),
    }};
    observation.confidence = std::min(
        left.score.value_or(0.0), right.score.value_or(0.0));
    observation.reprojection_error_px = epipolar_error;
    if (!core::validate(observation)) {
        return {std::nullopt, "triangulated observation is invalid"};
    }
    return {observation, {}};
}

}  // namespace volleyball::algorithms::stereo
