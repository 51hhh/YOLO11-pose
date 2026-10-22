#pragma once

#include "volleyball/core/calibration.hpp"
#include "volleyball/core/observation.hpp"

#include <optional>
#include <string>

namespace volleyball::algorithms::stereo {

struct TriangulationConfig {
    double center_sigma_px{0.5};
    double min_disparity_px{0.01};
};

struct TriangulationResult {
    std::optional<core::TriangulatedObservation> observation;
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return observation.has_value();
    }
};

// ROI/特征匹配输出的精化视差。left_anchor_px 是该次深度测量在左校正图
// 上的实际锚点；它不必等于检测框中心，disparity_px 也不应反写 2D 关联证据。
struct RefinedDisparityMeasurement {
    core::Vec2 left_anchor_px;
    double disparity_px{0.0};
    double disparity_stddev_px{0.0};
    double anchor_stddev_px{0.0};
    std::optional<double> confidence;
};

[[nodiscard]] bool isValidTriangulationConfig(
    const core::RectifiedStereoCalibration& calibration,
    const TriangulationConfig& config) noexcept;

[[nodiscard]] TriangulationResult triangulateRectifiedCenters(
    const core::Detection2D& left,
    const core::Detection2D& right,
    const core::StereoAssociationAttempt& association,
    const core::RectifiedStereoCalibration& calibration,
    const TriangulationConfig& config,
    core::ObservationId observation_id);

[[nodiscard]] TriangulationResult triangulateRefinedDisparity(
    const core::Detection2D& left,
    const core::Detection2D& right,
    const core::StereoAssociationAttempt& association,
    const RefinedDisparityMeasurement& measurement,
    const core::RectifiedStereoCalibration& calibration,
    const TriangulationConfig& config,
    core::ObservationId observation_id);

}  // namespace volleyball::algorithms::stereo
