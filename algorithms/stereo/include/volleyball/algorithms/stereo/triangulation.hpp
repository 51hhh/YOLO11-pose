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

[[nodiscard]] TriangulationResult triangulateRectifiedCenters(
    const core::Detection2D& left,
    const core::Detection2D& right,
    const core::StereoAssociationAttempt& association,
    const core::RectifiedStereoCalibration& calibration,
    const TriangulationConfig& config,
    core::ObservationId observation_id);

}  // namespace volleyball::algorithms::stereo
