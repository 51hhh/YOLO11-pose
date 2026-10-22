#pragma once

#include "volleyball/algorithms/stereo/association.hpp"
#include "volleyball/algorithms/stereo/triangulation.hpp"

#include <cstdint>

namespace volleyball::test_support {

inline core::Timestamp time(std::int64_t value) {
    return {value, core::ClockDomainId{"gpio-clock"}};
}

inline core::Detection2D detection(
    core::CameraRole role,
    const char* camera,
    core::CandidateId id,
    double x,
    double y,
    double score) {
    core::Detection2D output;
    output.frame = {core::CameraId{camera}, role, 7};
    output.candidate_id = id;
    output.center_px = {x, y};
    output.bounding_box = {x - 20.0, y - 20.0, 40.0, 40.0};
    output.score = score;
    output.source_pipeline = core::PipelineId{"fixture-detector"};
    return output;
}

inline algorithms::stereo::AssociationConfig associationConfig() {
    algorithms::stereo::AssociationConfig output;
    output.max_capture_delta_ns = 100;
    output.epipolar_tolerance_px = 2.0;
    output.max_size_ratio = 2.0;
    output.min_shifted_iou = 0.5;
    return output;
}

inline core::RectifiedStereoCalibration calibration() {
    core::RectifiedStereoCalibration output;
    output.left = {1000.0, 1000.0, 720.0, 540.0};
    output.right = output.left;
    output.baseline_m = 0.8;
    output.rig_frame = core::CoordinateFrameId{"stereo-rig"};
    return output;
}

inline algorithms::stereo::TriangulationConfig triangulationConfig() {
    algorithms::stereo::TriangulationConfig output;
    output.center_sigma_px = 0.5;
    return output;
}

}  // namespace volleyball::test_support
