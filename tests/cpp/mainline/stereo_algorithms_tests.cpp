#include "volleyball/algorithms/stereo/association.hpp"
#include "volleyball/algorithms/stereo/triangulation.hpp"

#include <cstdlib>
#include <iostream>

namespace algorithms = volleyball::algorithms::stereo;
namespace core = volleyball::core;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

core::Timestamp time(std::int64_t value) {
    return {value, core::ClockDomainId{"gpio-clock"}};
}

core::Detection2D detection(
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

algorithms::AssociationConfig associationConfig() {
    algorithms::AssociationConfig output;
    output.max_capture_delta_ns = 100;
    output.epipolar_tolerance_px = 2.0;
    output.max_size_ratio = 2.0;
    output.min_shifted_iou = 0.5;
    return output;
}

core::RectifiedStereoCalibration calibration() {
    core::RectifiedStereoCalibration output;
    output.left = {1000.0, 1000.0, 720.0, 540.0};
    output.right = output.left;
    output.baseline_m = 0.8;
    output.rig_frame = core::CoordinateFrameId{"stereo-rig"};
    return output;
}

struct ValidInputs {
    core::Detection2D left = detection(
        core::CameraRole::left, "left", 3, 700.0, 500.0, 0.95);
    core::Detection2D right = detection(
        core::CameraRole::right, "right", 4, 650.0, 500.5, 0.90);
    core::StereoAssociationAttempt association;
    core::RectifiedStereoCalibration calibration = ::calibration();
    algorithms::TriangulationConfig config;

    ValidInputs() {
        auto batch = algorithms::associateCandidates(
            {left},
            {right},
            time(1000),
            time(1000),
            associationConfig(),
            10);
        association = batch.attempts.front().attempt;
    }
};

void testValidInputsTriangulate() {
    const ValidInputs inputs;
    const auto result = algorithms::triangulateRectifiedCenters(
        inputs.left,
        inputs.right,
        inputs.association,
        inputs.calibration,
        inputs.config,
        7);
    check(static_cast<bool>(result), "consistent accepted pair must triangulate");
}

void testAssociationMustReferenceSuppliedCandidates() {
    auto inputs = ValidInputs{};
    ++inputs.association.left_candidate_id;
    const auto result = algorithms::triangulateRectifiedCenters(
        inputs.left,
        inputs.right,
        inputs.association,
        inputs.calibration,
        inputs.config,
        7);
    check(!result, "candidate identity mismatch must prevent triangulation");
}

void testAssociationGeometryMustMatchCandidates() {
    auto inputs = ValidInputs{};
    *inputs.association.disparity_px += 1.0;
    const auto result = algorithms::triangulateRectifiedCenters(
        inputs.left,
        inputs.right,
        inputs.association,
        inputs.calibration,
        inputs.config,
        7);
    check(!result, "stale association geometry must prevent triangulation");
}

void testCalibrationMustDescribeSharedRectifiedProjection() {
    auto inputs = ValidInputs{};
    inputs.calibration.right.cx += 1.0;
    const auto result = algorithms::triangulateRectifiedCenters(
        inputs.left,
        inputs.right,
        inputs.association,
        inputs.calibration,
        inputs.config,
        7);
    check(!result, "incompatible rectified intrinsics must be rejected");
}

}  // namespace

int main() {
    testValidInputsTriangulate();
    testAssociationMustReferenceSuppliedCandidates();
    testAssociationGeometryMustMatchCandidates();
    testCalibrationMustDescribeSharedRectifiedProjection();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stereo algorithm tests passed\n";
    return EXIT_SUCCESS;
}
