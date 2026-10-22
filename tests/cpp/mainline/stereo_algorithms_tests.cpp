#include "volleyball/algorithms/stereo/association.hpp"
#include "volleyball/algorithms/stereo/triangulation.hpp"

#include "stereo_domain_fixtures.hpp"

#include <cstdlib>
#include <iostream>

namespace algorithms = volleyball::algorithms::stereo;
namespace core = volleyball::core;
namespace support = volleyball::test_support;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

struct ValidInputs {
    core::Detection2D left = support::detection(
        core::CameraRole::left, "left", 3, 700.0, 500.0, 0.95);
    core::Detection2D right = support::detection(
        core::CameraRole::right, "right", 4, 650.0, 500.5, 0.90);
    core::StereoAssociationAttempt association;
    core::RectifiedStereoCalibration calibration = support::calibration();
    algorithms::TriangulationConfig config = support::triangulationConfig();

    ValidInputs() {
        auto batch = algorithms::associateCandidates(
            {left},
            {right},
            support::time(1000),
            support::time(1000),
            support::associationConfig(),
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
