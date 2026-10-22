#include "volleyball/algorithms/stereo/association.hpp"
#include "volleyball/algorithms/stereo/triangulation.hpp"

#include "stereo_domain_fixtures.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

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

void testInvalidAssociationConfigurationIsNotANoMatch() {
    const auto inputs = ValidInputs{};
    auto config = support::associationConfig();
    config.min_shifted_iou = std::numeric_limits<double>::quiet_NaN();
    check(!algorithms::isValidAssociationConfig(config),
          "association config validator must reject NaN thresholds");
    const auto batch = algorithms::associateCandidates(
        {inputs.left},
        {inputs.right},
        support::time(1000),
        support::time(1000),
        config,
        0);
    check(!batch.ok(), "invalid association config must fail the batch");
    check(batch.attempts.empty(), "invalid association config must emit no evidence");
}

void testAssociationRejectsExhaustedPairIdRange() {
    const auto inputs = ValidInputs{};
    const auto batch = algorithms::associateCandidates(
        {inputs.left},
        {inputs.right},
        support::time(1000),
        support::time(1000),
        support::associationConfig(),
        std::numeric_limits<core::PairId>::max());
    check(!batch.ok(), "pair ID exhaustion must fail before wraparound");
    check(batch.attempts.empty(), "pair ID exhaustion must emit no wrapped ID");
}

void testTriangulationConfigurationValidation() {
    const auto inputs = ValidInputs{};
    check(
        algorithms::isValidTriangulationConfig(
            inputs.calibration, inputs.config),
        "valid rectified triangulation config must pass");

    auto calibration = inputs.calibration;
    calibration.right.fx += 1.0;
    check(
        !algorithms::isValidTriangulationConfig(calibration, inputs.config),
        "mismatched rectified projections must fail config validation");

    auto config = inputs.config;
    config.center_sigma_px = -1.0;
    check(
        !algorithms::isValidTriangulationConfig(inputs.calibration, config),
        "negative center uncertainty must fail config validation");
}

}  // namespace

int main() {
    testValidInputsTriangulate();
    testAssociationMustReferenceSuppliedCandidates();
    testAssociationGeometryMustMatchCandidates();
    testCalibrationMustDescribeSharedRectifiedProjection();
    testInvalidAssociationConfigurationIsNotANoMatch();
    testAssociationRejectsExhaustedPairIdRange();
    testTriangulationConfigurationValidation();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stereo algorithm tests passed\n";
    return EXIT_SUCCESS;
}
