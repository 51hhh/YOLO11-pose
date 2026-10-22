#include "stereo/depth_match_contract.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

stereo3d::Detection detection(
    float cx,
    float cy,
    float width,
    float height,
    float confidence,
    int class_id) {
    stereo3d::Detection result;
    result.cx = cx;
    result.cy = cy;
    result.width = width;
    result.height = height;
    result.confidence = confidence;
    result.class_id = class_id;
    return result;
}

void testEveryCandidateCombinationIsRetained() {
    const std::vector<stereo3d::Detection> left{
        detection(100.0f, 50.0f, 20.0f, 20.0f, 0.9f, 0),
        detection(200.0f, 50.0f, 20.0f, 20.0f, 0.8f, 1),
    };
    const std::vector<stereo3d::Detection> right{
        detection(90.0f, 50.5f, 20.0f, 20.0f, 0.7f, 0),
        detection(80.0f, 50.0f, 20.0f, 20.0f, 0.6f, 2),
    };
    stereo3d::StereoRoiPairGateConfig config;
    config.epipolar_y_tolerance = 2.0f;
    config.min_shifted_iou = 0.1f;

    stereo3d::StereoRoiPairStats stats;
    const auto attempts = stereo3d::collectStereoRoiPairAttempts(
        left, right, config, &stats);
    check(attempts.size() == 4, "all left/right combinations must be retained");
    check(attempts[0].passed_gate, "first pair must pass the gate");
    check(
        attempts[0].reject_reason == stereo3d::StereoRoiPairRejectReason::NONE,
        "accepted gate candidate must have no rejection reason");
    check(attempts[0].pair.left_index == 0, "left candidate ID must be stable");
    check(attempts[0].pair.right_index == 0, "right candidate ID must be stable");
    check(attempts[0].pair.initial_disparity == 10.0f, "disparity must be auditable");
    check(stats.class_mismatch == 3, "all class mismatch rejects must be counted");

    const auto candidates = stereo3d::collectStereoRoiPairCandidates(
        left, right, config, 10, nullptr);
    check(candidates.size() == 1, "legacy candidate API must preserve accepted output");
}

void testRejectedAttemptRetainsRawGeometry() {
    const std::vector<stereo3d::Detection> left{
        detection(100.0f, 60.0f, 20.0f, 20.0f, 0.9f, 0),
    };
    const std::vector<stereo3d::Detection> right{
        detection(105.0f, 70.0f, 20.0f, 20.0f, 0.9f, 0),
    };
    stereo3d::StereoRoiPairGateConfig config;
    config.epipolar_y_tolerance = 2.0f;

    const auto attempts = stereo3d::collectStereoRoiPairAttempts(
        left, right, config, nullptr);
    check(attempts.size() == 1, "one combination must yield one attempt");
    check(!attempts[0].passed_gate, "negative disparity must be rejected");
    check(
        attempts[0].reject_reason ==
            stereo3d::StereoRoiPairRejectReason::NONPOSITIVE_DISPARITY,
        "rejection reason must identify nonpositive disparity");
    check(
        attempts[0].pair.initial_disparity == -5.0f,
        "rejected attempt must retain measured disparity");
    check(
        attempts[0].pair.epipolar_dy == 10.0f,
        "rejected attempt must retain epipolar residual");
}

void testRuntimeRejectReasonsHaveStableNames() {
    check(
        std::string(stereo3d::stereoRoiPairRejectReasonName(
            stereo3d::StereoRoiPairRejectReason::NOT_SELECTED)) == "not_selected",
        "not-selected reason must have stable serialization");
    check(
        std::string(stereo3d::stereoRoiPairRejectReasonName(
            stereo3d::StereoRoiPairRejectReason::DEPTH_REJECTED)) == "depth_rejected",
        "depth rejection reason must have stable serialization");
}

stereo3d::StereoRoiPairAttempt triangulatedAttempt(
    float confidence,
    float score,
    float z_m) {
    stereo3d::StereoRoiPairAttempt attempt;
    attempt.passed_gate = true;
    attempt.selected = true;
    attempt.triangulated = true;
    attempt.pair.semantic_confidence = confidence;
    attempt.pair.score = score;
    attempt.x_rig_m = 0.1f;
    attempt.y_rig_m = 0.2f;
    attempt.z_rig_m = z_m;
    return attempt;
}

void testSingleBallSelectionIsDeterministic() {
    std::vector<stereo3d::StereoRoiPairAttempt> attempts{
        triangulatedAttempt(0.90f, 0.10f, 5.0f),
        triangulatedAttempt(0.95f, 0.50f, 4.0f),
        triangulatedAttempt(0.95f, 0.20f, 3.0f),
        triangulatedAttempt(0.95f, 0.20f, 2.0f),
    };
    attempts.push_back(triangulatedAttempt(1.0f, 0.0f, -1.0f));

    const auto selected =
        stereo3d::selectSingleBallObservationPair(attempts);
    check(selected && *selected == 2,
          "single-ball selection must prefer confidence, score, then order");

    attempts[0].triangulated = false;
    attempts[1].triangulated = false;
    attempts[2].triangulated = false;
    attempts[3].triangulated = false;
    check(
        !stereo3d::selectSingleBallObservationPair(attempts),
        "invalid triangulations must not produce a direct observation");
}

}  // namespace

int main() {
    testEveryCandidateCombinationIsRetained();
    testRejectedAttemptRetainsRawGeometry();
    testRuntimeRejectReasonsHaveStableNames();
    testSingleBallSelectionIsDeterministic();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "legacy pair audit tests passed\n";
    return EXIT_SUCCESS;
}
