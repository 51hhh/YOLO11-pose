#include "stereo/depth_candidate_builder.h"

#include <cstdlib>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

}  // namespace

int main() {
    stereo3d::DepthCandidateBuilderInput input;
    input.left_detection.cx = 100.0f;
    input.left_detection.cy = 80.0f;
    input.left_circle.valid = true;
    input.left_circle.cx = 101.0f;
    input.left_circle.cy = 81.0f;

    input.subpixel_valid = true;
    input.subpixel_result.valid = true;
    input.subpixel_result.disparity = 100.0f;
    input.subpixel_result.confidence = 0.9f;
    input.z_subpixel = 10.0f;

    input.circle_candidate_valid = true;
    input.circle_disparity = 90.0f;
    input.circle_confidence = 0.8f;
    input.z_circle_raw = 11.0f;

    const auto built = stereo3d::buildDepthCandidateObservations(input);
    const auto subpixel = stereo3d::selectPreferredDepthOutputCandidate(
        built.candidates, stereo3d::DepthCandidateMethod::ROI_MULTI_POINT);
    check(subpixel.valid, "preferred subpixel candidate must be selected");
    check(subpixel.observation.method ==
              stereo3d::DepthCandidateMethod::ROI_MULTI_POINT,
          "selected candidate must retain the requested method");
    check(subpixel.observation.depth_m == 10.0f,
          "selected subpixel candidate must retain its depth");

    const auto circle = stereo3d::selectPreferredDepthOutputCandidate(
        built.candidates, stereo3d::DepthCandidateMethod::CIRCLE_CENTER);
    check(circle.valid, "preferred circle candidate must be selected");
    check(circle.observation.method ==
              stereo3d::DepthCandidateMethod::CIRCLE_CENTER,
          "selected circle candidate must retain its method");
    check(circle.observation.depth_m == 11.0f,
          "selected circle candidate must retain its depth");

    input.subpixel_valid = false;
    input.z_subpixel = -1.0f;
    const auto fallback_built =
        stereo3d::buildDepthCandidateObservations(input);
    const auto fallback = stereo3d::selectPreferredDepthOutputCandidate(
        fallback_built.candidates,
        stereo3d::DepthCandidateMethod::ROI_MULTI_POINT);
    check(fallback.valid, "missing preferred method must select a fallback");
    check(fallback.observation.method ==
              stereo3d::DepthCandidateMethod::CIRCLE_CENTER,
          "fallback selection must use the legacy circle candidate");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "depth candidate selection tests passed\n";
    return EXIT_SUCCESS;
}
