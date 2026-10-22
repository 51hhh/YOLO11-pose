#include "stereo/depth_candidate_builder.h"

#include <cassert>

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
    assert(subpixel.valid);
    assert(subpixel.observation.method ==
           stereo3d::DepthCandidateMethod::ROI_MULTI_POINT);
    assert(subpixel.observation.depth_m == 10.0f);

    const auto circle = stereo3d::selectPreferredDepthOutputCandidate(
        built.candidates, stereo3d::DepthCandidateMethod::CIRCLE_CENTER);
    assert(circle.valid);
    assert(circle.observation.method ==
           stereo3d::DepthCandidateMethod::CIRCLE_CENTER);
    assert(circle.observation.depth_m == 11.0f);

    input.subpixel_valid = false;
    input.z_subpixel = -1.0f;
    const auto fallback_built =
        stereo3d::buildDepthCandidateObservations(input);
    const auto fallback = stereo3d::selectPreferredDepthOutputCandidate(
        fallback_built.candidates,
        stereo3d::DepthCandidateMethod::ROI_MULTI_POINT);
    assert(fallback.valid);
    assert(fallback.observation.method ==
           stereo3d::DepthCandidateMethod::CIRCLE_CENTER);
    return 0;
}
