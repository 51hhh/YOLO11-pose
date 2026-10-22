#include "ros/nx_observation_quality.h"

#include <cmath>
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
    stereo3d::Object3D circle;
    circle.stereo_depth_source = 1;
    circle.stereo_match_source = 1;
    circle.raw_z = 7.6f;
    circle.confidence = 0.64f;
    circle.pair_score = -0.25f;

    const float fallback_confidence =
        stereo3d::selectedMatchConfidence(circle);
    check(std::abs(fallback_confidence - 0.64f) < 1e-6f,
          "missing method trust must fall back to overall confidence");
    check(stereo3d::depthSigmaFromObservation(circle, fallback_confidence) <
              0.75,
          "valid direct circle observation must keep bounded depth sigma");

    circle.p0p1_circle_center_trust = 0.81f;
    const float method_confidence =
        stereo3d::selectedMatchConfidence(circle);
    check(std::abs(method_confidence - 0.81f) < 1e-6f,
          "circle method trust must override overall confidence");

    stereo3d::Object3D bbox;
    bbox.stereo_depth_source = 3;
    bbox.confidence = 0.55f;
    bbox.p0p1_bbox_center_trust = 0.72f;
    check(std::abs(stereo3d::selectedMatchConfidence(bbox) - 0.72f) < 1e-6f,
          "bbox method must use bbox trust");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "NX observation quality tests passed\n";
    return EXIT_SUCCESS;
}
