#include "../src/ros/nx_observation_quality.h"

#include <cassert>
#include <cmath>

int main() {
    stereo3d::Object3D circle;
    circle.stereo_depth_source = 1;
    circle.stereo_match_source = 1;
    circle.raw_z = 7.6f;
    circle.confidence = 0.64f;
    circle.pair_score = -0.25f;

    const float fallback_confidence =
        stereo3d::selectedMatchConfidence(circle);
    assert(std::abs(fallback_confidence - 0.64f) < 1e-6f);
    assert(stereo3d::depthSigmaFromObservation(circle, fallback_confidence) < 0.75);

    circle.p0p1_circle_center_trust = 0.81f;
    const float method_confidence =
        stereo3d::selectedMatchConfidence(circle);
    assert(std::abs(method_confidence - 0.81f) < 1e-6f);

    stereo3d::Object3D bbox;
    bbox.stereo_depth_source = 3;
    bbox.confidence = 0.55f;
    bbox.p0p1_bbox_center_trust = 0.72f;
    assert(std::abs(stereo3d::selectedMatchConfidence(bbox) - 0.72f) < 1e-6f);
    return 0;
}
