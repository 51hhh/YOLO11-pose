#include "fusion/hybrid_depth.h"

#include <cmath>
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

stereo3d::Detection detection(float cx, float confidence, int class_id = 0) {
    stereo3d::Detection det;
    det.cx = cx;
    det.cy = 0.0f;
    det.width = 20.0f;
    det.height = 20.0f;
    det.confidence = confidence;
    det.class_id = class_id;
    return det;
}

stereo3d::Object3D stereoObservation(float z, float confidence = 1.0f) {
    stereo3d::Object3D obj;
    obj.z = z;
    obj.confidence = confidence;
    obj.stereo_match_source = 1;
    obj.stereo_depth_source = 1;
    return obj;
}

stereo3d::HybridDepthConfig config() {
    stereo3d::HybridDepthConfig cfg;
    cfg.min_depth = 0.8f;
    cfg.max_depth = 15.0f;
    cfg.stereo_min_z = 0.8f;
    cfg.mono_max_z = 1.2f;
    cfg.stereo_bias_correction_enabled = false;
    cfg.stereo_bias_initial = 1.0f;
    cfg.match_iou_threshold = 0.1f;
    cfg.match_center_gate = 2.5f;
    cfg.innovation_gate_sigma = 6.0f;
    return cfg;
}

}  // namespace

int main() {
    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto result = estimator.estimate(
            {detection(0.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.01);
        check(result.size() == 1, "valid stereo observation must produce output");
        check(!result.empty() && std::abs(result[0].raw_z - 10.0f) < 1e-4f,
              "valid stereo depth must remain available as raw observation");
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto result = estimator.estimate(
            {detection(0.0f, 0.9f)}, {stereoObservation(15.1f)}, 0.01);
        check(result.empty(), "out-of-range stereo observation must be rejected");
        check(estimator.activeTrackCount() == 0,
              "rejected new observation must not leave an active track");
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto result = estimator.estimate(
            {detection(-100.0f, 0.2f), detection(100.0f, 0.9f)},
            {stereoObservation(5.0f), stereoObservation(10.0f)}, 0.01);
        check(result.size() == 2, "two valid detections must produce two tracks");
        check(std::abs(estimator.predictPrimaryDepth() - 10.0f) < 1e-3f,
              "primary depth must follow the most confident track");
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto first = estimator.estimate(
            {detection(0.0f, 0.9f, 0), detection(100.0f, 0.9f, 1)},
            {stereoObservation(10.0f), stereoObservation(10.0f)}, 0.01);
        check(first.size() == 2, "class association fixture must create two tracks");
        if (first.size() != 2) {
            return EXIT_FAILURE;
        }
        const int class0_track = first[0].track_id;
        const int class1_track = first[1].track_id;

        const auto second = estimator.estimate(
            {detection(30.0f, 0.9f, 0), detection(70.0f, 0.9f, 1)},
            {stereoObservation(10.0f), stereoObservation(10.0f)}, 0.011);
        check(second.size() == 2, "class association must retain both tracks");
        check(second.size() == 2 && second[0].track_id == class0_track,
              "class zero detection must retain its track identity");
        check(second.size() == 2 && second[1].track_id == class1_track,
              "class one detection must retain its track identity");
    }

    {
        auto cfg = config();
        cfg.max_tracks = 2;
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, cfg);
        const auto result = estimator.estimate(
            {detection(-300.0f, 0.6f), detection(0.0f, 0.7f),
             detection(300.0f, 0.8f)},
            {stereoObservation(10.0f), stereoObservation(10.0f),
             stereoObservation(10.0f)},
            0.01);
        check(result.size() == 3,
              "current frame may emit all observations before pruning tracks");
        check(estimator.activeTrackCount() == 2,
              "active track storage must obey its configured hard limit");
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto first = estimator.estimate(
            {detection(0.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.01);
        check(first.size() == 1, "innovation fixture must initialize one track");
        if (first.empty()) {
            return EXIT_FAILURE;
        }
        const int track_id = first[0].track_id;

        const auto fast = estimator.estimate(
            {detection(30.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.011);
        check(fast.size() == 1, "bounded lateral motion must remain accepted");
        check(!fast.empty() && fast[0].track_id == track_id,
              "bounded motion must retain track identity");

        const auto fast_again = estimator.estimate(
            {detection(60.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.011);
        check(fast_again.size() == 1,
              "second bounded motion update must remain accepted");
        check(!fast_again.empty() && fast_again[0].track_id == track_id,
              "second motion update must retain track identity");

        const auto depth_outlier = estimator.estimate(
            {detection(90.0f, 0.9f)}, {stereoObservation(1.0f)}, 0.011);
        check(depth_outlier.empty(), "depth innovation outlier must be rejected");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "hybrid depth tests passed\n";
    return EXIT_SUCCESS;
}
