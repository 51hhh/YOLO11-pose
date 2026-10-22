#include "fusion/hybrid_depth.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

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
        assert(result.size() == 1);
        assert(std::abs(result[0].raw_z - 10.0f) < 1e-4f);
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto result = estimator.estimate(
            {detection(0.0f, 0.9f)}, {stereoObservation(15.1f)}, 0.01);
        assert(result.empty());
        assert(estimator.activeTrackCount() == 0);
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto result = estimator.estimate(
            {detection(-100.0f, 0.2f), detection(100.0f, 0.9f)},
            {stereoObservation(5.0f), stereoObservation(10.0f)}, 0.01);
        assert(result.size() == 2);
        assert(std::abs(estimator.predictPrimaryDepth() - 10.0f) < 1e-3f);
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto first = estimator.estimate(
            {detection(0.0f, 0.9f, 0), detection(100.0f, 0.9f, 1)},
            {stereoObservation(10.0f), stereoObservation(10.0f)}, 0.01);
        assert(first.size() == 2);
        const int class0_track = first[0].track_id;
        const int class1_track = first[1].track_id;

        const auto second = estimator.estimate(
            {detection(30.0f, 0.9f, 0), detection(70.0f, 0.9f, 1)},
            {stereoObservation(10.0f), stereoObservation(10.0f)}, 0.011);
        assert(second.size() == 2);
        assert(second[0].track_id == class0_track);
        assert(second[1].track_id == class1_track);
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
        assert(result.size() == 3);
        assert(estimator.activeTrackCount() == 2);
    }

    {
        stereo3d::HybridDepthEstimator estimator;
        estimator.init(1000.0f, 0.853154f, 0.0f, 0.0f, config());
        const auto first = estimator.estimate(
            {detection(0.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.01);
        assert(first.size() == 1);
        const int track_id = first[0].track_id;

        const auto fast = estimator.estimate(
            {detection(30.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.011);
        assert(fast.size() == 1);
        assert(fast[0].track_id == track_id);

        const auto fast_again = estimator.estimate(
            {detection(60.0f, 0.9f)}, {stereoObservation(10.0f)}, 0.011);
        assert(fast_again.size() == 1);
        assert(fast_again[0].track_id == track_id);

        const auto depth_outlier = estimator.estimate(
            {detection(90.0f, 0.9f)}, {stereoObservation(1.0f)}, 0.011);
        assert(depth_outlier.empty());
    }

    return 0;
}
