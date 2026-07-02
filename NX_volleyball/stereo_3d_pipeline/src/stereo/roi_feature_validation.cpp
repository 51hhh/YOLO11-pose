#include "roi_feature_validation.h"

#include "roi_feature_match_common.h"

#include <algorithm>
#include <cmath>

namespace stereo3d {

bool pointInsideDetectionEllipseForFeature(
    const Detection& det,
    float x,
    float y,
    float scale) {
    return pointInsideDetectionEllipse(det, x, y, scale);
}

bool validateSparseFeatureGeometry(
    const SparseFeatureDisparityResult& result,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    float focal,
    float baseline) {
    if (!result.valid || result.disparity <= 0.5f ||
        initial_disp <= 0.5f || focal <= 1e-3f || baseline <= 1e-6f) {
        return false;
    }
    const int min_support = std::max(1, cfg.subpixel_min_points);
    if (result.support < min_support) {
        return false;
    }
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    if (result.stddev > max_stddev) {
        return false;
    }

    const float left_x = result.anchor_cx;
    const float left_y = result.anchor_cy;
    const float expected_y = expectedFeatureYDelta(left_x, left_det, cfg);
    const bool has_right_anchor =
        std::isfinite(result.right_anchor_cx) &&
        std::isfinite(result.right_anchor_cy);
    RobustMatchSample sample;
    sample.left_x = left_x;
    sample.left_y = left_y;
    sample.right_x = has_right_anchor
        ? result.right_anchor_cx
        : left_x - result.disparity;
    sample.right_y = has_right_anchor
        ? result.right_anchor_cy
        : left_y - expected_y;
    sample.disparity = result.disparity;
    sample.score = result.confidence;

    return std::abs(featureYResidual(sample, left_det, cfg)) <=
               strictFeatureYTolerance(cfg) &&
           passesFeatureOverlapGate(sample, left_det, right_det,
                                    initial_disp, cfg) &&
           passesSphereRadiusGate(sample, left_det, initial_disp,
                                  focal, baseline, cfg);
}

}  // namespace stereo3d
