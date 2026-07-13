#include "nx_observation_quality.h"

#include <algorithm>
#include <cmath>

namespace stereo3d {
namespace {

float clamp01(float value) {
    if (!std::isfinite(value)) return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

bool validPositive(float value) {
    return std::isfinite(value) && value > 0.0f;
}

float methodConfidenceOrOverall(const Object3D& obj, float method_confidence) {
    if (validPositive(method_confidence)) return clamp01(method_confidence);
    return clamp01(obj.confidence);
}

}  // namespace

float selectedMatchConfidence(const Object3D& obj) {
    switch (obj.stereo_depth_source) {
    case 1:
        return methodConfidenceOrOverall(obj, obj.p0p1_circle_center_trust);
    case 2:  return clamp01(obj.subpixel_confidence);
    case 3:
        return methodConfidenceOrOverall(obj, obj.p0p1_bbox_center_trust);
    case 4:  return clamp01(obj.p0p1_center_patch_trust);
    case 5:  return clamp01(obj.p0p1_edge_centroid_trust);
    case 6:
        return methodConfidenceOrOverall(obj, obj.p0p1_bbox_center_trust);
    case 8:  return clamp01(obj.p0p1_radial_center_trust);
    case 9:  return clamp01(obj.p0p1_edge_pair_center_trust);
    case 10: return clamp01(obj.roi_corner_points_confidence);
    case 11: return clamp01(obj.roi_texture_points_confidence);
    case 12: return clamp01(obj.fallback_feature_points_confidence);
    case 13: return clamp01(obj.roi_binary_points_confidence);
    case 14: return clamp01(obj.roi_orb_points_confidence);
    case 15: return clamp01(obj.roi_brisk_points_confidence);
    case 16: return clamp01(obj.roi_akaze_points_confidence);
    case 17: return clamp01(obj.roi_sift_points_confidence);
    case 18: return clamp01(obj.roi_iou_region_color_patch_confidence);
    case 19: return clamp01(obj.roi_patch_iou_color_edge_confidence);
    case 20:
        return clamp01(std::max(obj.roi_neural_xfeat_confidence,
                                obj.roi_neural_feature_confidence));
    case 21: return clamp01(obj.roi_cuda_template_match_confidence);
    case 22: return clamp01(obj.roi_cuda_stereo_bm_confidence);
    case 23: return clamp01(obj.roi_cuda_stereo_sgm_confidence);
    case 24: return clamp01(obj.roi_ring_edge_profile_confidence);
    default: break;
    }
    return clamp01(obj.confidence);
}

double depthSigmaFromObservation(const Object3D& obj, float confidence) {
    const double z = std::max(0.1, static_cast<double>(obj.raw_z));
    double sigma = std::max(0.05, 0.025 * z);
    const double conf = std::max(0.05, static_cast<double>(confidence));
    sigma /= std::sqrt(conf);
    if (obj.stereo_match_source != 1) sigma *= 2.0;
    if (obj.stereo_depth_source == 0) sigma *= 3.0;
    return sigma;
}

}  // namespace stereo3d
