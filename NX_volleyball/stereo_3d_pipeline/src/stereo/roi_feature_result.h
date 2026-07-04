#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_

#include <array>
#include <limits>
#include <vector>

namespace stereo3d {

struct SparseFeatureDebugMatch {
    float left_x = 0.0f;
    float left_y = 0.0f;
    float right_x = 0.0f;
    float right_y = 0.0f;
    float disparity = 0.0f;
    float score = 0.0f;
};

constexpr int kMaxSparseFeatureDebugMatches = 64;
constexpr int kMaxSparseFeatureDebugPatchSide = 32;

struct SparseFeatureDebugPatch {
    bool valid = false;
    bool disparity_is_score = false;
    bool has_confidence = false;
    int width = 0;
    int height = 0;
    float left_x0 = 0.0f;
    float left_y0 = 0.0f;
    float step_x = 1.0f;
    float step_y = 1.0f;
    float disparity_min = std::numeric_limits<float>::quiet_NaN();
    float disparity_max = std::numeric_limits<float>::quiet_NaN();
    float confidence_min = std::numeric_limits<float>::quiet_NaN();
    float confidence_max = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> disparity;
    std::vector<float> confidence;
};

struct SparseFeatureDisparityResult {
    bool valid = false;
    bool low_confidence = false;
    float disparity = 0.0f;
    float confidence = 0.0f;
    float stddev = 0.0f;
    float anchor_cx = 0.0f;
    float anchor_cy = 0.0f;
    float right_anchor_cx = std::numeric_limits<float>::quiet_NaN();
    float right_anchor_cy = std::numeric_limits<float>::quiet_NaN();
    int support = 0;
    int attempted = 0;
    bool unsupported = false;
    int debug_match_count = 0;
    std::array<SparseFeatureDebugMatch, kMaxSparseFeatureDebugMatches> debug_matches{};
    SparseFeatureDebugPatch debug_patch;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_
