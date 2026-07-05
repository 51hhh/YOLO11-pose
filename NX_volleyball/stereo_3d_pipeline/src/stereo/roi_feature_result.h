#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace stereo3d {

enum class SparseFeatureDebugStage : int {
    RAW = 0,
    MATCH = 1,
    GEOMETRY = 2,
    INLIER = 3,
};

enum class SparseFeatureRejectReason : int {
    NONE = 0,
    STATUS = 1,
    NO_MUTUAL = 2,
    LOW_SCORE = 3,
    BAD_DISPARITY = 4,
    DISP_DELTA = 5,
    Y_RESIDUAL = 6,
    OVERLAP = 7,
    SPHERE = 8,
    RATIO = 9,
    REVERSE = 10,
    MAD_OUTLIER = 11,
    FINAL_GEOMETRY = 12,
    LOW_CONFIDENCE = 13,
    SUPPORT = 14,
    STDDEV = 15,
    OTHER = 16,
};

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
constexpr int kMaxSparseFeatureDebugPoints = 256;

struct SparseFeatureDebugPoint {
    float left_x = std::numeric_limits<float>::quiet_NaN();
    float left_y = std::numeric_limits<float>::quiet_NaN();
    float right_x = std::numeric_limits<float>::quiet_NaN();
    float right_y = std::numeric_limits<float>::quiet_NaN();
    float disparity = std::numeric_limits<float>::quiet_NaN();
    float score = std::numeric_limits<float>::quiet_NaN();
    float second_score = std::numeric_limits<float>::quiet_NaN();
    float y_delta = std::numeric_limits<float>::quiet_NaN();
    float y_residual = std::numeric_limits<float>::quiet_NaN();
    float disp_delta = std::numeric_limits<float>::quiet_NaN();
    int stage = static_cast<int>(SparseFeatureDebugStage::RAW);
    int reject_reason = static_cast<int>(SparseFeatureRejectReason::NONE);
};

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
    int debug_point_count = 0;
    std::array<SparseFeatureDebugPoint, kMaxSparseFeatureDebugPoints> debug_points{};
    SparseFeatureDebugPatch debug_patch;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_
