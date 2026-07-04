#ifndef STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_
#define STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_

#include <limits>

namespace stereo3d {

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
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ROI_FEATURE_RESULT_H_
