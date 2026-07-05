#ifndef STEREO_3D_PIPELINE_NEURAL_FEATURE_XFEAT_GPU_POSTPROCESS_H_
#define STEREO_3D_PIPELINE_NEURAL_FEATURE_XFEAT_GPU_POSTPROCESS_H_

#include <cuda_runtime.h>

#include <vector>

namespace stereo3d {

struct XFeatGpuCandidate {
    float x = 0.0f;
    float y = 0.0f;
    float feat_x = 0.0f;
    float feat_y = 0.0f;
    float score = -1.0f;
};

struct XFeatGpuFeature {
    float x = 0.0f;
    float y = 0.0f;
    float score = -1.0f;
};

struct XFeatGpuMatch {
    float left_x = 0.0f;
    float left_y = 0.0f;
    float right_x = 0.0f;
    float right_y = 0.0f;
    float disparity = -1.0f;
    float score = 0.0f;
};

struct XFeatGpuWorkspace {
    int feat_h = 0;
    int feat_w = 0;
    int cells = 0;
    int desc_dim = 0;
    int top_k = 0;

    float* left_feats = nullptr;
    float* left_keypoints = nullptr;
    float* left_heatmap = nullptr;

    XFeatGpuCandidate* left_candidates = nullptr;
    XFeatGpuCandidate* right_candidates = nullptr;
    float* left_scores = nullptr;
    float* right_scores = nullptr;
    int* left_indices = nullptr;
    int* right_indices = nullptr;

    XFeatGpuFeature* left_features = nullptr;
    XFeatGpuFeature* right_features = nullptr;
    float* left_desc = nullptr;
    float* right_desc = nullptr;

    int* left_best = nullptr;
    int* right_best = nullptr;
    float* left_best_score = nullptr;
    float* right_best_score = nullptr;
    float* left_second_score = nullptr;
    float* right_second_score = nullptr;

    XFeatGpuMatch* matches = nullptr;
    int* match_count = nullptr;

    XFeatGpuMatch* host_matches = nullptr;
    int* host_match_count = nullptr;
};

void releaseXFeatGpuWorkspace(XFeatGpuWorkspace& workspace);

bool ensureXFeatGpuWorkspace(XFeatGpuWorkspace& workspace,
                             int feat_h,
                             int feat_w,
                             int desc_dim,
                             int top_k);

bool runXFeatGpuPostprocess(XFeatGpuWorkspace& workspace,
                            const float* left_feats,
                            const float* left_keypoints,
                            const float* left_heatmap,
                            const float* right_feats,
                            const float* right_keypoints,
                            const float* right_heatmap,
                            int roi_size,
                            int min_matches,
                            float min_score,
                            float match_margin,
                            float max_y_error_px,
                            float max_disp_delta_px,
                            float initial_disparity,
                            int max_disparity,
                            float left_cx,
                            float left_cy,
                            float left_w,
                            float left_h,
                            float right_cx,
                            float right_cy,
                            float right_w,
                            float right_h,
                            cudaStream_t stream,
                            std::vector<XFeatGpuMatch>* matches);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NEURAL_FEATURE_XFEAT_GPU_POSTPROCESS_H_
