#ifndef STEREO_3D_PIPELINE_NEURAL_FEATURE_DIRECT_GPU_POSTPROCESS_H_
#define STEREO_3D_PIPELINE_NEURAL_FEATURE_DIRECT_GPU_POSTPROCESS_H_

#include <cuda_runtime.h>

#include <vector>

namespace stereo3d {

struct DirectFeatureGpuMatch {
    float left_x = 0.0f;
    float left_y = 0.0f;
    float right_x = 0.0f;
    float right_y = 0.0f;
    float disparity = -1.0f;
    float score = 0.0f;
};

struct DirectFeatureGpuWorkspace {
    int top_k = 0;
    int desc_dim = 0;

    int* left_best = nullptr;
    int* right_best = nullptr;
    float* left_best_score = nullptr;
    float* right_best_score = nullptr;
    float* left_second_score = nullptr;

    DirectFeatureGpuMatch* matches = nullptr;
    int* match_count = nullptr;

    DirectFeatureGpuMatch* host_matches = nullptr;
    int* host_match_count = nullptr;
};

enum DirectFeatureKeypointLayout {
    DIRECT_KPTS_K2 = 0,
    DIRECT_KPTS_2K = 1,
};

enum DirectFeatureDescriptorLayout {
    DIRECT_DESC_KD = 0,
    DIRECT_DESC_DK = 1,
};

void releaseDirectFeatureGpuWorkspace(DirectFeatureGpuWorkspace& workspace);

bool ensureDirectFeatureGpuWorkspace(DirectFeatureGpuWorkspace& workspace,
                                     int top_k,
                                     int desc_dim);

bool runDirectFeatureGpuPostprocess(
    DirectFeatureGpuWorkspace& workspace,
    const float* left_keypoints,
    const float* right_keypoints,
    const float* left_descriptors,
    const float* right_descriptors,
    const float* left_scores,
    const float* right_scores,
    int keypoint_count,
    int descriptor_count,
    int score_count,
    int desc_dim,
    DirectFeatureKeypointLayout keypoint_layout,
    DirectFeatureDescriptorLayout descriptor_layout,
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
    std::vector<DirectFeatureGpuMatch>* matches);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NEURAL_FEATURE_DIRECT_GPU_POSTPROCESS_H_
