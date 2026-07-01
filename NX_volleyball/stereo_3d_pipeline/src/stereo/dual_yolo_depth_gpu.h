#ifndef STEREO_3D_PIPELINE_DUAL_YOLO_DEPTH_GPU_H_
#define STEREO_3D_PIPELINE_DUAL_YOLO_DEPTH_GPU_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace stereo3d {

struct DualYoloGpuDetection {
    float cx;
    float cy;
    float width;
    float height;
    float confidence;
    int class_id;
};

struct DualYoloDepthGpuConfig {
    int max_disparity = 256;
    int patch_radius = 5;
    int search_radius_px = 8;
    int max_points = 12;
    int min_points = 4;
    int circle_max_roi_pixels = 18000;
    float min_confidence = 0.25f;
    float max_disp_delta_px = 2.0f;
    float max_disp_delta_ratio = 0.03f;
    float max_depth_delta_m = 0.5f;
    float max_stddev_px = 1.0f;
    float epipolar_y_tolerance = 12.0f;
    float min_depth = 0.3f;
    float max_depth = 15.0f;
    bool compute_geometry = true;
    bool compute_center_patch = true;
    bool compute_multi_point = true;
    bool compute_corner_points = false;
    bool compute_texture_points = false;
    bool compute_binary_points = false;
};

struct DualYoloGpuDetectionPair {
    DualYoloGpuDetection left;
    DualYoloGpuDetection right;
    int left_index = -1;
    int right_index = -1;
};

struct DualYoloGpuCircle {
    float cx = 0.0f;
    float cy = 0.0f;
    float radius = 0.0f;
    float confidence = 0.0f;
    int source = 0;
    int valid = 0;
};

struct DualYoloGpuPointMeasure {
    float cx = 0.0f;
    float cy = 0.0f;
    float confidence = 0.0f;
    int valid = 0;
};

struct DualYoloGpuDisparity {
    float disparity = -1.0f;
    float confidence = 0.0f;
    float stddev = -1.0f;
    float delta_gate_px = 0.0f;
    float anchor_cx = 0.0f;
    float anchor_cy = 0.0f;
    int support = 0;
    int attempted = 0;
    int low_confidence = 0;
    int valid = 0;
};

struct DualYoloGpuCandidate {
    int left_index = -1;
    int right_index = -1;

    DualYoloGpuCircle left_circle;
    DualYoloGpuCircle right_circle;
    DualYoloGpuPointMeasure left_edge_centroid;
    DualYoloGpuPointMeasure right_edge_centroid;
    DualYoloGpuPointMeasure left_radial_center;
    DualYoloGpuPointMeasure right_radial_center;
    DualYoloGpuPointMeasure left_edge_pair_center;
    DualYoloGpuPointMeasure right_edge_pair_center;

    DualYoloGpuDisparity center_patch;
    DualYoloGpuDisparity multi_point;
    DualYoloGpuDisparity corner_points;
    DualYoloGpuDisparity texture_points;
    DualYoloGpuDisparity binary_points;
};

class DualYoloDepthGpuMatcher {
public:
    DualYoloDepthGpuMatcher() = default;
    ~DualYoloDepthGpuMatcher();

    DualYoloDepthGpuMatcher(const DualYoloDepthGpuMatcher&) = delete;
    DualYoloDepthGpuMatcher& operator=(const DualYoloDepthGpuMatcher&) = delete;

    bool init(float focal, float baseline, float cx, float cy,
              const DualYoloDepthGpuConfig& config,
              int max_pairs = 64);

    bool ready() const { return ready_; }
    int maxPairs() const { return max_pairs_; }

    std::vector<DualYoloGpuCandidate> matchPairs(
        const uint8_t* left_gpu, int left_pitch,
        const uint8_t* right_gpu, int right_pitch,
        int img_width, int img_height,
        const std::vector<DualYoloGpuDetectionPair>& pairs,
        cudaStream_t stream);

private:
    void freeBuffers();

    float focal_ = 0.0f;
    float baseline_ = 0.0f;
    float cx_ = 0.0f;
    float cy_ = 0.0f;
    DualYoloDepthGpuConfig config_;
    int max_pairs_ = 0;
    bool ready_ = false;

    DualYoloGpuDetectionPair* pairs_host_ = nullptr;
    DualYoloGpuDetectionPair* pairs_device_ = nullptr;
    DualYoloGpuCandidate* results_host_ = nullptr;
    DualYoloGpuCandidate* results_device_ = nullptr;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_DUAL_YOLO_DEPTH_GPU_H_
