#include "dual_yolo_depth_gpu.h"

#include "../utils/logger.h"

#include <algorithm>
#include <cstring>

extern "C" void launchDualYoloDepthCandidatesGpu(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_width, int img_height,
    const stereo3d::DualYoloGpuDetectionPair* pairs,
    int num_pairs,
    stereo3d::DualYoloGpuCandidate* results,
    int max_disparity,
    int patch_radius,
    int search_radius_px,
    int max_points,
    int min_points,
    int circle_max_roi_pixels,
    float min_confidence,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m,
    float max_stddev_px,
    float epipolar_y_tolerance,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    int compute_geometry,
    int compute_center_patch,
    int compute_multi_point,
    int compute_corner_points,
    int compute_texture_points,
    int compute_binary_points,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    cudaStream_t stream);

namespace stereo3d {

DualYoloDepthGpuMatcher::~DualYoloDepthGpuMatcher() {
    freeBuffers();
}

bool DualYoloDepthGpuMatcher::init(float focal, float baseline, float cx, float cy,
                                   const DualYoloDepthGpuConfig& config,
                                   int max_pairs) {
    freeBuffers();
    focal_ = focal;
    baseline_ = baseline;
    cx_ = cx;
    cy_ = cy;
    config_ = config;
    max_pairs_ = std::clamp(max_pairs, 1, 256);

    cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&pairs_host_),
                                    max_pairs_ * sizeof(DualYoloGpuDetectionPair),
                                    cudaHostAllocDefault);
    if (err != cudaSuccess) {
        LOG_ERROR("DualYoloDepthGpuMatcher: cudaHostAlloc pairs failed: %s",
                  cudaGetErrorString(err));
        freeBuffers();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&pairs_device_),
                     max_pairs_ * sizeof(DualYoloGpuDetectionPair));
    if (err != cudaSuccess) {
        LOG_ERROR("DualYoloDepthGpuMatcher: cudaMalloc pairs failed: %s",
                  cudaGetErrorString(err));
        freeBuffers();
        return false;
    }
    err = cudaHostAlloc(reinterpret_cast<void**>(&results_host_),
                        max_pairs_ * sizeof(DualYoloGpuCandidate),
                        cudaHostAllocDefault);
    if (err != cudaSuccess) {
        LOG_ERROR("DualYoloDepthGpuMatcher: cudaHostAlloc results failed: %s",
                  cudaGetErrorString(err));
        freeBuffers();
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&results_device_),
                     max_pairs_ * sizeof(DualYoloGpuCandidate));
    if (err != cudaSuccess) {
        LOG_ERROR("DualYoloDepthGpuMatcher: cudaMalloc results failed: %s",
                  cudaGetErrorString(err));
        freeBuffers();
        return false;
    }

    ready_ = true;
    LOG_INFO("DualYoloDepthGpuMatcher: maxPairs=%d maxDisp=%d patch=%d search=%d points=%d/%d",
             max_pairs_, config_.max_disparity, config_.patch_radius,
             config_.search_radius_px, config_.min_points, config_.max_points);
    LOG_INFO("  GPU candidate modes: geom=%d centerPatch=%d multi=%d corner=%d texture=%d binary=%d",
             config_.compute_geometry,
             config_.compute_center_patch,
             config_.compute_multi_point,
             config_.compute_corner_points,
             config_.compute_texture_points,
             config_.compute_binary_points);
    return true;
}

void DualYoloDepthGpuMatcher::freeBuffers() {
    ready_ = false;
    if (pairs_host_) {
        cudaFreeHost(pairs_host_);
        pairs_host_ = nullptr;
    }
    if (pairs_device_) {
        cudaFree(pairs_device_);
        pairs_device_ = nullptr;
    }
    if (results_host_) {
        cudaFreeHost(results_host_);
        results_host_ = nullptr;
    }
    if (results_device_) {
        cudaFree(results_device_);
        results_device_ = nullptr;
    }
    max_pairs_ = 0;
}

std::vector<DualYoloGpuCandidate> DualYoloDepthGpuMatcher::matchPairs(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_width, int img_height,
    const std::vector<DualYoloGpuDetectionPair>& pairs,
    cudaStream_t stream) {
    std::vector<DualYoloGpuCandidate> out;
    if (!ready_ || !left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_width <= 0 || img_height <= 0 || pairs.empty()) {
        return out;
    }

    const int n = std::min(static_cast<int>(pairs.size()), max_pairs_);
    out.resize(static_cast<size_t>(n));
    std::memcpy(pairs_host_, pairs.data(),
                static_cast<size_t>(n) * sizeof(DualYoloGpuDetectionPair));
    std::memset(results_host_, 0,
                static_cast<size_t>(n) * sizeof(DualYoloGpuCandidate));

    cudaError_t err = cudaMemcpyAsync(
        pairs_device_, pairs_host_,
        static_cast<size_t>(n) * sizeof(DualYoloGpuDetectionPair),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        LOG_WARN("DualYoloDepthGpuMatcher: H2D pairs failed: %s",
                 cudaGetErrorString(err));
        return {};
    }

    launchDualYoloDepthCandidatesGpu(
        left_gpu, left_pitch, right_gpu, right_pitch,
        img_width, img_height,
        pairs_device_, n, results_device_,
        config_.max_disparity,
        config_.patch_radius,
        config_.search_radius_px,
        config_.max_points,
        config_.min_points,
        config_.circle_max_roi_pixels,
        config_.min_confidence,
        config_.max_disp_delta_px,
        config_.max_disp_delta_ratio,
        config_.max_depth_delta_m,
        config_.max_stddev_px,
        config_.epipolar_y_tolerance,
        config_.feature_y_tolerance_px,
        config_.feature_y_slope,
        config_.feature_y_offset_px,
        config_.feature_reverse_check_px,
        config_.feature_overlap_scale,
        config_.feature_mad_scale,
        config_.feature_ransac_gate_px,
        config_.feature_sphere_radius_m,
        config_.feature_sphere_radius_scale,
        config_.feature_sphere_margin_m,
        config_.compute_geometry ? 1 : 0,
        config_.compute_center_patch ? 1 : 0,
        config_.compute_multi_point ? 1 : 0,
        config_.compute_corner_points ? 1 : 0,
        config_.compute_texture_points ? 1 : 0,
        config_.compute_binary_points ? 1 : 0,
        focal_,
        baseline_,
        config_.min_depth,
        config_.max_depth,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_WARN("DualYoloDepthGpuMatcher: kernel launch failed: %s",
                 cudaGetErrorString(err));
        return {};
    }

    err = cudaMemcpyAsync(
        results_host_, results_device_,
        static_cast<size_t>(n) * sizeof(DualYoloGpuCandidate),
        cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        LOG_WARN("DualYoloDepthGpuMatcher: D2H results failed: %s",
                 cudaGetErrorString(err));
        return {};
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        LOG_WARN("DualYoloDepthGpuMatcher: stream sync failed: %s",
                 cudaGetErrorString(err));
        return {};
    }

    std::memcpy(out.data(), results_host_,
                static_cast<size_t>(n) * sizeof(DualYoloGpuCandidate));
    return out;
}

}  // namespace stereo3d
