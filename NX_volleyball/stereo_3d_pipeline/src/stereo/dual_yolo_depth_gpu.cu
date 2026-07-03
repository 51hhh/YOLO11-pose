#include "dual_yolo_depth_gpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxEdges = 1024;
constexpr int kMaxFeaturePoints = 64;
constexpr int kThreadsPerPoint = 4;
constexpr int kMaxParallelFeaturePoints = kThreads / kThreadsPerPoint;

#include "dual_yolo_depth_gpu_device_helpers.cuh"
#include "dual_yolo_depth_gpu_roi_match.cuh"

__global__ void dualYoloDepthCandidatesKernel(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int img_w, int img_h,
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
    int compute_orb_points,
    int compute_brisk_points,
    int compute_akaze_points,
    int compute_sift_points,
    int compute_iou_region_color_patch,
    int compute_patch_iou_color_edge,
    float focal,
    float baseline,
    float min_depth,
    float max_depth) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    __shared__ float edge_x[kMaxEdges];
    __shared__ float edge_y[kMaxEdges];
    __shared__ float edge_w[kMaxEdges];
    __shared__ int edge_n;
    __shared__ float max_grad;
    __shared__ float sample_disp[kMaxFeaturePoints];
    __shared__ float sample_score[kMaxFeaturePoints];
    __shared__ float sample_x[kMaxFeaturePoints];
    __shared__ float sample_y[kMaxFeaturePoints];
    __shared__ float point_x[kMaxFeaturePoints];
    __shared__ float point_y[kMaxFeaturePoints];
    __shared__ float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint];
    __shared__ float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint];
    __shared__ float best_dy_parts[kMaxFeaturePoints][kThreadsPerPoint];
    __shared__ int valid_count;

    stereo3d::DualYoloGpuCandidate* out = &results[pair_idx];
    const stereo3d::DualYoloGpuDetectionPair pair = pairs[pair_idx];
    if (threadIdx.x == 0) {
        out->left_index = pair.left_index;
        out->right_index = pair.right_index;
        clearCircle(&out->left_circle);
        clearCircle(&out->right_circle);
        clearPoint(&out->left_edge_centroid);
        clearPoint(&out->right_edge_centroid);
        clearPoint(&out->left_radial_center);
        clearPoint(&out->right_radial_center);
        clearPoint(&out->left_edge_pair_center);
        clearPoint(&out->right_edge_pair_center);
        clearDisparity(&out->center_patch);
        clearDisparity(&out->multi_point);
        clearDisparity(&out->corner_points);
        clearDisparity(&out->texture_points);
        clearDisparity(&out->binary_points);
        clearDisparity(&out->orb_points);
        clearDisparity(&out->brisk_points);
        clearDisparity(&out->akaze_points);
        clearDisparity(&out->sift_points);
        clearDisparity(&out->iou_region_color_patch);
        clearDisparity(&out->patch_iou_color_edge);
    }
    __syncthreads();

    if (compute_geometry) {
        fitGeometryInBBox(left_img, left_pitch, img_w, img_h, pair.left,
                          circle_max_roi_pixels,
                          edge_x, edge_y, edge_w, &edge_n, &max_grad,
                          &out->left_circle,
                          &out->left_edge_centroid,
                          &out->left_radial_center,
                          &out->left_edge_pair_center);
        fitGeometryInBBox(right_img, right_pitch, img_w, img_h, pair.right,
                          circle_max_roi_pixels,
                          edge_x, edge_y, edge_w, &edge_n, &max_grad,
                          &out->right_circle,
                          &out->right_edge_centroid,
                          &out->right_radial_center,
                          &out->right_edge_pair_center);
        __syncthreads();
    }

    const float left_cx = out->left_circle.valid ? out->left_circle.cx : pair.left.cx;
    const float left_cy = out->left_circle.valid ? out->left_circle.cy : pair.left.cy;
    const float left_r = out->left_circle.valid
        ? out->left_circle.radius
        : fmaxf(3.0f, 0.25f * (pair.left.width + pair.left.height));
    const float right_cx = out->right_circle.valid ? out->right_circle.cx : pair.right.cx;
    const float initial_disp = left_cx - right_cx;
    const float max_delta = disparityDeltaGate(initial_disp, focal, baseline,
                                               max_disp_delta_px,
                                               max_disp_delta_ratio,
                                               max_depth_delta_m);
    const int fast_points = clampInt(max_points, 4, 6);
    const int fast_min_points = clampInt(min_points, 2, fast_points);
    const int fast_search_radius = clampInt(search_radius_px, 2, 3);
    const int color_points = clampInt(max_points * 2, 16, 24);
    const int color_min_points = clampInt(max(min_points, 4), 3, color_points);
    const int color_search_radius = clampInt((search_radius_px + 2) / 3, 2, 3);
    if (initial_disp <= 0.5f ||
        initial_disp > static_cast<float>(max_disparity) ||
        fabsf(left_cy - (out->right_circle.valid ? out->right_circle.cy : pair.right.cy)) >
            fmaxf(1.0f, epipolar_y_tolerance)) {
        return;
    }

    if (compute_center_patch) {
        matchPatchAtPoint(left_img, left_pitch, right_img, right_pitch,
                          img_w, img_h,
                          left_cx, left_cy,
                          initial_disp,
                          patch_radius,
                          search_radius_px,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          &out->center_patch);
    }
    if (compute_multi_point) {
        matchMultiPointPatch(left_img, left_pitch, right_img, right_pitch,
                             img_w, img_h,
                             pair.left, pair.right,
                             left_cx, left_cy, left_r,
                             initial_disp,
                             patch_radius,
                             search_radius_px,
                             max_points,
                             min_points,
                             max_disparity,
                             min_confidence,
                             max_delta,
                             max_stddev_px,
                             focal,
                             baseline,
                             min_depth,
                             max_depth,
                             feature_y_tolerance_px,
                             feature_y_slope,
                             feature_y_offset_px,
                             feature_reverse_check_px,
                             feature_overlap_scale,
                             feature_mad_scale,
                             feature_ransac_gate_px,
                             feature_sphere_radius_m,
                             feature_sphere_radius_scale,
                             feature_sphere_margin_m,
                             sample_disp, sample_score, sample_x, sample_y,
                             point_x, point_y,
                             best_score_parts, best_disp_parts,
                             best_dy_parts, &valid_count,
                             &out->multi_point);
    }
    if (compute_corner_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          0,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->corner_points);
    }
    if (compute_texture_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->texture_points);
    }
    if (compute_binary_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          min(patch_radius, 3),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->binary_points);
    }
    if (compute_orb_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          min(patch_radius, 3),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->orb_points);
    }
    if (compute_brisk_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          2,
                          min(patch_radius, 3),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->brisk_points);
    }
    if (compute_akaze_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->akaze_points);
    }
    if (compute_sift_points) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          nullptr, 0, nullptr, 0,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          1,
                          min(patch_radius, 4),
                          fast_search_radius,
                          fast_points,
                          fast_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->sift_points);
    }
    if (compute_iou_region_color_patch && left_bgr && right_bgr) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          left_bgr, left_bgr_pitch, right_bgr, right_bgr_pitch,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          3,
                          min(patch_radius, 2),
                          color_search_radius,
                          color_points,
                          color_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->iou_region_color_patch);
    }
    if (compute_patch_iou_color_edge && left_bgr && right_bgr) {
        matchSparsePoints(left_img, left_pitch, right_img, right_pitch,
                          left_bgr, left_bgr_pitch, right_bgr, right_bgr_pitch,
                          img_w, img_h,
                          pair.left, pair.right,
                          left_cx, left_cy, left_r,
                          initial_disp,
                          4,
                          min(patch_radius, 2),
                          color_search_radius,
                          color_points,
                          color_min_points,
                          max_disparity,
                          min_confidence,
                          max_delta,
                          max_stddev_px,
                          focal,
                          baseline,
                          min_depth,
                          max_depth,
                          feature_y_tolerance_px,
                          feature_y_slope,
                          feature_y_offset_px,
                          feature_reverse_check_px,
                          feature_overlap_scale,
                          feature_mad_scale,
                          feature_ransac_gate_px,
                          feature_sphere_radius_m,
                          feature_sphere_radius_scale,
                          feature_sphere_margin_m,
                          sample_disp, sample_score, sample_x, sample_y,
                          point_x, point_y,
                          best_score_parts, best_disp_parts,
                          best_dy_parts, &valid_count,
                          &out->patch_iou_color_edge);
    }
}

}  // namespace

extern "C" void launchDualYoloDepthCandidatesGpu(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
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
    int compute_orb_points,
    int compute_brisk_points,
    int compute_akaze_points,
    int compute_sift_points,
    int compute_iou_region_color_patch,
    int compute_patch_iou_color_edge,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    cudaStream_t stream) {
    if (num_pairs <= 0) return;
    const int blocks = min(num_pairs, 256);
    dualYoloDepthCandidatesKernel<<<blocks, kThreads, 0, stream>>>(
        left_img, left_pitch,
        right_img, right_pitch,
        left_bgr, left_bgr_pitch,
        right_bgr, right_bgr_pitch,
        img_width, img_height,
        pairs,
        num_pairs,
        results,
        max_disparity,
        patch_radius,
        search_radius_px,
        max_points,
        min_points,
        circle_max_roi_pixels,
        min_confidence,
        max_disp_delta_px,
        max_disp_delta_ratio,
        max_depth_delta_m,
        max_stddev_px,
        epipolar_y_tolerance,
        feature_y_tolerance_px,
        feature_y_slope,
        feature_y_offset_px,
        feature_reverse_check_px,
        feature_overlap_scale,
        feature_mad_scale,
        feature_ransac_gate_px,
        feature_sphere_radius_m,
        feature_sphere_radius_scale,
        feature_sphere_margin_m,
        compute_geometry,
        compute_center_patch,
        compute_multi_point,
        compute_corner_points,
        compute_texture_points,
        compute_binary_points,
        compute_orb_points,
        compute_brisk_points,
        compute_akaze_points,
        compute_sift_points,
        compute_iou_region_color_patch,
        compute_patch_iou_color_edge,
        focal,
        baseline,
        min_depth,
        max_depth);
}
