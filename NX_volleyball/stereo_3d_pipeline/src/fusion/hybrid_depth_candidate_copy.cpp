#include "hybrid_depth_candidate_copy.h"

namespace stereo3d {

void copyRoiCandidateFields(const Object3D& roi, Object3D& obj) {
    obj.z_bbox_center = roi.z_bbox_center;
    obj.z_bbox_left_edge = roi.z_bbox_left_edge;
    obj.z_bbox_right_edge = roi.z_bbox_right_edge;
    obj.z_circle_center = roi.z_circle_center;
    obj.z_circle_left_edge = roi.z_circle_left_edge;
    obj.z_circle_right_edge = roi.z_circle_right_edge;
    obj.z_roi_edge_centroid = roi.z_roi_edge_centroid;
    obj.z_roi_radial_center = roi.z_roi_radial_center;
    obj.z_roi_edge_pair_center = roi.z_roi_edge_pair_center;
    obj.z_roi_corner_points = roi.z_roi_corner_points;
    obj.z_roi_texture_points = roi.z_roi_texture_points;
    obj.z_roi_binary_points = roi.z_roi_binary_points;
    obj.z_roi_orb_points = roi.z_roi_orb_points;
    obj.z_roi_brisk_points = roi.z_roi_brisk_points;
    obj.z_roi_akaze_points = roi.z_roi_akaze_points;
    obj.z_roi_sift_points = roi.z_roi_sift_points;
    obj.z_roi_iou_region_color_patch = roi.z_roi_iou_region_color_patch;
    obj.z_roi_patch_iou_color_edge = roi.z_roi_patch_iou_color_edge;
    obj.z_roi_neural_feature = roi.z_roi_neural_feature;
    obj.z_roi_center_patch = roi.z_roi_center_patch;
    obj.z_roi_multi_point = roi.z_roi_multi_point;
    obj.z_yolo_bbox_pair = roi.z_yolo_bbox_pair;
    obj.z_circle = roi.z_circle;
    obj.z_subpixel = roi.z_subpixel;
    obj.z_fallback = roi.z_fallback;
    obj.z_fallback_epipolar = roi.z_fallback_epipolar;
    obj.z_fallback_template = roi.z_fallback_template;
    obj.z_fallback_feature_points = roi.z_fallback_feature_points;
    obj.disparity_bbox_center = roi.disparity_bbox_center;
    obj.disparity_bbox_left_edge = roi.disparity_bbox_left_edge;
    obj.disparity_bbox_right_edge = roi.disparity_bbox_right_edge;
    obj.disparity_circle_center = roi.disparity_circle_center;
    obj.disparity_circle_left_edge = roi.disparity_circle_left_edge;
    obj.disparity_circle_right_edge = roi.disparity_circle_right_edge;
    obj.disparity_roi_edge_centroid = roi.disparity_roi_edge_centroid;
    obj.disparity_roi_radial_center = roi.disparity_roi_radial_center;
    obj.disparity_roi_edge_pair_center = roi.disparity_roi_edge_pair_center;
    obj.disparity_roi_corner_points = roi.disparity_roi_corner_points;
    obj.disparity_roi_texture_points = roi.disparity_roi_texture_points;
    obj.disparity_roi_binary_points = roi.disparity_roi_binary_points;
    obj.disparity_roi_orb_points = roi.disparity_roi_orb_points;
    obj.disparity_roi_brisk_points = roi.disparity_roi_brisk_points;
    obj.disparity_roi_akaze_points = roi.disparity_roi_akaze_points;
    obj.disparity_roi_sift_points = roi.disparity_roi_sift_points;
    obj.disparity_roi_iou_region_color_patch =
        roi.disparity_roi_iou_region_color_patch;
    obj.disparity_roi_patch_iou_color_edge =
        roi.disparity_roi_patch_iou_color_edge;
    obj.disparity_roi_neural_feature =
        roi.disparity_roi_neural_feature;
    obj.disparity_roi_center_patch = roi.disparity_roi_center_patch;
    obj.disparity_roi_multi_point = roi.disparity_roi_multi_point;
    obj.disparity_fallback_epipolar = roi.disparity_fallback_epipolar;
    obj.disparity_fallback_template = roi.disparity_fallback_template;
    obj.disparity_fallback_feature_points = roi.disparity_fallback_feature_points;
    obj.disparity_yolo = roi.disparity_yolo;
    obj.disparity_circle = roi.disparity_circle;
    obj.disparity_subpixel = roi.disparity_subpixel;
    obj.left_bbox_cx = roi.left_bbox_cx;
    obj.left_bbox_cy = roi.left_bbox_cy;
    obj.left_bbox_w = roi.left_bbox_w;
    obj.left_bbox_h = roi.left_bbox_h;
    obj.left_bbox_conf = roi.left_bbox_conf;
    obj.right_bbox_cx = roi.right_bbox_cx;
    obj.right_bbox_cy = roi.right_bbox_cy;
    obj.right_bbox_w = roi.right_bbox_w;
    obj.right_bbox_h = roi.right_bbox_h;
    obj.right_bbox_conf = roi.right_bbox_conf;
    obj.left_circle_cx = roi.left_circle_cx;
    obj.left_circle_cy = roi.left_circle_cy;
    obj.left_circle_r = roi.left_circle_r;
    obj.right_circle_cx = roi.right_circle_cx;
    obj.right_circle_cy = roi.right_circle_cy;
    obj.right_circle_r = roi.right_circle_r;
    obj.left_circle_source = roi.left_circle_source;
    obj.right_circle_source = roi.right_circle_source;
    obj.epipolar_dy = roi.epipolar_dy;
    obj.size_ratio = roi.size_ratio;
    obj.left_circle_conf = roi.left_circle_conf;
    obj.right_circle_conf = roi.right_circle_conf;
    obj.subpixel_valid = roi.subpixel_valid;
    obj.subpixel_attempted = roi.subpixel_attempted;
    obj.subpixel_support = roi.subpixel_support;
    obj.subpixel_std_px = roi.subpixel_std_px;
    obj.subpixel_confidence = roi.subpixel_confidence;
    obj.subpixel_gate_px = roi.subpixel_gate_px;
    obj.roi_corner_points_support = roi.roi_corner_points_support;
    obj.roi_corner_points_std_px = roi.roi_corner_points_std_px;
    obj.roi_corner_points_confidence = roi.roi_corner_points_confidence;
    obj.roi_texture_points_support = roi.roi_texture_points_support;
    obj.roi_texture_points_std_px = roi.roi_texture_points_std_px;
    obj.roi_texture_points_confidence = roi.roi_texture_points_confidence;
    obj.roi_binary_points_support = roi.roi_binary_points_support;
    obj.roi_binary_points_std_px = roi.roi_binary_points_std_px;
    obj.roi_binary_points_confidence = roi.roi_binary_points_confidence;
    obj.roi_orb_points_support = roi.roi_orb_points_support;
    obj.roi_orb_points_std_px = roi.roi_orb_points_std_px;
    obj.roi_orb_points_confidence = roi.roi_orb_points_confidence;
    obj.roi_brisk_points_support = roi.roi_brisk_points_support;
    obj.roi_brisk_points_std_px = roi.roi_brisk_points_std_px;
    obj.roi_brisk_points_confidence = roi.roi_brisk_points_confidence;
    obj.roi_akaze_points_support = roi.roi_akaze_points_support;
    obj.roi_akaze_points_std_px = roi.roi_akaze_points_std_px;
    obj.roi_akaze_points_confidence = roi.roi_akaze_points_confidence;
    obj.roi_sift_points_support = roi.roi_sift_points_support;
    obj.roi_sift_points_std_px = roi.roi_sift_points_std_px;
    obj.roi_sift_points_confidence = roi.roi_sift_points_confidence;
    obj.roi_iou_region_color_patch_support =
        roi.roi_iou_region_color_patch_support;
    obj.roi_iou_region_color_patch_std_px =
        roi.roi_iou_region_color_patch_std_px;
    obj.roi_iou_region_color_patch_confidence =
        roi.roi_iou_region_color_patch_confidence;
    obj.roi_patch_iou_color_edge_support =
        roi.roi_patch_iou_color_edge_support;
    obj.roi_patch_iou_color_edge_std_px =
        roi.roi_patch_iou_color_edge_std_px;
    obj.roi_patch_iou_color_edge_confidence =
        roi.roi_patch_iou_color_edge_confidence;
    obj.roi_neural_feature_support =
        roi.roi_neural_feature_support;
    obj.roi_neural_feature_std_px =
        roi.roi_neural_feature_std_px;
    obj.roi_neural_feature_confidence =
        roi.roi_neural_feature_confidence;
    obj.fallback_feature_points_support = roi.fallback_feature_points_support;
    obj.fallback_feature_points_std_px = roi.fallback_feature_points_std_px;
    obj.fallback_feature_points_confidence = roi.fallback_feature_points_confidence;
    obj.pair_initial_disparity = roi.pair_initial_disparity;
    obj.pair_epipolar_dy = roi.pair_epipolar_dy;
    obj.pair_y_tolerance = roi.pair_y_tolerance;
    obj.pair_size_ratio = roi.pair_size_ratio;
    obj.pair_shifted_iou = roi.pair_shifted_iou;
    obj.pair_score = roi.pair_score;
    obj.pair_bbox_prior_penalty = roi.pair_bbox_prior_penalty;
    obj.pair_positive_disparity = roi.pair_positive_disparity;
    obj.stereo_match_source = roi.stereo_match_source;
    obj.stereo_depth_source = roi.stereo_depth_source;
}

}  // namespace stereo3d
