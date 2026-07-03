#!/usr/bin/env python3
"""Feature array construction for trajectory fusion datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    from .dataset_io import safe_float
    from .dataset_schema import METHOD_COLUMNS, METHOD_NAMES, LegacySequence
except ImportError:
    from dataset_io import safe_float  # type: ignore
    from dataset_schema import METHOD_COLUMNS, METHOD_NAMES, LegacySequence  # type: ignore


def _metadata_float(metadata: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return safe_float(value, default)
    return default


def _metadata_bool(metadata: Dict[str, Any], keys: Sequence[str], default: bool = False) -> bool:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if value is not None:
            return str(value).strip().lower() in {"1", "true", "yes", "on", "static"}
    return default


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    return _median([abs(value - med) for value in values])


def build_legacy_arrays(sequence: LegacySequence) -> Dict[str, List[List[float]]]:
    """Build feature, measurement and validity arrays from a recorder sequence.

    measurements order: METHOD_NAMES.
    """

    features: List[List[float]] = []
    measurements: List[List[float]] = []
    valid: List[List[float]] = []
    labels: List[List[float]] = []
    prev_ts = None
    prev_valid_ts = None
    prev_median_z = 0.0
    prev_candidate_dz = 0.0
    have_prev_median = False
    metadata = sequence.metadata
    known_z = _metadata_float(metadata, ("known_z_m", "known_z", "known_distance_m"), 0.0)
    known_z_tol = _metadata_float(metadata, ("known_z_tolerance_m", "known_z_tolerance"), 0.0)
    known_z_min = _metadata_float(metadata, ("known_z_min_m", "known_z_min"), 0.0)
    known_z_max = _metadata_float(metadata, ("known_z_max_m", "known_z_max"), 0.0)
    if known_z > 0.0 and known_z_tol > 0.0 and (known_z_min <= 0.0 or known_z_max <= 0.0):
        known_z_min = known_z - known_z_tol
        known_z_max = known_z + known_z_tol
    known_z_valid = 1.0 if known_z > 0.0 else 0.0
    known_z_range_valid = 1.0 if known_z_min > 0.0 and known_z_max > known_z_min else 0.0
    static_flag = 1.0 if _metadata_bool(metadata, ("static", "is_static"), False) else 0.0
    landing_frame = _metadata_float(metadata, ("landing_frame",), -1.0)
    landing_frame_valid = 1.0 if landing_frame >= 0.0 else 0.0

    for row in sequence.rows:
        ts = row["timestamp"]
        if prev_ts is None:
            dt = 0.01
        else:
            dt = max(1e-4, min(0.2, ts - prev_ts))
        prev_ts = ts

        measurements_row = []
        valid_row = []
        for _, key in METHOD_COLUMNS:
            value = row[key]
            is_valid = 1.0 if value > 0.1 else 0.0
            measurements_row.append(value if is_valid else 0.0)
            valid_row.append(is_valid)
        valid_by_key = {
            key: valid_row[idx] for idx, (_, key) in enumerate(METHOD_COLUMNS)
        }
        candidate_values = [value for value, is_valid in zip(measurements_row, valid_row) if is_valid > 0.0]
        candidate_median_z = _median(candidate_values)
        candidate_mad_z = _mad(candidate_values)
        candidate_valid_count = float(len(candidate_values))
        if candidate_valid_count <= 0.0:
            candidate_dz = 0.0
            candidate_ddz = 0.0
        elif have_prev_median and prev_valid_ts is not None:
            raw_valid_dt = ts - prev_valid_ts
            valid_dt = max(1e-4, min(0.5, raw_valid_dt))
            candidate_dz = (candidate_median_z - prev_median_z) / valid_dt
            if raw_valid_dt > dt * 1.5:
                candidate_ddz = 0.0
            else:
                candidate_ddz = (candidate_dz - prev_candidate_dz) / dt
        else:
            candidate_dz = 0.0
            candidate_ddz = 0.0
        if candidate_valid_count > 0.0:
            prev_median_z = candidate_median_z
            prev_candidate_dz = candidate_dz
            prev_valid_ts = ts
            have_prev_median = True

        features.append(
            [
                dt,
                candidate_median_z,
                candidate_mad_z,
                candidate_valid_count,
                candidate_dz,
                candidate_ddz,
                row["z_mono"] if valid_by_key["z_mono"] else 0.0,
                row["z_bbox_center"] if valid_by_key["z_bbox_center"] else 0.0,
                row["z_bbox_left_edge"] if valid_by_key["z_bbox_left_edge"] else 0.0,
                row["z_bbox_right_edge"] if valid_by_key["z_bbox_right_edge"] else 0.0,
                row["z_circle_center"] if valid_by_key["z_circle_center"] else 0.0,
                row["z_circle_left_edge"] if valid_by_key["z_circle_left_edge"] else 0.0,
                row["z_circle_right_edge"] if valid_by_key["z_circle_right_edge"] else 0.0,
                row["z_roi_edge_centroid"] if valid_by_key["z_roi_edge_centroid"] else 0.0,
                row["z_roi_radial_center"] if valid_by_key["z_roi_radial_center"] else 0.0,
                row["z_roi_edge_pair_center"] if valid_by_key["z_roi_edge_pair_center"] else 0.0,
                row["z_roi_corner_points"] if valid_by_key["z_roi_corner_points"] else 0.0,
                row["z_roi_texture_points"] if valid_by_key["z_roi_texture_points"] else 0.0,
                row["z_roi_binary_points"] if valid_by_key["z_roi_binary_points"] else 0.0,
                row["z_roi_orb_points"] if valid_by_key["z_roi_orb_points"] else 0.0,
                row["z_roi_brisk_points"] if valid_by_key["z_roi_brisk_points"] else 0.0,
                row["z_roi_akaze_points"] if valid_by_key["z_roi_akaze_points"] else 0.0,
                row["z_roi_sift_points"] if valid_by_key["z_roi_sift_points"] else 0.0,
                row["z_roi_iou_region_color_patch"] if valid_by_key["z_roi_iou_region_color_patch"] else 0.0,
                row["z_roi_patch_iou_color_edge"] if valid_by_key["z_roi_patch_iou_color_edge"] else 0.0,
                row["z_roi_cuda_template_match"] if valid_by_key["z_roi_cuda_template_match"] else 0.0,
                row["z_roi_cuda_stereo_bm"] if valid_by_key["z_roi_cuda_stereo_bm"] else 0.0,
                row["z_roi_cuda_stereo_sgm"] if valid_by_key["z_roi_cuda_stereo_sgm"] else 0.0,
                row["z_roi_neural_feature"] if valid_by_key["z_roi_neural_feature"] else 0.0,
                row["z_roi_center_patch"] if valid_by_key["z_roi_center_patch"] else 0.0,
                row["z_roi_multi_point"] if valid_by_key["z_roi_multi_point"] else 0.0,
                row["z_fallback"] if valid_by_key["z_fallback"] else 0.0,
                row["z_fallback_epipolar"] if valid_by_key["z_fallback_epipolar"] else 0.0,
                row["z_fallback_template"] if valid_by_key["z_fallback_template"] else 0.0,
                row["z_fallback_feature_points"] if valid_by_key["z_fallback_feature_points"] else 0.0,
                row["confidence"],
                row["class_id"],
                row["disparity_bbox_center"],
                row["disparity_bbox_left_edge"],
                row["disparity_bbox_right_edge"],
                row["disparity_circle_center"],
                row["disparity_circle_left_edge"],
                row["disparity_circle_right_edge"],
                row["disparity_roi_edge_centroid"],
                row["disparity_roi_radial_center"],
                row["disparity_roi_edge_pair_center"],
                row["disparity_roi_corner_points"],
                row["disparity_roi_texture_points"],
                row["disparity_roi_binary_points"],
                row["disparity_roi_orb_points"],
                row["disparity_roi_brisk_points"],
                row["disparity_roi_akaze_points"],
                row["disparity_roi_sift_points"],
                row["disparity_roi_iou_region_color_patch"],
                row["disparity_roi_patch_iou_color_edge"],
                row["disparity_roi_cuda_template_match"],
                row["disparity_roi_cuda_stereo_bm"],
                row["disparity_roi_cuda_stereo_sgm"],
                row["disparity_roi_neural_feature"],
                row["disparity_roi_center_patch"],
                row["disparity_roi_multi_point"],
                row["disparity_fallback_epipolar"],
                row["disparity_fallback_template"],
                row["disparity_fallback_feature_points"],
                row["epipolar_dy"],
                row["size_ratio"],
                row["pair_initial_disparity"],
                row["pair_epipolar_dy"],
                row["pair_y_tolerance"],
                row["pair_size_ratio"],
                row["pair_shifted_iou"],
                row["pair_score"],
                row["pair_bbox_prior_penalty"],
                row["pair_positive_disparity"],
                row["left_circle_conf"],
                row["right_circle_conf"],
                row["subpixel_valid"],
                row["subpixel_attempted"],
                row["subpixel_support"],
                row["subpixel_std_px"],
                row["subpixel_confidence"],
                row["subpixel_gate_px"],
                row["roi_corner_points_support"],
                row["roi_corner_points_std_px"],
                row["roi_corner_points_confidence"],
                row["roi_texture_points_support"],
                row["roi_texture_points_std_px"],
                row["roi_texture_points_confidence"],
                row["roi_binary_points_support"],
                row["roi_binary_points_std_px"],
                row["roi_binary_points_confidence"],
                row["roi_orb_points_support"],
                row["roi_orb_points_std_px"],
                row["roi_orb_points_confidence"],
                row["roi_brisk_points_support"],
                row["roi_brisk_points_std_px"],
                row["roi_brisk_points_confidence"],
                row["roi_akaze_points_support"],
                row["roi_akaze_points_std_px"],
                row["roi_akaze_points_confidence"],
                row["roi_sift_points_support"],
                row["roi_sift_points_std_px"],
                row["roi_sift_points_confidence"],
                row["roi_iou_region_color_patch_support"],
                row["roi_iou_region_color_patch_std_px"],
                row["roi_iou_region_color_patch_confidence"],
                row["roi_patch_iou_color_edge_support"],
                row["roi_patch_iou_color_edge_std_px"],
                row["roi_patch_iou_color_edge_confidence"],
                row["roi_neural_feature_support"],
                row["roi_neural_feature_std_px"],
                row["roi_neural_feature_confidence"],
                row["fallback_feature_points_support"],
                row["fallback_feature_points_std_px"],
                row["fallback_feature_points_confidence"],
                row["raw_observation_valid"],
                row["left_circle_source"],
                row["right_circle_source"],
                row["stereo_match_source"],
                row["frame_counter_delta"],
                row["frame_number_delta"],
                row["timestamp_delta_us"],
                row["left_bbox_cx"],
                row["left_bbox_cy"],
                row["left_bbox_w"],
                row["left_bbox_h"],
                row["left_bbox_conf"],
                row["right_bbox_cx"],
                row["right_bbox_cy"],
                row["right_bbox_w"],
                row["right_bbox_h"],
                row["right_bbox_conf"],
                row["left_circle_cx"],
                row["left_circle_cy"],
                row["left_circle_r"],
                row["right_circle_cx"],
                row["right_circle_cy"],
                row["right_circle_r"],
                *valid_row,
            ]
        )
        measurements.append(measurements_row)
        valid.append(valid_row)
        labels.append(
            [
                known_z,
                known_z_valid,
                known_z_min,
                known_z_max,
                known_z_range_valid,
                static_flag,
                landing_frame,
                landing_frame_valid,
            ]
        )

    return {"features": features, "measurements": measurements, "valid": valid, "labels": labels}
