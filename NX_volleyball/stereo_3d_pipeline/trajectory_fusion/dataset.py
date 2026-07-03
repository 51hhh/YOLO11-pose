#!/usr/bin/env python3
"""Dataset helpers for trajectory fusion experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    from .dataset_io import (
        derive_frame_summary_path,
        find_metadata_for_csv,
        iter_extended_rows,
        read_csv_rows,
        read_metadata,
        safe_float as _safe_float,
        safe_int as _safe_int,
    )
    from .dataset_features import build_legacy_arrays
    from .dataset_normalization import (
        apply_feature_normalizer,
        compute_feature_normalizer,
        normalize_features,
    )
    from .dataset_schema import (
        METHOD_COLUMNS,
        METHOD_NAMES,
        LegacySequence,
        legacy_feature_names,
        weak_label_names,
    )
except ImportError:
    from dataset_io import (  # type: ignore
        derive_frame_summary_path,
        find_metadata_for_csv,
        iter_extended_rows,
        read_csv_rows,
        read_metadata,
        safe_float as _safe_float,
        safe_int as _safe_int,
    )
    from dataset_features import build_legacy_arrays  # type: ignore
    from dataset_normalization import (  # type: ignore
        apply_feature_normalizer,
        compute_feature_normalizer,
        normalize_features,
    )
    from dataset_schema import (  # type: ignore
        METHOD_COLUMNS,
        METHOD_NAMES,
        LegacySequence,
        legacy_feature_names,
        weak_label_names,
    )


def load_legacy_sequences(
    path: str | Path,
    min_track_len: int = 3,
    metadata_path: str | Path | None = None,
) -> List[LegacySequence]:
    """Load current trajectory recorder CSV and group rows by track_id."""

    metadata = read_metadata(metadata_path or find_metadata_for_csv(path))
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in read_csv_rows(path):
        track_id = _safe_int(row.get("track_id"), -1)
        if track_id < 0:
            continue
        parsed = {
            "frame_id": _safe_float(row.get("frame_id")),
            "timestamp": _safe_float(row.get("timestamp")),
            "x": _safe_float(row.get("x")),
            "y": _safe_float(row.get("y")),
            "z": _safe_float(row.get("z")),
            "vx": _safe_float(row.get("vx")),
            "vy": _safe_float(row.get("vy")),
            "vz": _safe_float(row.get("vz")),
            "ax": _safe_float(row.get("ax")),
            "ay": _safe_float(row.get("ay")),
            "az": _safe_float(row.get("az")),
            "z_mono": _safe_float(row.get("z_mono"), -1.0),
            "z_bbox_center": _safe_float(row.get("z_bbox_center", row.get("z_yolo_bbox_pair")), -1.0),
            "z_bbox_left_edge": _safe_float(row.get("z_bbox_left_edge"), -1.0),
            "z_bbox_right_edge": _safe_float(row.get("z_bbox_right_edge"), -1.0),
            "z_circle_center": _safe_float(row.get("z_circle_center", row.get("z_circle")), -1.0),
            "z_circle_left_edge": _safe_float(row.get("z_circle_left_edge"), -1.0),
            "z_circle_right_edge": _safe_float(row.get("z_circle_right_edge"), -1.0),
            "z_roi_edge_centroid": _safe_float(row.get("z_roi_edge_centroid"), -1.0),
            "z_roi_radial_center": _safe_float(row.get("z_roi_radial_center"), -1.0),
            "z_roi_edge_pair_center": _safe_float(row.get("z_roi_edge_pair_center"), -1.0),
            "z_roi_corner_points": _safe_float(row.get("z_roi_corner_points"), -1.0),
            "z_roi_texture_points": _safe_float(row.get("z_roi_texture_points"), -1.0),
            "z_roi_binary_points": _safe_float(row.get("z_roi_binary_points"), -1.0),
            "z_roi_orb_points": _safe_float(row.get("z_roi_orb_points"), -1.0),
            "z_roi_brisk_points": _safe_float(row.get("z_roi_brisk_points"), -1.0),
            "z_roi_akaze_points": _safe_float(row.get("z_roi_akaze_points"), -1.0),
            "z_roi_sift_points": _safe_float(row.get("z_roi_sift_points"), -1.0),
            "z_roi_iou_region_color_patch": _safe_float(row.get("z_roi_iou_region_color_patch"), -1.0),
            "z_roi_patch_iou_color_edge": _safe_float(row.get("z_roi_patch_iou_color_edge"), -1.0),
            "z_roi_cuda_template_match": _safe_float(row.get("z_roi_cuda_template_match"), -1.0),
            "z_roi_cuda_stereo_bm": _safe_float(row.get("z_roi_cuda_stereo_bm"), -1.0),
            "z_roi_cuda_stereo_sgm": _safe_float(row.get("z_roi_cuda_stereo_sgm"), -1.0),
            "z_roi_neural_feature": _safe_float(row.get("z_roi_neural_feature"), -1.0),
            "z_roi_center_patch": _safe_float(row.get("z_roi_center_patch"), -1.0),
            "z_roi_multi_point": _safe_float(row.get("z_roi_multi_point", row.get("z_subpixel")), -1.0),
            "z_yolo_bbox_pair": _safe_float(row.get("z_yolo_bbox_pair"), -1.0),
            "z_circle": _safe_float(row.get("z_circle"), -1.0),
            "z_subpixel": _safe_float(row.get("z_subpixel"), -1.0),
            "z_fallback": _safe_float(row.get("z_fallback"), -1.0),
            "z_fallback_epipolar": _safe_float(row.get("z_fallback_epipolar", row.get("z_fallback")), -1.0),
            "z_fallback_template": _safe_float(row.get("z_fallback_template"), -1.0),
            "z_fallback_feature_points": _safe_float(row.get("z_fallback_feature_points"), -1.0),
            "z_stereo": _safe_float(row.get("z_stereo"), -1.0),
            "z": _safe_float(row.get("z"), -1.0),
            "depth_method": _safe_float(row.get("depth_method")),
            "confidence": _safe_float(row.get("confidence"), 1.0),
            "class_id": _safe_float(row.get("class_id"), 0.0),
            "disparity_bbox_center": _safe_float(row.get("disparity_bbox_center", row.get("disparity_yolo")), -1.0),
            "disparity_bbox_left_edge": _safe_float(row.get("disparity_bbox_left_edge"), -1.0),
            "disparity_bbox_right_edge": _safe_float(row.get("disparity_bbox_right_edge"), -1.0),
            "disparity_circle_center": _safe_float(row.get("disparity_circle_center", row.get("disparity_circle")), -1.0),
            "disparity_circle_left_edge": _safe_float(row.get("disparity_circle_left_edge"), -1.0),
            "disparity_circle_right_edge": _safe_float(row.get("disparity_circle_right_edge"), -1.0),
            "disparity_roi_edge_centroid": _safe_float(row.get("disparity_roi_edge_centroid"), -1.0),
            "disparity_roi_radial_center": _safe_float(row.get("disparity_roi_radial_center"), -1.0),
            "disparity_roi_edge_pair_center": _safe_float(row.get("disparity_roi_edge_pair_center"), -1.0),
            "disparity_roi_corner_points": _safe_float(row.get("disparity_roi_corner_points"), -1.0),
            "disparity_roi_texture_points": _safe_float(row.get("disparity_roi_texture_points"), -1.0),
            "disparity_roi_binary_points": _safe_float(row.get("disparity_roi_binary_points"), -1.0),
            "disparity_roi_orb_points": _safe_float(row.get("disparity_roi_orb_points"), -1.0),
            "disparity_roi_brisk_points": _safe_float(row.get("disparity_roi_brisk_points"), -1.0),
            "disparity_roi_akaze_points": _safe_float(row.get("disparity_roi_akaze_points"), -1.0),
            "disparity_roi_sift_points": _safe_float(row.get("disparity_roi_sift_points"), -1.0),
            "disparity_roi_iou_region_color_patch": _safe_float(row.get("disparity_roi_iou_region_color_patch"), -1.0),
            "disparity_roi_patch_iou_color_edge": _safe_float(row.get("disparity_roi_patch_iou_color_edge"), -1.0),
            "disparity_roi_cuda_template_match": _safe_float(row.get("disparity_roi_cuda_template_match"), -1.0),
            "disparity_roi_cuda_stereo_bm": _safe_float(row.get("disparity_roi_cuda_stereo_bm"), -1.0),
            "disparity_roi_cuda_stereo_sgm": _safe_float(row.get("disparity_roi_cuda_stereo_sgm"), -1.0),
            "disparity_roi_neural_feature": _safe_float(row.get("disparity_roi_neural_feature"), -1.0),
            "disparity_roi_center_patch": _safe_float(row.get("disparity_roi_center_patch"), -1.0),
            "disparity_roi_multi_point": _safe_float(row.get("disparity_roi_multi_point", row.get("disparity_subpixel")), -1.0),
            "disparity_fallback_epipolar": _safe_float(row.get("disparity_fallback_epipolar"), -1.0),
            "disparity_fallback_template": _safe_float(row.get("disparity_fallback_template"), -1.0),
            "disparity_fallback_feature_points": _safe_float(row.get("disparity_fallback_feature_points"), -1.0),
            "disparity_yolo": _safe_float(row.get("disparity_yolo"), -1.0),
            "disparity_circle": _safe_float(row.get("disparity_circle"), -1.0),
            "disparity_subpixel": _safe_float(row.get("disparity_subpixel"), -1.0),
            "epipolar_dy": _safe_float(row.get("epipolar_dy"), -1.0),
            "size_ratio": _safe_float(row.get("size_ratio"), -1.0),
            "pair_initial_disparity": _safe_float(row.get("pair_initial_disparity"), -1.0),
            "pair_epipolar_dy": _safe_float(row.get("pair_epipolar_dy"), -1.0),
            "pair_y_tolerance": _safe_float(row.get("pair_y_tolerance"), -1.0),
            "pair_size_ratio": _safe_float(row.get("pair_size_ratio"), -1.0),
            "pair_shifted_iou": _safe_float(row.get("pair_shifted_iou"), -1.0),
            "pair_score": _safe_float(row.get("pair_score"), 0.0),
            "pair_bbox_prior_penalty": _safe_float(row.get("pair_bbox_prior_penalty"), 0.0),
            "pair_positive_disparity": _safe_float(row.get("pair_positive_disparity"), 0.0),
            "left_circle_conf": _safe_float(row.get("left_circle_conf"), 0.0),
            "right_circle_conf": _safe_float(row.get("right_circle_conf"), 0.0),
            "subpixel_valid": _safe_float(row.get("subpixel_valid"), 0.0),
            "subpixel_attempted": _safe_float(row.get("subpixel_attempted"), 0.0),
            "subpixel_support": _safe_float(row.get("subpixel_support"), 0.0),
            "subpixel_std_px": _safe_float(row.get("subpixel_std_px"), -1.0),
            "subpixel_confidence": _safe_float(row.get("subpixel_confidence"), 0.0),
            "subpixel_gate_px": _safe_float(row.get("subpixel_gate_px"), 0.0),
            "roi_corner_points_support": _safe_float(row.get("roi_corner_points_support"), 0.0),
            "roi_corner_points_std_px": _safe_float(row.get("roi_corner_points_std_px"), -1.0),
            "roi_corner_points_confidence": _safe_float(row.get("roi_corner_points_confidence"), 0.0),
            "roi_texture_points_support": _safe_float(row.get("roi_texture_points_support"), 0.0),
            "roi_texture_points_std_px": _safe_float(row.get("roi_texture_points_std_px"), -1.0),
            "roi_texture_points_confidence": _safe_float(row.get("roi_texture_points_confidence"), 0.0),
            "roi_binary_points_support": _safe_float(row.get("roi_binary_points_support"), 0.0),
            "roi_binary_points_std_px": _safe_float(row.get("roi_binary_points_std_px"), -1.0),
            "roi_binary_points_confidence": _safe_float(row.get("roi_binary_points_confidence"), 0.0),
            "roi_orb_points_support": _safe_float(row.get("roi_orb_points_support"), 0.0),
            "roi_orb_points_std_px": _safe_float(row.get("roi_orb_points_std_px"), -1.0),
            "roi_orb_points_confidence": _safe_float(row.get("roi_orb_points_confidence"), 0.0),
            "roi_brisk_points_support": _safe_float(row.get("roi_brisk_points_support"), 0.0),
            "roi_brisk_points_std_px": _safe_float(row.get("roi_brisk_points_std_px"), -1.0),
            "roi_brisk_points_confidence": _safe_float(row.get("roi_brisk_points_confidence"), 0.0),
            "roi_akaze_points_support": _safe_float(row.get("roi_akaze_points_support"), 0.0),
            "roi_akaze_points_std_px": _safe_float(row.get("roi_akaze_points_std_px"), -1.0),
            "roi_akaze_points_confidence": _safe_float(row.get("roi_akaze_points_confidence"), 0.0),
            "roi_sift_points_support": _safe_float(row.get("roi_sift_points_support"), 0.0),
            "roi_sift_points_std_px": _safe_float(row.get("roi_sift_points_std_px"), -1.0),
            "roi_sift_points_confidence": _safe_float(row.get("roi_sift_points_confidence"), 0.0),
            "roi_iou_region_color_patch_support": _safe_float(row.get("roi_iou_region_color_patch_support"), 0.0),
            "roi_iou_region_color_patch_std_px": _safe_float(row.get("roi_iou_region_color_patch_std_px"), -1.0),
            "roi_iou_region_color_patch_confidence": _safe_float(row.get("roi_iou_region_color_patch_confidence"), 0.0),
            "roi_patch_iou_color_edge_support": _safe_float(row.get("roi_patch_iou_color_edge_support"), 0.0),
            "roi_patch_iou_color_edge_std_px": _safe_float(row.get("roi_patch_iou_color_edge_std_px"), -1.0),
            "roi_patch_iou_color_edge_confidence": _safe_float(row.get("roi_patch_iou_color_edge_confidence"), 0.0),
            "roi_neural_feature_support": _safe_float(row.get("roi_neural_feature_support"), 0.0),
            "roi_neural_feature_std_px": _safe_float(row.get("roi_neural_feature_std_px"), -1.0),
            "roi_neural_feature_confidence": _safe_float(row.get("roi_neural_feature_confidence"), 0.0),
            "fallback_feature_points_support": _safe_float(row.get("fallback_feature_points_support"), 0.0),
            "fallback_feature_points_std_px": _safe_float(row.get("fallback_feature_points_std_px"), -1.0),
            "fallback_feature_points_confidence": _safe_float(row.get("fallback_feature_points_confidence"), 0.0),
            "raw_observation_valid": _safe_float(row.get("raw_observation_valid"), 1.0),
            "predicted_z": _safe_float(row.get("predicted_z"), -1.0),
            "innovation_z": _safe_float(row.get("innovation_z"), 0.0),
            "innovation_norm": _safe_float(row.get("innovation_norm"), 0.0),
            "kalman_sigma_z": _safe_float(row.get("kalman_sigma_z"), -1.0),
            "left_circle_source": _safe_float(row.get("left_circle_source"), 0.0),
            "right_circle_source": _safe_float(row.get("right_circle_source"), 0.0),
            "stereo_match_source": _safe_float(row.get("stereo_match_source"), 0.0),
            "stereo_depth_source": _safe_float(row.get("stereo_depth_source"), 0.0),
            "frame_counter_delta": _safe_float(row.get("frame_counter_delta"), 0.0),
            "frame_number_delta": _safe_float(row.get("frame_number_delta"), 0.0),
            "timestamp_delta_us": _safe_float(row.get("timestamp_delta_us"), 0.0),
            "left_bbox_cx": _safe_float(row.get("left_bbox_cx"), -1.0),
            "left_bbox_cy": _safe_float(row.get("left_bbox_cy"), -1.0),
            "left_bbox_w": _safe_float(row.get("left_bbox_w"), -1.0),
            "left_bbox_h": _safe_float(row.get("left_bbox_h"), -1.0),
            "left_bbox_conf": _safe_float(row.get("left_bbox_conf"), 0.0),
            "right_bbox_cx": _safe_float(row.get("right_bbox_cx"), -1.0),
            "right_bbox_cy": _safe_float(row.get("right_bbox_cy"), -1.0),
            "right_bbox_w": _safe_float(row.get("right_bbox_w"), -1.0),
            "right_bbox_h": _safe_float(row.get("right_bbox_h"), -1.0),
            "right_bbox_conf": _safe_float(row.get("right_bbox_conf"), 0.0),
            "left_circle_cx": _safe_float(row.get("left_circle_cx"), -1.0),
            "left_circle_cy": _safe_float(row.get("left_circle_cy"), -1.0),
            "left_circle_r": _safe_float(row.get("left_circle_r"), -1.0),
            "right_circle_cx": _safe_float(row.get("right_circle_cx"), -1.0),
            "right_circle_cy": _safe_float(row.get("right_circle_cy"), -1.0),
            "right_circle_r": _safe_float(row.get("right_circle_r"), -1.0),
        }
        grouped.setdefault(track_id, []).append(parsed)

    sequences: List[LegacySequence] = []
    for track_id, rows in grouped.items():
        rows.sort(key=lambda r: (r["timestamp"], r["frame_id"]))
        if len(rows) >= min_track_len:
            sequences.append(LegacySequence(track_id=track_id, rows=rows, metadata=metadata))
    sequences.sort(key=lambda seq: seq.track_id)
    return sequences
