#!/usr/bin/env python3
"""Dataset helpers for trajectory fusion experiments."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


METHOD_COLUMNS = (
    ("mono", "z_mono"),
    ("bbox_center", "z_bbox_center"),
    ("bbox_left_edge", "z_bbox_left_edge"),
    ("bbox_right_edge", "z_bbox_right_edge"),
    ("circle_center", "z_circle_center"),
    ("circle_left_edge", "z_circle_left_edge"),
    ("circle_right_edge", "z_circle_right_edge"),
    ("roi_edge_centroid", "z_roi_edge_centroid"),
    ("roi_radial_center", "z_roi_radial_center"),
    ("roi_edge_pair_center", "z_roi_edge_pair_center"),
    ("roi_corner_points", "z_roi_corner_points"),
    ("roi_texture_points", "z_roi_texture_points"),
    ("roi_binary_points", "z_roi_binary_points"),
    ("roi_orb_points", "z_roi_orb_points"),
    ("roi_brisk_points", "z_roi_brisk_points"),
    ("roi_akaze_points", "z_roi_akaze_points"),
    ("roi_sift_points", "z_roi_sift_points"),
    ("roi_iou_region_color_patch", "z_roi_iou_region_color_patch"),
    ("roi_patch_iou_color_edge", "z_roi_patch_iou_color_edge"),
    ("roi_neural_feature", "z_roi_neural_feature"),
    ("roi_center_patch", "z_roi_center_patch"),
    ("roi_multi_point", "z_roi_multi_point"),
    ("epipolar_fallback", "z_fallback"),
    ("fallback_template", "z_fallback_template"),
    ("fallback_feature_points", "z_fallback_feature_points"),
    ("stereo", "z_stereo"),
    ("online", "z"),
)
METHOD_NAMES = tuple(name for name, _ in METHOD_COLUMNS)


@dataclass
class LegacySequence:
    """One track from the current TrajectoryRecorder CSV."""

    track_id: int
    rows: List[Dict[str, float]]

    @property
    def length(self) -> int:
        return len(self.rows)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read CSV rows while tolerating accidental NUL bytes in log files."""

    data = Path(path).read_bytes().replace(b"\x00", b"")
    text = data.decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def load_legacy_sequences(path: str | Path, min_track_len: int = 3) -> List[LegacySequence]:
    """Load current trajectory recorder CSV and group rows by track_id."""

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
            "z_roi_neural_feature": _safe_float(row.get("z_roi_neural_feature"), -1.0),
            "z_roi_center_patch": _safe_float(row.get("z_roi_center_patch"), -1.0),
            "z_roi_multi_point": _safe_float(row.get("z_roi_multi_point", row.get("z_subpixel")), -1.0),
            "z_yolo_bbox_pair": _safe_float(row.get("z_yolo_bbox_pair"), -1.0),
            "z_circle": _safe_float(row.get("z_circle"), -1.0),
            "z_subpixel": _safe_float(row.get("z_subpixel"), -1.0),
            "z_fallback": _safe_float(row.get("z_fallback"), -1.0),
            "z_fallback_template": _safe_float(row.get("z_fallback_template"), -1.0),
            "z_fallback_feature_points": _safe_float(row.get("z_fallback_feature_points"), -1.0),
            "z_stereo": _safe_float(row.get("z_stereo"), -1.0),
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
            "disparity_roi_neural_feature": _safe_float(row.get("disparity_roi_neural_feature"), -1.0),
            "disparity_roi_center_patch": _safe_float(row.get("disparity_roi_center_patch"), -1.0),
            "disparity_roi_multi_point": _safe_float(row.get("disparity_roi_multi_point", row.get("disparity_subpixel")), -1.0),
            "disparity_fallback_template": _safe_float(row.get("disparity_fallback_template"), -1.0),
            "disparity_fallback_feature_points": _safe_float(row.get("disparity_fallback_feature_points"), -1.0),
            "disparity_yolo": _safe_float(row.get("disparity_yolo"), -1.0),
            "disparity_circle": _safe_float(row.get("disparity_circle"), -1.0),
            "disparity_subpixel": _safe_float(row.get("disparity_subpixel"), -1.0),
            "epipolar_dy": _safe_float(row.get("epipolar_dy"), -1.0),
            "size_ratio": _safe_float(row.get("size_ratio"), -1.0),
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
            sequences.append(LegacySequence(track_id=track_id, rows=rows))
    sequences.sort(key=lambda seq: seq.track_id)
    return sequences


def legacy_feature_names() -> List[str]:
    """Feature order used by build_legacy_arrays()."""

    return [
        "dt",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "z_mono",
        "z_bbox_center",
        "z_bbox_left_edge",
        "z_bbox_right_edge",
        "z_circle_center",
        "z_circle_left_edge",
        "z_circle_right_edge",
        "z_roi_edge_centroid",
        "z_roi_radial_center",
        "z_roi_edge_pair_center",
        "z_roi_corner_points",
        "z_roi_texture_points",
        "z_roi_binary_points",
        "z_roi_orb_points",
        "z_roi_brisk_points",
        "z_roi_akaze_points",
        "z_roi_sift_points",
        "z_roi_iou_region_color_patch",
        "z_roi_patch_iou_color_edge",
        "z_roi_neural_feature",
        "z_roi_center_patch",
        "z_roi_multi_point",
        "z_fallback",
        "z_fallback_template",
        "z_fallback_feature_points",
        "z_stereo",
        "confidence",
        "class_id",
        "disparity_bbox_center",
        "disparity_bbox_left_edge",
        "disparity_bbox_right_edge",
        "disparity_circle_center",
        "disparity_circle_left_edge",
        "disparity_circle_right_edge",
        "disparity_roi_edge_centroid",
        "disparity_roi_radial_center",
        "disparity_roi_edge_pair_center",
        "disparity_roi_corner_points",
        "disparity_roi_texture_points",
        "disparity_roi_binary_points",
        "disparity_roi_orb_points",
        "disparity_roi_brisk_points",
        "disparity_roi_akaze_points",
        "disparity_roi_sift_points",
        "disparity_roi_iou_region_color_patch",
        "disparity_roi_patch_iou_color_edge",
        "disparity_roi_neural_feature",
        "disparity_roi_center_patch",
        "disparity_roi_multi_point",
        "disparity_fallback_template",
        "disparity_fallback_feature_points",
        "epipolar_dy",
        "size_ratio",
        "left_circle_conf",
        "right_circle_conf",
        "subpixel_valid",
        "subpixel_attempted",
        "subpixel_support",
        "subpixel_std_px",
        "subpixel_confidence",
        "subpixel_gate_px",
        "roi_corner_points_support",
        "roi_corner_points_std_px",
        "roi_corner_points_confidence",
        "roi_texture_points_support",
        "roi_texture_points_std_px",
        "roi_texture_points_confidence",
        "roi_binary_points_support",
        "roi_binary_points_std_px",
        "roi_binary_points_confidence",
        "roi_orb_points_support",
        "roi_orb_points_std_px",
        "roi_orb_points_confidence",
        "roi_brisk_points_support",
        "roi_brisk_points_std_px",
        "roi_brisk_points_confidence",
        "roi_akaze_points_support",
        "roi_akaze_points_std_px",
        "roi_akaze_points_confidence",
        "roi_sift_points_support",
        "roi_sift_points_std_px",
        "roi_sift_points_confidence",
        "roi_iou_region_color_patch_support",
        "roi_iou_region_color_patch_std_px",
        "roi_iou_region_color_patch_confidence",
        "roi_patch_iou_color_edge_support",
        "roi_patch_iou_color_edge_std_px",
        "roi_patch_iou_color_edge_confidence",
        "roi_neural_feature_support",
        "roi_neural_feature_std_px",
        "roi_neural_feature_confidence",
        "fallback_feature_points_support",
        "fallback_feature_points_std_px",
        "fallback_feature_points_confidence",
        "raw_observation_valid",
        "predicted_z",
        "innovation_z",
        "innovation_norm",
        "kalman_sigma_z",
        "left_circle_source",
        "right_circle_source",
        "stereo_match_source",
        "stereo_depth_source",
        "frame_counter_delta",
        "frame_number_delta",
        "timestamp_delta_us",
        "left_bbox_cx",
        "left_bbox_cy",
        "left_bbox_w",
        "left_bbox_h",
        "left_bbox_conf",
        "right_bbox_cx",
        "right_bbox_cy",
        "right_bbox_w",
        "right_bbox_h",
        "right_bbox_conf",
        "left_circle_cx",
        "left_circle_cy",
        "left_circle_r",
        "right_circle_cx",
        "right_circle_cy",
        "right_circle_r",
        "method_is_mono",
        "method_is_stereo",
        "method_is_blend",
        *[f"{name}_valid" for name in METHOD_NAMES],
    ]


def build_legacy_arrays(sequence: LegacySequence) -> Dict[str, List[List[float]]]:
    """Build feature, measurement and validity arrays from a recorder sequence.

    measurements order: METHOD_NAMES.
    """

    features: List[List[float]] = []
    measurements: List[List[float]] = []
    valid: List[List[float]] = []
    prev_ts = None

    for row in sequence.rows:
        ts = row["timestamp"]
        if prev_ts is None:
            dt = 0.01
        else:
            dt = max(1e-4, min(0.2, ts - prev_ts))
        prev_ts = ts

        method = int(row["depth_method"])
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

        features.append(
            [
                dt,
                row["x"],
                row["y"],
                row["z"],
                row["vx"],
                row["vy"],
                row["vz"],
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
                row["z_roi_neural_feature"] if valid_by_key["z_roi_neural_feature"] else 0.0,
                row["z_roi_center_patch"] if valid_by_key["z_roi_center_patch"] else 0.0,
                row["z_roi_multi_point"] if valid_by_key["z_roi_multi_point"] else 0.0,
                row["z_fallback"] if valid_by_key["z_fallback"] else 0.0,
                row["z_fallback_template"] if valid_by_key["z_fallback_template"] else 0.0,
                row["z_fallback_feature_points"] if valid_by_key["z_fallback_feature_points"] else 0.0,
                row["z_stereo"] if valid_by_key["z_stereo"] else 0.0,
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
                row["disparity_roi_neural_feature"],
                row["disparity_roi_center_patch"],
                row["disparity_roi_multi_point"],
                row["disparity_fallback_template"],
                row["disparity_fallback_feature_points"],
                row["epipolar_dy"],
                row["size_ratio"],
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
                row["predicted_z"] if row["predicted_z"] > 0.0 else 0.0,
                row["innovation_z"],
                row["innovation_norm"],
                row["kalman_sigma_z"] if row["kalman_sigma_z"] > 0.0 else 0.0,
                row["left_circle_source"],
                row["right_circle_source"],
                row["stereo_match_source"],
                row["stereo_depth_source"],
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
                1.0 if method == 0 else 0.0,
                1.0 if method == 1 else 0.0,
                1.0 if method == 2 else 0.0,
                *valid_row,
            ]
        )
        measurements.append(measurements_row)
        valid.append(valid_row)

    return {"features": features, "measurements": measurements, "valid": valid}


def iter_extended_rows(path: str | Path) -> Iterable[Dict[str, str]]:
    """Yield rows from a future schema.md-compatible CSV file."""

    yield from read_csv_rows(path)


def normalize_features(features: Sequence[Sequence[float]]) -> List[List[float]]:
    """Simple per-sequence normalization for small offline experiments."""

    if not features:
        return []
    cols = len(features[0])
    means = [0.0] * cols
    for row in features:
        for i, value in enumerate(row):
            means[i] += value
    means = [value / len(features) for value in means]

    stds = [1e-6] * cols
    for row in features:
        for i, value in enumerate(row):
            diff = value - means[i]
            stds[i] += diff * diff
    stds = [(value / len(features)) ** 0.5 for value in stds]

    return [[(value - means[i]) / max(stds[i], 1e-6) for i, value in enumerate(row)] for row in features]
