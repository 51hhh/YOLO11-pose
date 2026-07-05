#!/usr/bin/env python3
"""Dataset helpers for trajectory fusion experiments."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    ("roi_cuda_template_match", "z_roi_cuda_template_match"),
    ("roi_cuda_stereo_bm", "z_roi_cuda_stereo_bm"),
    ("roi_cuda_stereo_sgm", "z_roi_cuda_stereo_sgm"),
    ("roi_vpi_template_match", "z_roi_vpi_template_match"),
    ("roi_vpi_orb", "z_roi_vpi_orb"),
    ("roi_opencv_cuda_gftt_lk", "z_roi_opencv_cuda_gftt_lk"),
    ("roi_ring_edge_profile", "z_roi_ring_edge_profile"),
    ("roi_neural_feature", "z_roi_neural_feature"),
    ("roi_center_patch", "z_roi_center_patch"),
    ("roi_multi_point", "z_roi_multi_point"),
    ("epipolar_fallback", "z_fallback_epipolar"),
    ("fallback_template", "z_fallback_template"),
    ("fallback_feature_points", "z_fallback_feature_points"),
)
METHOD_NAMES = tuple(name for name, _ in METHOD_COLUMNS)

P2_DIAGNOSTIC_MODE_COLUMNS = {
    "vpi_template_match": {
        "z": "z_roi_vpi_template_match",
        "disparity": "disparity_roi_vpi_template_match",
        "support": "roi_vpi_template_match_support",
        "std": "roi_vpi_template_match_std_px",
        "confidence": "roi_vpi_template_match_confidence",
    },
    "vpi_orb": {
        "z": "z_roi_vpi_orb",
        "disparity": "disparity_roi_vpi_orb",
        "support": "roi_vpi_orb_support",
        "std": "roi_vpi_orb_std_px",
        "confidence": "roi_vpi_orb_confidence",
    },
    "opencv_cuda_gftt_lk": {
        "z": "z_roi_opencv_cuda_gftt_lk",
        "disparity": "disparity_roi_opencv_cuda_gftt_lk",
        "support": "roi_opencv_cuda_gftt_lk_support",
        "std": "roi_opencv_cuda_gftt_lk_std_px",
        "confidence": "roi_opencv_cuda_gftt_lk_confidence",
    },
    "neural_feature": {
        "z": "z_roi_neural_feature",
        "disparity": "disparity_roi_neural_feature",
        "support": "roi_neural_feature_support",
        "std": "roi_neural_feature_std_px",
        "confidence": "roi_neural_feature_confidence",
    },
}


@dataclass
class LegacySequence:
    """One track from the current TrajectoryRecorder CSV."""

    track_id: int
    rows: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

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


def _parse_metadata_value(value: str) -> Any:
    value = value.strip()
    if value == "" or value.lower() in {"null", "none", "~"}:
        return None
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_metadata(path: str | Path | None) -> Dict[str, Any]:
    """Read optional weak-label metadata.

    PyYAML is used when present. A minimal key/value parser is kept as a
    dependency-free fallback for the flat metadata files recommended in the
    wiki.
    """

    if path is None:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    text = meta_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return dict(loaded or {}) if isinstance(loaded, dict) else {}
    except ImportError:
        pass

    metadata: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_metadata_value(value)
    return metadata


def find_metadata_for_csv(path: str | Path) -> Path | None:
    """Find a sidecar metadata file for a trajectory CSV."""

    csv_path = Path(path)
    candidates = (
        csv_path.with_suffix(".metadata.yaml"),
        csv_path.with_suffix(".metadata.yml"),
        csv_path.parent / "metadata.yaml",
        csv_path.parent / "metadata.yml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def derive_frame_summary_path(path: str | Path) -> Path:
    """Derive the recorder sidecar path matching the C++ recorder."""

    csv_path = Path(path)
    if csv_path.name.endswith(".csv"):
        return csv_path.with_name(csv_path.name[:-4] + ".frames.csv")
    return Path(str(csv_path) + ".frames.csv")


def derive_p2_diagnostic_path(path: str | Path) -> Path:
    """Derive the P2 diagnostic sidecar path used by the runtime recorder."""

    csv_path = Path(path)
    if csv_path.name.endswith(".csv"):
        return csv_path.with_name(csv_path.name[:-4] + ".p2_diagnostic.csv")
    return Path(str(csv_path) + ".p2_diagnostic.csv")


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read CSV rows while tolerating accidental NUL bytes in log files."""

    data = Path(path).read_bytes().replace(b"\x00", b"")
    text = data.decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def _diagnostic_candidate_rank(row: Dict[str, str]) -> Tuple[int, float, float]:
    return (
        _safe_int(row.get("valid"), 0),
        _safe_float(row.get("confidence"), 0.0),
        _safe_float(row.get("support"), 0.0),
    )


def read_p2_diagnostic_candidates(
    path: str | Path,
) -> Dict[int, Dict[str, Dict[str, str]]]:
    """Read optional P2 diagnostic sidecar rows keyed by frame and mode."""

    sidecar_path = derive_p2_diagnostic_path(path)
    if not sidecar_path.exists():
        return {}

    by_frame: Dict[int, Dict[str, Dict[str, str]]] = {}
    for row in read_csv_rows(sidecar_path):
        mode = (row.get("mode") or "").strip()
        if mode not in P2_DIAGNOSTIC_MODE_COLUMNS:
            continue
        frame_id = _safe_int(row.get("frame_id"), -1)
        if frame_id < 0:
            continue
        frame_modes = by_frame.setdefault(frame_id, {})
        previous = frame_modes.get(mode)
        if (
            previous is None
            or _diagnostic_candidate_rank(row) > _diagnostic_candidate_rank(previous)
        ):
            frame_modes[mode] = row
    return by_frame


def _set_if_missing(row: Dict[str, str], key: str, value: object) -> None:
    if key not in row or row[key] == "":
        row[key] = str(value)


def merge_p2_diagnostic_candidates(
    row: Dict[str, str],
    frame_candidates: Dict[str, Dict[str, str]] | None,
) -> None:
    """Expose selected diagnostic sidecar rows as optional training columns."""

    for mode, columns in P2_DIAGNOSTIC_MODE_COLUMNS.items():
        candidate = frame_candidates.get(mode) if frame_candidates else None
        if candidate is None or _safe_int(candidate.get("valid"), 0) != 1:
            _set_if_missing(row, columns["z"], "-1")
            _set_if_missing(row, columns["disparity"], "-1")
            _set_if_missing(row, columns["support"], "0")
            _set_if_missing(row, columns["std"], "-1")
            _set_if_missing(row, columns["confidence"], "0")
            continue
        row[columns["z"]] = candidate.get("z_m", "-1")
        row[columns["disparity"]] = candidate.get("disparity", "-1")
        row[columns["support"]] = candidate.get("support", "0")
        row[columns["std"]] = candidate.get("stddev", "-1")
        row[columns["confidence"]] = candidate.get("confidence", "0")


def load_legacy_sequences(
    path: str | Path,
    min_track_len: int = 3,
    metadata_path: str | Path | None = None,
) -> List[LegacySequence]:
    """Load current trajectory recorder CSV and group rows by track_id."""

    metadata = read_metadata(metadata_path or find_metadata_for_csv(path))
    p2_diagnostic_by_frame = read_p2_diagnostic_candidates(path)
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for raw_row in read_csv_rows(path):
        row = dict(raw_row)
        frame_id_int = _safe_int(row.get("frame_id"), -1)
        merge_p2_diagnostic_candidates(
            row,
            p2_diagnostic_by_frame.get(frame_id_int),
        )
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
            "z_roi_vpi_template_match": _safe_float(row.get("z_roi_vpi_template_match"), -1.0),
            "z_roi_vpi_orb": _safe_float(row.get("z_roi_vpi_orb"), -1.0),
            "z_roi_opencv_cuda_gftt_lk": _safe_float(row.get("z_roi_opencv_cuda_gftt_lk"), -1.0),
            "z_roi_ring_edge_profile": _safe_float(row.get("z_roi_ring_edge_profile"), -1.0),
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
            "disparity_roi_vpi_template_match": _safe_float(row.get("disparity_roi_vpi_template_match"), -1.0),
            "disparity_roi_vpi_orb": _safe_float(row.get("disparity_roi_vpi_orb"), -1.0),
            "disparity_roi_opencv_cuda_gftt_lk": _safe_float(row.get("disparity_roi_opencv_cuda_gftt_lk"), -1.0),
            "disparity_roi_ring_edge_profile": _safe_float(row.get("disparity_roi_ring_edge_profile"), -1.0),
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
            "roi_cuda_stereo_sgm_support": _safe_float(row.get("roi_cuda_stereo_sgm_support"), 0.0),
            "roi_cuda_stereo_sgm_std_px": _safe_float(row.get("roi_cuda_stereo_sgm_std_px"), -1.0),
            "roi_cuda_stereo_sgm_confidence": _safe_float(row.get("roi_cuda_stereo_sgm_confidence"), 0.0),
            "roi_vpi_template_match_support": _safe_float(row.get("roi_vpi_template_match_support"), 0.0),
            "roi_vpi_template_match_std_px": _safe_float(row.get("roi_vpi_template_match_std_px"), -1.0),
            "roi_vpi_template_match_confidence": _safe_float(row.get("roi_vpi_template_match_confidence"), 0.0),
            "roi_vpi_orb_support": _safe_float(row.get("roi_vpi_orb_support"), 0.0),
            "roi_vpi_orb_std_px": _safe_float(row.get("roi_vpi_orb_std_px"), -1.0),
            "roi_vpi_orb_confidence": _safe_float(row.get("roi_vpi_orb_confidence"), 0.0),
            "roi_opencv_cuda_gftt_lk_support": _safe_float(row.get("roi_opencv_cuda_gftt_lk_support"), 0.0),
            "roi_opencv_cuda_gftt_lk_std_px": _safe_float(row.get("roi_opencv_cuda_gftt_lk_std_px"), -1.0),
            "roi_opencv_cuda_gftt_lk_confidence": _safe_float(row.get("roi_opencv_cuda_gftt_lk_confidence"), 0.0),
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


def legacy_feature_names() -> List[str]:
    """Feature order used by build_legacy_arrays()."""

    return [
        "dt",
        "candidate_median_z",
        "candidate_mad_z",
        "candidate_valid_count",
        "candidate_dz",
        "candidate_ddz",
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
        "z_roi_cuda_template_match",
        "z_roi_cuda_stereo_bm",
        "z_roi_cuda_stereo_sgm",
        "z_roi_vpi_template_match",
        "z_roi_vpi_orb",
        "z_roi_opencv_cuda_gftt_lk",
        "z_roi_ring_edge_profile",
        "z_roi_neural_feature",
        "z_roi_center_patch",
        "z_roi_multi_point",
        "z_fallback",
        "z_fallback_epipolar",
        "z_fallback_template",
        "z_fallback_feature_points",
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
        "disparity_roi_cuda_template_match",
        "disparity_roi_cuda_stereo_bm",
        "disparity_roi_cuda_stereo_sgm",
        "disparity_roi_vpi_template_match",
        "disparity_roi_vpi_orb",
        "disparity_roi_opencv_cuda_gftt_lk",
        "disparity_roi_ring_edge_profile",
        "disparity_roi_neural_feature",
        "disparity_roi_center_patch",
        "disparity_roi_multi_point",
        "disparity_fallback_epipolar",
        "disparity_fallback_template",
        "disparity_fallback_feature_points",
        "epipolar_dy",
        "size_ratio",
        "pair_initial_disparity",
        "pair_epipolar_dy",
        "pair_y_tolerance",
        "pair_size_ratio",
        "pair_shifted_iou",
        "pair_score",
        "pair_bbox_prior_penalty",
        "pair_positive_disparity",
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
        "roi_cuda_stereo_sgm_support",
        "roi_cuda_stereo_sgm_std_px",
        "roi_cuda_stereo_sgm_confidence",
        "roi_vpi_template_match_support",
        "roi_vpi_template_match_std_px",
        "roi_vpi_template_match_confidence",
        "roi_vpi_orb_support",
        "roi_vpi_orb_std_px",
        "roi_vpi_orb_confidence",
        "roi_opencv_cuda_gftt_lk_support",
        "roi_opencv_cuda_gftt_lk_std_px",
        "roi_opencv_cuda_gftt_lk_confidence",
        "roi_neural_feature_support",
        "roi_neural_feature_std_px",
        "roi_neural_feature_confidence",
        "fallback_feature_points_support",
        "fallback_feature_points_std_px",
        "fallback_feature_points_confidence",
        "raw_observation_valid",
        "left_circle_source",
        "right_circle_source",
        "stereo_match_source",
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
        *[f"{name}_valid" for name in METHOD_NAMES],
    ]


def _metadata_float(metadata: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return _safe_float(value, default)
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


def weak_label_names() -> List[str]:
    """Weak-label tensor order used by build_legacy_arrays()."""

    return [
        "known_z",
        "known_z_valid",
        "known_z_min",
        "known_z_max",
        "known_z_range_valid",
        "static",
        "landing_frame",
        "landing_frame_valid",
    ]


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
                row["z_roi_vpi_template_match"] if valid_by_key["z_roi_vpi_template_match"] else 0.0,
                row["z_roi_vpi_orb"] if valid_by_key["z_roi_vpi_orb"] else 0.0,
                row["z_roi_opencv_cuda_gftt_lk"] if valid_by_key["z_roi_opencv_cuda_gftt_lk"] else 0.0,
                row["z_roi_ring_edge_profile"] if valid_by_key["z_roi_ring_edge_profile"] else 0.0,
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
                row["disparity_roi_vpi_template_match"],
                row["disparity_roi_vpi_orb"],
                row["disparity_roi_opencv_cuda_gftt_lk"],
                row["disparity_roi_ring_edge_profile"],
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
                row["roi_cuda_stereo_sgm_support"],
                row["roi_cuda_stereo_sgm_std_px"],
                row["roi_cuda_stereo_sgm_confidence"],
                row["roi_vpi_template_match_support"],
                row["roi_vpi_template_match_std_px"],
                row["roi_vpi_template_match_confidence"],
                row["roi_vpi_orb_support"],
                row["roi_vpi_orb_std_px"],
                row["roi_vpi_orb_confidence"],
                row["roi_opencv_cuda_gftt_lk_support"],
                row["roi_opencv_cuda_gftt_lk_std_px"],
                row["roi_opencv_cuda_gftt_lk_confidence"],
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


def iter_extended_rows(path: str | Path) -> Iterable[Dict[str, str]]:
    """Yield rows from a future schema.md-compatible CSV file."""

    yield from read_csv_rows(path)


def compute_feature_normalizer(features: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Compute a fixed feature normalizer for training/deployment."""

    if not features:
        return [], []
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
    stds = [max(value, 1e-6) for value in stds]
    return means, stds


def apply_feature_normalizer(
    features: Sequence[Sequence[float]],
    means: Sequence[float],
    stds: Sequence[float],
) -> List[List[float]]:
    """Apply a fixed feature normalizer."""

    if not features:
        return []
    if len(means) != len(features[0]) or len(stds) != len(features[0]):
        raise ValueError("feature normalizer dimension mismatch")
    return [[(value - means[i]) / max(stds[i], 1e-6) for i, value in enumerate(row)] for row in features]


def normalize_features(features: Sequence[Sequence[float]]) -> List[List[float]]:
    """Backward-compatible helper using a normalizer fit on the input."""

    means, stds = compute_feature_normalizer(features)
    return apply_feature_normalizer(features, means, stds)
