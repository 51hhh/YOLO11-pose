"""Output helpers for offline volleyball keypoint probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import cv2

from stereo_feature_matching.realtime_contract import DEPTH_CANDIDATE_PRIORITY, STEREO_DEPTH_SOURCE

from offline_volleyball_probe_roi import MatchResult
from offline_volleyball_probe_triangulation import (
    method_validation_status,
    runtime_feature_geometry_status,
    triangulation_stats,
)


def build_result_rows(
    results: Sequence[MatchResult],
    triangulated_rows_by_method: Dict[str, Sequence[Dict[str, object]]],
    thresholds,
    feature_gate,
    lroi,
    rroi,
    initial_disp: float,
    focal_px: float,
    baseline_m: float,
    method_elapsed_ms: Dict[str, float],
) -> list[Dict[str, object]]:
    rows = []
    for res in results:
        tri_stats = triangulation_stats(triangulated_rows_by_method.get(res.name, []))
        validation_status, validation_fail_reasons = method_validation_status(tri_stats, thresholds)
        runtime_feature_ok, runtime_anchor_x, runtime_anchor_y = runtime_feature_geometry_status(
            res,
            lroi,
            rroi,
            initial_disp,
            focal_px,
            baseline_m,
            feature_gate,
        )
        rows.append({
            "method": res.name,
            "left_keypoints": len(res.left_keypoints),
            "right_keypoints": len(res.right_keypoints),
            "candidates": res.candidates,
            "matches": len(res.matches),
            "disparity_px": res.disparity,
            "std_px": res.std_px,
            "depth_m": res.depth_m,
            "confidence": res.confidence,
            "elapsed_ms": method_elapsed_ms.get(res.name, 0.0),
            "validation_status": validation_status,
            "validation_fail_reasons": validation_fail_reasons,
            "runtime_feature_geometry_ok": runtime_feature_ok,
            "runtime_feature_anchor_x": runtime_anchor_x,
            "runtime_feature_anchor_y": runtime_anchor_y,
            **tri_stats,
        })
    return rows


def build_probe_summary(
    args,
    focal_px: float,
    baseline_m: float,
    ball_center_3d,
    thresholds,
    pair_gate,
    runtime_pair,
    runtime_pair_reject: str,
    initial_disp: float,
    initial_depth: float,
    lroi,
    rroi,
    feature_gate,
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "left_image": args.left,
        "right_image": args.right,
        "calibration": args.calib,
        "focal_px": focal_px,
        "baseline_m": baseline_m,
        "ball_diameter_m": args.ball_diameter_m,
        "ball_center_3d_m": {
            "x": ball_center_3d[0],
            "y": ball_center_3d[1],
            "z": ball_center_3d[2],
            "source": "roi_center_disparity",
        },
        "validation_thresholds": {
            "min_valid_matches": thresholds.min_valid_matches,
            "max_y_error_px": thresholds.max_y_error_px,
            "max_disparity_mad_px": thresholds.max_disparity_mad_px,
            "max_disparity_range_px": thresholds.max_disparity_range_px,
            "max_z_mad_m": thresholds.max_z_mad_m,
            "max_z_range_m": thresholds.max_z_range_m,
            "max_sphere_residual_m": thresholds.max_sphere_residual_m,
            "max_depth_vs_center_m": thresholds.max_depth_vs_center_m,
        },
        "runtime_contract": {
            "roi_pair_gate": {
                "max_disparity": pair_gate.max_disparity,
                "epipolar_y_tolerance": pair_gate.epipolar_y_tolerance,
                "max_size_ratio": pair_gate.max_size_ratio,
                "adaptive_y_ratio": pair_gate.adaptive_y_ratio,
                "min_shifted_iou": pair_gate.min_shifted_iou,
                "min_depth_m": args.min_depth_m,
                "max_depth_m": args.max_depth_m,
            },
            "roi_pair": {
                "valid": runtime_pair is not None,
                "reject_reason": runtime_pair_reject,
                "initial_disparity_px": runtime_pair.initial_disparity if runtime_pair else initial_disp,
                "epipolar_dy_px": runtime_pair.epipolar_dy if runtime_pair else abs(lroi.center[1] - rroi.center[1]),
                "shifted_bbox_iou": runtime_pair.shifted_bbox_iou if runtime_pair else 0.0,
                "score": runtime_pair.score if runtime_pair else None,
            },
            "depth_candidate_priority": list(DEPTH_CANDIDATE_PRIORITY),
            "stereo_depth_source": STEREO_DEPTH_SOURCE,
            "feature_validation_gate": {
                "min_support": feature_gate.min_support,
                "max_stddev_px": feature_gate.max_stddev_px,
                "feature_y_tolerance_px": feature_gate.feature_y_tolerance_px,
                "feature_y_slope": feature_gate.feature_y_slope,
                "feature_y_offset_px": feature_gate.feature_y_offset_px,
                "feature_overlap_scale": feature_gate.feature_overlap_scale,
                "feature_sphere_radius_m": feature_gate.feature_sphere_radius_m,
                "feature_sphere_radius_scale": feature_gate.feature_sphere_radius_scale,
                "feature_sphere_margin_m": feature_gate.feature_sphere_margin_m,
            },
        },
        "matcher_params": {
            "edge_percentile": args.edge_percentile,
            "iou_patch_radius": args.iou_patch_radius,
            "iou_search_radius": args.iou_search_radius,
            "iou_y_radius": args.iou_y_radius,
            "iou_min_score": args.iou_min_score,
            "iou_reverse_tolerance_px": args.iou_reverse_tolerance_px,
            "iou_max_points": args.iou_max_points,
        },
        "initial_disparity_px": initial_disp,
        "initial_depth_m": initial_depth,
        "left_roi": {"bbox": lroi.bbox, "center": lroi.center, "radius": lroi.radius, "source": lroi.source},
        "right_roi": {"bbox": rroi.bbox, "center": rroi.center, "radius": rroi.radius, "source": rroi.source},
        "results": list(rows),
    }


def write_keypoint_summary(
    out_dir: Path,
    summary: Dict[str, object],
    rows: Sequence[Dict[str, object]],
) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not rows:
        (out_dir / "summary.csv").write_text("", encoding="utf-8")
        return
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheet(image_paths: Sequence[Path], out_path: Path) -> None:
    sheets = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            sheets.append(img)
    if not sheets:
        return

    target_w = sheets[0].shape[1]
    normalized = []
    for img in sheets:
        if img.shape[1] != target_w:
            scale = target_w / img.shape[1]
            img = cv2.resize(img, (target_w, max(1, round(img.shape[0] * scale))))
        normalized.append(img)
    cv2.imwrite(str(out_path), cv2.vconcat(normalized))


def write_match_contact_sheets(
    out_dir: Path,
    results: Sequence[MatchResult],
) -> None:
    match_paths = [out_dir / f"{res.name}_matches.png" for res in results]
    zoom_paths = [out_dir / f"{res.name}_matches_zoom.png" for res in results]
    _write_contact_sheet(match_paths, out_dir / "contact_sheet.png")
    _write_contact_sheet(zoom_paths, out_dir / "zoom_contact_sheet.png")
