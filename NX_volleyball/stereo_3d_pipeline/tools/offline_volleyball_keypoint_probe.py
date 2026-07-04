#!/usr/bin/env python3
"""Offline CPU probe for volleyball stereo keypoint/depth candidates.

This is intentionally diagnostic: it uses one saved stereo pair, rectifies it
with the current calibration, segments the ball ROI, then compares descriptor
matches against a patch IoU/ZNCC epipolar search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import evaluate_stereo_roi_pair

from offline_volleyball_keypoint_config import (
    build_feature_gate,
    build_pair_gate,
    build_validation_thresholds,
    parse_args,
)
from offline_volleyball_probe_roi import (
    MatchResult,
    _circle_roi,
    _roi_to_runtime_detection,
    detect_ball_rois,
    draw_roi_debug,
    load_calibration,
    rectify_pair,
)
from offline_volleyball_probe_triangulation import (
    depth_from_disparity,
    estimate_ball_center_3d,
    triangulate_match_rows,
    validate_triangulated_rows,
    write_triangulated_points,
)
from offline_volleyball_probe_visualization import (
    draw_matches,
    draw_matches_zoom,
)
from offline_volleyball_keypoint_methods import (
    prepare_keypoint_feature_inputs,
    run_keypoint_probe_methods,
)
from offline_volleyball_keypoint_report import (
    build_probe_summary,
    build_result_rows,
    write_keypoint_summary,
    write_match_contact_sheets,
)

def main() -> int:
    args = parse_args()
    thresholds = build_validation_thresholds(args)
    pair_gate = build_pair_gate(args)
    feature_gate = build_feature_gate(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    left = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError("failed to load input images")

    calib = load_calibration(Path(args.calib))
    left_rect, right_rect = rectify_pair(left, right, calib)
    cv2.imwrite(str(out_dir / "left_rect_color.png"), left_rect)
    cv2.imwrite(str(out_dir / "right_rect_color.png"), right_rect)

    focal_px = float(np.asarray(calib["P1"])[0, 0])
    baseline_m = float(calib["baseline_m"])
    if (args.left_circle is None) != (args.right_circle is None):
        raise ValueError("--left-circle and --right-circle must be provided together")
    if args.left_circle is not None and args.right_circle is not None:
        lx, ly, lr = args.left_circle
        rx, ry, rr = args.right_circle
        lroi = _circle_roi(left_rect.shape[:2], (lx, ly), lr, "manual", args.mask_margin)
        rroi = _circle_roi(right_rect.shape[:2], (rx, ry), rr, "manual", args.mask_margin)
    else:
        lroi, rroi = detect_ball_rois(
            left_rect,
            right_rect,
            focal_px,
            baseline_m,
            args.mask_margin,
            args.ball_diameter_m,
            pair_gate,
            args.min_depth_m,
            args.max_depth_m,
        )
    initial_disp = float(lroi.center[0] - rroi.center[0])
    initial_depth = depth_from_disparity(initial_disp, focal_px, baseline_m)
    runtime_pair, runtime_pair_reject = evaluate_stereo_roi_pair(
        _roi_to_runtime_detection(lroi),
        _roi_to_runtime_detection(rroi),
        0,
        0,
        pair_gate,
    )

    roi_debug = draw_roi_debug(left_rect, right_rect, lroi, rroi)
    cv2.imwrite(str(out_dir / "rectified_roi_debug.png"), roi_debug)

    features, left_overlap, right_overlap = prepare_keypoint_feature_inputs(
        left_rect,
        right_rect,
        lroi,
        rroi,
        initial_disp,
        args,
        out_dir,
    )
    results, method_elapsed_ms = run_keypoint_probe_methods(
        features,
        lroi,
        rroi,
        left_overlap,
        right_overlap,
        initial_disp,
        focal_px,
        baseline_m,
        args,
    )

    for res in results:
        draw_matches(left_rect, right_rect, res, out_dir / f"{res.name}_matches.png")
        draw_matches_zoom(left_rect, right_rect, lroi, rroi, res, out_dir / f"{res.name}_matches_zoom.png")

    triangulated_rows_by_method: Dict[str, List[Dict[str, float | int | str]]] = {}
    all_triangulated_rows: List[Dict[str, float | int | str]] = []
    ball_center_3d = estimate_ball_center_3d(calib, lroi, initial_disp, baseline_m)
    ball_radius_m = 0.5 * args.ball_diameter_m
    for res in results:
        tri_rows = triangulate_match_rows(res, calib, baseline_m)
        tri_rows = validate_triangulated_rows(
            tri_rows,
            lroi,
            rroi,
            left_overlap,
            right_overlap,
            ball_center_3d,
            ball_radius_m,
            thresholds,
        )
        triangulated_rows_by_method[res.name] = tri_rows
        all_triangulated_rows.extend(tri_rows)
        write_triangulated_points(out_dir / f"{res.name}_triangulated_points.csv", tri_rows)
    write_triangulated_points(out_dir / "triangulated_points.csv", all_triangulated_rows)

    rows = build_result_rows(
        results,
        triangulated_rows_by_method,
        thresholds,
        feature_gate,
        lroi,
        rroi,
        initial_disp,
        focal_px,
        baseline_m,
        method_elapsed_ms,
    )
    summary = build_probe_summary(
        args,
        focal_px,
        baseline_m,
        ball_center_3d,
        thresholds,
        pair_gate,
        runtime_pair,
        runtime_pair_reject,
        initial_disp,
        initial_depth,
        lroi,
        rroi,
        feature_gate,
        rows,
    )
    write_keypoint_summary(out_dir, summary, rows)
    write_match_contact_sheets(out_dir, results)

    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
