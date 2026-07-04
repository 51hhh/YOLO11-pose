#!/usr/bin/env python3
"""Offline neural feature probe for volleyball stereo ROIs.

This script keeps neural experimentation out of the realtime C++ pipeline.
It reuses the current calibration, ROI detection, triangulation, validation,
and visualization helpers from offline_volleyball_keypoint_probe.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import offline_volleyball_keypoint_probe as probe
from neural_feature_probe_config import backend_names, parse_args
from neural_feature_probe_runner import evaluate_neural_backend, write_zoom_contact_sheet
from stereo_feature_matching.probe_utils import (
    crop_square,
    write_csv_rows,
)
from stereo_feature_matching.visualization import (
    draw_crop_debug,
)


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    left = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError("failed to load input images")

    calib = probe.load_calibration(Path(args.calib))
    left_rect, right_rect = probe.rectify_pair(left, right, calib)
    cv2.imwrite(str(out_dir / "left_rect_color.png"), left_rect)
    cv2.imwrite(str(out_dir / "right_rect_color.png"), right_rect)

    focal_px = float(np.asarray(calib["P1"])[0, 0])
    baseline_m = float(calib["baseline_m"])
    if (args.left_circle is None) != (args.right_circle is None):
        raise ValueError("--left-circle and --right-circle must be provided together")
    if args.left_circle is not None and args.right_circle is not None:
        lx, ly, lr = args.left_circle
        rx, ry, rr = args.right_circle
        lroi = probe._circle_roi(left_rect.shape[:2], (lx, ly), lr, "manual", args.mask_margin)
        rroi = probe._circle_roi(right_rect.shape[:2], (rx, ry), rr, "manual", args.mask_margin)
    else:
        lroi, rroi = probe.detect_ball_rois(
            left_rect, right_rect, focal_px, baseline_m, args.mask_margin, 0.210
        )
    initial_disp = float(lroi.center[0] - rroi.center[0])
    initial_depth = probe.depth_from_disparity(initial_disp, focal_px, baseline_m)
    cv2.imwrite(str(out_dir / "rectified_roi_debug.png"), probe.draw_roi_debug(left_rect, right_rect, lroi, rroi))

    left_crop, left_crop_mask, lt = crop_square(
        left_rect, lroi.mask, lroi.bbox, pad=args.crop_pad, output_size=args.roi_size
    )
    right_crop, right_crop_mask, rt = crop_square(
        right_rect, rroi.mask, rroi.bbox, pad=args.crop_pad, output_size=args.roi_size
    )
    draw_crop_debug(left_crop, right_crop, out_dir / "neural_roi_crops.png")

    left_overlap, right_overlap = probe._overlap_masks_for_disparity(lroi.mask, rroi.mask, initial_disp)
    thresholds = probe.ValidationThresholds(
        min_valid_matches=8,
        max_y_error_px=args.max_y_error_px,
    )
    ball_center_3d = probe.estimate_ball_center_3d(calib, lroi, initial_disp, baseline_m)

    selected_backends = backend_names(args.backends)
    rows: List[Dict[str, object]] = []
    results: List[probe.MatchResult] = []
    missing = []

    for backend_name in selected_backends:
        row, result, missing_backend = evaluate_neural_backend(
            backend_name,
            args,
            out_dir,
            left_rect,
            right_rect,
            lroi,
            rroi,
            left_crop,
            right_crop,
            left_crop_mask,
            right_crop_mask,
            lt,
            rt,
            calib,
            focal_px,
            baseline_m,
            initial_disp,
            left_overlap,
            right_overlap,
            ball_center_3d,
            thresholds,
        )
        rows.append(row)
        if result is not None:
            results.append(result)
        if missing_backend is not None:
            missing.append(missing_backend)

    write_csv_rows(out_dir / "summary.csv", rows)
    summary = {
        "left_image": args.left,
        "right_image": args.right,
        "calibration": args.calib,
        "initial_disparity_px": initial_disp,
        "initial_depth_m": initial_depth,
        "roi_size": args.roi_size,
        "top_k": args.top_k,
        "results": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_zoom_contact_sheet(out_dir, results)

    print(json.dumps(summary, indent=2))
    if missing and args.fail_on_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
