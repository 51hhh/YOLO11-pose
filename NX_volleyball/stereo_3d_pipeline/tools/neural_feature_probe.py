#!/usr/bin/env python3
"""Offline neural feature probe for volleyball stereo ROIs.

This script keeps neural experimentation out of the realtime C++ pipeline.
It reuses the current calibration, ROI detection, triangulation, validation,
and visualization helpers from offline_volleyball_keypoint_probe.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

import offline_volleyball_keypoint_probe as probe
from stereo_feature_matching import (
    BackendUnavailable,
    MatchFilterConfig,
    match_descriptors,
)
from stereo_feature_matching.common import RawMatch
from stereo_feature_matching.geometry import filter_feature_set_by_mask, filter_stereo_matches
from stereo_feature_matching.neural_backends import BackendConfig, create_backend
from stereo_feature_matching.probe_utils import (
    crop_square,
    filter_matches_by_roi_masks,
    write_csv_rows,
)
from stereo_feature_matching.visualization import (
    cv_keypoints_from_global,
    cv_matches_from_filtered,
    draw_crop_debug,
)


def _build_match_result(
    name: str,
    left_features,
    right_features,
    filtered,
    attempted: int,
    depth_m: float,
    timings_ms: Dict[str, float],
    left_to_global: Callable[[float, float], Tuple[float, float]],
    right_to_global: Callable[[float, float], Tuple[float, float]],
) -> probe.MatchResult:
    disparities = np.asarray([m.disparity for m in filtered], dtype=np.float32)
    if disparities.size:
        disparity = float(np.median(disparities))
        std_px = float(np.std(disparities))
        confidence = float(np.clip(len(filtered) / 12.0 / (1.0 + std_px), 0.0, 1.0))
    else:
        disparity = -1.0
        std_px = -1.0
        confidence = 0.0

    notes = ",".join(f"{k}={v:.3f}ms" for k, v in timings_ms.items())
    return probe.MatchResult(
        name=name,
        left_keypoints=cv_keypoints_from_global(left_features, left_to_global),
        right_keypoints=cv_keypoints_from_global(right_features, right_to_global),
        matches=cv_matches_from_filtered(filtered),
        candidates=attempted,
        disparity=disparity,
        std_px=std_px,
        depth_m=depth_m if disparity > 0.0 else -1.0,
        confidence=confidence,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/left/0000.png")
    parser.add_argument("--right", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/right/0000.png")
    parser.add_argument("--calib", default="NX_volleyball/calibration/stereo_calib.yaml")
    parser.add_argument("--out", default="NX_volleyball/stereo_3d_pipeline/test_logs/neural_feature_probe_latest")
    parser.add_argument("--backends", default="xfeat,aliked,superpoint_lightglue",
                        help="comma-separated: xfeat,aliked,superpoint_lightglue")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--crop-pad", type=int, default=24)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--max-y-error-px", type=float, default=2.0)
    parser.add_argument("--max-disp-delta-px", type=float, default=32.0)
    parser.add_argument("--final-disp-gate-px", type=float, default=0.0)
    parser.add_argument("--max-disparity", type=float, default=2048.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--xfeat-repo", default="")
    parser.add_argument("--allow-torch-hub", action="store_true")
    parser.add_argument("--aliked-lightglue", action="store_true",
                        help="Use LightGlue matcher for ALIKED instead of descriptor NN")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

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
    lroi, rroi = probe.detect_ball_rois(left_rect, right_rect, focal_px, baseline_m, 10.0, 0.210)
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

    backend_names = [v.strip() for v in args.backends.split(",") if v.strip()]
    rows: List[Dict[str, object]] = []
    results: List[probe.MatchResult] = []
    missing = []

    for backend_name in backend_names:
        cfg = BackendConfig(
            name=backend_name,
            top_k=args.top_k,
            device=args.device,
            xfeat_repo=args.xfeat_repo or None,
            allow_torch_hub=args.allow_torch_hub,
            use_lightglue=args.aliked_lightglue if backend_name == "aliked" else True,
        )
        try:
            backend = create_backend(cfg)
        except BackendUnavailable as exc:
            rows.append({
                "method": backend_name,
                "status": "missing_dependency",
                "reason": str(exc),
            })
            missing.append(backend_name)
            continue

        t0 = time.perf_counter()
        timed = backend.run(left_crop, right_crop)
        total_ms = (time.perf_counter() - t0) * 1000.0
        timed.timings_ms["total"] = total_ms

        left_features = filter_feature_set_by_mask(timed.left, left_crop_mask, lambda x, y: (x, y))
        right_features = filter_feature_set_by_mask(timed.right, right_crop_mask, lambda x, y: (x, y))
        raw_matches: List[RawMatch]
        if timed.matches:
            # Direct matcher output refers to unfiltered feature indices.
            left_features = timed.left
            right_features = timed.right
            raw_matches = timed.matches
        else:
            raw_matches = match_descriptors(
                left_features,
                right_features,
                ratio=args.ratio,
                mutual=True,
                top_k=max(args.top_k, 64),
            )

        filtered = filter_stereo_matches(
            left_features,
            right_features,
            raw_matches,
            lt.to_global,
            rt.to_global,
            MatchFilterConfig(
                initial_disparity=initial_disp,
                max_y_error_px=args.max_y_error_px,
                max_disp_delta_px=args.max_disp_delta_px,
                final_disp_gate_px=args.final_disp_gate_px,
                max_disparity_px=args.max_disparity,
                min_score=args.min_score,
            ),
        )
        filtered = filter_matches_by_roi_masks(filtered, lroi.mask, rroi.mask)

        disparity = float(np.median([m.disparity for m in filtered])) if filtered else -1.0
        depth_m = probe.depth_from_disparity(disparity, focal_px, baseline_m)
        method_name = f"neural_{backend_name}"
        result = _build_match_result(
            method_name,
            left_features,
            right_features,
            filtered,
            attempted=len(raw_matches),
            depth_m=depth_m,
            timings_ms=timed.timings_ms,
            left_to_global=lt.to_global,
            right_to_global=rt.to_global,
        )
        results.append(result)

        probe.draw_matches(left_rect, right_rect, result, out_dir / f"{method_name}_matches.png")
        probe.draw_matches_zoom(left_rect, right_rect, lroi, rroi, result, out_dir / f"{method_name}_matches_zoom.png")

        tri_rows = probe.triangulate_match_rows(result, calib, baseline_m)
        tri_rows = probe.validate_triangulated_rows(
            tri_rows,
            lroi,
            rroi,
            left_overlap,
            right_overlap,
            ball_center_3d,
            0.105,
            thresholds,
        )
        probe.write_triangulated_points(out_dir / f"{method_name}_triangulated_points.csv", tri_rows)
        tri_stats = probe.triangulation_stats(tri_rows)
        validation_status, validation_fail_reasons = probe.method_validation_status(tri_stats, thresholds)
        rows.append({
            "method": method_name,
            "status": "ok",
            "backend_notes": timed.notes,
            "left_keypoints": left_features.count,
            "right_keypoints": right_features.count,
            "candidates": len(raw_matches),
            "matches": len(filtered),
            "disparity_px": result.disparity,
            "depth_m": result.depth_m,
            "confidence": result.confidence,
            "timing_ms": json.dumps(timed.timings_ms, sort_keys=True),
            "validation_status": validation_status,
            "validation_fail_reasons": validation_fail_reasons,
            **tri_stats,
        })

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

    zoom_sheets = []
    for res in results:
        img = cv2.imread(str(out_dir / f"{res.name}_matches_zoom.png"))
        if img is not None:
            zoom_sheets.append(img)
    if zoom_sheets:
        target_w = zoom_sheets[0].shape[1]
        normalized = []
        for img in zoom_sheets:
            if img.shape[1] != target_w:
                scale = target_w / img.shape[1]
                img = cv2.resize(img, (target_w, max(1, round(img.shape[0] * scale))))
            normalized.append(img)
        cv2.imwrite(str(out_dir / "zoom_contact_sheet.png"), cv2.vconcat(normalized))

    print(json.dumps(summary, indent=2))
    if missing and args.fail_on_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
