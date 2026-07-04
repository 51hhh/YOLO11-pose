"""Backend execution helpers for the offline neural feature probe."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from stereo_feature_matching.probe_utils import filter_matches_by_roi_masks
from neural_feature_probe_outputs import build_match_result, write_zoom_contact_sheet


def evaluate_neural_backend(
    backend_name: str,
    args,
    out_dir: Path,
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    lroi: probe.BallROI,
    rroi: probe.BallROI,
    left_crop: np.ndarray,
    right_crop: np.ndarray,
    left_crop_mask: np.ndarray,
    right_crop_mask: np.ndarray,
    left_transform,
    right_transform,
    calib: Dict[str, object],
    focal_px: float,
    baseline_m: float,
    initial_disp: float,
    left_overlap: np.ndarray,
    right_overlap: np.ndarray,
    ball_center_3d: Tuple[float, float, float],
    thresholds: probe.ValidationThresholds,
) -> Tuple[Dict[str, object], probe.MatchResult | None, str | None]:
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
        return {
            "method": backend_name,
            "status": "missing_dependency",
            "reason": str(exc),
        }, None, backend_name

    t0 = time.perf_counter()
    timed = backend.run(left_crop, right_crop)
    total_ms = (time.perf_counter() - t0) * 1000.0
    timed.timings_ms["total"] = total_ms

    left_features = filter_feature_set_by_mask(
        timed.left, left_crop_mask, lambda x, y: (x, y)
    )
    right_features = filter_feature_set_by_mask(
        timed.right, right_crop_mask, lambda x, y: (x, y)
    )
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
        left_transform.to_global,
        right_transform.to_global,
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
    result = build_match_result(
        method_name,
        left_features,
        right_features,
        filtered,
        attempted=len(raw_matches),
        depth_m=depth_m,
        timings_ms=timed.timings_ms,
        left_to_global=left_transform.to_global,
        right_to_global=right_transform.to_global,
    )

    probe.draw_matches(left_rect, right_rect, result, out_dir / f"{method_name}_matches.png")
    probe.draw_matches_zoom(
        left_rect,
        right_rect,
        lroi,
        rroi,
        result,
        out_dir / f"{method_name}_matches_zoom.png",
    )

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
    return {
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
    }, result, None
