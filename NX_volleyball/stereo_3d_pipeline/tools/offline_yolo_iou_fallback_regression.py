#!/usr/bin/env python3
"""Offline regression tests for YOLO ROI pairing and single-side fallback.

The tool reads a baseline clip with frames.csv, uses the recorded YOLO boxes as
ground truth proxies, then simulates degraded cases:

- normal left/right YOLO pairing;
- fake same-y/same-size right detections at wrong disparities;
- fake same-y/same-size left detections at wrong disparities;
- right missing: left ROI searches the right epipolar band;
- left missing: right ROI searches the left epipolar band.

It is intended to catch ROI/IoU baseline search drift before enabling a method
in the realtime pipeline.
"""

from __future__ import annotations

import json
from typing import Dict, List

from offline_yolo_iou_config import build_bbox_prior, build_pair_gate, parse_args, parse_fake_scales
from offline_yolo_iou_inputs import (
    load_baseline_from_calib,
    read_rows,
)
from offline_yolo_iou_regression_cases import evaluate_regression_row
from offline_yolo_iou_report import build_summary, write_regression_outputs


def main() -> int:
    args = parse_args()
    baseline_m = load_baseline_from_calib(args.calib)
    pair_gate = build_pair_gate(args)
    prior = build_bbox_prior(args)
    fake_scales = parse_fake_scales(args.fake_disparity_scales)

    rows_in = read_rows(args.clip / "frames.csv", args.max_frames)
    args.out.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, object]] = []

    for row in rows_in:
        evaluated = evaluate_regression_row(
            row,
            args.clip,
            args,
            baseline_m,
            pair_gate,
            prior,
            fake_scales,
        )
        if evaluated is not None:
            out_rows.append(evaluated)

    summary = {
        "clip": str(args.clip),
        "calibration": str(args.calib),
        "baseline_m": baseline_m,
        "assignment": "global_score_one_to_one",
        "pair_gate": {
            "max_disparity": pair_gate.max_disparity,
            "epipolar_y_tolerance": pair_gate.epipolar_y_tolerance,
            "max_size_ratio": pair_gate.max_size_ratio,
            "min_shifted_iou": pair_gate.min_shifted_iou,
        },
        "bbox_prior": {
            "object_diameter_m": prior.object_diameter_m,
            "bbox_scale": prior.bbox_scale,
            "consistency_ratio": prior.consistency_ratio,
            "consistency_min_px": prior.consistency_min_px,
            "penalty_scale": prior.penalty_scale,
        },
        "thresholds": {
            "max_center_error_px": args.max_center_error_px,
            "max_y_error_px": args.max_y_error_px,
            "max_disparity_error_px": args.max_disparity_error_px,
            "template_min_score": args.template_min_score,
            "template_min_score_gap": args.template_min_score_gap,
            "template_peak_exclusion_radius": args.template_peak_exclusion_radius,
        },
        "metrics": build_summary(out_rows),
    }
    write_regression_outputs(args.out, out_rows, summary, args.clip)
    print(json.dumps(summary, indent=2))
    if args.fail_on_regression:
        metrics = summary["metrics"]
        required_rates = [
            "normal_pair_pass_rate",
            "fake_right_low_selected_true_rate",
            "fake_right_high_selected_true_rate",
            "fake_left_low_selected_true_rate",
            "fake_left_high_selected_true_rate",
            "right_missing_pass_rate",
            "left_missing_pass_rate",
        ]
        if any(float(metrics[name]) < args.min_pass_rate for name in required_rates):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
