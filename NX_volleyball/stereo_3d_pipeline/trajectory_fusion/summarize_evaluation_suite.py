#!/usr/bin/env python3
"""Summarize trajectory-fusion suite outputs into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


VARIANT_JSON_KEYS = (
    ("raw", "raw_eval_json"),
    ("robust_smooth", "robust_smooth_eval_json"),
    ("robust_rts_smooth", "robust_rts_smooth_eval_json"),
    ("calibrated_smoother", "calibrated_smoother_eval_json"),
    ("calibrated_rts_smoother", "calibrated_rts_smoother_eval_json"),
    ("reliability_direct", "reliability_direct_eval_json"),
    ("reliability_smoother", "reliability_smoother_eval_json"),
    ("reliability_rts_smoother", "reliability_rts_smoother_eval_json"),
)


def _load_json(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _suite_summary_path(path: str | Path) -> Path:
    suite_path = Path(path)
    return suite_path / "suite_summary.json" if suite_path.is_dir() else suite_path


def _value(mapping: Dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    return "" if value is None else value


def _check_context(clip: Dict[str, Any]) -> Dict[str, Any]:
    check = _load_json(clip.get("check_dataset_json"))
    frame_summary = check.get("frame_summary", {})
    totals = frame_summary.get("totals", {}) if isinstance(frame_summary, dict) else {}
    return {
        "rows": check.get("rows", clip.get("check_rows", "")),
        "fps_intervals": check.get("fps_intervals", ""),
        "frame_gaps": (check.get("frame_gaps") or {}).get("count", ""),
        "direct_pair_count": totals.get("direct_pair_count", ""),
        "fallback_l2r_count": totals.get("fallback_l2r_count", ""),
        "fallback_r2l_count": totals.get("fallback_r2l_count", ""),
    }


def iter_summary_rows(suite_summary: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for clip in suite_summary.get("clips", []):
        context = _check_context(clip)
        for variant, key in VARIANT_JSON_KEYS:
            report = _load_json(clip.get(key))
            if not report:
                continue
            known_z = report.get("known_z", "")
            for track_id, track in report.get("tracks", {}).items():
                metrics = track.get("raw", {}) if variant == "raw" else track.get("smooth", {})
                if not metrics:
                    continue
                p0 = track.get("p0_median", {})
                yield {
                    "clip": clip.get("name", ""),
                    "split": clip.get("split", ""),
                    "variant": variant,
                    "track_id": track_id,
                    "known_z": "" if known_z is None else known_z,
                    "rows": context["rows"],
                    "fps_intervals": context["fps_intervals"],
                    "frame_gaps": context["frame_gaps"],
                    "direct_pair_count": context["direct_pair_count"],
                    "fallback_l2r_count": context["fallback_l2r_count"],
                    "fallback_r2l_count": context["fallback_r2l_count"],
                    "z_mean": _value(metrics, "z_mean"),
                    "z_std": _value(metrics, "z_std"),
                    "z_peak_to_peak": _value(metrics, "z_peak_to_peak"),
                    "dz_rms": _value(metrics, "dz_rms"),
                    "ddz_rms": _value(metrics, "ddz_rms"),
                    "jerk_rms": _value(metrics, "dddz_rms"),
                    "duration_s": _value(metrics, "duration_s"),
                    "fps_estimate": _value(metrics, "fps_estimate"),
                    "speed_rms_mps": _value(metrics, "speed_rms_mps"),
                    "speed_p95_mps": _value(metrics, "speed_p95_mps"),
                    "accel_z_rms_mps2": _value(metrics, "accel_z_rms_mps2"),
                    "accel_y_residual_rms_mps2": _value(metrics, "accel_y_residual_rms_mps2"),
                    "ballistic_residual_rms_mps2": _value(metrics, "ballistic_residual_rms_mps2"),
                    "motion_jerk_rms_mps3": _value(metrics, "jerk_rms_mps3"),
                    "gravity_y_used_mps2": _value(metrics, "gravity_y_used_mps2"),
                    "raw_z_std_ratio": _value(metrics, "raw_z_std_ratio"),
                    "known_z_bias": _value(metrics, "known_z_bias"),
                    "known_z_mad": _value(metrics, "known_z_mad"),
                    "p0_median_mean": _value(p0, "mean"),
                    "p0_median_mad": _value(p0, "mad"),
                    "p0_known_z_bias": _value(p0, "known_z_bias"),
                    "p0_known_z_mad": _value(p0, "known_z_mad"),
                }


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return "nan"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("clip,split,variant,track,z_mean,z_std,p2p,ratio,known_z_bias")
    for row in rows:
        print(
            "{clip},{split},{variant},{track},{mean},{std},{p2p},{ratio},{bias}".format(
                clip=row["clip"],
                split=row["split"],
                variant=row["variant"],
                track=row["track_id"],
                mean=_fmt(row["z_mean"]),
                std=_fmt(row["z_std"]),
                p2p=_fmt(row["z_peak_to_peak"]),
                ratio=_fmt(row["raw_z_std_ratio"], 3),
                bias=_fmt(row["known_z_bias"]),
            )
        )


def summarize_suite(path: str | Path, output_csv: str | Path | None = None) -> List[Dict[str, Any]]:
    summary_path = _suite_summary_path(path)
    summary = _load_json(summary_path)
    if not summary:
        raise FileNotFoundError(f"missing suite summary: {summary_path}")
    rows = list(iter_summary_rows(summary))
    if output_csv:
        write_csv(output_csv, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", help="Suite directory or suite_summary.json")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    output = args.output
    if output is None:
        summary_path = _suite_summary_path(args.suite)
        output = str(summary_path.with_name("suite_metrics.csv"))
    rows = summarize_suite(args.suite, output)
    print_table(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
