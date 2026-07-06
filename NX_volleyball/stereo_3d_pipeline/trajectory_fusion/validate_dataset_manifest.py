#!/usr/bin/env python3
"""Validate a trajectory dataset manifest before training/evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from .check_dataset import P0_DEPTH_KEYS, P1_DEPTH_KEYS, analyze_dataset
    from .manifest import load_manifest
except ImportError:  # pragma: no cover - direct script execution
    from check_dataset import P0_DEPTH_KEYS, P1_DEPTH_KEYS, analyze_dataset
    from manifest import load_manifest


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _depth_hit(report: Dict[str, Any], key: str) -> float:
    return float((report.get("depth", {}).get(key) or {}).get("hit_rate", 0.0) or 0.0)


def _min_hit(report: Dict[str, Any], keys: tuple[str, ...]) -> float:
    values = [_depth_hit(report, key) for key in keys if key in report.get("depth", {})]
    return min(values) if values else 0.0


def _frame_summary_totals(report: Dict[str, Any]) -> Dict[str, Any]:
    frame_summary = report.get("frame_summary", {})
    return frame_summary.get("totals", {}) if isinstance(frame_summary, dict) else {}


def _clip_warnings(
    clip_report: Dict[str, Any],
    *,
    min_rows: int,
    min_fps: float,
    min_p0_hit: float,
    max_frame_gaps: int,
) -> List[str]:
    warnings: List[str] = []
    if clip_report["missing"]:
        warnings.append("missing_file")
        return warnings
    if clip_report["rows"] < min_rows:
        warnings.append(f"rows<{min_rows}")
    fps = _safe_float(clip_report.get("fps_intervals"), None)
    if fps is None or fps < min_fps:
        warnings.append(f"fps<{min_fps:g}")
    if clip_report["frame_gaps"] > max_frame_gaps:
        warnings.append(f"frame_gaps>{max_frame_gaps}")
    if clip_report["missing_fields"]:
        warnings.append("missing_required_fields")
    if clip_report["watermark_nonzero"]:
        warnings.append("watermark_delta_nonzero")
    if clip_report["p0_min_hit_rate"] < min_p0_hit:
        warnings.append(f"p0_hit<{min_p0_hit:.2f}")
    if clip_report["known_z"] is not None and clip_report["split"] not in {"train", "val", "test"}:
        warnings.append("known_z_clip_has_nonstandard_split")
    return warnings


def analyze_manifest(
    manifest_path: str | Path,
    *,
    min_rows: int = 100,
    min_fps: float = 80.0,
    min_p0_hit: float = 0.85,
    max_frame_gaps: int = 0,
    require_stratified_known_z: bool = False,
) -> Dict[str, Any]:
    clips = load_manifest(manifest_path)
    clip_reports: List[Dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    known_by_split: Counter[str] = Counter()
    warnings_by_kind: Counter[str] = Counter()
    known_z_values: Dict[str, List[float]] = defaultdict(list)
    known_z_bucket_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for clip in clips:
        split_counts[clip.split] += 1
        missing = not clip.csv.exists()
        if missing:
            clip_report: Dict[str, Any] = {
                "name": clip.name,
                "split": clip.split,
                "csv": str(clip.csv),
                "metadata": str(clip.metadata) if clip.metadata else None,
                "missing": True,
                "rows": 0,
                "fps_intervals": None,
                "frame_gaps": 0,
                "missing_fields": [],
                "watermark_nonzero": False,
                "known_z": None,
                "p0_min_hit_rate": 0.0,
                "p1_min_hit_rate": 0.0,
                "direct_pair_count": 0,
                "fallback_l2r_count": 0,
                "fallback_r2l_count": 0,
            }
        else:
            report = analyze_dataset(clip.csv, metadata_path=clip.metadata)
            known_z = _safe_float(report.get("known_z"), None)
            if known_z is not None and known_z > 0.0:
                known_by_split[clip.split] += 1
                known_z_values[clip.split].append(known_z)
                known_z_bucket_counts[f"{known_z:.3f}"][clip.split] += 1
            totals = _frame_summary_totals(report)
            watermarks = report.get("watermarks", {})
            watermark_nonzero = any(
                bool((stats or {}).get("nonzero", 0))
                for stats in watermarks.values()
                if isinstance(stats, dict)
            )
            clip_report = {
                "name": clip.name,
                "split": clip.split,
                "csv": str(clip.csv),
                "metadata": report.get("metadata"),
                "missing": False,
                "rows": report.get("rows", 0),
                "duration_sec": report.get("duration_sec"),
                "fps_intervals": report.get("fps_intervals"),
                "frame_gaps": (report.get("frame_gaps") or {}).get("count", 0),
                "missing_fields": report.get("missing_fields", []),
                "watermark_nonzero": watermark_nonzero,
                "known_z": known_z,
                "p0_min_hit_rate": _min_hit(report, P0_DEPTH_KEYS),
                "p1_min_hit_rate": _min_hit(report, P1_DEPTH_KEYS),
                "direct_pair_count": totals.get("direct_pair_count", 0),
                "fallback_l2r_count": totals.get("fallback_l2r_count", 0),
                "fallback_r2l_count": totals.get("fallback_r2l_count", 0),
            }
        warnings = _clip_warnings(
            clip_report,
            min_rows=min_rows,
            min_fps=min_fps,
            min_p0_hit=min_p0_hit,
            max_frame_gaps=max_frame_gaps,
        )
        clip_report["warnings"] = warnings
        warnings_by_kind.update(warnings)
        clip_reports.append(clip_report)

    dataset_warnings: List[str] = []
    if not split_counts:
        dataset_warnings.append("empty_manifest")
    if split_counts and split_counts.get("val", 0) == 0:
        dataset_warnings.append("missing_val_split")
    if sum(known_by_split.values()) == 0:
        dataset_warnings.append("missing_known_z_clips")
    if known_by_split and known_by_split.get("val", 0) == 0:
        dataset_warnings.append("missing_known_z_val_split")
    known_z_bucket_warnings: List[Dict[str, Any]] = []
    if require_stratified_known_z:
        for bucket, counts in sorted(known_z_bucket_counts.items()):
            for split in ("train", "val"):
                if counts.get(split, 0) > 0:
                    continue
                warning = f"known_z_bucket_missing_{split}_split"
                dataset_warnings.append(warning)
                known_z_bucket_warnings.append(
                    {
                        "known_z_bucket": bucket,
                        "missing_split": split,
                        "counts": dict(counts),
                    }
                )
    warnings_by_kind.update(dataset_warnings)

    return {
        "manifest": str(manifest_path),
        "clip_count": len(clips),
        "split_counts": dict(split_counts),
        "known_z_counts": dict(known_by_split),
        "known_z_values": {key: sorted(values) for key, values in known_z_values.items()},
        "known_z_bucket_counts": {
            bucket: dict(counts) for bucket, counts in sorted(known_z_bucket_counts.items())
        },
        "known_z_bucket_warnings": known_z_bucket_warnings,
        "warnings": dataset_warnings,
        "warning_counts": dict(warnings_by_kind),
        "clips": clip_reports,
        "thresholds": {
            "min_rows": min_rows,
            "min_fps": min_fps,
            "min_p0_hit": min_p0_hit,
            "max_frame_gaps": max_frame_gaps,
            "require_stratified_known_z": require_stratified_known_z,
        },
    }


def _write_json(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    print(f"manifest={report['manifest']}")
    print(f"clips={report['clip_count']} splits={report['split_counts']} known_z={report['known_z_counts']}")
    if report.get("known_z_bucket_counts"):
        print(f"known_z_buckets={report['known_z_bucket_counts']}")
    print(f"warnings={report['warning_counts']}")
    for clip in report["clips"]:
        print(
            "clip={name} split={split} rows={rows} fps={fps} gaps={gaps} "
            "p0_min_hit={p0:.3f} known_z={known_z} warnings={warnings}".format(
                name=clip["name"],
                split=clip["split"],
                rows=clip["rows"],
                fps=clip.get("fps_intervals"),
                gaps=clip["frame_gaps"],
                p0=clip["p0_min_hit_rate"],
                known_z=clip["known_z"],
                warnings=clip["warnings"],
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--json-out")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-fps", type=float, default=80.0)
    parser.add_argument("--min-p0-hit", type=float, default=0.85)
    parser.add_argument("--max-frame-gaps", type=int, default=0)
    parser.add_argument(
        "--require-stratified-known-z",
        action="store_true",
        help="Warn if any known_z bucket lacks train or val coverage.",
    )
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    report = analyze_manifest(
        args.manifest,
        min_rows=args.min_rows,
        min_fps=args.min_fps,
        min_p0_hit=args.min_p0_hit,
        max_frame_gaps=args.max_frame_gaps,
        require_stratified_known_z=args.require_stratified_known_z,
    )
    print_report(report)
    if args.json_out:
        _write_json(args.json_out, report)
    return 1 if args.fail_on_warning and report["warning_counts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
