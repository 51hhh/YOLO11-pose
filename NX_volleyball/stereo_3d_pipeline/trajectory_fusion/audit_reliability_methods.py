#!/usr/bin/env python3
"""Audit ReliabilityNet per-method diagnostics from a suite or sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


AUDIT_FIELDNAMES = [
    "config",
    "variant",
    "split",
    "clip_track_count",
    "method_count",
    "frame_count",
    "top_total",
    "dominant_top_method",
    "dominant_top_count",
    "dominant_top_share",
    "low_coverage_top_count",
    "low_coverage_top_share",
    "small_sigma_low_coverage_methods",
    "large_bias_methods",
    "low_inlier_top_methods",
    "warnings",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _read_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        row.get("config") or "suite",
        row.get("variant") or "",
        row.get("split") or "",
    )


def _track_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (row.get("clip") or "", row.get("track_id") or "")


def _fmt_methods(methods: Iterable[str]) -> str:
    return ";".join(sorted(method for method in methods if method))


def audit_reliability_methods(
    path: str | Path,
    *,
    min_valid_rate: float = 0.2,
    dominant_top_share: float = 0.85,
    low_coverage_top_share: float = 0.25,
    small_sigma: float = 0.018,
    large_abs_bias: float = 0.08,
    low_inlier_prob: float = 0.25,
) -> List[Dict[str, Any]]:
    """Return risk summaries for ReliabilityNet per-method diagnostics."""

    rows = _read_rows(path)
    grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[_group_key(row)].append(row)

    audit_rows: List[Dict[str, Any]] = []
    for (config, variant, split), items in grouped_rows.items():
        method_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "valid": 0.0,
                "frames": 0.0,
                "top_count": 0.0,
                "sigma_sum": 0.0,
                "bias_sum": 0.0,
                "abs_bias_sum": 0.0,
                "inlier_sum": 0.0,
            }
        )
        frame_by_track: Dict[Tuple[str, str], float] = {}
        for row in items:
            method = row.get("method") or ""
            if not method:
                continue
            frames = _safe_float(row.get("rows"))
            valid = _safe_float(row.get("valid"))
            top_count = _safe_float(row.get("top_count"))
            sigma = _safe_float(row.get("mean_sigma"))
            bias = _safe_float(row.get("mean_bias"))
            abs_bias = _safe_float(row.get("mean_abs_bias"))
            inlier = _safe_float(row.get("mean_inlier_prob"))

            stats = method_stats[method]
            stats["valid"] += valid
            stats["frames"] += frames
            stats["top_count"] += top_count
            if valid > 0.0:
                stats["sigma_sum"] += sigma * valid
                stats["bias_sum"] += bias * valid
                stats["abs_bias_sum"] += abs_bias * valid
                stats["inlier_sum"] += inlier * valid
            track_key = _track_key(row)
            frame_by_track[track_key] = max(frame_by_track.get(track_key, 0.0), frames)

        frame_count = sum(frame_by_track.values())
        top_total = sum(stats["top_count"] for stats in method_stats.values())
        dominant_method = ""
        dominant_count = 0.0
        low_coverage_top_count = 0.0
        small_sigma_low_coverage: List[str] = []
        large_bias_methods: List[str] = []
        low_inlier_top_methods: List[str] = []

        for method, stats in method_stats.items():
            valid_rate = stats["valid"] / stats["frames"] if stats["frames"] > 0.0 else 0.0
            top_count = stats["top_count"]
            mean_sigma = stats["sigma_sum"] / stats["valid"] if stats["valid"] > 0.0 else 0.0
            mean_abs_bias = stats["abs_bias_sum"] / stats["valid"] if stats["valid"] > 0.0 else 0.0
            mean_inlier = stats["inlier_sum"] / stats["valid"] if stats["valid"] > 0.0 else 0.0
            if top_count > dominant_count:
                dominant_method = method
                dominant_count = top_count
            if valid_rate < min_valid_rate:
                low_coverage_top_count += top_count
                if mean_sigma > 0.0 and mean_sigma < small_sigma:
                    small_sigma_low_coverage.append(method)
            if mean_abs_bias > large_abs_bias:
                large_bias_methods.append(method)
            if top_count > 0.0 and mean_inlier > 0.0 and mean_inlier < low_inlier_prob:
                low_inlier_top_methods.append(method)

        dominant_share = dominant_count / top_total if top_total > 0.0 else 0.0
        low_coverage_share = low_coverage_top_count / top_total if top_total > 0.0 else 0.0
        warnings: List[str] = []
        if top_total <= 0.0 and variant != "reliability_direct":
            warnings.append("missing_top_counts")
        if dominant_share > dominant_top_share:
            warnings.append("dominant_method_top_share")
        if low_coverage_share > low_coverage_top_share:
            warnings.append("low_coverage_methods_receive_top_weight")
        if small_sigma_low_coverage:
            warnings.append("low_coverage_methods_have_tiny_sigma")
        if large_bias_methods:
            warnings.append("large_method_bias")
        if low_inlier_top_methods:
            warnings.append("low_inlier_method_receives_top_weight")

        audit_rows.append(
            {
                "config": config,
                "variant": variant,
                "split": split,
                "clip_track_count": len(frame_by_track),
                "method_count": len(method_stats),
                "frame_count": frame_count,
                "top_total": top_total,
                "dominant_top_method": dominant_method,
                "dominant_top_count": dominant_count,
                "dominant_top_share": dominant_share,
                "low_coverage_top_count": low_coverage_top_count,
                "low_coverage_top_share": low_coverage_share,
                "small_sigma_low_coverage_methods": _fmt_methods(small_sigma_low_coverage),
                "large_bias_methods": _fmt_methods(large_bias_methods),
                "low_inlier_top_methods": _fmt_methods(low_inlier_top_methods),
                "warnings": ";".join(warnings),
            }
        )

    audit_rows.sort(key=lambda row: (row["config"], row["variant"], row["split"]))
    return audit_rows


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in AUDIT_FIELDNAMES})


def write_json(path: str | Path, rows: List[Dict[str, Any]], source: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"source": str(source), "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def print_audit(rows: List[Dict[str, Any]]) -> None:
    print("config,variant,split,dominant_method,dominant_share,low_coverage_share,warnings")
    for row in rows:
        print(
            "{config},{variant},{split},{method},{dominant:.3f},{lowcov:.3f},{warnings}".format(
                config=row["config"],
                variant=row["variant"],
                split=row["split"],
                method=row["dominant_top_method"],
                dominant=float(row["dominant_top_share"]),
                lowcov=float(row["low_coverage_top_share"]),
                warnings=row["warnings"],
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods_csv", help="suite_reliability_methods.csv or sweep_reliability_methods.csv")
    parser.add_argument("-o", "--output", help="Audit CSV output")
    parser.add_argument("--json-out", help="Audit JSON output")
    parser.add_argument("--min-valid-rate", type=float, default=0.2)
    parser.add_argument("--dominant-top-share", type=float, default=0.85)
    parser.add_argument("--low-coverage-top-share", type=float, default=0.25)
    parser.add_argument("--small-sigma", type=float, default=0.018)
    parser.add_argument("--large-abs-bias", type=float, default=0.08)
    parser.add_argument("--low-inlier-prob", type=float, default=0.25)
    args = parser.parse_args()

    rows = audit_reliability_methods(
        args.methods_csv,
        min_valid_rate=args.min_valid_rate,
        dominant_top_share=args.dominant_top_share,
        low_coverage_top_share=args.low_coverage_top_share,
        small_sigma=args.small_sigma,
        large_abs_bias=args.large_abs_bias,
        low_inlier_prob=args.low_inlier_prob,
    )
    output = args.output or str(Path(args.methods_csv).with_name("reliability_method_audit.csv"))
    write_csv(output, rows)
    if args.json_out:
        write_json(args.json_out, rows, args.methods_csv)
    print_audit(rows)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
