#!/usr/bin/env python3
"""Summarize P0/P1 depth candidate coverage from recorded trajectory CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from typing import Iterable


DEPTH_FIELDS = [
    "z_bbox_center",
    "z_circle_center",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
    "z_roi_center_patch",
    "z_roi_multi_point",
    "z_roi_cuda_template_match",
    "z_roi_neural_xfeat",
    "z_roi_neural_superpoint",
    "z_roi_neural_aliked",
]

QUALITY_FIELDS = [
    "p0p1_dy_center",
    "p0p1_dy_mad",
    "p0p1_dy_sample_count",
    "p0p1_untrusted_mask",
    "subpixel_attempted",
    "subpixel_valid",
    "subpixel_support",
    "roi_cuda_template_match_support",
    "roi_neural_xfeat_support",
    "roi_neural_superpoint_support",
    "roi_neural_aliked_support",
]

P1_FIELDS = [
    "z_roi_center_patch",
    "z_roi_multi_point",
    "z_roi_cuda_template_match",
]

TRUST_FIELDS = [
    "p0p1_bbox_center_trust",
    "p0p1_circle_center_trust",
    "p0p1_edge_centroid_trust",
    "p0p1_radial_center_trust",
    "p0p1_edge_pair_center_trust",
    "p0p1_center_patch_trust",
    "p0p1_multi_point_trust",
    "p0p1_cuda_template_match_trust",
    "p0p1_neural_xfeat_trust",
]

CIRCLE_SOURCE_NAMES = {
    1: "bbox_proxy",
    2: "roi_fit",
    3: "epipolar_search",
    4: "template_search",
    5: "feature_proxy",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def valid_depth_values(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if math.isfinite(value) and value > 0.0:
            values.append(value)
    return values


def valid_depth_count(rows: Iterable[dict[str, str]], field: str) -> int:
    return len(valid_depth_values(rows, field))


def finite_values(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if math.isfinite(value):
            values.append(value)
    return values


def mad(values: list[float]) -> float:
    if not values:
        return math.nan
    center = median(values)
    return median([abs(value - center) for value in values])


def safe_int(value: str | None) -> int | None:
    number = safe_float(value)
    if not math.isfinite(number):
        return None
    return int(number)


def circle_source_name(value: int | None) -> str:
    if value is None:
        return "missing"
    return CIRCLE_SOURCE_NAMES.get(value, f"source_{value}")


def print_circle_source_summary(rows: list[dict[str, str]]) -> None:
    if not rows or "left_circle_source" not in rows[0] or "right_circle_source" not in rows[0]:
        return

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        left_source = safe_int(row.get("left_circle_source"))
        right_source = safe_int(row.get("right_circle_source"))
        pair_name = f"{circle_source_name(left_source)}/{circle_source_name(right_source)}"
        groups.setdefault(pair_name, []).append(row)

    print("\nCircle source groups:")
    header_fields = [
        "group",
        "rows",
        "z_circle_rate",
        "bbox_rate",
        "center_patch_rate",
        "multi_point_rate",
        "cuda_template_rate",
    ]
    print(",".join(header_fields))
    for group, group_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        total = len(group_rows)
        def rate(field: str) -> float:
            if field not in group_rows[0] or total == 0:
                return math.nan
            return 100.0 * valid_depth_count(group_rows, field) / total

        print(
            f"{group},{total},"
            f"{rate('z_circle_center'):.1f},"
            f"{rate('z_bbox_center'):.1f},"
            f"{rate('z_roi_center_patch'):.1f},"
            f"{rate('z_roi_multi_point'):.1f},"
            f"{rate('z_roi_cuda_template_match'):.1f}"
        )

    non_roi_rows = [
        row for row in rows
        if safe_int(row.get("left_circle_source")) != 2 or
           safe_int(row.get("right_circle_source")) != 2
    ]
    if non_roi_rows:
        total = len(non_roi_rows)
        print("\nNon-ROI circle-source P1 signal:")
        bbox_count = (
            valid_depth_count(non_roi_rows, "z_bbox_center")
            if "z_bbox_center" in rows[0] else 0
        )
        best_p1_count = 0
        for field in P1_FIELDS:
            if field not in rows[0]:
                continue
            count = valid_depth_count(non_roi_rows, field)
            best_p1_count = max(best_p1_count, count)
            print(f"{field}: {count}/{total} ({100.0 * count / total:.1f}%)")
        if total > 0:
            bbox_rate = 100.0 * bbox_count / total
            best_p1_rate = 100.0 * best_p1_count / total
            if bbox_rate > 80.0 and best_p1_rate < 20.0:
                print(
                    "Signal: non-ROI circle-source frames have high bbox coverage "
                    f"({bbox_rate:.1f}%) but low P1 coverage "
                    f"({best_p1_rate:.1f}%). Check P1 GPU bbox-fallback seed path."
                )


def companion(path: Path, suffix: str) -> Path:
    if path.name.endswith(".csv"):
        return path.with_name(path.name[:-4] + suffix)
    return Path(str(path) + suffix)


def print_depth_summary(path: Path, rows: list[dict[str, str]]) -> None:
    total = len(rows)
    print(f"\n== {path} ==")
    print(f"rows: {total}")
    if not rows:
        return
    print("\nDepth fields:")
    print("field,valid,total,rate,median_m,mad_m")
    for field in DEPTH_FIELDS:
        if field not in rows[0]:
            continue
        values = valid_depth_values(rows, field)
        rate = 100.0 * len(values) / total if total else 0.0
        med = median(values) if values else math.nan
        field_mad = mad(values)
        print(
            f"{field},{len(values)},{total},{rate:.1f},"
            f"{med:.4f},{field_mad:.4f}"
        )

    print("\nQuality fields:")
    print("field,count,median,mad,min,max")
    for field in QUALITY_FIELDS + TRUST_FIELDS:
        if field not in rows[0]:
            continue
        values = finite_values(rows, field)
        if not values:
            print(f"{field},0,,,,")
            continue
        print(
            f"{field},{len(values)},{median(values):.4f},{mad(values):.4f},"
            f"{min(values):.4f},{max(values):.4f}"
        )

    dy_values = finite_values(rows, "p0p1_dy_center")
    p0_values = valid_depth_values(rows, "z_bbox_center")
    p1_values = [
        len(valid_depth_values(rows, field))
        for field in ("z_roi_center_patch", "z_roi_multi_point", "z_roi_cuda_template_match")
        if field in rows[0]
    ]
    if dy_values and p0_values and p1_values:
        p1_best = max(p1_values)
        p1_best_rate = 100.0 * p1_best / total if total else 0.0
        p0_rate = 100.0 * len(p0_values) / total if total else 0.0
        if abs(median(dy_values)) > 3.0 and p0_rate > 80.0 and p1_best_rate < 80.0:
            print(
                "\nSignal: P0 is mostly valid but P1 is low while "
                f"median p0p1_dy_center={median(dy_values):.2f}px. "
                "Check per-pair signed y prior in P1 search."
            )

    print_circle_source_summary(rows)


def print_sidecar_summary(path: Path) -> None:
    sidecar = companion(path, ".p2_diagnostic.csv")
    if not sidecar.exists():
        return
    rows = read_csv(sidecar)
    print(f"\nP2 sidecar: {sidecar}")
    print(f"rows: {len(rows)}")
    if not rows:
        return
    by_mode: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_mode.setdefault(row.get("mode", ""), []).append(row)
    print("mode,valid,total,rate,median_z,mad_z,median_algo_ms,p95_algo_ms")
    for mode in sorted(by_mode):
        mode_rows = by_mode[mode]
        valid_rows = [
            row for row in mode_rows
            if row.get("valid") in ("1", "true", "True")
        ]
        z_values = valid_depth_values(valid_rows, "z_m")
        algo_values = finite_values(mode_rows, "algo_ms")
        algo_values.sort()
        p95 = (
            algo_values[min(len(algo_values) - 1, int(0.95 * (len(algo_values) - 1)))]
            if algo_values else math.nan
        )
        rate = 100.0 * len(valid_rows) / len(mode_rows) if mode_rows else 0.0
        print(
            f"{mode},{len(valid_rows)},{len(mode_rows)},{rate:.1f},"
            f"{(median(z_values) if z_values else math.nan):.4f},"
            f"{mad(z_values):.4f},"
            f"{(median(algo_values) if algo_values else math.nan):.4f},"
            f"{p95:.4f}"
        )


def print_frames_summary(path: Path) -> None:
    frames = companion(path, ".frames.csv")
    if not frames.exists():
        return
    rows = read_csv(frames)
    print(f"\nFrames sidecar: {frames}")
    print(f"rows: {len(rows)}")
    if not rows:
        return
    header = rows[0].keys()
    for field in (
        "roi_outputs",
        "stale_roi_frames",
        "async_roi_over_deadline",
        "dropped_roi_frames",
        "subpixel_attempted",
        "subpixel_valid",
    ):
        if field not in header:
            continue
        values = finite_values(rows, field)
        if values:
            print(f"{field}: median={median(values):.4f} max={max(values):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze P0/P1 depth candidate coverage in trajectory CSVs."
    )
    parser.add_argument("csv", nargs="+", type=Path, help="trajectory CSV path")
    args = parser.parse_args()

    for path in args.csv:
        rows = read_csv(path)
        print_depth_summary(path, rows)
        print_sidecar_summary(path)
        print_frames_summary(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
