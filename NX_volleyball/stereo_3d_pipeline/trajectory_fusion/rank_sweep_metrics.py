#!/usr/bin/env python3
"""Rank sweep configurations or variant baselines from sweep_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


BASELINE_VARIANTS = {
    "raw",
    "robust_smooth",
    "robust_rts_smooth",
    "calibrated_smoother",
    "calibrated_rts_smoother",
}


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _read_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return mean(values) if values else None


def _score(row: Dict[str, Any]) -> float:
    known_terms = []
    if row["known_clip_count"] > 0:
        if row["mean_abs_known_z_bias"] is not None:
            known_terms.append(row["mean_abs_known_z_bias"])
        if row["mean_known_z_mad"] is not None:
            known_terms.append(row["mean_known_z_mad"])
    stability = 0.0
    if row["mean_z_std"] is not None:
        stability += row["mean_z_std"]
    if row["mean_z_peak_to_peak"] is not None:
        stability += 0.25 * row["mean_z_peak_to_peak"]
    if known_terms:
        return sum(known_terms) + 0.25 * stability
    return stability


def rank_metrics(
    metrics_csv: str | Path,
    *,
    variant: str = "reliability_smoother",
    split: str | None = "auto",
) -> List[Dict[str, Any]]:
    include_all_variants = variant == "all"
    rows = _read_rows(metrics_csv)
    if not include_all_variants:
        rows = [row for row in rows if row.get("variant") == variant]
    selected_split = None
    if split == "auto":
        splits = {row.get("split", "") for row in rows}
        selected_split = "val" if "val" in splits else None
    elif split and split != "all":
        selected_split = split
    if selected_split is not None:
        rows = [row for row in rows if row.get("split", "") == selected_split]

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_baselines: set[tuple[str, str, str, str]] = set()
    for row in rows:
        row_variant = row.get("variant", "")
        if include_all_variants and row_variant in BASELINE_VARIANTS:
            dedupe_key = (
                row_variant,
                row.get("split", ""),
                row.get("clip", ""),
                row.get("track_id", ""),
            )
            if dedupe_key in seen_baselines:
                continue
            seen_baselines.add(dedupe_key)
            group_key = f"baseline:{row_variant}"
        elif include_all_variants:
            group_key = f"{row.get('config', '')}:{row_variant}"
        else:
            group_key = row.get("config", "")
        grouped[group_key].append(row)

    ranked: List[Dict[str, Any]] = []
    for group_key, items in grouped.items():
        row_variant = items[0].get("variant", variant)
        config = "baseline" if row_variant in BASELINE_VARIANTS and include_all_variants else items[0].get("config", "")
        z_std = [_safe_float(row.get("z_std")) for row in items]
        z_p2p = [_safe_float(row.get("z_peak_to_peak")) for row in items]
        bias = [_safe_float(row.get("known_z_bias")) for row in items]
        mad = [_safe_float(row.get("known_z_mad")) for row in items]
        z_std_values = [value for value in z_std if value is not None]
        z_p2p_values = [value for value in z_p2p if value is not None]
        bias_values = [abs(value) for value in bias if value is not None]
        mad_values = [value for value in mad if value is not None]
        ranked_row: Dict[str, Any] = {
            "config": config,
            "checkpoint": items[0].get("checkpoint", ""),
            "suite_dir": items[0].get("suite_dir", ""),
            "variant": row_variant,
            "split": selected_split or "all",
            "clip_count": len(items),
            "known_clip_count": len(bias_values),
            "mean_z_std": _mean(z_std_values),
            "mean_z_peak_to_peak": _mean(z_p2p_values),
            "mean_abs_known_z_bias": _mean(bias_values),
            "mean_known_z_mad": _mean(mad_values),
        }
        ranked_row["score"] = _score(ranked_row)
        ranked.append(ranked_row)

    ranked.sort(key=lambda row: (row["score"], row["variant"], row["config"]))
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def write_ranking(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "config",
        "checkpoint",
        "suite_dir",
        "variant",
        "split",
        "score",
        "clip_count",
        "known_clip_count",
        "mean_abs_known_z_bias",
        "mean_known_z_mad",
        "mean_z_std",
        "mean_z_peak_to_peak",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "nan"
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def print_ranking(rows: List[Dict[str, Any]], limit: int = 10) -> None:
    print("rank,config,variant,split,score,known_bias,known_mad,z_std,p2p")
    for row in rows[:limit]:
        print(
            "{rank},{config},{variant},{split},{score},{bias},{mad},{std},{p2p}".format(
                rank=row["rank"],
                config=row["config"],
                variant=row["variant"],
                split=row["split"],
                score=_fmt(row["score"]),
                bias=_fmt(row["mean_abs_known_z_bias"]),
                mad=_fmt(row["mean_known_z_mad"]),
                std=_fmt(row["mean_z_std"]),
                p2p=_fmt(row["mean_z_peak_to_peak"]),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", help="sweep_metrics.csv")
    parser.add_argument("-o", "--output")
    parser.add_argument("--variant", default="reliability_smoother", help="Variant to rank, or 'all' to compare variants.")
    parser.add_argument("--split", default="auto", help="Use a split for ranking. 'auto' prefers val when present.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = rank_metrics(args.metrics_csv, variant=args.variant, split=args.split)
    output = args.output or str(Path(args.metrics_csv).with_name("sweep_ranking.csv"))
    write_ranking(output, rows)
    print_ranking(rows, limit=args.limit)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
