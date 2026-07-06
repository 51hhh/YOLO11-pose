#!/usr/bin/env python3
"""Select ReliabilityNet candidates by combining metric ranking and method-risk audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .rank_sweep_metrics import rank_metrics
except ImportError:  # pragma: no cover - direct script execution
    from rank_sweep_metrics import rank_metrics


SEVERE_WARNINGS = {
    "dominant_method_top_share",
    "low_coverage_methods_receive_top_weight",
    "low_coverage_methods_have_tiny_sigma",
    "low_inlier_method_receives_top_weight",
}
CAUTION_WARNINGS = {
    "large_method_bias",
    "missing_top_counts",
}
SELECTION_FIELDNAMES = [
    "selection_rank",
    "decision",
    "decision_reason",
    "metric_rank",
    "config",
    "checkpoint",
    "suite_dir",
    "variant",
    "split",
    "score",
    "known_clip_count",
    "mean_abs_known_z_bias",
    "mean_known_z_mad",
    "mean_z_std",
    "mean_z_peak_to_peak",
    "mean_ballistic_residual_rms_mps2",
    "mean_accel_z_rms_mps2",
    "audit_warnings",
    "dominant_top_method",
    "dominant_top_share",
    "low_coverage_top_share",
]


def _read_csv(path: str | Path | None) -> List[Dict[str, str]]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_warnings(value: object) -> set[str]:
    return {item for item in str(value or "").split(";") if item}


def _audit_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        row.get("config") or "suite",
        row.get("variant") or "",
        row.get("split") or "",
    )


def _ranking_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("config") or "suite"),
        str(row.get("variant") or ""),
        str(row.get("split") or ""),
    )


def _default_audit_path(metrics_csv: str | Path) -> Path:
    return Path(metrics_csv).with_name("sweep_reliability_method_audit.csv")


def _aggregate_audit_rows(rows: List[Dict[str, str]]) -> Dict[str, str] | None:
    if not rows:
        return None
    warning_set: set[str] = set()
    dominant_counts: Dict[str, float] = {}
    top_total = 0.0
    low_coverage_top_count = 0.0
    for row in rows:
        warning_set.update(_split_warnings(row.get("warnings")))
        method = row.get("dominant_top_method") or ""
        dominant_count = _safe_float(row.get("dominant_top_count"))
        if method:
            dominant_counts[method] = dominant_counts.get(method, 0.0) + dominant_count
        top_total += _safe_float(row.get("top_total"))
        low_coverage_top_count += _safe_float(row.get("low_coverage_top_count"))
    dominant_method = ""
    dominant_count = 0.0
    if dominant_counts:
        dominant_method, dominant_count = max(dominant_counts.items(), key=lambda item: item[1])
    return {
        "config": rows[0].get("config", ""),
        "variant": rows[0].get("variant", ""),
        "split": "all",
        "warnings": ";".join(sorted(warning_set)),
        "dominant_top_method": dominant_method,
        "dominant_top_share": _fmt_ratio(dominant_count, top_total),
        "low_coverage_top_share": _fmt_ratio(low_coverage_top_count, top_total),
    }


def _safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return 0.0
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0.0:
        return "0.0"
    return f"{numerator / denominator:.12g}"


def _decision_for_audit(audit: Dict[str, str] | None) -> Tuple[str, str]:
    if audit is None:
        return "caution", "missing_method_audit"
    warnings = _split_warnings(audit.get("warnings"))
    severe = sorted(warnings & SEVERE_WARNINGS)
    caution = sorted(warnings & CAUTION_WARNINGS)
    if severe:
        return "reject", ";".join(severe)
    if caution:
        return "caution", ";".join(caution)
    return "recommended", ""


def select_reliability_models(
    metrics_csv: str | Path,
    *,
    audit_csv: str | Path | None = None,
    variant: str = "reliability_smoother",
    split: str | None = "auto",
) -> List[Dict[str, Any]]:
    """Return ranked candidates with audit-derived decisions."""

    rankings = rank_metrics(metrics_csv, variant=variant, split=split)
    audit_path = Path(audit_csv) if audit_csv else _default_audit_path(metrics_csv)
    audit_rows = _read_csv(audit_path)
    audit_by_key = {_audit_key(row): row for row in audit_rows}
    audit_all_by_key: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    grouped_audits: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in audit_rows:
        key = (row.get("config") or "suite", row.get("variant") or "")
        grouped_audits.setdefault(key, []).append(row)
    for (config, audit_variant), rows in grouped_audits.items():
        aggregate = _aggregate_audit_rows(rows)
        if aggregate is not None:
            audit_all_by_key[(config, audit_variant, "all")] = aggregate

    selected: List[Dict[str, Any]] = []
    for ranking in rankings:
        ranking_key = _ranking_key(ranking)
        audit = audit_by_key.get(ranking_key)
        if audit is None and ranking_key[2] == "all":
            audit = audit_all_by_key.get(ranking_key)
        decision, reason = _decision_for_audit(audit)
        selected.append(
            {
                "selection_rank": 0,
                "decision": decision,
                "decision_reason": reason,
                "metric_rank": ranking.get("rank", ""),
                "config": ranking.get("config", ""),
                "checkpoint": ranking.get("checkpoint", ""),
                "suite_dir": ranking.get("suite_dir", ""),
                "variant": ranking.get("variant", ""),
                "split": ranking.get("split", ""),
                "score": ranking.get("score", ""),
                "known_clip_count": ranking.get("known_clip_count", ""),
                "mean_abs_known_z_bias": ranking.get("mean_abs_known_z_bias", ""),
                "mean_known_z_mad": ranking.get("mean_known_z_mad", ""),
                "mean_z_std": ranking.get("mean_z_std", ""),
                "mean_z_peak_to_peak": ranking.get("mean_z_peak_to_peak", ""),
                "mean_ballistic_residual_rms_mps2": ranking.get("mean_ballistic_residual_rms_mps2", ""),
                "mean_accel_z_rms_mps2": ranking.get("mean_accel_z_rms_mps2", ""),
                "audit_warnings": "" if audit is None else audit.get("warnings", ""),
                "dominant_top_method": "" if audit is None else audit.get("dominant_top_method", ""),
                "dominant_top_share": "" if audit is None else audit.get("dominant_top_share", ""),
                "low_coverage_top_share": "" if audit is None else audit.get("low_coverage_top_share", ""),
            }
        )

    decision_order = {"recommended": 0, "caution": 1, "reject": 2}
    selected.sort(
        key=lambda row: (
            decision_order.get(str(row["decision"]), 99),
            int(row["metric_rank"] or 999999),
        )
    )
    for index, row in enumerate(selected, start=1):
        row["selection_rank"] = index
    return selected


def write_selection(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in SELECTION_FIELDNAMES})


def write_json(path: str | Path, rows: List[Dict[str, Any]], metrics_csv: str | Path, audit_csv: str | Path | None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metrics_csv": str(metrics_csv),
                "audit_csv": str(audit_csv) if audit_csv else str(_default_audit_path(metrics_csv)),
                "recommended": [row for row in rows if row["decision"] == "recommended"],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def print_selection(rows: List[Dict[str, Any]], limit: int = 10) -> None:
    print("selection_rank,decision,metric_rank,config,variant,score,warnings")
    for row in rows[:limit]:
        print(
            "{selection_rank},{decision},{metric_rank},{config},{variant},{score},{warnings}".format(
                selection_rank=row["selection_rank"],
                decision=row["decision"],
                metric_rank=row["metric_rank"],
                config=row["config"],
                variant=row["variant"],
                score=row["score"],
                warnings=row["audit_warnings"],
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", help="sweep_metrics.csv")
    parser.add_argument("--audit", help="sweep_reliability_method_audit.csv; defaults next to metrics")
    parser.add_argument("-o", "--output", help="Selection CSV output")
    parser.add_argument("--json-out", help="Selection JSON output")
    parser.add_argument("--variant", default="reliability_smoother")
    parser.add_argument("--split", default="auto")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = select_reliability_models(
        args.metrics_csv,
        audit_csv=args.audit,
        variant=args.variant,
        split=args.split,
    )
    output = args.output or str(Path(args.metrics_csv).with_name("sweep_model_selection.csv"))
    write_selection(output, rows)
    if args.json_out:
        write_json(args.json_out, rows, args.metrics_csv, args.audit)
    print_selection(rows, limit=args.limit)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
