#!/usr/bin/env python3
"""Compare multiple trajectory workflow outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .summarize_workflow import build_workflow_report
except ImportError:  # pragma: no cover - direct script execution
    from summarize_workflow import build_workflow_report


FIELDNAMES = [
    "rank",
    "workflow",
    "path",
    "readiness",
    "ready_for_sweep",
    "ready_for_model_selection",
    "split_counts",
    "known_z_counts",
    "known_z_bucket_count",
    "warnings",
    "top_decision",
    "top_decision_reason",
    "top_config",
    "top_method_allowlist",
    "top_known_clip_count",
    "top_mean_abs_known_z_bias",
    "top_mean_known_z_mad",
    "top_mean_z_std",
    "top_audit_warnings",
    "best_variant_config",
    "best_variant",
    "best_variant_method_allowlist",
    "best_variant_score",
    "best_variant_z_std",
    "best_variant_known_bias",
    "candidate_top_method",
    "candidate_top_hit_rate",
    "candidate_top_residual_median",
    "candidate_top_residual_mad",
    "calibration_used",
    "calibration_method_count",
]


def _workflow_summary_path(path: str | Path) -> Path:
    workflow_path = Path(path)
    return workflow_path / "workflow_summary.json" if workflow_path.is_dir() else workflow_path


def _json_compact(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _name_for_workflow(path: str | Path, report: Dict[str, Any]) -> str:
    output_dir = report.get("output_dir") or ""
    if output_dir:
        return Path(output_dir).name
    summary_path = _workflow_summary_path(path)
    return summary_path.parent.name if summary_path.name == "workflow_summary.json" else summary_path.stem


def _first_row(rows: Any) -> Dict[str, Any]:
    if isinstance(rows, list) and rows:
        first = rows[0]
        return first if isinstance(first, dict) else {}
    return {}


def _top_selection(report: Dict[str, Any]) -> Dict[str, Any]:
    top = (report.get("sweep", {}).get("selection_status", {}) or {}).get("top")
    return top if isinstance(top, dict) else _first_row(report.get("sweep", {}).get("top_selection"))


def _top_variant(report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_row(report.get("sweep", {}).get("top_variant_ranking"))


def _top_candidate(report: Dict[str, Any]) -> Dict[str, Any]:
    return _first_row(report.get("candidate_consistency", {}).get("top_aggregate"))


def _rank_key(row: Dict[str, Any]) -> tuple[int, int, float, float, str]:
    readiness_order = {
        "ready_for_model_selection": 0,
        "ready_for_sweep": 1,
        "smoke_only": 2,
        "blocked": 3,
    }
    decision_order = {"recommended": 0, "caution": 1, "reject": 2, "missing": 3}
    readiness = str(row.get("readiness", ""))
    decision = str(row.get("top_decision", ""))
    try:
        known_bias = float(row.get("top_mean_abs_known_z_bias") or 1e9)
    except (TypeError, ValueError):
        known_bias = 1e9
    try:
        z_std = float(row.get("top_mean_z_std") or 1e9)
    except (TypeError, ValueError):
        z_std = 1e9
    return (
        readiness_order.get(readiness, 9),
        decision_order.get(decision, 9),
        known_bias,
        z_std,
        str(row.get("workflow", "")),
    )


def compare_workflows(inputs: Sequence[str | Path]) -> List[Dict[str, Any]]:
    """Return one comparison row per workflow directory or workflow_summary.json."""

    rows: List[Dict[str, Any]] = []
    for item in inputs:
        report = build_workflow_report(item)
        readiness = report.get("readiness", {})
        validation = report.get("validation", {})
        calibration = report.get("calibration", {})
        candidate = _top_candidate(report)
        top = _top_selection(report)
        variant = _top_variant(report)
        rows.append(
            {
                "rank": 0,
                "workflow": _name_for_workflow(item, report),
                "path": str(_workflow_summary_path(item)),
                "readiness": readiness.get("status", ""),
                "ready_for_sweep": readiness.get("ready_for_sweep", False),
                "ready_for_model_selection": readiness.get("ready_for_model_selection", False),
                "split_counts": _json_compact(validation.get("split_counts", {})),
                "known_z_counts": _json_compact(validation.get("known_z_counts", {})),
                "known_z_bucket_count": len(validation.get("known_z_bucket_counts", {}) or {}),
                "warnings": ";".join(report.get("warnings", [])),
                "top_decision": top.get("decision", ""),
                "top_decision_reason": top.get("decision_reason", ""),
                "top_config": top.get("config", ""),
                "top_method_allowlist": top.get("method_allowlist", ""),
                "top_known_clip_count": top.get("known_clip_count", ""),
                "top_mean_abs_known_z_bias": top.get("mean_abs_known_z_bias", ""),
                "top_mean_known_z_mad": top.get("mean_known_z_mad", ""),
                "top_mean_z_std": top.get("mean_z_std", ""),
                "top_audit_warnings": top.get("audit_warnings", ""),
                "best_variant_config": variant.get("config", ""),
                "best_variant": variant.get("variant", ""),
                "best_variant_method_allowlist": variant.get("method_allowlist", ""),
                "best_variant_score": variant.get("score", ""),
                "best_variant_z_std": variant.get("mean_z_std", ""),
                "best_variant_known_bias": variant.get("mean_abs_known_z_bias", ""),
                "candidate_top_method": candidate.get("method", ""),
                "candidate_top_hit_rate": candidate.get("hit_rate", ""),
                "candidate_top_residual_median": candidate.get("residual_median", ""),
                "candidate_top_residual_mad": candidate.get("residual_mad", ""),
                "calibration_used": calibration.get("used_for_suite", False),
                "calibration_method_count": calibration.get("method_count", 0),
            }
        )

    rows.sort(key=_rank_key)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in FIELDNAMES})


def write_json(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"workflows": list(rows)}, indent=2, sort_keys=True), encoding="utf-8")


def print_report(rows: Sequence[Dict[str, Any]], *, limit: int = 20) -> None:
    print("rank,workflow,readiness,decision,config,methods,known,bias,z_std,best_variant,warnings")
    for row in rows[:limit]:
        print(
            "{rank},{workflow},{readiness},{decision},{config},{methods},{known},{bias},{z_std},{variant},{warnings}".format(
                rank=row.get("rank", ""),
                workflow=row.get("workflow", ""),
                readiness=row.get("readiness", ""),
                decision=row.get("top_decision", ""),
                config=row.get("top_config", ""),
                methods=row.get("top_method_allowlist", ""),
                known=row.get("top_known_clip_count", ""),
                bias=row.get("top_mean_abs_known_z_bias", ""),
                z_std=row.get("top_mean_z_std", ""),
                variant=row.get("best_variant", ""),
                warnings=row.get("warnings", ""),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows", nargs="+", help="Workflow directories or workflow_summary.json files")
    parser.add_argument("--csv-out")
    parser.add_argument("--json-out")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = compare_workflows(args.workflows)
    if args.csv_out:
        write_csv(args.csv_out, rows)
    if args.json_out:
        write_json(args.json_out, rows)
    print_report(rows, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
