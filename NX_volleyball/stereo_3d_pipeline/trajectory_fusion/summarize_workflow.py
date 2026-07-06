#!/usr/bin/env python3
"""Summarize post-recording trajectory workflow outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _read_csv(path: str | Path | None) -> List[Dict[str, str]]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _workflow_summary_path(path: str | Path) -> Path:
    workflow_path = Path(path)
    return workflow_path / "workflow_summary.json" if workflow_path.is_dir() else workflow_path


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _fmt(value: object, digits: int = 4) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def _top_rows(rows: List[Dict[str, str]], *, limit: int = 8) -> List[Dict[str, str]]:
    return rows[:limit]


def _selection_status(selection_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not selection_rows:
        return {
            "decision": "missing",
            "reason": "missing_model_selection",
            "top": None,
        }
    top = selection_rows[0]
    return {
        "decision": top.get("decision", ""),
        "reason": top.get("decision_reason", ""),
        "top": top,
    }


def _derive_warnings(summary: Dict[str, Any], selection_rows: List[Dict[str, str]]) -> List[str]:
    warnings: List[str] = []
    validation_counts = summary.get("validation", {}).get("warning_counts", {})
    for key in sorted(validation_counts):
        if key:
            warnings.append(f"validation:{key}")

    calibration = summary.get("calibration", {})
    if not bool(calibration.get("used_for_suite", False)):
        warnings.append("calibration:not_used")

    sweep = summary.get("sweep", {})
    if bool(sweep.get("skipped", False)):
        warnings.append("sweep:skipped")
    elif not selection_rows:
        warnings.append("sweep:missing_selection")
    else:
        top = selection_rows[0]
        decision = top.get("decision", "")
        if decision != "recommended":
            warnings.append(f"selection:{decision or 'unknown'}")
        if _safe_float(top.get("known_clip_count")) in (None, 0.0):
            warnings.append("selection:no_known_z")
        audit_warnings = [item for item in str(top.get("audit_warnings", "")).split(";") if item]
        for warning in audit_warnings:
            warnings.append(f"audit:{warning}")
    return warnings


def _derive_actions(summary: Dict[str, Any], selection_rows: List[Dict[str, str]]) -> List[str]:
    actions: List[str] = []
    validation_counts = summary.get("validation", {}).get("warning_counts", {})
    if "missing_known_z_clips" in validation_counts:
        actions.append("Record static known-distance clips before selecting a final model.")
    if "missing_val_split" in validation_counts:
        actions.append("Add heldout val clips or regenerate the manifest with a val split.")
    if "frame_gaps>0" in validation_counts:
        actions.append("Inspect frame gaps and synchronization before trusting dynamic metrics.")
    if not bool(summary.get("calibration", {}).get("used_for_suite", False)):
        actions.append("Fit per-method calibration after train split known_z clips are available.")

    if selection_rows:
        top = selection_rows[0]
        if top.get("decision") != "recommended":
            actions.append("Do not promote the top ReliabilityNet checkpoint until audit warnings are resolved.")
        if _safe_float(top.get("known_clip_count")) in (None, 0.0):
            actions.append("Treat this sweep as a toolchain smoke test only; it has no absolute-depth evidence.")
    else:
        actions.append("Run the ReliabilityNet sweep when model selection is required.")

    actions.append("Compare calibrated_smoother, robust_smooth, and reliability_smoother on heldout known_z clips.")
    return actions


def build_workflow_report(path: str | Path) -> Dict[str, Any]:
    """Build a compact report from a workflow directory or workflow_summary.json."""

    summary_path = _workflow_summary_path(path)
    summary = _read_json(summary_path)
    if not summary:
        raise FileNotFoundError(f"missing workflow summary: {summary_path}")

    baseline_metrics = _read_csv(summary.get("baseline_suite", {}).get("metrics_csv"))
    sweep = summary.get("sweep", {})
    selection_rows = _read_csv(sweep.get("sweep_model_selection"))
    variant_rows = _read_csv(sweep.get("sweep_variant_ranking"))
    audit_rows = _read_csv(sweep.get("sweep_reliability_method_audit"))

    report = {
        "workflow_summary": str(summary_path),
        "output_dir": summary.get("output_dir", ""),
        "validation": summary.get("validation", {}),
        "calibration": summary.get("calibration", {}),
        "baseline_suite": {
            "metrics_csv": summary.get("baseline_suite", {}).get("metrics_csv"),
            "variants": summary.get("baseline_suite", {}).get("variants", []),
            "top_metrics": _top_rows(baseline_metrics),
        },
        "sweep": {
            "skipped": bool(sweep.get("skipped", False)),
            "selection_csv": sweep.get("sweep_model_selection"),
            "variant_ranking_csv": sweep.get("sweep_variant_ranking"),
            "method_audit_csv": sweep.get("sweep_reliability_method_audit"),
            "selection_status": _selection_status(selection_rows),
            "top_selection": _top_rows(selection_rows, limit=5),
            "top_variant_ranking": _top_rows(variant_rows, limit=8),
            "audit_rows": _top_rows(audit_rows, limit=8),
        },
    }
    report["warnings"] = _derive_warnings(summary, selection_rows)
    report["recommended_actions"] = _derive_actions(summary, selection_rows)
    return report


def write_json_report(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown_report(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    validation = report.get("validation", {})
    calibration = report.get("calibration", {})
    selection = report.get("sweep", {}).get("selection_status", {})
    top = selection.get("top") or {}
    lines = [
        "# Trajectory Workflow Report",
        "",
        f"- output_dir: `{report.get('output_dir', '')}`",
        f"- validation warnings: `{validation.get('warning_counts', {})}`",
        f"- splits: `{validation.get('split_counts', {})}`",
        f"- known_z: `{validation.get('known_z_counts', {})}`",
        f"- calibration methods: `{calibration.get('method_count', 0)}`",
        f"- calibration used: `{calibration.get('used_for_suite', False)}`",
        f"- top selection: `{top.get('config', '')}` / `{top.get('decision', selection.get('decision', ''))}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings", [])
    lines.extend(f"- `{warning}`" for warning in warnings)
    if not warnings:
        lines.append("- none")

    lines.extend(["", "## Recommended Actions", ""])
    lines.extend(f"- {action}" for action in report.get("recommended_actions", []))

    selection_rows = report.get("sweep", {}).get("top_selection", [])
    selection_table = _markdown_table(
        [
            "rank",
            "decision",
            "config",
            "split",
            "known",
            "z_std",
            "known_bias",
            "warnings",
        ],
        [
            [
                str(row.get("selection_rank", "")),
                str(row.get("decision", "")),
                str(row.get("config", "")),
                str(row.get("split", "")),
                str(row.get("known_clip_count", "")),
                _fmt(row.get("mean_z_std")),
                _fmt(row.get("mean_abs_known_z_bias")),
                str(row.get("audit_warnings", "")),
            ]
            for row in selection_rows
        ],
    )
    lines.extend(["", "## Model Selection", ""])
    lines.append(selection_table or "No model selection rows.")

    variant_rows = report.get("sweep", {}).get("top_variant_ranking", [])
    variant_table = _markdown_table(
        ["rank", "config", "variant", "split", "known", "z_std", "known_bias", "score"],
        [
            [
                str(row.get("rank", "")),
                str(row.get("config", "")),
                str(row.get("variant", "")),
                str(row.get("split", "")),
                str(row.get("known_clip_count", "")),
                _fmt(row.get("mean_z_std")),
                _fmt(row.get("mean_abs_known_z_bias")),
                _fmt(row.get("score")),
            ]
            for row in variant_rows
        ],
    )
    lines.extend(["", "## Variant Ranking", ""])
    lines.append(variant_table or "No variant ranking rows.")

    baseline_rows = report.get("baseline_suite", {}).get("top_metrics", [])
    baseline_table = _markdown_table(
        ["clip", "split", "variant", "z_std", "known_bias", "p0_median"],
        [
            [
                str(row.get("clip", "")),
                str(row.get("split", "")),
                str(row.get("variant", "")),
                _fmt(row.get("z_std")),
                _fmt(row.get("known_z_bias")),
                _fmt(row.get("p0_median_mean")),
            ]
            for row in baseline_rows
        ],
    )
    lines.extend(["", "## Baseline Metrics", ""])
    lines.append(baseline_table or "No baseline metric rows.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    selection = report.get("sweep", {}).get("selection_status", {})
    top = selection.get("top") or {}
    print(f"workflow={report.get('output_dir', '')}")
    print(f"warnings={report.get('warnings', [])}")
    print(f"top_selection={top.get('config', '')} decision={selection.get('decision', '')}")
    for action in report.get("recommended_actions", []):
        print(f"action={action}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", help="Workflow directory or workflow_summary.json")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    report = build_workflow_report(args.workflow)
    summary_path = _workflow_summary_path(args.workflow)
    json_out = Path(args.json_out) if args.json_out else summary_path.with_name("workflow_report.json")
    markdown_out = Path(args.markdown_out) if args.markdown_out else summary_path.with_name("workflow_report.md")
    write_json_report(json_out, report)
    write_markdown_report(markdown_out, report)
    print_report(report)
    print(f"wrote {json_out}")
    print(f"wrote {markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
