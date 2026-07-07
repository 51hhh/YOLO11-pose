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


def _candidate_sort_key(row: Dict[str, str]) -> tuple[float, float, float, float, str]:
    residual_median_value = _safe_float(row.get("residual_median"))
    residual_median = abs(residual_median_value) if residual_median_value is not None else float("inf")
    residual_mad = _safe_float(row.get("residual_mad"))
    residual_p95 = _safe_float(row.get("residual_abs_p95"))
    hit_rate = _safe_float(row.get("hit_rate")) or 0.0
    return (
        residual_median,
        residual_mad if residual_mad is not None else float("inf"),
        residual_p95 if residual_p95 is not None else float("inf"),
        -hit_rate,
        str(row.get("method", "")),
    )


def _top_candidate_rows(
    rows: List[Dict[str, str]],
    *,
    scope: str,
    limit: int = 8,
) -> List[Dict[str, str]]:
    scoped = [
        row
        for row in rows
        if row.get("scope") == scope and int(_safe_float(row.get("valid")) or 0) > 0
    ]
    scoped.sort(key=_candidate_sort_key)
    return scoped[:limit]


def _top_candidate_bucket_rows(
    rows: List[Dict[str, str]],
    *,
    per_bucket: int = 3,
    limit: int = 18,
) -> List[Dict[str, str]]:
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("scope") != "known_z_bucket":
            continue
        if int(_safe_float(row.get("valid")) or 0) <= 0:
            continue
        key = (row.get("split", ""), row.get("known_z_bucket", ""))
        grouped.setdefault(key, []).append(row)
    selected: List[Dict[str, str]] = []
    for key in sorted(grouped):
        group = grouped[key]
        group.sort(key=_candidate_sort_key)
        selected.extend(group[:per_bucket])
    return selected[:limit]


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


def _warning_count(summary: Dict[str, Any], key: str) -> int:
    return int(summary.get("validation", {}).get("warning_counts", {}).get(key, 0) or 0)


def _derive_readiness(summary: Dict[str, Any], selection_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Classify whether a workflow output is ready for model training/selection."""

    blocking: List[str] = []
    cautions: List[str] = []
    validation_counts = summary.get("validation", {}).get("warning_counts", {})
    training_audit = summary.get("training_input_audit", {})
    training_warnings = set(training_audit.get("warnings", []))
    candidate_consistency = summary.get("candidate_consistency", {})
    calibration = summary.get("calibration", {})
    sweep = summary.get("sweep", {})

    for key in ("empty_manifest", "missing_file", "missing_required_fields"):
        if _warning_count(summary, key):
            blocking.append(f"validation:{key}")
    for warning in training_warnings:
        if warning in {"no_input_clips", "no_training_sequences", "no_training_frames"}:
            blocking.append(f"training_input:{warning}")
        if str(warning).startswith("legacy_"):
            blocking.append(f"training_input:{warning}")

    known_z_counts = summary.get("validation", {}).get("known_z_counts", {})
    has_known_train = int(known_z_counts.get("train", 0) or 0) > 0
    has_known_val = int(known_z_counts.get("val", 0) or 0) > 0
    has_candidate_consistency = (
        not bool(candidate_consistency.get("skipped", False))
        and int(candidate_consistency.get("frames", 0) or 0) > 0
    )
    calibration_used = bool(calibration.get("used_for_suite", False))

    if not has_known_train:
        cautions.append("known_z:missing_train")
    if not has_known_val:
        cautions.append("known_z:missing_val")
    if "missing_known_z_clips" in validation_counts:
        cautions.append("known_z:missing_all")
    if "missing_val_split" in validation_counts:
        cautions.append("split:missing_val")
    if "missing_known_z_val_split" in validation_counts:
        cautions.append("known_z:missing_val_split")
    if any(str(key).startswith("known_z_bucket_missing_") for key in validation_counts):
        cautions.append("known_z:bucket_not_stratified")
    if _warning_count(summary, "frame_gaps>0"):
        cautions.append("frames:has_gaps")
    if not has_candidate_consistency:
        cautions.append("candidate_consistency:missing")
    if not calibration_used:
        cautions.append("calibration:not_used")
    if any(str(item).startswith("low_method_coverage") for item in training_warnings):
        cautions.append("training_input:low_method_coverage")
    if any(str(item).startswith("mostly_zero_features") for item in training_warnings):
        cautions.append("training_input:mostly_zero_features")

    ready_for_sweep = (
        not blocking
        and has_known_train
        and has_known_val
        and has_candidate_consistency
        and calibration_used
    )

    top = selection_rows[0] if selection_rows else {}
    top_known_count = _safe_float(top.get("known_clip_count")) if top else None
    top_decision = str(top.get("decision", "")) if top else ""
    if bool(sweep.get("skipped", False)):
        cautions.append("sweep:skipped")
    elif not selection_rows:
        cautions.append("sweep:missing_selection")
    elif top_decision != "recommended":
        cautions.append(f"selection:{top_decision or 'unknown'}")
    if selection_rows and (top_known_count is None or top_known_count <= 0.0):
        cautions.append("selection:no_known_z")

    ready_for_model_selection = (
        ready_for_sweep
        and not bool(sweep.get("skipped", False))
        and bool(selection_rows)
        and top_decision == "recommended"
        and top_known_count is not None
        and top_known_count > 0.0
    )

    if blocking:
        status = "blocked"
    elif ready_for_model_selection:
        status = "ready_for_model_selection"
    elif ready_for_sweep:
        status = "ready_for_sweep"
    else:
        status = "smoke_only"

    return {
        "status": status,
        "ready_for_sweep": ready_for_sweep,
        "ready_for_model_selection": ready_for_model_selection,
        "blocking_reasons": blocking,
        "cautions": sorted(set(cautions)),
        "known_z_train_clips": int(known_z_counts.get("train", 0) or 0),
        "known_z_val_clips": int(known_z_counts.get("val", 0) or 0),
        "candidate_consistency_frames": int(candidate_consistency.get("frames", 0) or 0),
        "calibration_used": calibration_used,
    }


def _derive_warnings(summary: Dict[str, Any], selection_rows: List[Dict[str, str]]) -> List[str]:
    warnings: List[str] = []
    validation_counts = summary.get("validation", {}).get("warning_counts", {})
    for key in sorted(validation_counts):
        if key:
            warnings.append(f"validation:{key}")

    training_audit = summary.get("training_input_audit", {})
    for warning in training_audit.get("warnings", []):
        warnings.append(f"training_input:{warning}")

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
    if "known_z_bucket_missing_val_split" in validation_counts:
        actions.append("For stratified known_z runs, record at least two clips per static distance or add heldout val clips for missing buckets.")
    if "known_z_bucket_missing_train_split" in validation_counts:
        actions.append("For stratified known_z runs, ensure every evaluated distance also has train coverage or intentionally disable stratified validation.")
    if "frame_gaps>0" in validation_counts:
        actions.append("Inspect frame gaps and synchronization before trusting dynamic metrics.")
    training_audit = summary.get("training_input_audit", {})
    audit_warnings = set(training_audit.get("warnings", []))
    if "no_training_sequences" in audit_warnings or "no_training_frames" in audit_warnings:
        actions.append("Fix manifest/CSV parsing before training; the model input audit found no usable sequences.")
    if any(str(item).startswith("legacy_") for item in audit_warnings):
        actions.append("Remove legacy online state/depth leakage from training inputs before running a sweep.")
    if any(str(item).startswith("low_method_coverage") for item in audit_warnings):
        actions.append("Inspect training_method_coverage.csv and confirm expected P0/P1/NCC/XFeat fields are present.")
    if any(str(item).startswith("mostly_zero_features") for item in audit_warnings):
        actions.append("Inspect training_feature_coverage.csv; many model features are constant or absent in this dataset.")
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
    candidate_consistency = dict(summary.get("candidate_consistency", {}))
    candidate_rows = _read_csv(candidate_consistency.get("method_csv"))
    candidate_consistency["top_aggregate"] = _top_candidate_rows(candidate_rows, scope="aggregate")
    candidate_consistency["top_known_z_buckets"] = _top_candidate_bucket_rows(candidate_rows)

    report = {
        "workflow_summary": str(summary_path),
        "output_dir": summary.get("output_dir", ""),
        "validation": summary.get("validation", {}),
        "training_input_audit": summary.get("training_input_audit", {}),
        "candidate_consistency": candidate_consistency,
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
    report["readiness"] = _derive_readiness(summary, selection_rows)
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
    training_audit = report.get("training_input_audit", {})
    candidate_consistency = report.get("candidate_consistency", {})
    calibration = report.get("calibration", {})
    selection = report.get("sweep", {}).get("selection_status", {})
    top = selection.get("top") or {}
    readiness = report.get("readiness", {})
    lines = [
        "# Trajectory Workflow Report",
        "",
        f"- output_dir: `{report.get('output_dir', '')}`",
        f"- readiness: `{readiness.get('status', '')}` (sweep `{readiness.get('ready_for_sweep', False)}`, model selection `{readiness.get('ready_for_model_selection', False)}`)",
        f"- validation warnings: `{validation.get('warning_counts', {})}`",
        f"- splits: `{validation.get('split_counts', {})}`",
        f"- known_z: `{validation.get('known_z_counts', {})}`",
        f"- known_z buckets: `{validation.get('known_z_bucket_counts', {})}`",
        f"- training inputs: frames `{training_audit.get('frame_count', 0)}`, methods `{training_audit.get('method_count', 0)}`, features `{training_audit.get('feature_count', 0)}`",
        f"- training input warnings: `{training_audit.get('warnings', [])}`",
        f"- candidate consistency: frames `{candidate_consistency.get('frames', 0)}`, methods `{candidate_consistency.get('method_count', 0)}`, known_z buckets `{candidate_consistency.get('known_z_bucket_count', 0)}`",
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

    lines.extend(["", "## Readiness", ""])
    lines.append(f"- status: `{readiness.get('status', '')}`")
    lines.append(f"- ready_for_sweep: `{readiness.get('ready_for_sweep', False)}`")
    lines.append(f"- ready_for_model_selection: `{readiness.get('ready_for_model_selection', False)}`")
    blockers = readiness.get("blocking_reasons", [])
    cautions = readiness.get("cautions", [])
    lines.append(f"- blocking: `{blockers}`")
    lines.append(f"- cautions: `{cautions}`")

    lines.extend(["", "## Recommended Actions", ""])
    lines.extend(f"- {action}" for action in report.get("recommended_actions", []))

    aggregate_candidates = candidate_consistency.get("top_aggregate", [])
    candidate_table = _markdown_table(
        ["scope", "method", "hit", "z_med", "z_mad", "res_med", "res_mad", "res_p95"],
        [
            [
                str(row.get("scope", "")),
                str(row.get("method", "")),
                _fmt(row.get("hit_rate"), digits=3),
                _fmt(row.get("z_median")),
                _fmt(row.get("z_mad")),
                _fmt(row.get("residual_median")),
                _fmt(row.get("residual_mad")),
                _fmt(row.get("residual_abs_p95")),
            ]
            for row in aggregate_candidates
        ],
    )
    lines.extend(["", "## Candidate Consistency", ""])
    lines.append(candidate_table or "No candidate consistency rows.")

    bucket_candidates = candidate_consistency.get("top_known_z_buckets", [])
    bucket_table = _markdown_table(
        ["split", "known_z", "method", "hit", "res_med", "res_mad", "res_p95"],
        [
            [
                str(row.get("split", "")),
                str(row.get("known_z_bucket", "")),
                str(row.get("method", "")),
                _fmt(row.get("hit_rate"), digits=3),
                _fmt(row.get("residual_median")),
                _fmt(row.get("residual_mad")),
                _fmt(row.get("residual_abs_p95")),
            ]
            for row in bucket_candidates
        ],
    )
    lines.extend(["", "## Known-Z Bucket Candidates", ""])
    lines.append(bucket_table or "No known_z bucket candidate rows.")

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
    print(f"readiness={report.get('readiness', {}).get('status', '')}")
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
