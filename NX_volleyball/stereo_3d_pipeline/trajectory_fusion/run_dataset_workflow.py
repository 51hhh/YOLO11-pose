#!/usr/bin/env python3
"""Run the post-recording trajectory dataset workflow end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

try:
    from .build_dataset_manifest import build_manifest, write_manifest
    from .fit_method_calibration import CalibrationConfig, fit_method_calibration, write_calibration
    from .manifest import is_manifest_path, load_manifest
    from .run_evaluation_suite import run_suite
    from .run_reliability_sweep import run_sweep
    from .summarize_evaluation_suite import summarize_reliability_methods, summarize_suite
    from .summarize_workflow import build_workflow_report, write_json_report, write_markdown_report
    from .validate_dataset_manifest import analyze_manifest
except ImportError:  # pragma: no cover - direct script execution
    from build_dataset_manifest import build_manifest, write_manifest
    from fit_method_calibration import CalibrationConfig, fit_method_calibration, write_calibration
    from manifest import is_manifest_path, load_manifest
    from run_evaluation_suite import run_suite
    from run_reliability_sweep import run_sweep
    from summarize_evaluation_suite import summarize_reliability_methods, summarize_suite
    from summarize_workflow import build_workflow_report, write_json_report, write_markdown_report
    from validate_dataset_manifest import analyze_manifest


def _write_json(path: str | Path, data: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_manifest(
    inputs: Sequence[str | Path],
    *,
    output_dir: Path,
    manifest: str | Path | None,
    recursive: bool,
    split_mode: str,
    unlabeled_split: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    require_metadata: bool,
    absolute_paths: bool,
    stratify_known_z: bool,
) -> tuple[Path, Dict[str, Any]]:
    input_is_manifest = len(inputs) == 1 and is_manifest_path(inputs[0])
    if input_is_manifest and manifest is not None:
        raise ValueError("--manifest cannot be combined with a manifest positional input")
    if input_is_manifest:
        manifest_path = Path(inputs[0]).expanduser()
        return manifest_path, {
            "path": str(manifest_path),
            "generated": False,
            "source": "input_manifest",
        }

    if manifest is not None:
        manifest_path = Path(manifest).expanduser()
        if manifest_path.exists() and not inputs:
            return manifest_path, {
                "path": str(manifest_path),
                "generated": False,
                "source": "manifest_option",
            }
    else:
        manifest_path = output_dir / "dataset_manifest.yaml"

    if not inputs:
        raise ValueError("inputs are required when no existing manifest is provided")

    entries = build_manifest(
        inputs,
        output_path=manifest_path,
        recursive=recursive,
        split_mode=split_mode,
        unlabeled_split=unlabeled_split,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        require_metadata=require_metadata,
        absolute_paths=absolute_paths,
        stratify_known_z=stratify_known_z,
    )
    write_manifest(manifest_path, entries)
    return manifest_path, {
        "path": str(manifest_path),
        "generated": True,
        "source": "discovered_inputs",
        "clip_count": len(entries),
        "split_counts": _count_splits(entry.split for entry in entries),
        "known_z_counts": _count_splits(entry.split for entry in entries if entry.known_z is not None),
        "stratify_known_z": stratify_known_z,
    }


def _count_splits(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _fit_calibration(
    manifest_path: Path,
    output_path: Path,
    *,
    train_split: str,
    min_count: int,
    min_sigma: float,
) -> tuple[Path | None, Dict[str, Any]]:
    clips = load_manifest(manifest_path)
    calibration = fit_method_calibration(
        clips,
        cfg=CalibrationConfig(
            train_split=train_split,
            min_count=min_count,
            min_sigma=min_sigma,
        ),
    )
    write_calibration(output_path, calibration)
    method_count = len(calibration.get("methods", {}))
    return (
        output_path if method_count > 0 else None,
        {
            "path": str(output_path),
            "written": True,
            "used_for_suite": method_count > 0,
            "method_count": method_count,
            "insufficient_method_count": len(calibration.get("insufficient_methods", {})),
            "used_clip_count": len(calibration.get("used_clips", [])),
            "skipped_clip_count": len(calibration.get("skipped_clips", [])),
            "frame_count": calibration.get("frame_count", 0),
        },
    )


def _run_suite_and_summarize(
    manifest_path: Path,
    output_dir: Path,
    *,
    calibration_path: Path | None,
    checkpoint: str | Path | None,
    device: str,
    gravity_y: float,
    use_online_position: bool,
    use_static_known_z: bool,
    include_depth_polyfit: bool,
    include_rts_smoother: bool,
    include_candidate_consistency: bool,
    candidate_reference: str,
) -> Dict[str, Any]:
    suite_report = run_suite(
        [str(manifest_path)],
        output_dir,
        checkpoint=checkpoint,
        calibration=calibration_path,
        device=device,
        gravity_y=gravity_y,
        use_online_position=use_online_position,
        use_static_known_z=use_static_known_z,
        include_depth_polyfit=include_depth_polyfit,
        include_rts_smoother=include_rts_smoother,
        include_candidate_consistency=include_candidate_consistency,
        candidate_reference=candidate_reference,
    )
    metrics_path = output_dir / "suite_metrics.csv"
    metrics_rows = summarize_suite(output_dir, metrics_path)
    method_rows = []
    methods_path = None
    if checkpoint:
        methods_path = output_dir / "suite_reliability_methods.csv"
        method_rows = summarize_reliability_methods(output_dir, methods_path)
    variants = sorted({str(row.get("variant", "")) for row in metrics_rows if row.get("variant", "")})
    return {
        "dir": str(output_dir),
        "summary_json": str(output_dir / "suite_summary.json"),
        "metrics_csv": str(metrics_path),
        "reliability_methods_csv": str(methods_path) if methods_path else None,
        "clip_count": len(suite_report.get("clips", [])),
        "metric_rows": len(metrics_rows),
        "reliability_method_rows": len(method_rows),
        "variants": variants,
        "calibration": str(calibration_path) if calibration_path else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def run_workflow(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    manifest: str | Path | None = None,
    recursive: bool = False,
    split_mode: str = "auto",
    unlabeled_split: str = "train",
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    require_metadata: bool = False,
    absolute_paths: bool = False,
    stratify_known_z: bool = False,
    min_rows: int = 100,
    min_fps: float = 80.0,
    min_p0_hit: float = 0.85,
    max_frame_gaps: int = 0,
    skip_calibration: bool = False,
    calibration_min_count: int = 8,
    calibration_min_sigma: float = 0.015,
    train_split: str = "train",
    checkpoint: str | Path | None = None,
    configs: str | Path | None = None,
    device: str = "cpu",
    gravity_y: float = 9.81,
    rank_split: str = "auto",
    use_online_position: bool = False,
    use_static_known_z: bool = False,
    skip_sweep: bool = False,
    include_depth_polyfit: bool = True,
    include_rts_smoother: bool = True,
    include_candidate_consistency: bool = True,
    candidate_reference: str = "auto",
) -> Dict[str, Any]:
    """Build/validate a dataset, run baselines, and optionally run a ReliabilityNet sweep."""

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_summary = _resolve_manifest(
        inputs,
        output_dir=root,
        manifest=manifest,
        recursive=recursive,
        split_mode=split_mode,
        unlabeled_split=unlabeled_split,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        require_metadata=require_metadata,
        absolute_paths=absolute_paths,
        stratify_known_z=stratify_known_z,
    )

    validation = analyze_manifest(
        manifest_path,
        min_rows=min_rows,
        min_fps=min_fps,
        min_p0_hit=min_p0_hit,
        max_frame_gaps=max_frame_gaps,
        require_stratified_known_z=stratify_known_z,
    )
    validation_json = root / "manifest_validation.json"
    _write_json(validation_json, validation)

    calibration_path: Path | None = None
    if skip_calibration:
        calibration_summary: Dict[str, Any] = {
            "skipped": True,
            "reason": "skip_calibration",
            "used_for_suite": False,
        }
    else:
        calibration_path, calibration_summary = _fit_calibration(
            manifest_path,
            root / "method_calibration.json",
            train_split=train_split,
            min_count=calibration_min_count,
            min_sigma=calibration_min_sigma,
        )
        calibration_summary["skipped"] = False
        if calibration_path is None:
            calibration_summary["reason"] = "no_calibrated_methods"

    baseline_suite = _run_suite_and_summarize(
        manifest_path,
        root / "baseline_suite",
        calibration_path=calibration_path,
        checkpoint=checkpoint,
        device=device,
        gravity_y=gravity_y,
        use_online_position=use_online_position,
        use_static_known_z=use_static_known_z,
        include_depth_polyfit=include_depth_polyfit,
        include_rts_smoother=include_rts_smoother,
        include_candidate_consistency=include_candidate_consistency,
        candidate_reference=candidate_reference,
    )

    if skip_sweep:
        sweep_summary: Dict[str, Any] = {
            "skipped": True,
            "reason": "skip_sweep",
        }
    else:
        sweep_report = run_sweep(
            [str(manifest_path)],
            root / "reliability_sweep",
            configs_path=configs,
            calibration=calibration_path,
            train_split=train_split,
            device=device,
            gravity_y=gravity_y,
            use_static_known_z=use_static_known_z,
            rank_split=rank_split,
            include_depth_polyfit=include_depth_polyfit,
            include_rts_smoother=include_rts_smoother,
            include_candidate_consistency=include_candidate_consistency,
            candidate_reference=candidate_reference,
        )
        sweep_summary = {
            "skipped": False,
            "dir": str(root / "reliability_sweep"),
            "summary_json": str(root / "reliability_sweep" / "sweep_summary.json"),
            "sweep_metrics": str(root / "reliability_sweep" / "sweep_metrics.csv"),
            "sweep_ranking": sweep_report.get("sweep_ranking"),
            "sweep_variant_ranking": sweep_report.get("sweep_variant_ranking"),
            "sweep_reliability_methods": sweep_report.get("sweep_reliability_methods"),
            "sweep_reliability_method_audit": sweep_report.get("sweep_reliability_method_audit"),
            "sweep_model_selection": sweep_report.get("sweep_model_selection"),
            "run_count": len(sweep_report.get("runs", [])),
        }

    summary: Dict[str, Any] = {
        "output_dir": str(root),
        "manifest": manifest_summary,
        "validation": {
            "json": str(validation_json),
            "clip_count": validation.get("clip_count", 0),
            "split_counts": validation.get("split_counts", {}),
            "known_z_counts": validation.get("known_z_counts", {}),
            "known_z_bucket_counts": validation.get("known_z_bucket_counts", {}),
            "known_z_bucket_warnings": validation.get("known_z_bucket_warnings", []),
            "warning_counts": validation.get("warning_counts", {}),
            "warnings": validation.get("warnings", []),
        },
        "calibration": calibration_summary,
        "baseline_suite": baseline_suite,
        "sweep": sweep_summary,
        "config": {
            "device": device,
            "gravity_y": gravity_y,
            "train_split": train_split,
            "rank_split": rank_split,
            "use_online_position": use_online_position,
            "use_static_known_z": use_static_known_z,
            "include_depth_polyfit": include_depth_polyfit,
            "include_rts_smoother": include_rts_smoother,
            "include_candidate_consistency": include_candidate_consistency,
            "candidate_reference": candidate_reference,
            "stratify_known_z": stratify_known_z,
        },
        "workflow_report_json": str(root / "workflow_report.json"),
        "workflow_report_md": str(root / "workflow_report.md"),
    }
    _write_json(root / "workflow_summary.json", summary)
    report = build_workflow_report(root / "workflow_summary.json")
    write_json_report(root / "workflow_report.json", report)
    write_markdown_report(root / "workflow_report.md", report)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="CSV files/directories, or one dataset manifest")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--manifest",
        help="Existing manifest to use when inputs are omitted, or output path for a generated manifest.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--split-mode", choices=("auto", "train", "eval"), default="auto")
    parser.add_argument("--unlabeled-split", choices=("train", "eval"), default="train")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument(
        "--stratify-known-z",
        action="store_true",
        help="When generating a manifest, split each known_z distance bucket independently.",
    )
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-fps", type=float, default=80.0)
    parser.add_argument("--min-p0-hit", type=float, default=0.85)
    parser.add_argument("--max-frame-gaps", type=int, default=0)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-min-count", type=int, default=8)
    parser.add_argument("--calibration-min-sigma", type=float, default=0.015)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--checkpoint", help="Optional ReliabilityNet checkpoint for baseline suite comparison")
    parser.add_argument("--configs", help="Optional ReliabilityNet sweep YAML/JSON config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gravity-y", type=float, default=9.81)
    parser.add_argument("--rank-split", default="auto")
    parser.add_argument("--use-online-position", action="store_true")
    parser.add_argument(
        "--use-static-known-z",
        action="store_true",
        help="Use static known_z as smoother update. Off by default to avoid label leakage.",
    )
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-depth-polyfit", action="store_true")
    parser.add_argument("--skip-rts-smoother", action="store_true")
    parser.add_argument("--skip-candidate-consistency", action="store_true")
    parser.add_argument("--candidate-reference", default="auto")
    args = parser.parse_args()

    summary = run_workflow(
        args.inputs,
        args.output_dir,
        manifest=args.manifest,
        recursive=args.recursive,
        split_mode=args.split_mode,
        unlabeled_split=args.unlabeled_split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        require_metadata=args.require_metadata,
        absolute_paths=args.absolute_paths,
        stratify_known_z=args.stratify_known_z,
        min_rows=args.min_rows,
        min_fps=args.min_fps,
        min_p0_hit=args.min_p0_hit,
        max_frame_gaps=args.max_frame_gaps,
        skip_calibration=args.skip_calibration,
        calibration_min_count=args.calibration_min_count,
        calibration_min_sigma=args.calibration_min_sigma,
        train_split=args.train_split,
        checkpoint=args.checkpoint,
        configs=args.configs,
        device=args.device,
        gravity_y=args.gravity_y,
        rank_split=args.rank_split,
        use_online_position=args.use_online_position,
        use_static_known_z=args.use_static_known_z,
        skip_sweep=args.skip_sweep,
        include_depth_polyfit=not args.skip_depth_polyfit,
        include_rts_smoother=not args.skip_rts_smoother,
        include_candidate_consistency=not args.skip_candidate_consistency,
        candidate_reference=args.candidate_reference,
    )
    print(f"workflow output: {summary['output_dir']}")
    print(f"manifest: {summary['manifest']['path']}")
    print(f"validation warnings: {summary['validation']['warning_counts']}")
    print(f"baseline suite: {summary['baseline_suite']['metrics_csv']}")
    print(f"workflow report: {summary['workflow_report_md']}")
    if summary["sweep"]["skipped"]:
        print(f"sweep: skipped ({summary['sweep']['reason']})")
    else:
        print(f"sweep: {summary['sweep']['dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
