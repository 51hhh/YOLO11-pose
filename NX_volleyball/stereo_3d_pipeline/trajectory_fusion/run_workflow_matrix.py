#!/usr/bin/env python3
"""Run a standard trajectory workflow matrix for post-recording model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .build_dataset_manifest import build_manifest
    from .compare_workflows import compare_workflows, write_csv as write_compare_csv, write_json as write_compare_json
    from .manifest import is_manifest_path
    from .run_dataset_workflow import run_workflow
except ImportError:  # pragma: no cover - direct script execution
    from build_dataset_manifest import build_manifest
    from compare_workflows import compare_workflows, write_csv as write_compare_csv, write_json as write_compare_json
    from manifest import is_manifest_path
    from run_dataset_workflow import run_workflow


def _default_config(name: str) -> Path:
    return Path(__file__).with_name("configs") / name


def _safe_bucket(value: float) -> str:
    return f"{value:.3f}"


def _safe_dir_suffix(value: str) -> str:
    return value.replace(".", "m").replace("-", "neg")


def _discover_known_z_buckets(
    inputs: Sequence[str | Path],
    *,
    output_dir: Path,
    recursive: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    require_metadata: bool,
) -> List[str]:
    if len(inputs) == 1 and is_manifest_path(inputs[0]):
        # Matrix holdout rewrites splits, so keep this first version scoped to
        # raw CSV/directory discovery where build_manifest owns the split plan.
        return []
    entries = build_manifest(
        inputs,
        output_path=output_dir / "_matrix_probe_manifest.yaml",
        recursive=recursive,
        split_mode="auto",
        unlabeled_split="train",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        require_metadata=require_metadata,
        stratify_known_z=True,
    )
    return sorted({_safe_bucket(float(entry.known_z)) for entry in entries if entry.known_z is not None})


def _workflow_kwargs(
    *,
    recursive: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    require_metadata: bool,
    train_split: str,
    device: str,
    gravity_y: float,
    rank_split: str,
    min_rows: int,
    min_fps: float,
    min_p0_hit: float,
    max_frame_gaps: int,
    skip_sweep: bool,
    include_depth_polyfit: bool,
    include_rts_smoother: bool,
    calibration_min_count: int,
    calibration_min_sigma: float,
    candidate_reference: str,
) -> Dict[str, Any]:
    return {
        "recursive": recursive,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "require_metadata": require_metadata,
        "train_split": train_split,
        "device": device,
        "gravity_y": gravity_y,
        "rank_split": rank_split,
        "min_rows": min_rows,
        "min_fps": min_fps,
        "min_p0_hit": min_p0_hit,
        "max_frame_gaps": max_frame_gaps,
        "skip_sweep": skip_sweep,
        "include_depth_polyfit": include_depth_polyfit,
        "include_rts_smoother": include_rts_smoother,
        "calibration_min_count": calibration_min_count,
        "calibration_min_sigma": calibration_min_sigma,
        "candidate_reference": candidate_reference,
    }


def run_workflow_matrix(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    recursive: bool = False,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 7,
    require_metadata: bool = False,
    train_split: str = "train",
    device: str = "cpu",
    gravity_y: float = 0.0,
    rank_split: str = "auto",
    min_rows: int = 100,
    min_fps: float = 80.0,
    min_p0_hit: float = 0.85,
    max_frame_gaps: int = 0,
    known_configs: str | Path | None = None,
    dynamic_configs: str | Path | None = None,
    include_dynamic: bool = False,
    skip_sweep: bool = False,
    include_depth_polyfit: bool = True,
    include_rts_smoother: bool = True,
    calibration_min_count: int = 8,
    calibration_min_sigma: float = 0.015,
    candidate_reference: str = "auto",
) -> Dict[str, Any]:
    """Run standard workflow variants and emit a comparison table."""

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    known_config_path = Path(known_configs) if known_configs else _default_config("sweep_known_distance_selection.json")
    dynamic_config_path = Path(dynamic_configs) if dynamic_configs else _default_config("sweep_dynamic_regularization.json")
    common = _workflow_kwargs(
        recursive=recursive,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        require_metadata=require_metadata,
        train_split=train_split,
        device=device,
        gravity_y=gravity_y,
        rank_split=rank_split,
        min_rows=min_rows,
        min_fps=min_fps,
        min_p0_hit=min_p0_hit,
        max_frame_gaps=max_frame_gaps,
        skip_sweep=skip_sweep,
        include_depth_polyfit=include_depth_polyfit,
        include_rts_smoother=include_rts_smoother,
        calibration_min_count=calibration_min_count,
        calibration_min_sigma=calibration_min_sigma,
        candidate_reference=candidate_reference,
    )

    runs: List[Dict[str, Any]] = []
    workflow_dirs: List[Path] = []

    def run_named(name: str, **kwargs: Any) -> None:
        workflow_dir = root / name
        summary = run_workflow(
            inputs,
            workflow_dir,
            **common,
            **kwargs,
        )
        runs.append(
            {
                "name": name,
                "dir": str(workflow_dir),
                "workflow_summary": str(workflow_dir / "workflow_summary.json"),
                "workflow_report": summary.get("workflow_report_md"),
            }
        )
        workflow_dirs.append(workflow_dir)

    run_named(
        "known_stratified",
        configs=known_config_path,
        stratify_known_z=True,
    )

    known_z_buckets = _discover_known_z_buckets(
        inputs,
        output_dir=root,
        recursive=recursive,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        require_metadata=require_metadata,
    )
    if len(known_z_buckets) >= 2:
        for bucket in known_z_buckets:
            run_named(
                f"holdout_{_safe_dir_suffix(bucket)}",
                configs=known_config_path,
                holdout_known_z=bucket,
                holdout_split="val",
            )
    if include_dynamic:
        run_named(
            "dynamic_regularization",
            configs=dynamic_config_path,
            stratify_known_z=True,
        )

    comparison_rows = compare_workflows(workflow_dirs)
    compare_csv = root / "workflow_compare.csv"
    compare_json = root / "workflow_compare.json"
    write_compare_csv(compare_csv, comparison_rows)
    write_compare_json(compare_json, comparison_rows)
    summary = {
        "output_dir": str(root),
        "inputs": [str(item) for item in inputs],
        "known_configs": str(known_config_path),
        "dynamic_configs": str(dynamic_config_path),
        "known_z_buckets": known_z_buckets,
        "include_dynamic": include_dynamic,
        "skip_sweep": skip_sweep,
        "workflow_count": len(runs),
        "workflows": runs,
        "workflow_compare_csv": str(compare_csv),
        "workflow_compare_json": str(compare_json),
    }
    (root / "workflow_matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trajectory CSV files/directories. Manifest input skips holdout discovery.")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gravity-y", type=float, default=0.0)
    parser.add_argument("--rank-split", default="auto")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-fps", type=float, default=80.0)
    parser.add_argument("--min-p0-hit", type=float, default=0.85)
    parser.add_argument("--max-frame-gaps", type=int, default=0)
    parser.add_argument("--known-configs", default=str(_default_config("sweep_known_distance_selection.json")))
    parser.add_argument("--dynamic-configs", default=str(_default_config("sweep_dynamic_regularization.json")))
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-depth-polyfit", action="store_true")
    parser.add_argument("--skip-rts-smoother", action="store_true")
    parser.add_argument("--calibration-min-count", type=int, default=8)
    parser.add_argument("--calibration-min-sigma", type=float, default=0.015)
    parser.add_argument("--candidate-reference", default="auto")
    args = parser.parse_args()

    summary = run_workflow_matrix(
        args.inputs,
        args.output_dir,
        recursive=args.recursive,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        require_metadata=args.require_metadata,
        train_split=args.train_split,
        device=args.device,
        gravity_y=args.gravity_y,
        rank_split=args.rank_split,
        min_rows=args.min_rows,
        min_fps=args.min_fps,
        min_p0_hit=args.min_p0_hit,
        max_frame_gaps=args.max_frame_gaps,
        known_configs=args.known_configs,
        dynamic_configs=args.dynamic_configs,
        include_dynamic=args.include_dynamic,
        skip_sweep=args.skip_sweep,
        include_depth_polyfit=not args.skip_depth_polyfit,
        include_rts_smoother=not args.skip_rts_smoother,
        calibration_min_count=args.calibration_min_count,
        calibration_min_sigma=args.calibration_min_sigma,
        candidate_reference=args.candidate_reference,
    )
    print(f"workflow matrix: {summary['output_dir']}")
    print(f"workflows: {summary['workflow_count']}")
    print(f"known_z_buckets: {summary['known_z_buckets']}")
    print(f"compare csv: {summary['workflow_compare_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
