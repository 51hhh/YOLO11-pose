#!/usr/bin/env python3
"""Train and evaluate multiple ReliabilityNet configurations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .rank_sweep_metrics import rank_metrics, write_ranking
    from .run_evaluation_suite import run_suite
    from .summarize_evaluation_suite import summarize_reliability_methods, summarize_suite
except ImportError:  # pragma: no cover - direct script execution
    from rank_sweep_metrics import rank_metrics, write_ranking
    from run_evaluation_suite import run_suite
    from summarize_evaluation_suite import summarize_reliability_methods, summarize_suite


DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "h32_e80_bias10",
        "epochs": 80,
        "hidden": 32,
        "static_jitter_weight": 1.0,
        "bias_reg_weight": 1.0,
        "known_z_weight": 1.0,
        "known_z_range_weight": 0.5,
        "leave_one_weight": 0.0,
    },
    {
        "name": "h32_e120_bias10",
        "epochs": 120,
        "hidden": 32,
        "static_jitter_weight": 1.0,
        "bias_reg_weight": 1.0,
        "known_z_weight": 1.0,
        "known_z_range_weight": 0.5,
        "leave_one_weight": 0.0,
    },
    {
        "name": "h64_e80_bias10",
        "epochs": 80,
        "hidden": 64,
        "static_jitter_weight": 1.0,
        "bias_reg_weight": 1.0,
        "known_z_weight": 1.0,
        "known_z_range_weight": 0.5,
        "leave_one_weight": 0.0,
    },
    {
        "name": "h32_e80_bias10_leave002",
        "epochs": 80,
        "hidden": 32,
        "static_jitter_weight": 1.0,
        "bias_reg_weight": 1.0,
        "known_z_weight": 1.0,
        "known_z_range_weight": 0.5,
        "leave_one_weight": 0.02,
        "leave_one_max_methods": 8,
    },
]

TRAIN_OPTION_KEYS = (
    "epochs",
    "lr",
    "hidden",
    "known_z_weight",
    "known_z_range_weight",
    "static_jitter_weight",
    "bias_reg_weight",
    "leave_one_weight",
    "leave_one_max_methods",
)


def _safe_name(value: str) -> str:
    allowed = []
    for char in value.strip():
        allowed.append(char if char.isalnum() or char in "._-" else "_")
    name = "".join(allowed).strip("._")
    return name or "config"


def _load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def load_sweep_configs(path: str | Path | None) -> List[Dict[str, Any]]:
    """Load sweep configs from YAML/JSON, or return conservative defaults."""

    if path is None:
        return [dict(item) for item in DEFAULT_CONFIGS]
    raw = _load_yaml_or_json(Path(path))
    configs = raw.get("configs", raw) if isinstance(raw, dict) else raw
    if not isinstance(configs, list):
        raise ValueError("sweep config must be a list or a mapping with 'configs'")
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(configs):
        if not isinstance(item, dict):
            raise ValueError(f"sweep config #{index} must be a mapping")
        config = dict(item)
        config["name"] = _safe_name(str(config.get("name") or f"config_{index + 1}"))
        out.append(config)
    if not out:
        raise ValueError("sweep config is empty")
    return out


def _option_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_train_command(
    inputs: Sequence[str],
    checkpoint: Path,
    config: Dict[str, Any],
    *,
    metadata: str | None = None,
    train_split: str = "train",
    device: str = "cpu",
) -> List[str]:
    script = Path(__file__).with_name("train_reliability.py")
    cmd = [
        sys.executable,
        str(script),
        *inputs,
        "--train-split",
        train_split,
        "--device",
        device,
        "-o",
        str(checkpoint),
    ]
    if metadata:
        cmd.extend(["--metadata", metadata])
    for key in TRAIN_OPTION_KEYS:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool):
            if value:
                cmd.append(_option_name(key))
            continue
        cmd.extend([_option_name(key), str(value)])
    return cmd


def _write_sweep_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _write_combined_metrics(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    prefix = {"config", "checkpoint", "suite_dir"}
    fieldnames = [
        "config",
        "checkpoint",
        "suite_dir",
        *[key for key in rows[0].keys() if key not in prefix],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    inputs: Sequence[str],
    output_dir: str | Path,
    *,
    metadata: str | None = None,
    configs_path: str | Path | None = None,
    calibration: str | Path | None = None,
    train_split: str = "train",
    device: str = "cpu",
    gravity_y: float = 9.81,
    use_static_known_z: bool = False,
    rank_split: str = "auto",
) -> Dict[str, Any]:
    root = Path(output_dir)
    checkpoints_dir = root / "checkpoints"
    suites_dir = root / "suites"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    suites_dir.mkdir(parents=True, exist_ok=True)

    configs = load_sweep_configs(configs_path)
    summary: Dict[str, Any] = {
        "output_dir": str(root),
        "configs_path": str(configs_path) if configs_path else None,
        "calibration": str(calibration) if calibration else None,
        "train_split": train_split,
        "device": device,
        "gravity_y": gravity_y,
        "use_static_known_z": use_static_known_z,
        "rank_split": rank_split,
        "sweep_ranking": str(root / "sweep_ranking.csv"),
        "sweep_variant_ranking": str(root / "sweep_variant_ranking.csv"),
        "sweep_reliability_methods": str(root / "sweep_reliability_methods.csv"),
        "runs": [],
    }
    combined_rows: List[Dict[str, Any]] = []
    combined_method_rows: List[Dict[str, Any]] = []

    for config in configs:
        name = _safe_name(str(config["name"]))
        checkpoint = checkpoints_dir / f"{name}.pt"
        suite_dir = suites_dir / name
        train_cmd = build_train_command(
            inputs,
            checkpoint,
            config,
            metadata=metadata,
            train_split=train_split,
            device=device,
        )
        print(f"training {name}: {' '.join(train_cmd)}", flush=True)
        subprocess.run(train_cmd, check=True)

        suite_report = run_suite(
            inputs,
            suite_dir,
            metadata=metadata,
            checkpoint=checkpoint,
            calibration=calibration,
            device=device,
            gravity_y=gravity_y,
            use_static_known_z=use_static_known_z,
        )
        metrics_path = suite_dir / "suite_metrics.csv"
        rows = summarize_suite(suite_dir, metrics_path)
        method_metrics_path = suite_dir / "suite_reliability_methods.csv"
        method_rows = summarize_reliability_methods(suite_dir, method_metrics_path)
        for row in rows:
            combined_rows.append(
                {
                    "config": name,
                    "checkpoint": str(checkpoint),
                    "suite_dir": str(suite_dir),
                    **row,
                }
            )
        for row in method_rows:
            combined_method_rows.append(
                {
                    "config": name,
                    "checkpoint": str(checkpoint),
                    "suite_dir": str(suite_dir),
                    **row,
                }
            )
        summary["runs"].append(
            {
                "name": name,
                "config": config,
                "checkpoint": str(checkpoint),
                "calibration": str(calibration) if calibration else None,
                "suite_dir": str(suite_dir),
                "suite_summary": str(suite_dir / "suite_summary.json"),
                "suite_metrics": str(metrics_path),
                "suite_reliability_methods": str(method_metrics_path),
                "train_command": train_cmd,
                "clip_count": len(suite_report.get("clips", [])),
            }
        )
        _write_sweep_summary(root / "sweep_summary.json", summary)
        metrics_path = root / "sweep_metrics.csv"
        _write_combined_metrics(metrics_path, combined_rows)
        _write_combined_metrics(root / "sweep_reliability_methods.csv", combined_method_rows)
        ranking = rank_metrics(metrics_path, split=rank_split)
        write_ranking(root / "sweep_ranking.csv", ranking)
        variant_ranking = rank_metrics(metrics_path, variant="all", split=rank_split)
        write_ranking(root / "sweep_variant_ranking.csv", variant_ranking)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="CSV file(s), or one dataset manifest YAML/JSON")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--metadata", help="Optional metadata YAML for a single CSV")
    parser.add_argument("--configs", help="Optional YAML/JSON config list. Defaults to a conservative sweep.")
    parser.add_argument("--calibration", help="Optional per-method calibration JSON passed into each suite.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gravity-y", type=float, default=9.81)
    parser.add_argument("--rank-split", default="auto", help="Ranking split. 'auto' prefers val when present.")
    parser.add_argument(
        "--use-static-known-z",
        action="store_true",
        help="Use known_z as smoother update during suite evaluation. Off by default to avoid label leakage.",
    )
    args = parser.parse_args()

    summary = run_sweep(
        args.inputs,
        args.output_dir,
        metadata=args.metadata,
        configs_path=args.configs,
        calibration=args.calibration,
        train_split=args.train_split,
        device=args.device,
        gravity_y=args.gravity_y,
        use_static_known_z=args.use_static_known_z,
        rank_split=args.rank_split,
    )
    print(f"wrote sweep for {len(summary['runs'])} config(s) to {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
