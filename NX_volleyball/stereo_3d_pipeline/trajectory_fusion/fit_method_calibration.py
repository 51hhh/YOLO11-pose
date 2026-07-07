#!/usr/bin/env python3
"""Fit per-method depth bias/sigma from known-distance clips."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence

try:
    from .dataset import METHOD_COLUMNS, load_legacy_sequences, resolve_method_allowlist
    from .manifest import DatasetClip, is_manifest_path, load_manifest
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS, load_legacy_sequences, resolve_method_allowlist
    from manifest import DatasetClip, is_manifest_path, load_manifest


@dataclass(frozen=True)
class CalibrationConfig:
    train_split: str = "train"
    min_count: int = 8
    min_sigma: float = 0.015
    mad_sigma_scale: float = 1.4826


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_float(metadata: Dict[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        value = _safe_float(metadata.get(key), 0.0)
        if value > 0.0:
            return value
    return 0.0


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: Sequence[float]) -> float | None:
    med = _median(values)
    if med is None:
        return None
    return _median([abs(value - med) for value in values])


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def resolve_clips(inputs: Sequence[str], metadata: str | None = None) -> List[DatasetClip]:
    if len(inputs) == 1 and is_manifest_path(inputs[0]):
        if metadata:
            raise SystemExit("--metadata cannot be used with a manifest")
        return load_manifest(inputs[0])
    if metadata and len(inputs) != 1:
        raise SystemExit("--metadata can only be used with one CSV")
    return [
        DatasetClip(
            csv=Path(item),
            metadata=Path(metadata) if metadata else None,
            split="train",
            name=Path(item).stem,
        )
        for item in inputs
    ]


def _fit_method_stats(residuals: Sequence[float], cfg: CalibrationConfig) -> Dict[str, float | int | None]:
    med = _median(residuals)
    mad = _mad(residuals)
    std = pstdev(residuals) if len(residuals) > 1 else 0.0
    robust_sigma = cfg.mad_sigma_scale * (mad or 0.0)
    sigma = max(cfg.min_sigma, robust_sigma, std * 0.5)
    return {
        "count": len(residuals),
        "bias_median": med,
        "bias_mean": mean(residuals) if residuals else None,
        "mad": mad,
        "std": std,
        "abs_p95": _percentile([abs(value) for value in residuals], 95.0),
        "sigma": sigma,
    }


def fit_method_calibration(
    clips: Sequence[DatasetClip],
    *,
    cfg: CalibrationConfig | None = None,
    method_names: Sequence[str] | str | None = None,
) -> Dict[str, Any]:
    cfg = cfg or CalibrationConfig()
    method_allowlist = resolve_method_allowlist(method_names)
    enabled = set(method_allowlist) if method_allowlist is not None else None
    residuals_by_method: Dict[str, List[float]] = {
        name: [] for name, _ in METHOD_COLUMNS if enabled is None or name in enabled
    }
    used_clips: List[Dict[str, Any]] = []
    skipped_clips: List[Dict[str, Any]] = []
    frame_count = 0

    for clip in clips:
        if clip.split != cfg.train_split:
            skipped_clips.append({"csv": str(clip.csv), "split": clip.split, "reason": "split"})
            continue
        sequences = load_legacy_sequences(clip.csv, metadata_path=clip.metadata)
        clip_known_frames = 0
        clip_known_z_values: List[float] = []
        for sequence in sequences:
            known_z = _metadata_float(sequence.metadata, ("known_z_m", "known_z", "known_distance_m"))
            if known_z <= 0.0:
                continue
            clip_known_z_values.append(known_z)
            for row in sequence.rows:
                frame_count += 1
                clip_known_frames += 1
                for method_name, key in METHOD_COLUMNS:
                    if enabled is not None and method_name not in enabled:
                        continue
                    value = float(row.get(key, -1.0))
                    if value > 0.1 and math.isfinite(value):
                        residuals_by_method[method_name].append(value - known_z)
        if clip_known_frames > 0:
            used_clips.append(
                {
                    "csv": str(clip.csv),
                    "metadata": str(clip.metadata) if clip.metadata else None,
                    "split": clip.split,
                    "name": clip.name,
                    "frames": clip_known_frames,
                    "known_z_values": sorted(set(clip_known_z_values)),
                }
            )
        else:
            skipped_clips.append({"csv": str(clip.csv), "split": clip.split, "reason": "missing_known_z"})

    methods: Dict[str, Dict[str, Any]] = {}
    insufficient: Dict[str, Dict[str, Any]] = {}
    for method_name, key in METHOD_COLUMNS:
        if enabled is not None and method_name not in enabled:
            continue
        residuals = residuals_by_method[method_name]
        stats = _fit_method_stats(residuals, cfg)
        stats["key"] = key
        if len(residuals) >= cfg.min_count:
            methods[method_name] = stats
        elif residuals:
            insufficient[method_name] = stats

    return {
        "version": 1,
        "type": "method_depth_bias_sigma",
        "config": {
            "train_split": cfg.train_split,
            "min_count": cfg.min_count,
            "min_sigma": cfg.min_sigma,
            "mad_sigma_scale": cfg.mad_sigma_scale,
            "method_allowlist": list(method_allowlist) if method_allowlist is not None else None,
        },
        "frame_count": frame_count,
        "used_clips": used_clips,
        "skipped_clips": skipped_clips,
        "methods": methods,
        "insufficient_methods": insufficient,
    }


def write_calibration(path: str | Path, calibration: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")


def load_calibration(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_report(calibration: Dict[str, Any]) -> None:
    print(
        "frames={frames} used_clips={used} calibrated_methods={methods} insufficient={insufficient}".format(
            frames=calibration["frame_count"],
            used=len(calibration["used_clips"]),
            methods=len(calibration["methods"]),
            insufficient=len(calibration["insufficient_methods"]),
        )
    )
    print("method,key,count,bias_median,sigma,mad,abs_p95")
    for method, stats in sorted(
        calibration["methods"].items(),
        key=lambda item: (-int(item[1]["count"]), item[0]),
    ):
        print(
            "{method},{key},{count},{bias},{sigma},{mad},{p95}".format(
                method=method,
                key=stats["key"],
                count=stats["count"],
                bias=_fmt(stats["bias_median"]),
                sigma=_fmt(stats["sigma"]),
                mad=_fmt(stats["mad"]),
                p95=_fmt(stats["abs_p95"]),
            )
        )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="CSV file(s), or one dataset manifest YAML/JSON")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--metadata", help="Optional metadata YAML for one CSV input")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--min-sigma", type=float, default=0.015)
    parser.add_argument("--methods", default=None, help="Optional method allowlist/preset such as p0, p0p1, p0p1_ncc_xfeat")
    args = parser.parse_args()

    clips = resolve_clips(args.inputs, metadata=args.metadata)
    calibration = fit_method_calibration(
        clips,
        cfg=CalibrationConfig(
            train_split=args.train_split,
            min_count=args.min_count,
            min_sigma=args.min_sigma,
        ),
        method_names=args.methods,
    )
    write_calibration(args.output, calibration)
    print_report(calibration)
    print(f"wrote {args.output}")
    if not calibration["methods"]:
        print("no calibrated methods; check train split and known_z metadata")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
