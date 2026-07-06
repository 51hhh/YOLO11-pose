#!/usr/bin/env python3
"""Audit the exact training inputs used by ReliabilityNet."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

try:
    from .dataset import (
        METHOD_COLUMNS,
        METHOD_NAMES,
        build_legacy_arrays,
        find_metadata_for_csv,
        legacy_feature_names,
        load_legacy_sequences,
    )
    from .manifest import DatasetClip, is_manifest_path, load_manifest
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        METHOD_COLUMNS,
        METHOD_NAMES,
        build_legacy_arrays,
        find_metadata_for_csv,
        legacy_feature_names,
        load_legacy_sequences,
    )
    from manifest import DatasetClip, is_manifest_path, load_manifest


METHOD_FIELDNAMES = [
    "method",
    "column",
    "split",
    "valid",
    "total",
    "hit_rate",
    "median",
    "mad",
    "mean",
    "min",
    "max",
]
FEATURE_FIELDNAMES = [
    "feature",
    "split",
    "nonzero",
    "total",
    "nonzero_rate",
    "finite",
    "mean",
    "std",
    "min",
    "max",
]
LEAKAGE_METHOD_COLUMNS = {"z", "z_stereo"}
LEAKAGE_FEATURE_NAMES = {"z", "z_stereo", "x", "y", "vx", "vy", "vz", "ax", "ay", "az"}


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


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _stats(values: Sequence[float]) -> Dict[str, float | None]:
    return {
        "mean": mean(values) if values else None,
        "std": _std(values),
        "median": _median(values),
        "mad": _mad(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def resolve_clips(inputs: Sequence[str | Path], metadata: str | Path | None = None) -> List[DatasetClip]:
    if len(inputs) == 1 and is_manifest_path(inputs[0]):
        if metadata:
            raise ValueError("metadata cannot be used with a manifest")
        return load_manifest(inputs[0])
    if metadata and len(inputs) != 1:
        raise ValueError("metadata can only be used with one CSV input")
    return [
        DatasetClip(
            csv=Path(item),
            metadata=Path(metadata) if metadata else find_metadata_for_csv(item),
            split="eval",
            name=Path(item).stem,
        )
        for item in inputs
    ]


def _empty_method_accumulator() -> Dict[str, Any]:
    return {"valid": 0, "total": 0, "values": []}


def _empty_feature_accumulator() -> Dict[str, Any]:
    return {"nonzero": 0, "total": 0, "finite": 0, "values": []}


def _add_method_rows(
    report_rows: List[Dict[str, Any]],
    method_accumulators: Dict[tuple[str, str], Dict[str, Any]],
) -> None:
    method_column_by_name = dict(METHOD_COLUMNS)
    for split in sorted({key[0] for key in method_accumulators}):
        for method_name in METHOD_NAMES:
            data = method_accumulators.get((split, method_name), _empty_method_accumulator())
            values = list(data["values"])
            stats = _stats(values)
            total = int(data["total"])
            valid = int(data["valid"])
            report_rows.append(
                {
                    "method": method_name,
                    "column": method_column_by_name[method_name],
                    "split": split,
                    "valid": valid,
                    "total": total,
                    "hit_rate": valid / total if total else 0.0,
                    "median": stats["median"],
                    "mad": stats["mad"],
                    "mean": stats["mean"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )


def _add_feature_rows(
    report_rows: List[Dict[str, Any]],
    feature_accumulators: Dict[tuple[str, str], Dict[str, Any]],
) -> None:
    for split in sorted({key[0] for key in feature_accumulators}):
        for feature_name in legacy_feature_names():
            data = feature_accumulators.get((split, feature_name), _empty_feature_accumulator())
            values = list(data["values"])
            stats = _stats(values)
            total = int(data["total"])
            nonzero = int(data["nonzero"])
            report_rows.append(
                {
                    "feature": feature_name,
                    "split": split,
                    "nonzero": nonzero,
                    "total": total,
                    "nonzero_rate": nonzero / total if total else 0.0,
                    "finite": int(data["finite"]),
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )


def _schema_warnings() -> List[str]:
    warnings: List[str] = []
    method_columns = {column for _, column in METHOD_COLUMNS}
    feature_names = set(legacy_feature_names())
    leaked_method_columns = sorted(method_columns & LEAKAGE_METHOD_COLUMNS)
    leaked_feature_names = sorted(feature_names & LEAKAGE_FEATURE_NAMES)
    if leaked_method_columns:
        warnings.append(f"legacy_depth_columns_in_methods:{','.join(leaked_method_columns)}")
    if leaked_feature_names:
        warnings.append(f"legacy_online_state_in_features:{','.join(leaked_feature_names)}")
    return warnings


def audit_training_inputs(
    inputs: Sequence[str | Path],
    *,
    metadata: str | Path | None = None,
    low_method_hit_rate: float = 0.01,
    low_feature_nonzero_rate: float = 0.001,
) -> Dict[str, Any]:
    """Summarize the exact arrays consumed by train_reliability.py."""

    clips = resolve_clips(inputs, metadata=metadata)
    method_accumulators: Dict[tuple[str, str], Dict[str, Any]] = {}
    feature_accumulators: Dict[tuple[str, str], Dict[str, Any]] = {}
    clip_reports: List[Dict[str, Any]] = []
    sequence_count = 0
    frame_count = 0
    feature_names = legacy_feature_names()

    for clip in clips:
        clip_sequence_count = 0
        clip_frame_count = 0
        sequences = load_legacy_sequences(clip.csv, metadata_path=clip.metadata)
        for sequence in sequences:
            arrays = build_legacy_arrays(sequence)
            clip_sequence_count += 1
            sequence_count += 1
            rows = len(arrays["features"])
            clip_frame_count += rows
            frame_count += rows
            for method_index, method_name in enumerate(METHOD_NAMES):
                key = (clip.split, method_name)
                data = method_accumulators.setdefault(key, _empty_method_accumulator())
                for measurement, valid in zip(arrays["measurements"], arrays["valid"]):
                    data["total"] += 1
                    if valid[method_index] > 0.0:
                        data["valid"] += 1
                        data["values"].append(float(measurement[method_index]))
            for feature_row in arrays["features"]:
                for feature_name, value in zip(feature_names, feature_row):
                    key = (clip.split, feature_name)
                    data = feature_accumulators.setdefault(key, _empty_feature_accumulator())
                    data["total"] += 1
                    value_f = float(value)
                    if math.isfinite(value_f):
                        data["finite"] += 1
                        data["values"].append(value_f)
                    if abs(value_f) > 1e-9:
                        data["nonzero"] += 1

        clip_reports.append(
            {
                "name": clip.name,
                "split": clip.split,
                "csv": str(clip.csv),
                "metadata": str(clip.metadata) if clip.metadata else None,
                "sequence_count": clip_sequence_count,
                "frame_count": clip_frame_count,
            }
        )

    method_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    _add_method_rows(method_rows, method_accumulators)
    _add_feature_rows(feature_rows, feature_accumulators)

    warnings = _schema_warnings()
    if not clips:
        warnings.append("no_input_clips")
    if sequence_count == 0:
        warnings.append("no_training_sequences")
    if frame_count == 0:
        warnings.append("no_training_frames")
    low_method_rows = [
        row
        for row in method_rows
        if int(row["total"]) > 0 and float(row["hit_rate"]) < low_method_hit_rate
    ]
    all_zero_features = [
        row
        for row in feature_rows
        if int(row["total"]) > 0 and float(row["nonzero_rate"]) <= low_feature_nonzero_rate
    ]
    if low_method_rows:
        warnings.append(f"low_method_coverage:{len(low_method_rows)}")
    if all_zero_features:
        warnings.append(f"mostly_zero_features:{len(all_zero_features)}")

    return {
        "clip_count": len(clips),
        "sequence_count": sequence_count,
        "frame_count": frame_count,
        "feature_count": len(feature_names),
        "method_count": len(METHOD_NAMES),
        "feature_names": feature_names,
        "method_names": list(METHOD_NAMES),
        "warnings": warnings,
        "clip_reports": clip_reports,
        "method_coverage": method_rows,
        "feature_coverage": feature_rows,
        "low_method_coverage": low_method_rows,
        "mostly_zero_features": all_zero_features,
        "thresholds": {
            "low_method_hit_rate": low_method_hit_rate,
            "low_feature_nonzero_rate": low_feature_nonzero_rate,
        },
    }


def write_json(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_method_csv(path: str | Path, report: Dict[str, Any]) -> None:
    _write_csv(path, list(report.get("method_coverage", [])), METHOD_FIELDNAMES)


def write_feature_csv(path: str | Path, report: Dict[str, Any]) -> None:
    _write_csv(path, list(report.get("feature_coverage", [])), FEATURE_FIELDNAMES)


def print_summary(report: Dict[str, Any], *, limit: int = 12) -> None:
    print(
        "clips={clip_count} sequences={sequence_count} frames={frame_count} "
        "features={feature_count} methods={method_count}".format(**report)
    )
    print(f"warnings={report['warnings']}")
    ranked = sorted(
        report["method_coverage"],
        key=lambda row: (-float(row["hit_rate"]), str(row["split"]), str(row["method"])),
    )
    for row in ranked[:limit]:
        print(
            "method={method} split={split} valid={valid}/{total} hit={hit_rate:.3f} "
            "median={median} mad={mad}".format(**row)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="CSV files or one dataset manifest")
    parser.add_argument("--metadata", help="Optional metadata YAML for a single CSV")
    parser.add_argument("--json-out")
    parser.add_argument("--method-csv-out")
    parser.add_argument("--feature-csv-out")
    parser.add_argument("--low-method-hit-rate", type=float, default=0.01)
    parser.add_argument("--low-feature-nonzero-rate", type=float, default=0.001)
    args = parser.parse_args()

    report = audit_training_inputs(
        args.inputs,
        metadata=args.metadata,
        low_method_hit_rate=args.low_method_hit_rate,
        low_feature_nonzero_rate=args.low_feature_nonzero_rate,
    )
    if args.json_out:
        write_json(args.json_out, report)
    if args.method_csv_out:
        write_method_csv(args.method_csv_out, report)
    if args.feature_csv_out:
        write_feature_csv(args.feature_csv_out, report)
    print_summary(report)
    return 1 if "no_training_sequences" in report["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
