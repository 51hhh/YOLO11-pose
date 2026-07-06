#!/usr/bin/env python3
"""Analyze depth-candidate consistency before training reliability models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from .dataset import METHOD_COLUMNS, LegacySequence, find_metadata_for_csv, load_legacy_sequences
    from .manifest import DatasetClip, is_manifest_path, load_manifest
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS, LegacySequence, find_metadata_for_csv, load_legacy_sequences
    from manifest import DatasetClip, is_manifest_path, load_manifest


P0_REFERENCE_KEYS = (
    "z_bbox_center",
    "z_circle_center",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
)
METHOD_KEY_BY_NAME = {name: key for name, key in METHOD_COLUMNS}
METHOD_NAME_BY_KEY = {key: name for name, key in METHOD_COLUMNS}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_float(metadata: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = _safe_float(metadata.get(key), None)
        if value is not None:
            return value
    return default


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


def _stats(values: Sequence[float], *, abs_p95: bool = False) -> Dict[str, float | int | None]:
    out: Dict[str, float | int | None] = {
        "count": len(values),
        "mean": mean(values) if values else None,
        "std": pstdev(values) if len(values) > 1 else 0.0 if values else None,
        "median": _median(values),
        "mad": _mad(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
    if abs_p95:
        out["abs_p95"] = _percentile([abs(value) for value in values], 95.0)
    return out


def _valid_method_depths(row: Dict[str, float]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, key in METHOD_COLUMNS:
        value = float(row.get(key, -1.0))
        if value > 0.1 and math.isfinite(value):
            values[name] = value
    return values


def _known_z(metadata: Dict[str, Any]) -> float:
    return _metadata_float(metadata, ("known_z_m", "known_z", "known_distance_m"), 0.0)


def _method_reference_name(value: str) -> str:
    raw = value.removeprefix("method:").strip()
    if raw in METHOD_KEY_BY_NAME:
        return raw
    if raw in METHOD_NAME_BY_KEY:
        return METHOD_NAME_BY_KEY[raw]
    raise ValueError(f"unknown reference method: {value}")


def _reference_value(
    row: Dict[str, float],
    valid: Dict[str, float],
    *,
    reference: str,
    known_z: float,
) -> Tuple[float | None, str]:
    if reference == "auto":
        if known_z > 0.0:
            return known_z, "known_z"
        return _reference_value(row, valid, reference="p0_median", known_z=known_z)
    if reference == "known_z":
        return (known_z, "known_z") if known_z > 0.0 else (None, "known_z_missing")
    if reference == "candidate_median":
        med = _median(list(valid.values()))
        return med, "candidate_median" if med is not None else "candidate_median_missing"
    if reference == "p0_median":
        p0_values = [
            float(row.get(key, -1.0))
            for key in P0_REFERENCE_KEYS
            if float(row.get(key, -1.0)) > 0.1 and math.isfinite(float(row.get(key, -1.0)))
        ]
        med = _median(p0_values)
        return med, "p0_median" if med is not None else "p0_median_missing"
    method_name = _method_reference_name(reference)
    value = valid.get(method_name)
    return value, method_name if value is not None else f"{method_name}_missing"


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
            metadata=Path(metadata) if metadata else find_metadata_for_csv(item),
            split="eval",
            name=Path(item).stem,
        )
        for item in inputs
    ]


def _method_summary(
    data: Dict[str, Dict[str, List[float] | int]],
    *,
    total_frames: int,
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name, values in data.items():
        z_values = list(values["z"])  # type: ignore[index]
        residuals = list(values["residual"])  # type: ignore[index]
        valid = int(values["valid"])  # type: ignore[arg-type]
        summary[name] = {
            "key": METHOD_KEY_BY_NAME[name],
            "valid": valid,
            "total": total_frames,
            "hit_rate": valid / total_frames if total_frames else 0.0,
            "z": _stats(z_values),
            "residual": _stats(residuals, abs_p95=True),
        }
    return summary


def _pairwise_summary(
    pair_diffs: Dict[Tuple[str, str], List[float]],
    *,
    min_pair_count: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (left, right), diffs in sorted(pair_diffs.items()):
        if len(diffs) < min_pair_count:
            continue
        stats = _stats(diffs, abs_p95=True)
        rows.append(
            {
                "left": left,
                "right": right,
                "left_key": METHOD_KEY_BY_NAME[left],
                "right_key": METHOD_KEY_BY_NAME[right],
                "count": len(diffs),
                "mean_diff_left_minus_right": stats["mean"],
                "median_diff_left_minus_right": stats["median"],
                "mad_diff": stats["mad"],
                "abs_p95_diff": stats["abs_p95"],
            }
        )
    rows.sort(key=lambda item: (-int(item["count"]), str(item["left"]), str(item["right"])))
    return rows


def _empty_method_data() -> Dict[str, List[float] | int]:
    return {"valid": 0, "z": [], "residual": []}


def _analyze_sequences(
    sequences: Iterable[LegacySequence],
    *,
    reference: str,
    min_pair_count: int,
) -> Dict[str, Any]:
    total_frames = 0
    reference_counts: Counter[str] = Counter()
    methods: Dict[str, Dict[str, List[float] | int]] = defaultdict(_empty_method_data)
    pair_diffs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    track_reports: Dict[str, Any] = {}

    for sequence in sequences:
        track_total = 0
        track_reference_counts: Counter[str] = Counter()
        track_methods: Dict[str, Dict[str, List[float] | int]] = defaultdict(_empty_method_data)
        track_pair_diffs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        known_z = _known_z(sequence.metadata)

        for row in sequence.rows:
            track_total += 1
            total_frames += 1
            valid = _valid_method_depths(row)
            ref_value, ref_kind = _reference_value(row, valid, reference=reference, known_z=known_z)
            reference_counts[ref_kind] += 1
            track_reference_counts[ref_kind] += 1

            for name, z_value in valid.items():
                methods[name]["valid"] = int(methods[name]["valid"]) + 1
                methods[name]["z"].append(z_value)  # type: ignore[union-attr]
                track_methods[name]["valid"] = int(track_methods[name]["valid"]) + 1
                track_methods[name]["z"].append(z_value)  # type: ignore[union-attr]
                if ref_value is not None and math.isfinite(ref_value):
                    residual = z_value - ref_value
                    methods[name]["residual"].append(residual)  # type: ignore[union-attr]
                    track_methods[name]["residual"].append(residual)  # type: ignore[union-attr]

            for left, right in combinations(sorted(valid), 2):
                diff = valid[left] - valid[right]
                pair_diffs[(left, right)].append(diff)
                track_pair_diffs[(left, right)].append(diff)

        track_key = str(sequence.track_id)
        if track_key in track_reports:
            track_key = f"{track_key}_{len(track_reports) + 1}"
        track_reports[track_key] = {
            "frames": track_total,
            "known_z": known_z if known_z > 0.0 else None,
            "reference_counts": dict(track_reference_counts),
            "methods": _method_summary(track_methods, total_frames=track_total),
            "pairwise": _pairwise_summary(track_pair_diffs, min_pair_count=min_pair_count),
        }

    return {
        "frames": total_frames,
        "reference_counts": dict(reference_counts),
        "methods": _method_summary(methods, total_frames=total_frames),
        "pairwise": _pairwise_summary(pair_diffs, min_pair_count=min_pair_count),
        "tracks": track_reports,
    }


def analyze_candidate_consistency(
    inputs: Sequence[str],
    *,
    metadata: str | None = None,
    reference: str = "auto",
    min_pair_count: int = 5,
) -> Dict[str, Any]:
    clips = resolve_clips(inputs, metadata)
    return analyze_candidate_clips(
        clips,
        inputs=list(inputs),
        reference=reference,
        min_pair_count=min_pair_count,
    )


def analyze_candidate_clips(
    clips: Sequence[DatasetClip],
    *,
    inputs: Sequence[str] | None = None,
    reference: str = "auto",
    min_pair_count: int = 5,
) -> Dict[str, Any]:
    clip_reports: List[Dict[str, Any]] = []
    aggregate_sequences: List[LegacySequence] = []

    for clip in clips:
        sequences = load_legacy_sequences(clip.csv, metadata_path=clip.metadata)
        aggregate_sequences.extend(sequences)
        clip_report = _analyze_sequences(
            sequences,
            reference=reference,
            min_pair_count=min_pair_count,
        )
        clip_report.update(
            {
                "name": clip.name or clip.csv.stem,
                "split": clip.split,
                "csv": str(clip.csv),
                "metadata": str(clip.metadata) if clip.metadata else None,
            }
        )
        clip_reports.append(clip_report)

    aggregate = _analyze_sequences(
        aggregate_sequences,
        reference=reference,
        min_pair_count=min_pair_count,
    )
    return {
        "inputs": list(inputs) if inputs is not None else [str(clip.csv) for clip in clips],
        "reference": reference,
        "min_pair_count": min_pair_count,
        "clip_count": len(clips),
        "aggregate": aggregate,
        "clips": clip_reports,
    }


def _write_json(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_method_csv(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "clip",
        "split",
        "track_id",
        "method",
        "key",
        "valid",
        "total",
        "hit_rate",
        "z_median",
        "z_mad",
        "residual_median",
        "residual_mad",
        "residual_abs_p95",
    ]
    rows: List[Dict[str, Any]] = []

    def add_methods(scope: str, clip: str, split: str, track_id: str, methods: Dict[str, Any]) -> None:
        for method, item in sorted(methods.items()):
            rows.append(
                {
                    "scope": scope,
                    "clip": clip,
                    "split": split,
                    "track_id": track_id,
                    "method": method,
                    "key": item["key"],
                    "valid": item["valid"],
                    "total": item["total"],
                    "hit_rate": item["hit_rate"],
                    "z_median": item["z"]["median"],
                    "z_mad": item["z"]["mad"],
                    "residual_median": item["residual"]["median"],
                    "residual_mad": item["residual"]["mad"],
                    "residual_abs_p95": item["residual"]["abs_p95"],
                }
            )

    add_methods("aggregate", "", "", "", report["aggregate"]["methods"])
    for clip in report["clips"]:
        add_methods("clip", clip["name"], clip["split"], "", clip["methods"])
        for track_id, track in clip["tracks"].items():
            add_methods("track", clip["name"], clip["split"], track_id, track["methods"])

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_csv(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "left",
        "right",
        "left_key",
        "right_key",
        "count",
        "mean_diff_left_minus_right",
        "median_diff_left_minus_right",
        "mad_diff",
        "abs_p95_diff",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["aggregate"]["pairwise"])


def print_report(report: Dict[str, Any], *, top: int = 20) -> None:
    aggregate = report["aggregate"]
    print(
        f"clips={report['clip_count']} frames={aggregate['frames']} "
        f"reference={report['reference']} ref_counts={aggregate['reference_counts']}"
    )
    print("method,key,valid,total,hit,z_median,z_mad,residual_median,residual_mad,residual_abs_p95")
    method_rows = sorted(
        aggregate["methods"].items(),
        key=lambda item: (-int(item[1]["valid"]), item[0]),
    )
    for method, item in method_rows[:top]:
        print(
            "{method},{key},{valid},{total},{hit:.3f},{zmed},{zmad},{rmed},{rmad},{rp95}".format(
                method=method,
                key=item["key"],
                valid=item["valid"],
                total=item["total"],
                hit=item["hit_rate"],
                zmed=_fmt(item["z"]["median"]),
                zmad=_fmt(item["z"]["mad"]),
                rmed=_fmt(item["residual"]["median"]),
                rmad=_fmt(item["residual"]["mad"]),
                rp95=_fmt(item["residual"]["abs_p95"]),
            )
        )
    print("pairwise,left,right,count,median_diff,mad_diff,abs_p95")
    for item in aggregate["pairwise"][:top]:
        print(
            "pairwise,{left},{right},{count},{median},{mad},{p95}".format(
                left=item["left"],
                right=item["right"],
                count=item["count"],
                median=_fmt(item["median_diff_left_minus_right"]),
                mad=_fmt(item["mad_diff"]),
                p95=_fmt(item["abs_p95_diff"]),
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
    parser.add_argument("--metadata", help="Optional metadata YAML for one CSV input")
    parser.add_argument(
        "--reference",
        default="auto",
        help="auto, known_z, p0_median, candidate_median, or method:<name/key>",
    )
    parser.add_argument("--min-pair-count", type=int, default=5)
    parser.add_argument("--json-out")
    parser.add_argument("--csv-out", help="Write method residual summary CSV")
    parser.add_argument("--pairwise-csv-out", help="Write aggregate pairwise bias CSV")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    report = analyze_candidate_consistency(
        args.inputs,
        metadata=args.metadata,
        reference=args.reference,
        min_pair_count=args.min_pair_count,
    )
    print_report(report, top=args.top)
    if args.json_out:
        _write_json(args.json_out, report)
    if args.csv_out:
        write_method_csv(args.csv_out, report)
    if args.pairwise_csv_out:
        write_pairwise_csv(args.pairwise_csv_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
