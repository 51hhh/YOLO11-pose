#!/usr/bin/env python3
"""Build a trajectory dataset manifest from recorded CSV clips."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .dataset import METHOD_COLUMNS, find_metadata_for_csv, read_metadata
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS, find_metadata_for_csv, read_metadata


SIDE_CAR_SUFFIXES = (
    ".frames.csv",
    ".p2_diagnostic.csv",
)
DERIVED_CSV_NAMES = {
    "candidate_consistency.csv",
    "candidate_pairwise.csv",
    "suite_metrics.csv",
    "sweep_metrics.csv",
    "sweep_ranking.csv",
    "sweep_variant_ranking.csv",
    "robust_smooth.csv",
    "robust_rts_smooth.csv",
    "depth_polyfit_smooth.csv",
    "calibrated_smoother.csv",
    "calibrated_rts_smoother.csv",
    "calibrated_smooth.csv",
    "reliability_direct.csv",
    "reliability_smoother.csv",
    "reliability_rts_smoother.csv",
    "reliability_consensus.csv",
    "trajectory_fusion_smooth.csv",
}
DERIVED_NAME_SUFFIXES = (
    ".eval.csv",
    ".check.csv",
    ".summary.csv",
)
REQUIRED_TRAJECTORY_FIELDS = {"frame_id", "timestamp", "track_id"}
CANDIDATE_DEPTH_FIELDS = {column for _, column in METHOD_COLUMNS}


@dataclass(frozen=True)
class ManifestClipEntry:
    csv: Path
    metadata: Path | None
    split: str
    name: str
    known_z: float | None = None


def _is_sidecar_or_derived_csv(path: Path) -> bool:
    name = path.name
    if any(name.endswith(suffix) for suffix in SIDE_CAR_SUFFIXES):
        return True
    if name in DERIVED_CSV_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in DERIVED_NAME_SUFFIXES)


def _csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except (OSError, csv.Error):
        return []


def _looks_like_trajectory_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv" or _is_sidecar_or_derived_csv(path):
        return False
    header = {item.strip() for item in _csv_header(path)}
    if not REQUIRED_TRAJECTORY_FIELDS.issubset(header):
        return False
    return bool(header & CANDIDATE_DEPTH_FIELDS)


def discover_csvs(paths: Sequence[str | Path], *, recursive: bool = False) -> List[Path]:
    """Discover TrajectoryRecorder main CSV clips from files or directories."""

    discovered: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            candidates = sorted(path.glob(pattern))
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate.exists() and _looks_like_trajectory_csv(candidate):
                discovered.append(candidate.resolve())

    unique: Dict[Path, Path] = {}
    for path in discovered:
        unique[path] = path
    return sorted(unique)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def _known_z_for_metadata(metadata_path: Path | None) -> float | None:
    if metadata_path is None:
        return None
    metadata = read_metadata(metadata_path)
    for key in ("known_z_m", "known_z", "known_distance_m"):
        value = _safe_float(metadata.get(key))
        if value is not None:
            return value
    return None


def _assign_split_counts(total: int, *, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    test_count = int(total * max(0.0, test_ratio))
    val_count = int(total * max(0.0, val_ratio))
    if val_ratio > 0.0 and total >= 2 and val_count == 0:
        val_count = 1
    if test_ratio > 0.0 and total >= 3 and test_count == 0:
        test_count = 1
    while val_count + test_count >= total and val_count > 0:
        val_count -= 1
    while val_count + test_count >= total and test_count > 0:
        test_count -= 1
    return val_count, test_count


def _assign_auto_splits(
    entries: Sequence[ManifestClipEntry],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    unlabeled_split: str,
) -> Dict[Path, str]:
    splits = {entry.csv: unlabeled_split for entry in entries}
    labeled = [entry for entry in entries if entry.known_z is not None]
    if not labeled:
        return splits

    shuffled = list(labeled)
    random.Random(seed).shuffle(shuffled)
    val_count, test_count = _assign_split_counts(
        len(shuffled),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    val_set = {entry.csv for entry in shuffled[:val_count]}
    test_set = {entry.csv for entry in shuffled[val_count : val_count + test_count]}
    for entry in labeled:
        if entry.csv in val_set:
            splits[entry.csv] = "val"
        elif entry.csv in test_set:
            splits[entry.csv] = "test"
        else:
            splits[entry.csv] = "train"
    return splits


def _known_z_bucket(known_z: float) -> str:
    # Millimeter buckets avoid tiny YAML/float formatting differences splitting one measured distance.
    return f"{known_z:.3f}"


def _known_z_bucket_from_value(value: str | float | None) -> str | None:
    if value is None:
        return None
    parsed = _safe_float(value)
    if parsed is None:
        raise ValueError(f"invalid known_z bucket: {value}")
    return _known_z_bucket(parsed)


def _assign_stratified_known_z_splits(
    entries: Sequence[ManifestClipEntry],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    unlabeled_split: str,
) -> Dict[Path, str]:
    splits = {entry.csv: unlabeled_split for entry in entries}
    groups: Dict[str, List[ManifestClipEntry]] = {}
    for entry in entries:
        if entry.known_z is None:
            continue
        groups.setdefault(_known_z_bucket(entry.known_z), []).append(entry)

    rng = random.Random(seed)
    for bucket in sorted(groups):
        shuffled = list(groups[bucket])
        rng.shuffle(shuffled)
        val_count, test_count = _assign_split_counts(
            len(shuffled),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        val_set = {entry.csv for entry in shuffled[:val_count]}
        test_set = {entry.csv for entry in shuffled[val_count : val_count + test_count]}
        for entry in shuffled:
            if entry.csv in val_set:
                splits[entry.csv] = "val"
            elif entry.csv in test_set:
                splits[entry.csv] = "test"
            else:
                splits[entry.csv] = "train"
    return splits


def _assign_holdout_known_z_splits(
    entries: Sequence[ManifestClipEntry],
    *,
    holdout_known_z: str | float,
    holdout_split: str,
    unlabeled_split: str,
) -> Dict[Path, str]:
    holdout_bucket = _known_z_bucket_from_value(holdout_known_z)
    splits = {entry.csv: unlabeled_split for entry in entries}
    for entry in entries:
        if entry.known_z is None:
            continue
        bucket = _known_z_bucket(entry.known_z)
        splits[entry.csv] = holdout_split if bucket == holdout_bucket else "train"
    return splits


def build_manifest(
    inputs: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
    recursive: bool = False,
    split_mode: str = "auto",
    unlabeled_split: str = "train",
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    require_metadata: bool = False,
    absolute_paths: bool = False,
    stratify_known_z: bool = False,
    holdout_known_z: str | float | None = None,
    holdout_split: str = "val",
) -> List[ManifestClipEntry]:
    """Build manifest entries from one or more dataset locations."""

    csv_paths = discover_csvs(inputs, recursive=recursive)
    entries: List[ManifestClipEntry] = []
    for csv_path in csv_paths:
        metadata_path = find_metadata_for_csv(csv_path)
        if require_metadata and metadata_path is None:
            continue
        entries.append(
            ManifestClipEntry(
                csv=csv_path,
                metadata=metadata_path,
                split="train",
                name=csv_path.stem,
                known_z=_known_z_for_metadata(metadata_path),
            )
        )

    if split_mode == "train":
        if holdout_known_z is not None:
            raise ValueError("--holdout-known-z requires --split-mode auto")
        split_by_csv = {entry.csv: "train" for entry in entries}
    elif split_mode == "eval":
        if holdout_known_z is not None:
            raise ValueError("--holdout-known-z requires --split-mode auto")
        split_by_csv = {entry.csv: "eval" for entry in entries}
    elif split_mode == "auto":
        if holdout_known_z is not None:
            split_by_csv = _assign_holdout_known_z_splits(
                entries,
                holdout_known_z=holdout_known_z,
                holdout_split=holdout_split,
                unlabeled_split=unlabeled_split,
            )
        else:
            assign_splits = _assign_stratified_known_z_splits if stratify_known_z else _assign_auto_splits
            split_by_csv = assign_splits(
                entries,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                unlabeled_split=unlabeled_split,
            )
    else:
        raise ValueError(f"unsupported split mode: {split_mode}")

    normalized: List[ManifestClipEntry] = []
    for entry in entries:
        csv_path = entry.csv if absolute_paths else _relative_to_output(entry.csv, output_path)
        metadata_path: Path | None = None
        if entry.metadata is not None:
            metadata_path = entry.metadata if absolute_paths else _relative_to_output(entry.metadata, output_path)
        normalized.append(
            ManifestClipEntry(
                csv=csv_path,
                metadata=metadata_path,
                split=split_by_csv[entry.csv],
                name=entry.name,
                known_z=entry.known_z,
            )
        )
    return normalized


def _relative_to_output(path: Path, output_path: str | Path | None) -> Path:
    base = Path(output_path).expanduser().resolve().parent if output_path else Path.cwd().resolve()
    try:
        return path.resolve().relative_to(base)
    except ValueError:
        return path.resolve()


def _yaml_quote(value: str) -> str:
    if value == "" or any(ch in value for ch in ":#{}[]&,"):
        return json.dumps(value, ensure_ascii=False)
    return value


def manifest_to_data(entries: Sequence[ManifestClipEntry]) -> Dict[str, Any]:
    clips: List[Dict[str, str]] = []
    for entry in entries:
        item = {
            "csv": str(entry.csv),
            "split": entry.split,
            "name": entry.name,
        }
        if entry.metadata is not None:
            item["metadata"] = str(entry.metadata)
        clips.append(item)
    return {"clips": clips}


def manifest_to_yaml(entries: Sequence[ManifestClipEntry]) -> str:
    lines = ["clips:"]
    for entry in entries:
        lines.append(f"  - csv: {_yaml_quote(str(entry.csv))}")
        if entry.metadata is not None:
            lines.append(f"    metadata: {_yaml_quote(str(entry.metadata))}")
        lines.append(f"    split: {_yaml_quote(entry.split)}")
        lines.append(f"    name: {_yaml_quote(entry.name)}")
    return "\n".join(lines) + "\n"


def write_manifest(path: str | Path, entries: Sequence[ManifestClipEntry]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(
            json.dumps(manifest_to_data(entries), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        output.write_text(manifest_to_yaml(entries), encoding="utf-8")


def print_summary(entries: Sequence[ManifestClipEntry]) -> None:
    split_counts: Dict[str, int] = {}
    known_counts: Dict[str, int] = {}
    for entry in entries:
        split_counts[entry.split] = split_counts.get(entry.split, 0) + 1
        if entry.known_z is not None:
            known_counts[entry.split] = known_counts.get(entry.split, 0) + 1
    print(f"clips={len(entries)} splits={split_counts} known_z={known_counts}")
    for entry in entries:
        known = "" if entry.known_z is None else f" known_z={entry.known_z:g}"
        print(f"clip={entry.name} split={entry.split} csv={entry.csv}{known}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Trajectory CSV files or directories")
    parser.add_argument("-o", "--output", required=True, help="Output dataset_manifest.yaml/json")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directory inputs")
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
        help="In auto split mode, split each known_z distance bucket independently.",
    )
    parser.add_argument(
        "--holdout-known-z",
        help="In auto split mode, assign this known_z bucket to --holdout-split and other known_z clips to train.",
    )
    parser.add_argument("--holdout-split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)

    entries = build_manifest(
        args.inputs,
        output_path=args.output,
        recursive=args.recursive,
        split_mode=args.split_mode,
        unlabeled_split=args.unlabeled_split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        require_metadata=args.require_metadata,
        absolute_paths=args.absolute_paths,
        stratify_known_z=args.stratify_known_z,
        holdout_known_z=args.holdout_known_z,
        holdout_split=args.holdout_split,
    )
    write_manifest(args.output, entries)
    print_summary(entries)
    if not entries:
        print("warning: no trajectory CSV clips discovered", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
