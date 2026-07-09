#!/usr/bin/env python3
"""Backfill known_z / static weak labels into recorder metadata files.

The 2026-07-07 dataset was recorded with the ``nx_p1_dy_regression.sh`` harness,
so every ``traj.metadata.yaml`` carries ``known_z: null`` even though the wiki
dataset catalogue records the tape-measured distance per run.

This tool parses the wiki catalogue (``wiki/数据集目录.md``), extracts the
tape-measured static distance from section headers that contain ``第二组``
(second-group evening recordings, which the user confirmed are tape/laser
ground truth), maps each run id to that distance, and writes the weak labels
that ``dataset.py`` actually reads for supervision:

    known_z, known_z_tolerance, static, known_z_training, motion_type

Only second-group ``固定位置 Nm（第二组...）`` sections are used. The first two
sections ("静止排球" / "固定位置 (3m/4m/5m)") record distance reflected FROM the
stereo depth reading itself, so using them as ``known_z`` would leak the
measurement into the label. Those are skipped by design.

Dry-run by default. Pass --apply to write.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# Section header example:
#   ## 八、固定位置 4m（第二组，晚间录制）
# We only accept headers that contain 第二组 (tape-measured ground truth).
_SECTION_RE = re.compile(r"^#+\s*[^\n]*固定位置\s*([0-9]+(?:\.[0-9]+)?)\s*m[^\n]*第二组")
_ANY_HEADER_RE = re.compile(r"^#+\s")
# Table row: | `165405` | 5290 | ... |
_RUN_RE = re.compile(r"^\|\s*`?([0-9]{6})`?\s*\|")


@dataclass
class RunLabel:
    run_id: str
    known_z: float
    section_title: str


def parse_catalogue(md_path: Path) -> List[RunLabel]:
    """Extract (run_id -> known_z) for second-group fixed-distance sections."""

    labels: List[RunLabel] = []
    current_distance: float | None = None
    current_title = ""
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if _ANY_HEADER_RE.match(line):
            match = _SECTION_RE.match(line)
            if match:
                current_distance = float(match.group(1))
                current_title = line.lstrip("# ").strip()
            else:
                # Entering an unrelated section closes the current capture.
                current_distance = None
                current_title = ""
            continue
        if current_distance is None:
            continue
        run_match = _RUN_RE.match(line)
        if run_match:
            labels.append(
                RunLabel(
                    run_id=run_match.group(1),
                    known_z=current_distance,
                    section_title=current_title,
                )
            )
    return labels


def _parse_flat_yaml(text: str) -> List[Tuple[str, str]]:
    """Parse the flat ``key: value`` recorder metadata, preserving order.

    The recorder metadata is a flat, single-level YAML with no nesting, lists,
    or block scalars, so a line-oriented parse is sufficient and avoids a
    PyYAML dependency. Comment/blank lines are preserved as (``None``, raw).
    """

    items: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            items.append((None, raw))
            continue
        if ":" not in raw:
            items.append((None, raw))
            continue
        key, _, value = raw.partition(":")
        items.append((key.strip(), value.strip()))
    return items


def _render_flat_yaml(items: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for key, value in items:
        if key is None:
            lines.append(value)
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def build_metadata(
    existing: str,
    known_z: float,
    tolerance: float,
) -> str:
    """Return updated flat-yaml metadata with weak labels applied."""

    items = _parse_flat_yaml(existing)
    updates = {
        "dataset_type": "static_known_z",
        "known_z": f"{known_z:.4g}",
        "known_z_tolerance": f"{tolerance:.4g}",
        "static": "true",
        "known_z_training": "true",
        "motion_type": "static",
    }
    seen: Dict[str, bool] = {key: False for key in updates}
    out: List[Tuple[str, str]] = []
    for key, value in items:
        if key in updates:
            out.append((key, updates[key]))
            seen[key] = True
        else:
            out.append((key, value))
    for key, applied in seen.items():
        if not applied:
            out.append((key, updates[key]))
    return _render_flat_yaml(out)


def resolve_metadata_path(runs_dir: Path, run_id: str) -> Path | None:
    """Find the traj.metadata.yaml for a given run id under runs_dir."""

    matches = sorted(runs_dir.glob(f"*{run_id}*/traj.metadata.yaml"))
    if not matches:
        return None
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        default="wiki/数据集目录.md",
        help="Path to the wiki dataset catalogue markdown (relative to stereo_3d_pipeline)",
    )
    parser.add_argument(
        "--runs-dir",
        default="test_logs/recording_runs_20260707",
        help="Directory containing per-run recording folders",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="known_z_tolerance in meters (soft band for known_z_range loss)",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent  # stereo_3d_pipeline/
    catalogue = (base / args.catalogue).resolve()
    runs_dir = (base / args.runs_dir).resolve()

    if not catalogue.exists():
        raise SystemExit(f"catalogue not found: {catalogue}")
    if not runs_dir.exists():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    labels = parse_catalogue(catalogue)
    if not labels:
        raise SystemExit("no second-group fixed-distance runs parsed from catalogue")

    print(f"catalogue: {catalogue}")
    print(f"runs dir:  {runs_dir}")
    print(f"mode:      {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"parsed {len(labels)} second-group runs (tape-measured known_z)\n")

    by_distance: Dict[float, List[str]] = {}
    for label in labels:
        by_distance.setdefault(label.known_z, []).append(label.run_id)
    for distance in sorted(by_distance):
        runs = ", ".join(sorted(by_distance[distance]))
        print(f"  known_z={distance:>5.1f}m : {runs}")
    print()

    applied = 0
    missing = 0
    for label in labels:
        meta_path = resolve_metadata_path(runs_dir, label.run_id)
        if meta_path is None:
            print(f"  MISSING metadata for run {label.run_id} (known_z={label.known_z}m)")
            missing += 1
            continue
        existing = meta_path.read_text(encoding="utf-8")
        updated = build_metadata(existing, label.known_z, args.tolerance)
        if args.apply:
            meta_path.write_text(updated, encoding="utf-8")
            applied += 1
        else:
            print(f"  [{label.run_id}] known_z={label.known_z}m -> {meta_path}")

    print()
    if args.apply:
        print(f"applied {applied} metadata files, {missing} missing")
    else:
        print(f"dry-run: would update {len(labels) - missing} files, {missing} missing")
        print("re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
