#!/usr/bin/env python3
"""Dataset helpers for trajectory fusion experiments."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


METHOD_NAMES = ("mono", "stereo", "online")


@dataclass
class LegacySequence:
    """One track from the current TrajectoryRecorder CSV."""

    track_id: int
    rows: List[Dict[str, float]]

    @property
    def length(self) -> int:
        return len(self.rows)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read CSV rows while tolerating accidental NUL bytes in log files."""

    data = Path(path).read_bytes().replace(b"\x00", b"")
    text = data.decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def load_legacy_sequences(path: str | Path, min_track_len: int = 3) -> List[LegacySequence]:
    """Load current trajectory recorder CSV and group rows by track_id."""

    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in read_csv_rows(path):
        track_id = _safe_int(row.get("track_id"), -1)
        if track_id < 0:
            continue
        parsed = {
            "frame_id": _safe_float(row.get("frame_id")),
            "timestamp": _safe_float(row.get("timestamp")),
            "x": _safe_float(row.get("x")),
            "y": _safe_float(row.get("y")),
            "z": _safe_float(row.get("z")),
            "vx": _safe_float(row.get("vx")),
            "vy": _safe_float(row.get("vy")),
            "vz": _safe_float(row.get("vz")),
            "ax": _safe_float(row.get("ax")),
            "ay": _safe_float(row.get("ay")),
            "az": _safe_float(row.get("az")),
            "z_mono": _safe_float(row.get("z_mono"), -1.0),
            "z_stereo": _safe_float(row.get("z_stereo"), -1.0),
            "depth_method": _safe_float(row.get("depth_method")),
            "confidence": _safe_float(row.get("confidence"), 1.0),
        }
        grouped.setdefault(track_id, []).append(parsed)

    sequences: List[LegacySequence] = []
    for track_id, rows in grouped.items():
        rows.sort(key=lambda r: (r["timestamp"], r["frame_id"]))
        if len(rows) >= min_track_len:
            sequences.append(LegacySequence(track_id=track_id, rows=rows))
    sequences.sort(key=lambda seq: seq.track_id)
    return sequences


def legacy_feature_names() -> List[str]:
    """Feature order used by build_legacy_arrays()."""

    return [
        "dt",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "z_mono",
        "z_stereo",
        "confidence",
        "method_is_mono",
        "method_is_stereo",
        "method_is_blend",
        "mono_valid",
        "stereo_valid",
    ]


def build_legacy_arrays(sequence: LegacySequence) -> Dict[str, List[List[float]]]:
    """Build feature, measurement and validity arrays from a legacy sequence.

    measurements order: mono, stereo, online.
    """

    features: List[List[float]] = []
    measurements: List[List[float]] = []
    valid: List[List[float]] = []
    prev_ts = None

    for row in sequence.rows:
        ts = row["timestamp"]
        if prev_ts is None:
            dt = 0.01
        else:
            dt = max(1e-4, min(0.2, ts - prev_ts))
        prev_ts = ts

        method = int(row["depth_method"])
        z_mono = row["z_mono"]
        z_stereo = row["z_stereo"]
        z_online = row["z"]
        mono_valid = 1.0 if z_mono > 0.1 else 0.0
        stereo_valid = 1.0 if z_stereo > 0.1 else 0.0
        online_valid = 1.0 if z_online > 0.1 else 0.0

        features.append(
            [
                dt,
                row["x"],
                row["y"],
                z_online,
                row["vx"],
                row["vy"],
                row["vz"],
                z_mono if mono_valid else 0.0,
                z_stereo if stereo_valid else 0.0,
                row["confidence"],
                1.0 if method == 0 else 0.0,
                1.0 if method == 1 else 0.0,
                1.0 if method == 2 else 0.0,
                mono_valid,
                stereo_valid,
            ]
        )
        measurements.append(
            [
                z_mono if mono_valid else 0.0,
                z_stereo if stereo_valid else 0.0,
                z_online if online_valid else 0.0,
            ]
        )
        valid.append([mono_valid, stereo_valid, online_valid])

    return {"features": features, "measurements": measurements, "valid": valid}


def iter_extended_rows(path: str | Path) -> Iterable[Dict[str, str]]:
    """Yield rows from a future schema.md-compatible CSV file."""

    yield from read_csv_rows(path)


def normalize_features(features: Sequence[Sequence[float]]) -> List[List[float]]:
    """Simple per-sequence normalization for small offline experiments."""

    if not features:
        return []
    cols = len(features[0])
    means = [0.0] * cols
    for row in features:
        for i, value in enumerate(row):
            means[i] += value
    means = [value / len(features) for value in means]

    stds = [1e-6] * cols
    for row in features:
        for i, value in enumerate(row):
            diff = value - means[i]
            stds[i] += diff * diff
    stds = [(value / len(features)) ** 0.5 for value in stds]

    return [[(value - means[i]) / max(stds[i], 1e-6) for i, value in enumerate(row)] for row in features]
