#!/usr/bin/env python3
"""Feature array construction for trajectory fusion datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    from .dataset_feature_rows import build_feature_row
    from .dataset_io import safe_float
    from .dataset_schema import METHOD_COLUMNS, METHOD_NAMES, LegacySequence
except ImportError:
    from dataset_feature_rows import build_feature_row  # type: ignore
    from dataset_io import safe_float  # type: ignore
    from dataset_schema import METHOD_COLUMNS, METHOD_NAMES, LegacySequence  # type: ignore


def _metadata_float(metadata: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return safe_float(value, default)
    return default


def _metadata_bool(metadata: Dict[str, Any], keys: Sequence[str], default: bool = False) -> bool:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if value is not None:
            return str(value).strip().lower() in {"1", "true", "yes", "on", "static"}
    return default


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    return _median([abs(value - med) for value in values])


def build_legacy_arrays(sequence: LegacySequence) -> Dict[str, List[List[float]]]:
    """Build feature, measurement and validity arrays from a recorder sequence.

    measurements order: METHOD_NAMES.
    """

    features: List[List[float]] = []
    measurements: List[List[float]] = []
    valid: List[List[float]] = []
    labels: List[List[float]] = []
    prev_ts = None
    prev_valid_ts = None
    prev_median_z = 0.0
    prev_candidate_dz = 0.0
    have_prev_median = False
    metadata = sequence.metadata
    known_z = _metadata_float(metadata, ("known_z_m", "known_z", "known_distance_m"), 0.0)
    known_z_tol = _metadata_float(metadata, ("known_z_tolerance_m", "known_z_tolerance"), 0.0)
    known_z_min = _metadata_float(metadata, ("known_z_min_m", "known_z_min"), 0.0)
    known_z_max = _metadata_float(metadata, ("known_z_max_m", "known_z_max"), 0.0)
    if known_z > 0.0 and known_z_tol > 0.0 and (known_z_min <= 0.0 or known_z_max <= 0.0):
        known_z_min = known_z - known_z_tol
        known_z_max = known_z + known_z_tol
    known_z_valid = 1.0 if known_z > 0.0 else 0.0
    known_z_range_valid = 1.0 if known_z_min > 0.0 and known_z_max > known_z_min else 0.0
    static_flag = 1.0 if _metadata_bool(metadata, ("static", "is_static"), False) else 0.0
    landing_frame = _metadata_float(metadata, ("landing_frame",), -1.0)
    landing_frame_valid = 1.0 if landing_frame >= 0.0 else 0.0

    for row in sequence.rows:
        ts = row["timestamp"]
        if prev_ts is None:
            dt = 0.01
        else:
            dt = max(1e-4, min(0.2, ts - prev_ts))
        prev_ts = ts

        measurements_row = []
        valid_row = []
        for _, key in METHOD_COLUMNS:
            value = row[key]
            is_valid = 1.0 if value > 0.1 else 0.0
            measurements_row.append(value if is_valid else 0.0)
            valid_row.append(is_valid)
        valid_by_key = {
            key: valid_row[idx] for idx, (_, key) in enumerate(METHOD_COLUMNS)
        }
        candidate_values = [value for value, is_valid in zip(measurements_row, valid_row) if is_valid > 0.0]
        candidate_median_z = _median(candidate_values)
        candidate_mad_z = _mad(candidate_values)
        candidate_valid_count = float(len(candidate_values))
        if candidate_valid_count <= 0.0:
            candidate_dz = 0.0
            candidate_ddz = 0.0
        elif have_prev_median and prev_valid_ts is not None:
            raw_valid_dt = ts - prev_valid_ts
            valid_dt = max(1e-4, min(0.5, raw_valid_dt))
            candidate_dz = (candidate_median_z - prev_median_z) / valid_dt
            if raw_valid_dt > dt * 1.5:
                candidate_ddz = 0.0
            else:
                candidate_ddz = (candidate_dz - prev_candidate_dz) / dt
        else:
            candidate_dz = 0.0
            candidate_ddz = 0.0
        if candidate_valid_count > 0.0:
            prev_median_z = candidate_median_z
            prev_candidate_dz = candidate_dz
            prev_valid_ts = ts
            have_prev_median = True

        features.append(
            build_feature_row(
                row,
                valid_by_key,
                valid_row,
                dt,
                candidate_median_z,
                candidate_mad_z,
                candidate_valid_count,
                candidate_dz,
                candidate_ddz,
            )
        )
        measurements.append(measurements_row)
        valid.append(valid_row)
        labels.append(
            [
                known_z,
                known_z_valid,
                known_z_min,
                known_z_max,
                known_z_range_valid,
                static_flag,
                landing_frame,
                landing_frame_valid,
            ]
        )

    return {"features": features, "measurements": measurements, "valid": valid, "labels": labels}
