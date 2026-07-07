#!/usr/bin/env python3
"""Apply fitted per-method depth calibration through the robust smoother."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .dataset import METHOD_COLUMNS, load_legacy_sequences, resolve_method_allowlist
    from .fit_method_calibration import load_calibration
    from .robust_smoother import (
        SmootherConfig,
        ZMeasurement,
        group_correlated_z_measurements,
        smooth_sequence,
        smooth_sequence_rts,
        write_output,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS, load_legacy_sequences, resolve_method_allowlist
    from fit_method_calibration import load_calibration
    from robust_smoother import (
        SmootherConfig,
        ZMeasurement,
        group_correlated_z_measurements,
        smooth_sequence,
        smooth_sequence_rts,
        write_output,
    )


def _empty_diagnostic() -> Dict[str, float | str]:
    return {
        "calibrated_smoother_valid_count": 0.0,
        "calibrated_smoother_group_count": 0.0,
        "calibrated_smoother_top_method_name": "",
        "calibrated_smoother_top_weight": 0.0,
        "calibrated_smoother_top_raw_z": 0.0,
        "calibrated_smoother_top_corrected_z": 0.0,
        "calibrated_smoother_top_bias": 0.0,
        "calibrated_smoother_top_sigma": 0.0,
    }


def _method_observations(
    row: Dict[str, float],
    calibration: Dict[str, Any],
    *,
    min_sigma: float,
    relative_floor: float,
    method_allowlist: Tuple[str, ...] | None = None,
) -> Tuple[List[ZMeasurement], Dict[str, float | str], Dict[str, float]]:
    methods = calibration.get("methods", {})
    enabled = set(method_allowlist) if method_allowlist is not None else None
    raw_candidates: List[ZMeasurement] = []
    diagnostic = _empty_diagnostic()
    top_weight = -1.0
    top: Dict[str, float | str] = {}
    method_counts: Dict[str, float] = {}

    for method_name, key in METHOD_COLUMNS:
        if enabled is not None and method_name not in enabled:
            continue
        method_stats = methods.get(method_name)
        if not method_stats:
            continue
        raw_z = float(row.get(key, -1.0))
        if raw_z <= 0.1 or not math.isfinite(raw_z):
            continue
        bias = float(method_stats.get("bias_median") or 0.0)
        sigma = max(min_sigma, float(method_stats.get("sigma") or min_sigma))
        corrected_z = raw_z - bias
        if corrected_z <= 0.1 or not math.isfinite(corrected_z):
            continue
        variance = sigma * sigma
        raw_candidates.append((corrected_z, variance, method_name))
        method_counts[method_name] = method_counts.get(method_name, 0.0) + 1.0
        weight = 1.0 / variance if variance > 0.0 else 0.0
        if weight > top_weight:
            top_weight = weight
            top = {
                "calibrated_smoother_top_method_name": method_name,
                "calibrated_smoother_top_weight": weight,
                "calibrated_smoother_top_raw_z": raw_z,
                "calibrated_smoother_top_corrected_z": corrected_z,
                "calibrated_smoother_top_bias": bias,
                "calibrated_smoother_top_sigma": sigma,
            }

    grouped = group_correlated_z_measurements(raw_candidates, relative_floor=relative_floor)
    diagnostic["calibrated_smoother_valid_count"] = float(len(raw_candidates))
    diagnostic["calibrated_smoother_group_count"] = float(len(grouped))
    diagnostic.update(top)
    return grouped, diagnostic, method_counts


def apply_calibrated_smoother(
    input_csv: str | Path,
    calibration_path: str | Path,
    output_csv: str | Path,
    *,
    metadata_path: str | Path | None = None,
    smoother_cfg: SmootherConfig | None = None,
    min_sigma: float = 0.015,
    relative_floor: float = 0.01,
    rts: bool = False,
    method_names: Tuple[str, ...] | str | None = None,
) -> Dict[str, Any]:
    calibration = load_calibration(calibration_path)
    method_allowlist = resolve_method_allowlist(method_names)
    sequences = load_legacy_sequences(input_csv, metadata_path=metadata_path)
    smoother_cfg = smoother_cfg or SmootherConfig()
    all_rows: List[Dict[str, float]] = []
    report: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "calibration": str(calibration_path),
        "method_allowlist": list(method_allowlist) if method_allowlist is not None else None,
        "calibrated_methods": sorted(calibration.get("methods", {}).keys()),
        "sequences": [],
    }
    smoother = smooth_sequence_rts if rts else smooth_sequence

    for sequence in sequences:
        diagnostics: List[Dict[str, float | str]] = []
        method_counts: Dict[str, float] = {}

        def provider(_index: int, row: Dict[str, float]) -> List[ZMeasurement]:
            observations, diagnostic, counts = _method_observations(
                row,
                calibration,
                min_sigma=min_sigma,
                relative_floor=relative_floor,
                method_allowlist=method_allowlist,
            )
            diagnostics.append(diagnostic)
            for method, count in counts.items():
                method_counts[method] = method_counts.get(method, 0.0) + count
            return observations

        rows, metrics = smoother(sequence, smoother_cfg, z_measurement_provider=provider)
        for index, row in enumerate(rows):
            if index < len(diagnostics):
                row.update(diagnostics[index])
        all_rows.extend(rows)
        report["sequences"].append(
            {
                **metrics,
                "method_counts": method_counts,
            }
        )

    write_output(output_csv, all_rows)
    report["rows"] = len(all_rows)
    report["output_csv"] = str(output_csv)
    report["rts"] = rts
    return report


def _write_json(path: str | Path, report: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--metadata", help="Optional metadata YAML")
    parser.add_argument("--json-out")
    parser.add_argument("--gravity-y", type=float, default=9.81)
    parser.add_argument("--min-sigma", type=float, default=0.015)
    parser.add_argument("--relative-floor", type=float, default=0.01)
    parser.add_argument("--methods", default=None, help="Optional method allowlist/preset such as p0, p0p1, p0p1_ncc_xfeat")
    parser.add_argument("--rts", action="store_true", help="Run offline RTS backward smoothing after calibrated observations")
    parser.add_argument(
        "--use-static-known-z",
        dest="use_static_known_z",
        action="store_true",
        help="Use static known_z metadata as a smoother update. Off by default to avoid label leakage in evaluation.",
    )
    parser.set_defaults(use_static_known_z=False)
    args = parser.parse_args()

    report = apply_calibrated_smoother(
        args.input,
        args.calibration,
        args.output,
        metadata_path=args.metadata,
        smoother_cfg=SmootherConfig(
            gravity_y=args.gravity_y,
            use_static_known_z=args.use_static_known_z,
        ),
        min_sigma=args.min_sigma,
        relative_floor=args.relative_floor,
        rts=args.rts,
        method_names=args.methods,
    )
    if args.json_out:
        _write_json(args.json_out, report)
    print(
        "rows={rows} calibrated_methods={methods} output={output}".format(
            rows=report["rows"],
            methods=len(report["calibrated_methods"]),
            output=report["output_csv"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
