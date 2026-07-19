#!/usr/bin/env python3
"""Calibrate the stereo disparity zero-point offset d0.

Background
----------
The recorder computes depth as ``z = fB / disparity``. Comparing static
known-distance clips against tape-measured ``known_z`` shows a systematic bias
that grows with distance (see docs/开发记忆 and the trajectory wiki): +0.06m at
3m up to +0.86m at 12m. Fitting ``z = fB / (disparity - d0)`` collapses that
bias to ~0.11m RMS with a single offset ``d0 ~= -11 px``, while the fitted
``fB`` stays within 0.3% of the calibration value ``fx * baseline``.

The offset is the rectification residual between the left/right principal
points; it is a deterministic calibration artefact, not a per-clip error. This
tool fits ``(fB, d0)`` from the backfilled static ``known_z`` clips so the
reprojection preprocessing can correct depth before it reaches the model.

It is data-driven on purpose: rerun after re-calibrating the rig or recording
new known-distance clips instead of hard-coding a magic number.

Usage
-----
    python3 calibrate_disparity_offset.py RUNS_DIR \
        --calib ../../calibration/stereo_calib.yaml \
        --disparity-field disparity_bbox_center \
        --depth-field z_bbox_center \
        -o disparity_offset.json

``RUNS_DIR`` is scanned recursively for ``*.metadata.yaml`` files that carry a
static ``known_z`` label. For each such clip the paired ``*.csv`` is read and
the median disparity for the requested field is used as one calibration point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .dataset import _metadata_bool, _metadata_float, find_metadata_for_csv, read_metadata
except ImportError:  # pragma: no cover - direct script execution
    from dataset import _metadata_bool, _metadata_float, find_metadata_for_csv, read_metadata


@dataclass
class CalibrationPoint:
    clip: str
    known_z: float
    disparity_median: float
    depth_median: float
    n_frames: int


@dataclass
class OffsetFit:
    fB: float
    d0: float
    rms_error_m: float
    max_abs_error_m: float
    n_points: int
    calib_fB: float
    fB_ratio: float


def _median(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _read_calib_fB(calib_path: Path) -> Optional[float]:
    """Return fx * baseline(m) from an OpenCV stereo calibration YAML."""

    try:
        import cv2
    except ImportError:
        return None
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    try:
        proj_left = fs.getNode("projection_left")
        fx = None
        if not proj_left.empty():
            fx = float(proj_left.mat()[0, 0])
        else:
            cam_left = fs.getNode("camera_matrix_left")
            if not cam_left.empty():
                fx = float(cam_left.mat()[0, 0])
        baseline_node = fs.getNode("baseline")
        baseline_mm = float(baseline_node.real()) if not baseline_node.empty() else None
        if fx is None or baseline_mm is None:
            return None
        return fx * (baseline_mm / 1000.0)
    finally:
        fs.release()


def _collect_points(
    runs_dir: Path,
    disparity_field: str,
    depth_field: str,
    min_frames: int,
) -> List[CalibrationPoint]:
    points: List[CalibrationPoint] = []
    for meta_path in sorted(runs_dir.rglob("*.metadata.yaml")):
        metadata = read_metadata(meta_path)
        if metadata is None:
            continue
        known_z = _metadata_float(metadata, ("known_z_m", "known_z", "known_distance_m"), 0.0)
        if known_z <= 0.0:
            continue
        is_static = _metadata_bool(metadata, ("static", "is_static"), False)
        known_z_training = is_static or _metadata_bool(
            metadata,
            ("known_z_training", "known_z_supervision", "use_known_z_for_training"),
            False,
        )
        if not known_z_training:
            continue
        csv_path = meta_path.with_name(meta_path.name.replace(".metadata.yaml", ".csv"))
        if not csv_path.exists():
            continue
        disparities: List[float] = []
        depths: List[float] = []
        with csv_path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            if disparity_field not in (reader.fieldnames or []):
                continue
            for row in reader:
                disp = _safe_float(row.get(disparity_field))
                depth = _safe_float(row.get(depth_field)) if depth_field else None
                if disp is not None and disp > 0.0:
                    disparities.append(disp)
                if depth is not None and depth > 0.1:
                    depths.append(depth)
        if len(disparities) < min_frames:
            continue
        points.append(
            CalibrationPoint(
                clip=csv_path.stem,
                known_z=known_z,
                disparity_median=_median(disparities),
                depth_median=_median(depths) if depths else float("nan"),
                n_frames=len(disparities),
            )
        )
    return points


def _safe_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _aggregate_by_distance(points: List[CalibrationPoint]) -> List[Tuple[float, float]]:
    """Collapse clips at the same known_z into one (known_z, disparity) point.

    Multiple clips are recorded per distance. Using the per-distance median
    disparity keeps every distance weighted equally in the fit regardless of
    how many clips exist for it.
    """

    buckets: Dict[float, List[float]] = {}
    for point in points:
        buckets.setdefault(point.known_z, []).append(point.disparity_median)
    return [(known_z, _median(disps)) for known_z, disps in sorted(buckets.items())]


def fit_offset(
    distance_points: List[Tuple[float, float]],
    calib_fB: Optional[float],
) -> OffsetFit:
    """Fit z = fB / (disp - d0) via linear least squares.

    Rearranged: disp = fB * (1/z) + d0, which is linear in unknowns (fB, d0)
    against the regressor (1/z). No iterative solver needed.
    """

    n = len(distance_points)
    if n < 2:
        raise SystemExit("need at least 2 distinct known distances to fit (fB, d0)")

    inv_z = [1.0 / z for z, _ in distance_points]
    disp = [d for _, d in distance_points]

    mean_x = sum(inv_z) / n
    mean_y = sum(disp) / n
    sxx = sum((x - mean_x) ** 2 for x in inv_z)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(inv_z, disp))
    if sxx <= 0.0:
        raise SystemExit("degenerate fit: all distances identical")
    fB = sxy / sxx
    d0 = mean_y - fB * mean_x

    errors = []
    for (z_known, d_obs) in distance_points:
        denom = d_obs - d0
        z_pred = fB / denom if denom > 1e-6 else float("inf")
        errors.append(z_pred - z_known)
    rms = math.sqrt(sum(e * e for e in errors) / n)
    max_abs = max(abs(e) for e in errors)

    return OffsetFit(
        fB=fB,
        d0=d0,
        rms_error_m=rms,
        max_abs_error_m=max_abs,
        n_points=n,
        calib_fB=calib_fB if calib_fB is not None else float("nan"),
        fB_ratio=(fB / calib_fB) if calib_fB else float("nan"),
    )


def _print_report(
    points: List[CalibrationPoint],
    distance_points: List[Tuple[float, float]],
    fit: OffsetFit,
) -> None:
    print(f"collected {len(points)} known-distance clips across {fit.n_points} distances\n")
    print(f"{'known_z':>8} {'disp_med':>10} {'z_pred':>8} {'err':>8}")
    for (z_known, d_obs) in distance_points:
        denom = d_obs - fit.d0
        z_pred = fit.fB / denom if denom > 1e-6 else float("inf")
        print(f"{z_known:8.2f} {d_obs:10.3f} {z_pred:8.3f} {z_pred - z_known:+8.3f}")
    print()
    print(f"  fitted fB = {fit.fB:.3f}   d0 = {fit.d0:.3f} px")
    if not math.isnan(fit.calib_fB):
        print(f"  calib  fB = {fit.calib_fB:.3f}   (ratio {fit.fB_ratio:.4f})")
    print(f"  RMS err = {fit.rms_error_m:.4f} m   max |err| = {fit.max_abs_error_m:.4f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", help="Directory scanned recursively for known-distance clips")
    parser.add_argument("--calib", help="Stereo calibration YAML, used to sanity-check fitted fB")
    parser.add_argument(
        "--disparity-field",
        default="disparity_bbox_center",
        help="Disparity column used as the calibration observable (default: disparity_bbox_center)",
    )
    parser.add_argument(
        "--depth-field",
        default="z_bbox_center",
        help="Depth column recorded for reference/reporting (default: z_bbox_center)",
    )
    parser.add_argument("--min-frames", type=int, default=100, help="Minimum valid frames per clip")
    parser.add_argument(
        "--exclude-known-z",
        type=float,
        nargs="*",
        default=[],
        help="known_z values to exclude from the fit (e.g. noisy far distances)",
    )
    parser.add_argument("-o", "--output", help="Write fit + points to this JSON path")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"not a directory: {runs_dir}")

    calib_fB = _read_calib_fB(Path(args.calib)) if args.calib else None

    points = _collect_points(runs_dir, args.disparity_field, args.depth_field, args.min_frames)
    if not points:
        raise SystemExit("no known-distance clips with the requested disparity field found")

    excluded = set(args.exclude_known_z)
    kept = [p for p in points if p.known_z not in excluded]
    distance_points = _aggregate_by_distance(kept)
    fit = fit_offset(distance_points, calib_fB)
    _print_report(kept, distance_points, fit)

    if args.output:
        payload = {
            "disparity_field": args.disparity_field,
            "depth_field": args.depth_field,
            "excluded_known_z": sorted(excluded),
            "fit": asdict(fit),
            "distance_points": [
                {"known_z": z, "disparity_median": d} for z, d in distance_points
            ],
            "clips": [asdict(p) for p in kept],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
