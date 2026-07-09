#!/usr/bin/env python3
"""Reproject per-method stereo depth candidates into metric 3D points.

Stage-1 preprocessing for the causal 3D trajectory state estimator. Each depth
method reports a disparity; combined with the shared left-image pixel of the
ball centre and the stereo intrinsics it becomes a metric ``[X, Y, Z]`` point in
the left rectified camera frame.

Two things this module deliberately does:

1. Applies the disparity zero-point correction ``Z = fB / (disparity - d0)``
   instead of the raw ``z_*`` columns. See ``calibrate_disparity_offset.py`` and
   the memory note ``stereo-disparity-offset-d0``: the raw depth carries a
   systematic 0.1-1.0m far-range bias that a single ``d0`` removes. Feeding the
   corrected depth means the learned model spends its capacity on random noise
   and multi-method fusion, not on relearning a deterministic calibration
   residual.

2. Reprojects every candidate through the *same* left pixel ``(u, v)``:

       X = (u - cx) * Z / fx
       Y = (v - cy) * Z / fy
       Z = fB / (disparity - d0)

   Because the pixel anchor is shared, a candidate whose disparity is noisy
   produces a proportionally noisy ``Z`` and therefore proportionally noisy
   ``X, Y``. This makes the coupling "XY jitter originates from Z jitter"
   explicit and visible to the model, exactly as intended.

The module reads intrinsics from the OpenCV ``stereo_calib.yaml`` and the fitted
``d0`` from the JSON produced by ``calibrate_disparity_offset.py``. It does not
touch the legacy online ``x/y/z`` state columns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StereoIntrinsics:
    """Rectified left-camera intrinsics plus the focal*baseline product."""

    fx: float
    fy: float
    cx: float
    cy: float
    fB: float  # fx * baseline_metres, the numerator of z = fB / (disp - d0)
    baseline_m: float

    @property
    def focal(self) -> float:
        return self.fx


@dataclass(frozen=True)
class ReprojectionModel:
    """Intrinsics + disparity offset used to lift disparities to metric 3D."""

    intrinsics: StereoIntrinsics
    d0: float  # disparity zero-point offset in pixels
    fB: float  # fB actually used for depth (fitted, falls back to calib)

    def depth_from_disparity(self, disparity: float) -> Optional[float]:
        denom = disparity - self.d0
        if denom <= 1e-6:
            return None
        z = self.fB / denom
        if not math.isfinite(z) or z <= 0.0:
            return None
        return z

    def reproject(self, u: float, v: float, disparity: float) -> Optional[Tuple[float, float, float]]:
        z = self.depth_from_disparity(disparity)
        if z is None:
            return None
        x = (u - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * z / self.intrinsics.fy
        return (x, y, z)


def load_intrinsics(calib_path: str | Path) -> StereoIntrinsics:
    """Load rectified left intrinsics from an OpenCV stereo calibration YAML.

    Prefers the rectified ``projection_left`` matrix (post-rectification fx, cx,
    cy) and derives the baseline from ``projection_right`` Tx = -fx*baseline, or
    the explicit ``baseline`` node (millimetres) when present.
    """

    import cv2  # local import so non-reprojection callers do not need cv2

    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    try:
        proj_left = fs.getNode("projection_left").mat()
        proj_right = fs.getNode("projection_right").mat()
        if proj_left is None:
            raise ValueError(f"projection_left missing in {calib_path}")
        fx = float(proj_left[0, 0])
        fy = float(proj_left[1, 1])
        cx = float(proj_left[0, 2])
        cy = float(proj_left[1, 2])

        baseline_m: Optional[float] = None
        if proj_right is not None:
            tx = float(proj_right[0, 3])  # -fx * baseline (in the calib's length unit)
            if abs(fx) > 1e-6 and abs(tx) > 1e-6:
                # Tx is expressed in fx * baseline; recovering metres needs the
                # baseline node's unit. Cross-check against the baseline node.
                baseline_from_tx = -tx / fx  # in the same unit as the stereo rig (mm here)
                baseline_m = baseline_from_tx / 1000.0

        baseline_node = fs.getNode("baseline")
        if not baseline_node.empty():
            baseline_mm = float(baseline_node.real())
            baseline_m = baseline_mm / 1000.0

        if baseline_m is None or baseline_m <= 0.0:
            raise ValueError(f"could not determine baseline from {calib_path}")

        fB = fx * baseline_m
        return StereoIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, fB=fB, baseline_m=baseline_m)
    finally:
        fs.release()


def load_reprojection_model(
    calib_path: str | Path,
    offset_fit_path: str | Path | None = None,
    d0_override: float | None = None,
) -> ReprojectionModel:
    """Build a ReprojectionModel from calibration + fitted disparity offset.

    ``offset_fit_path`` is the JSON written by ``calibrate_disparity_offset.py``.
    When it is provided the fitted ``fB`` and ``d0`` are used (the fit's fB is
    typically within ~1% of the calibration value and better matches the tape
    measurements). ``d0_override`` forces a specific offset for experiments.
    """

    intrinsics = load_intrinsics(calib_path)
    fB = intrinsics.fB
    d0 = 0.0

    if offset_fit_path is not None:
        fit = json.loads(Path(offset_fit_path).read_text())
        fit_block = fit.get("fit", fit)
        fitted_fB = fit_block.get("fB")
        fitted_d0 = fit_block.get("d0")
        if fitted_fB is not None and float(fitted_fB) > 0.0:
            fB = float(fitted_fB)
        if fitted_d0 is not None:
            d0 = float(fitted_d0)

    if d0_override is not None:
        d0 = float(d0_override)

    return ReprojectionModel(intrinsics=intrinsics, d0=d0, fB=fB)


# ---------------------------------------------------------------------------
# Per-method pixel anchor and disparity column mapping.
#
# Every depth method estimates the same ball, so they share the left-image
# centre pixel. Circle-based methods use the fitted circle centre; everything
# else uses the bbox centre. The disparity is method-specific.
# ---------------------------------------------------------------------------

# method_name -> (left pixel x column, left pixel y column)
_CIRCLE_PIXEL = ("left_circle_cx", "left_circle_cy")
_BBOX_PIXEL = ("left_bbox_cx", "left_bbox_cy")

_CIRCLE_METHODS = {
    "circle_center",
    "roi_radial_center",
    "roi_edge_pair_center",
}


def method_pixel_columns(method_name: str) -> Tuple[str, str]:
    """Return the (u, v) column names used to reproject a given method."""

    if method_name in _CIRCLE_METHODS:
        return _CIRCLE_PIXEL
    return _BBOX_PIXEL


def method_disparity_column(depth_column: str) -> str:
    """Map a ``z_*`` depth column to its ``disparity_*`` companion column."""

    if depth_column.startswith("z_"):
        return "disparity_" + depth_column[2:]
    return "disparity_" + depth_column


@dataclass
class ReprojectedPoint:
    method: str
    x: float
    y: float
    z: float
    disparity: float
    valid: bool


def reproject_row(
    row: Dict[str, float],
    model: ReprojectionModel,
    methods: Sequence[Tuple[str, str]],
    *,
    min_disparity: float = 0.1,
) -> Dict[str, ReprojectedPoint]:
    """Reproject each requested method's disparity for a single CSV row.

    ``methods`` is a sequence of ``(method_name, depth_column)`` pairs (the
    METHOD_COLUMNS shape). Returns a dict keyed by method name. Invalid or
    missing candidates come back with ``valid=False`` and zeroed coordinates so
    downstream masking stays uniform.
    """

    out: Dict[str, ReprojectedPoint] = {}
    for method_name, depth_column in methods:
        disp_col = method_disparity_column(depth_column)
        u_col, v_col = method_pixel_columns(method_name)
        disparity = _safe(row.get(disp_col))
        u = _safe(row.get(u_col))
        v = _safe(row.get(v_col))
        if disparity is None or disparity < min_disparity or u is None or v is None:
            out[method_name] = ReprojectedPoint(method_name, 0.0, 0.0, 0.0, 0.0, False)
            continue
        result = model.reproject(u, v, disparity)
        if result is None:
            out[method_name] = ReprojectedPoint(method_name, 0.0, 0.0, 0.0, 0.0, False)
            continue
        x, y, z = result
        out[method_name] = ReprojectedPoint(method_name, x, y, z, disparity, True)
    return out


def _safe(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _cli() -> int:
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Sanity-check reprojection on a recorded CSV")
    parser.add_argument("csv", help="TrajectoryRecorder CSV")
    parser.add_argument("--calib", required=True, help="stereo_calib.yaml")
    parser.add_argument("--offset-fit", help="disparity_offset_fit.json from calibrate_disparity_offset.py")
    parser.add_argument("--d0", type=float, help="Override disparity offset in pixels")
    parser.add_argument("--limit", type=int, default=5, help="Rows to print")
    args = parser.parse_args()

    model = load_reprojection_model(args.calib, args.offset_fit, args.d0)
    print(f"fB={model.fB:.3f}  d0={model.d0:.3f}px  fx={model.intrinsics.fx:.2f} "
          f"cx={model.intrinsics.cx:.2f} cy={model.intrinsics.cy:.2f} baseline={model.intrinsics.baseline_m*1000:.1f}mm")

    probe_methods = [
        ("bbox_center", "z_bbox_center"),
        ("circle_center", "z_circle_center"),
        ("roi_neural_xfeat", "z_roi_neural_xfeat"),
        ("roi_cuda_template_match", "z_roi_cuda_template_match"),
    ]
    with open(args.csv, newline="") as fp:
        reader = csv.DictReader(fp)
        printed = 0
        for row in reader:
            points = reproject_row(row, model, probe_methods)
            valid = {k: p for k, p in points.items() if p.valid}
            if not valid:
                continue
            print(f"\nframe {row.get('frame_id')}:")
            for name, p in valid.items():
                print(f"  {name:26s} X={p.x:+.3f} Y={p.y:+.3f} Z={p.z:.3f} (disp={p.disparity:.2f}, raw z_*={_safe(row.get(dict(probe_methods)[name])):.3f})")
            printed += 1
            if printed >= args.limit:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
