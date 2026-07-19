"""Shared data loading for landing evaluation.

Streams rows from p1_dy_regression_20260709 recordings. Static segments
carry hand-recorded known_z labels (see wiki/数据集目录.md section 18).
"""
import csv
import json
import math
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "test_logs")

STATIC_BUCKETS = {
    3.0: ["130138", "130300", "130433"],
    4.0: ["130707", "131813", "131944"],
    5.0: ["132131", "132305", "132526"],
    6.0: ["133347", "133528", "133829"],
    7.0: ["134557", "134809", "135115"],
    8.0: ["135244", "135424", "135854"],
    9.0: ["140149", "140335", "140512"],
    10.0: ["140851", "141034", "141222"],
    11.0: ["141456", "141742", "141939"],
    13.0: ["142201", "142340", "142648"],
}
THROWS = [
    "143434", "143504", "143620", "143642", "143702", "143721", "143739",
    "143824", "143847", "143909", "143930", "143949", "144011", "144030",
]
# 141742: 929 rows, bbox=18.29m — known bad segment (wiki).
# 13m bucket: label direction inconsistent — excluded from d0 fitting.
EXCLUDE_FROM_FIT = {"141742"}
LABEL_SUSPECT_BUCKETS = {13.0}

# Rectified projection (calibration/stereo_calib.yaml projection_left)
FX = 1613.873447565926
FY = 1613.873447565926
CX = 681.58988952636719
CY = 620.79562759399414
BASELINE_M = 0.92763108240240683
FB = FX * BASELINE_M  # ~1497.08 px*m


def run_path(run_id):
    return os.path.join(DATA_ROOT, f"p1_dy_regression_20260709_{run_id}", "traj.csv")


def f(row, key, default=-1.0):
    try:
        v = float(row[key])
    except (KeyError, ValueError, TypeError):
        return default
    return v if math.isfinite(v) else default


def stream_rows(run_id):
    """Yield dict rows one at a time (causal streaming contract)."""
    with open(run_path(run_id)) as fh:
        for row in csv.DictReader(fh):
            yield row


def observation(row, d0=0.0):
    """Per-frame 3D observation from raw bbox candidate (not online Kalman).

    Returns (t, p_cam[3], quality dict) or None if unusable this frame.
    Depth from z_bbox_center's disparity with optional d0 correction;
    pixel anchor = left bbox center backprojected with corrected depth.
    """
    t = f(row, "timestamp")
    disp = f(row, "disparity_bbox_center")
    u = f(row, "left_bbox_cx")
    v = f(row, "left_bbox_cy")
    if t <= 0 or disp <= 0 or u < 0 or v < 0:
        return None
    denom = disp - d0
    if denom <= 1.0:
        return None
    z = FB / denom
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    q = {
        "trust": f(row, "p0p1_bbox_center_trust", 0.0),
        "dy_mad": f(row, "p0p1_dy_mad", 0.0),
        "pair_score": f(row, "pair_score", 0.0),
    }
    return t, (x, y, z), q


def load_manifest():
    segs = []
    for kz, ids in STATIC_BUCKETS.items():
        for rid in ids:
            segs.append({"run_id": rid, "kind": "static", "known_z": kz,
                         "fit_ok": rid not in EXCLUDE_FROM_FIT
                         and kz not in LABEL_SUSPECT_BUCKETS})
    for rid in THROWS:
        segs.append({"run_id": rid, "kind": "throw", "known_z": None,
                     "fit_ok": False})
    return segs


if __name__ == "__main__":
    ok = 0
    for seg in load_manifest():
        p = run_path(seg["run_id"])
        n = sum(1 for _ in open(p)) - 1 if os.path.exists(p) else -1
        ok += n > 0
        print(f"{seg['run_id']} {seg['kind']:6} kz={seg['known_z']} rows={n}")
    print("segments ok:", ok)
