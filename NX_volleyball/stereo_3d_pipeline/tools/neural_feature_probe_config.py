"""CLI helpers for the offline neural feature probe."""

from __future__ import annotations

import argparse
from typing import Tuple


def parse_circle(value: str) -> Tuple[float, float, float]:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("circle must be x,y,r")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/left/0000.png")
    parser.add_argument("--right", default="NX_volleyball/stereo_3d_pipeline/test_logs/volleyball_raw_pair_latest/right/0000.png")
    parser.add_argument("--calib", default="NX_volleyball/calibration/stereo_calib.yaml")
    parser.add_argument("--out", default="NX_volleyball/stereo_3d_pipeline/test_logs/neural_feature_probe_latest")
    parser.add_argument("--backends", default="xfeat,aliked,superpoint_lightglue",
                        help="comma-separated: xfeat,aliked,superpoint_lightglue")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--crop-pad", type=int, default=24)
    parser.add_argument("--left-circle", type=parse_circle, help="rectified left ball circle: x,y,r")
    parser.add_argument("--right-circle", type=parse_circle, help="rectified right ball circle: x,y,r")
    parser.add_argument("--mask-margin", type=float, default=10.0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--max-y-error-px", type=float, default=2.0)
    parser.add_argument("--max-disp-delta-px", type=float, default=32.0)
    parser.add_argument("--final-disp-gate-px", type=float, default=0.0)
    parser.add_argument("--max-disparity", type=float, default=2048.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--xfeat-repo", default="")
    parser.add_argument("--allow-torch-hub", action="store_true")
    parser.add_argument("--aliked-lightglue", action="store_true",
                        help="Use LightGlue matcher for ALIKED instead of descriptor NN")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args()


def backend_names(backends: str) -> list[str]:
    return [v.strip() for v in backends.split(",") if v.strip()]
