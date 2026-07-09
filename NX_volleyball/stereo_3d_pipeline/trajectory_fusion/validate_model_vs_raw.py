#!/usr/bin/env python3
"""Compare model-filtered Z bias vs raw-candidate bias against known_z.

This is the decisive test for "let the model learn the distance-dependent
residual": on the far distances (9m/12m) the single global d0 leaves a
systematic residual (9m ~ -0.18m, 12m ~ -0.45m). If the causal state estimator
has learned it, its filtered Z should be closer to the tape-measured known_z
than any raw reprojected candidate.

For each val clip it reports:
  - known_z (tape truth)
  - raw fused median Z (median of valid per-method reprojected Z, d0-corrected)
  - model filtered median Z
  - bias of each vs known_z

Usage:
  python3 validate_model_vs_raw.py MANIFEST.json --checkpoint model.pt \
      --calib ../calibration/stereo_calib.yaml \
      --offset-fit test_logs/.../disparity_offset_fit.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

try:
    from .dataset import build_metric_state_arrays, load_legacy_sequences, resolve_method_allowlist, _median
    from .manifest import load_manifest
    from .reproject import load_reprojection_model
    from .state_estimator import CausalKalmanNet, FilterConfig
except ImportError:
    from dataset import build_metric_state_arrays, load_legacy_sequences, resolve_method_allowlist, _median
    from manifest import load_manifest
    from reproject import load_reprojection_model
    from state_estimator import CausalKalmanNet, FilterConfig


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fc = ckpt["filter_config"]
    cfg = FilterConfig(**fc)
    model = CausalKalmanNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--offset-fit")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="val", help="only evaluate clips in this split")
    args = ap.parse_args()

    model, ckpt = _load_model(args.checkpoint, args.device)
    methods = ckpt.get("method_allowlist")
    method_allowlist = resolve_method_allowlist(",".join(methods)) if methods else None
    reproj = load_reprojection_model(args.calib, args.offset_fit)

    rows: List[Dict] = []
    with torch.no_grad():
        for clip in load_manifest(args.manifest):
            if clip.split != args.split:
                continue
            csv_path = clip.csv
            meta = clip.metadata
            seqs = load_legacy_sequences(csv_path, metadata_path=meta)
            for seq in seqs:
                known_z = float(seq.metadata.get("known_z", 0.0) or 0.0)
                if known_z <= 0.0:
                    continue
                arrays = build_metric_state_arrays(seq, reproj, method_names=method_allowlist)
                if not arrays["dt"]:
                    continue
                points = torch.tensor(arrays["points"], dtype=torch.float32, device=args.device)  # [T,M,3]
                pvalid = torch.tensor(arrays["point_valid"], dtype=torch.float32, device=args.device)  # [T,M]
                quality = torch.tensor(arrays["features"], dtype=torch.float32, device=args.device)
                dt = torch.tensor([d[0] for d in arrays["dt"]], dtype=torch.float32, device=args.device)

                # raw fused Z: per-frame median of valid method Z, then clip median
                raw_z_per_frame = []
                for tt in range(points.shape[0]):
                    zs = [float(points[tt, m, 2]) for m in range(points.shape[1]) if pvalid[tt, m] > 0]
                    if zs:
                        raw_z_per_frame.append(_median(zs))
                raw_med = _median(raw_z_per_frame) if raw_z_per_frame else float("nan")

                states, _lv = model(
                    points.unsqueeze(0), pvalid.unsqueeze(0),
                    quality.unsqueeze(0), dt.unsqueeze(0),
                )
                filt_z = states[0, :, 2]
                filt_med = float(filt_z.median())

                rows.append({
                    "clip": clip.name or Path(csv_path).parent.name,
                    "known_z": known_z,
                    "raw_med": raw_med,
                    "raw_bias": raw_med - known_z,
                    "model_med": filt_med,
                    "model_bias": filt_med - known_z,
                })

    print(f"{'clip':22} {'known_z':>8} {'raw_Z':>8} {'raw_bias':>9} {'model_Z':>8} {'model_bias':>11}")
    print("-" * 74)
    raw_abs = []
    model_abs = []
    for r in sorted(rows, key=lambda x: x["known_z"]):
        print(f"{r['clip']:22} {r['known_z']:8.2f} {r['raw_med']:8.3f} {r['raw_bias']:+9.3f} "
              f"{r['model_med']:8.3f} {r['model_bias']:+11.3f}")
        raw_abs.append(abs(r["raw_bias"]))
        model_abs.append(abs(r["model_bias"]))
    if rows:
        print("-" * 74)
        print(f"mean |bias|:  raw={sum(raw_abs)/len(raw_abs):.4f}m   model={sum(model_abs)/len(model_abs):.4f}m")
        # far-distance focus
        far = [r for r in rows if r["known_z"] >= 9.0]
        if far:
            fr = sum(abs(r["raw_bias"]) for r in far) / len(far)
            fm = sum(abs(r["model_bias"]) for r in far) / len(far)
            print(f"far (>=9m):   raw={fr:.4f}m   model={fm:.4f}m   ({len(far)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
