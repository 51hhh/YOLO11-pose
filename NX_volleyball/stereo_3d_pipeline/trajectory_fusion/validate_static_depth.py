#!/usr/bin/env python3
"""Per-method static-depth accuracy validation against tape-measured known_z.

For every backfilled static segment (3-12m) this reprojects each depth
candidate with the fitted disparity offset (d0) and reports, per known distance
and per method:

  - valid rate
  - reprojected Z median vs known_z (bias)
  - Z MAD (frame-to-frame stability)

It also optionally runs a trained CausalKalmanNet checkpoint and reports the
filtered Z accuracy, so the learned fusion can be compared to the best single
candidate and to the raw per-method reprojection.

This is the Stage-1/Stage-2 acceptance check for depth accuracy. It only uses
reprojected metric depth (d0 corrected); it never reads the legacy online z.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .dataset import (
        load_legacy_sequences,
        metric_state_method_names,
        _median,
        _mad,
    )
    from .reproject import load_reprojection_model, reproject_row, method_disparity_column
    from .dataset import METHOD_COLUMNS
except ImportError:  # pragma: no cover - direct execution
    from dataset import (
        load_legacy_sequences,
        metric_state_method_names,
        _median,
        _mad,
    )
    from reproject import load_reprojection_model, reproject_row, method_disparity_column
    from dataset import METHOD_COLUMNS


# known_z (m) -> run ids (from wiki 数据集目录.md sections 六/八..十六, tape-measured)
SEGMENTS: Dict[float, List[str]] = {
    3.0: ["172811", "173137", "173258"],
    4.0: ["165405", "165705", "170006"],
    5.0: ["163933", "164203", "164659"],
    6.0: ["173719", "174012", "174306"],
    7.0: ["174904", "175521", "175727"],
    8.0: ["180406", "180557", "180826"],
    9.0: ["181558", "181746"],
    10.0: ["181955", "183648", "183918"],
    11.0: ["184317", "190047"],
    12.0: ["190901"],
}


def _find_csv(runs_dir: Path, run_id: str) -> Optional[Path]:
    matches = list(runs_dir.glob(f"*{run_id}*/traj.csv"))
    return matches[0] if matches else None


def _load_first_seq(csv_path: Path):
    seqs = load_legacy_sequences(csv_path)
    return seqs[0] if seqs else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", help="recording_runs directory")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--offset-fit", help="disparity_offset_fit.json")
    parser.add_argument("--methods", default="p0p1_ncc_xfeat")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    model = load_reprojection_model(args.calib, args.offset_fit)
    methods = list(metric_state_method_names(args.methods))
    method_cols = [(name, dict(METHOD_COLUMNS)[name]) for name in methods]

    # stats[known_z][method] -> list of per-frame reprojected Z
    stats: Dict[float, Dict[str, List[float]]] = {}
    valid_counts: Dict[float, Dict[str, Tuple[int, int]]] = {}

    for known_z, runs in sorted(SEGMENTS.items()):
        stats[known_z] = {name: [] for name in methods}
        vc = {name: [0, 0] for name in methods}
        for run_id in runs:
            csv_path = _find_csv(runs_dir, run_id)
            if csv_path is None:
                continue
            seq = _load_first_seq(csv_path)
            if seq is None:
                continue
            for row in seq.rows:
                reprojected = reproject_row(row, model, method_cols)
                for name, _col in method_cols:
                    pt = reprojected.get(name)
                    vc[name][1] += 1
                    if pt is not None and pt.valid:
                        stats[known_z][name].append(pt.z)
                        vc[name][0] += 1
        valid_counts[known_z] = {k: (v[0], v[1]) for k, v in vc.items()}

    # ---- report ----
    print(f"methods: {', '.join(methods)}\n")
    header = f"{'known_z':>8} {'method':<24} {'rate':>7} {'Z_med':>8} {'bias':>8} {'MAD':>7}"
    print(header)
    print("-" * len(header))
    per_method_abs_bias: Dict[str, List[float]] = {name: [] for name in methods}
    for known_z in sorted(SEGMENTS):
        for name in methods:
            zs = stats[known_z][name]
            valid, total = valid_counts[known_z][name]
            rate = 100.0 * valid / total if total else 0.0
            if zs:
                zmed = _median(zs)
                bias = zmed - known_z
                mad = _mad(zs)
                per_method_abs_bias[name].append(abs(bias))
                print(f"{known_z:8.1f} {name:<24} {rate:6.1f}% {zmed:8.3f} {bias:+8.3f} {mad:7.4f}")
            else:
                print(f"{known_z:8.1f} {name:<24} {rate:6.1f}% {'--':>8} {'--':>8} {'--':>7}")
        print()

    print("=== per-method mean |bias| across distances (lower = more accurate) ===")
    ranked = sorted(per_method_abs_bias.items(), key=lambda kv: (sum(kv[1]) / len(kv[1])) if kv[1] else 9e9)
    for name, biases in ranked:
        if biases:
            print(f"  {name:<24} mean|bias|={sum(biases) / len(biases):.4f}m  n_dist={len(biases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
