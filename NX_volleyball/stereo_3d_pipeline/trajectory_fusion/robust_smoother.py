#!/usr/bin/env python3
"""Robust physics-aware smoother for candidate-depth trajectory CSV files."""

from __future__ import annotations

import argparse
from typing import Dict, List

try:
    from .dataset import load_legacy_sequences
    from .robust_smoother_core import SmootherConfig, smooth_sequence
    from .robust_smoother_io import write_output
except ImportError:  # pragma: no cover - direct script execution
    from dataset import load_legacy_sequences
    from robust_smoother_core import SmootherConfig, smooth_sequence
    from robust_smoother_io import write_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="TrajectoryRecorder CSV")
    parser.add_argument("-o", "--output", default="trajectory_fusion_smooth.csv")
    parser.add_argument("--metadata", help="Optional metadata.yaml with weak labels")
    parser.add_argument("--no-method-depths", action="store_true", help="Do not use raw candidate z_* updates")
    parser.add_argument("--use-online-position", action="store_true", help="Also use legacy online x/y/z as a position update")
    parser.add_argument("--no-static-known-z", action="store_true", help="Do not use static known_z metadata as an update")
    parser.add_argument("--gravity-y", type=float, default=9.81, help="Camera-y gravity prior in m/s^2")
    args = parser.parse_args()

    cfg = SmootherConfig(
        use_method_depths=not args.no_method_depths,
        use_online_position=args.use_online_position,
        use_static_known_z=not args.no_static_known_z,
        gravity_y=args.gravity_y,
    )
    sequences = load_legacy_sequences(args.input, metadata_path=args.metadata)
    all_rows: List[Dict[str, float]] = []
    metrics: List[Dict[str, float]] = []
    for seq in sequences:
        rows, seq_metrics = smooth_sequence(seq, cfg)
        all_rows.extend(rows)
        metrics.append(seq_metrics)

    write_output(args.output, all_rows)
    for item in metrics:
        raw_std = item["raw_z_std"]
        smooth_std = item["smooth_z_std"]
        ratio = smooth_std / raw_std if raw_std > 1e-9 else 0.0
        print(
            "track={track_id:.0f} frames={frames:.0f} raw_z_std={raw:.4f} "
            "smooth_z_std={smooth:.4f} ratio={ratio:.3f} inn_mean={inn:.3f} inn_max={inn_max:.3f}".format(
                track_id=item["track_id"],
                frames=item["frames"],
                raw=raw_std,
                smooth=smooth_std,
                ratio=ratio,
                inn=item["innovation_norm_mean"],
                inn_max=item["innovation_norm_max"],
            )
        )
    print(f"wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
