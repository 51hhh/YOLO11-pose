#!/usr/bin/env python3
"""Run the landing pipeline on a trajectory CSV (causal, bbox-primary)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow `python -m trajectory_fusion.landing_pipeline.run_csv` and direct path exec.
_THIS = Path(__file__).resolve().parent
_TF = _THIS.parent
_ROOT = _TF.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trajectory_fusion.landing_pipeline import LandingPipeline  # noqa: E402
from trajectory_fusion.landing_pipeline.config import load_pipeline_config  # noqa: E402


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="traj.csv with disparity_bbox_center + left_bbox_cx/cy")
    ap.add_argument("--config", type=Path, default=None, help="optional JSON config overlay")
    ap.add_argument("--no-residual", action="store_true", help="disable TinyGRU residual")
    ap.add_argument("--checkpoint", type=Path, default=None, help="TinyGRU checkpoint")
    ap.add_argument("--max-rows", type=int, default=0, help="optional row cap for smoke runs")
    ap.add_argument("--json-out", type=Path, default=None, help="write per-frame predictions JSONL/JSON")
    ap.add_argument("--summary-out", type=Path, default=None, help="write summary JSON")
    args = ap.parse_args(argv)

    cfg = load_pipeline_config(
        args.config,
        enable_residual=not args.no_residual,
        residual_checkpoint=args.checkpoint,
    )
    pipe = LandingPipeline(cfg)

    preds: List[Dict] = []
    n_rows = 0
    n_obs = 0
    n_pred = 0
    sources: Dict[str, int] = {}

    with args.csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_rows += 1
            if args.max_rows and n_rows > args.max_rows:
                break
            out = pipe.update_row(row)
            if out is None:
                # Still count observation attempts when builder accepts row.
                continue
            n_pred += 1
            sources[out.source] = sources.get(out.source, 0) + 1
            preds.append(out.as_dict())

    # n_obs approximated from predictions + warm-up frames with accepted obs is hard
    # without instrumenting builder; report prediction count and residual status.
    summary = {
        "csv": str(args.csv),
        "rows_seen": n_rows,
        "predictions": n_pred,
        "sources": sources,
        "residual_enabled": bool(cfg.residual.enabled),
        "residual_available": bool(pipe.residual.available),
        "residual_reason": pipe.residual.reason,
        "d0": cfg.d0,
        "fB": cfg.fB,
        "ekf": {
            "cd": cfg.ekf.cd,
            "nu": cfg.ekf.nu,
            "sigma_d_px": cfg.ekf.sigma_d_px,
            "q_vel": cfg.ekf.q_vel,
        },
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.json_out:
        # JSON list keeps tooling simple for short clips; large logs can switch later.
        args.json_out.write_text(json.dumps(preds, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
