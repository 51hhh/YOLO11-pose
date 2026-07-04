#!/usr/bin/env python3
"""Compare offline probe depth candidates with online recorder columns.

The script intentionally compares only methods with a known semantic mapping.
Missing realtime implementations are reported as missing instead of being
matched to a different online heuristic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from compare_offline_online_depth_report import build_report, render_csv, render_markdown
from compare_offline_online_depth_values import load_offline_summary, read_csv_rows


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-summary", required=True, type=Path, help="offline probe summary.csv")
    parser.add_argument("--online-csv", required=True, type=Path, help="TrajectoryRecorder CSV")
    parser.add_argument("--out", type=Path, help="optional output file; .csv suffix writes CSV, otherwise Markdown")
    parser.add_argument("--online-rows", choices=("median", "first", "last"), default="median")
    parser.add_argument("--max-disparity-delta-px", type=float, default=0.5)
    parser.add_argument("--max-depth-delta-m", type=float, default=0.02)
    parser.add_argument("--fail-on-mismatch", action="store_true", help="exit 1 when any mapped method fails or is missing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    offline_by_method = load_offline_summary(args.offline_summary)
    online_rows = read_csv_rows(args.online_csv)
    rows, has_failure = build_report(
        offline_by_method,
        online_rows,
        online_mode=args.online_rows,
        max_disparity_delta_px=args.max_disparity_delta_px,
        max_depth_delta_m=args.max_depth_delta_m,
    )

    if args.out and args.out.suffix.lower() == ".csv":
        rendered = render_csv(rows)
    else:
        rendered = render_markdown(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if args.fail_on_mismatch and has_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
