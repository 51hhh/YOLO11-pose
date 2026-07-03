#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排球3D轨迹离线可视化工具。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visualize_trajectory_plotters import (
    animate_trajectory,
    export_html,
    load_csv,
    plot_bev,
    plot_static_3d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="排球3D轨迹可视化工具")
    parser.add_argument("--input", "-i", required=True, help="CSV轨迹文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径 (不指定则交互显示)")
    parser.add_argument("--format", "-f", choices=["png", "mp4", "html"], default="png",
                        help="输出格式 (默认: png)")
    parser.add_argument("--fps", type=int, default=30, help="动画帧率 (默认: 30)")
    parser.add_argument("--court", action="store_true", default=True, help="显示球场 (默认开启)")
    parser.add_argument("--no-court", dest="court", action="store_false", help="不显示球场")
    parser.add_argument("--view", choices=["3d", "bev"], default="3d",
                        help="视角: 3d或bev俯视 (默认: 3d)")
    parser.add_argument("--trail", type=int, default=20, help="动画拖尾帧数 (默认: 20)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"错误: 文件不存在 — {csv_path}")
        return 1

    df = load_csv(str(csv_path))
    print(f"已加载 {len(df)} 条记录, {df['track_id'].nunique()} 条轨迹, "
          f"帧范围 [{df['frame_id'].min()}, {df['frame_id'].max()}]")

    out = args.output
    if out is None and args.format != "png":
        out = str(csv_path.with_suffix(f".{args.format}"))

    if args.format == "html":
        export_html(df, show_court=args.court, save_path=out or str(csv_path.with_suffix(".html")))
    elif args.format == "mp4":
        animate_trajectory(df, fps=args.fps, show_court=args.court,
                           save_path=out or str(csv_path.with_suffix(".mp4")), trail=args.trail)
    else:
        if args.view == "bev":
            plot_bev(df, show_court=args.court, save_path=out)
        else:
            plot_static_3d(df, show_court=args.court, save_path=out)
        if out is None:
            animate_trajectory(df, fps=args.fps, show_court=args.court, trail=args.trail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
