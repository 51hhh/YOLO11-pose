# -*- coding: utf-8 -*-
"""Shared helpers for offline volleyball trajectory visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_Y = 9.0
NET_HEIGHT = 2.43


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["frame_id", "timestamp", "track_id", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必需列: {missing}")
    df.sort_values(["track_id", "frame_id"], inplace=True)
    return df


def draw_court(ax, is_3d=True):
    corners_x = [0, COURT_WIDTH, COURT_WIDTH, 0, 0]
    corners_y = [0, 0, COURT_LENGTH, COURT_LENGTH, 0]

    if is_3d:
        ax.plot(corners_x, corners_y, [0] * 5, "g-", linewidth=1.5, alpha=0.6)
        ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], [0, 0], "g--", linewidth=1, alpha=0.4)
        ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], [NET_HEIGHT, NET_HEIGHT],
                "k-", linewidth=2, alpha=0.7, label="球网")
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        net_verts = [[(0, NET_Y, 0), (COURT_WIDTH, NET_Y, 0),
                      (COURT_WIDTH, NET_Y, NET_HEIGHT), (0, NET_Y, NET_HEIGHT)]]
        ax.add_collection3d(Poly3DCollection(net_verts, alpha=0.1, facecolor="gray"))
        for offset in [NET_Y - 3, NET_Y + 3]:
            ax.plot([0, COURT_WIDTH], [offset, offset], [0, 0], "g:", linewidth=0.8, alpha=0.4)
    else:
        ax.plot(corners_x, corners_y, "g-", linewidth=1.5, alpha=0.6)
        ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], "k-", linewidth=2, alpha=0.7)
        for offset in [NET_Y - 3, NET_Y + 3]:
            ax.plot([0, COURT_WIDTH], [offset, offset], "g:", linewidth=0.8, alpha=0.4)


def get_track_colors(track_ids):
    unique_ids = sorted(track_ids.unique())
    cmap = plt.cm.get_cmap("tab10", max(len(unique_ids), 1))
    return {tid: cmap(i) for i, tid in enumerate(unique_ids)}
