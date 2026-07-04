# -*- coding: utf-8 -*-
"""Plotting helpers for offline volleyball trajectory visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from visualize_trajectory_animation import animate_trajectory
from visualize_trajectory_common import draw_court, get_track_colors, load_csv
from visualize_trajectory_html import export_html


def plot_static_3d(df, show_court=True, save_path=None):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    colors = get_track_colors(df["track_id"])

    for tid, group in df.groupby("track_id"):
        color = colors[tid]
        ax.plot(group["x"].values, group["y"].values, group["z"].values,
                "-", color=color, linewidth=1.5, label=f"Track {tid}")
        ax.scatter(*group.iloc[0][["x", "y", "z"]], color=color, marker="o", s=40)
        ax.scatter(*group.iloc[-1][["x", "y", "z"]], color=color, marker="s", s=40)

        if all(c in df.columns for c in ["vx", "vy", "vz"]):
            step = max(1, len(group) // 10)
            valid = group.iloc[::step].dropna(subset=["vx", "vy", "vz"])
            if len(valid) > 0:
                scale = 0.05
                ax.quiver(valid["x"], valid["y"], valid["z"],
                          valid["vx"] * scale, valid["vy"] * scale, valid["vz"] * scale,
                          color=color, alpha=0.5, arrow_length_ratio=0.3)

        if all(c in df.columns for c in ["landing_x", "landing_y"]):
            landings = group.dropna(subset=["landing_x", "landing_y"])
            if len(landings) > 0:
                last = landings.iloc[-1]
                ax.scatter(last["landing_x"], last["landing_y"], 0,
                           color=color, marker="x", s=120, linewidths=3, zorder=10)

    if show_court:
        draw_court(ax, is_3d=True)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("排球3D轨迹")
    if len(colors) <= 10:
        ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存静态图: {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_bev(df, show_court=True, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 16))
    colors = get_track_colors(df["track_id"])

    for tid, group in df.groupby("track_id"):
        color = colors[tid]
        ax.plot(group["x"].values, group["y"].values,
                "-", color=color, linewidth=1.5, label=f"Track {tid}")
        ax.scatter(group.iloc[0]["x"], group.iloc[0]["y"], color=color, marker="o", s=40)
        ax.scatter(group.iloc[-1]["x"], group.iloc[-1]["y"], color=color, marker="s", s=40)
        if all(c in df.columns for c in ["landing_x", "landing_y"]):
            landings = group.dropna(subset=["landing_x", "landing_y"])
            if len(landings) > 0:
                last = landings.iloc[-1]
                ax.scatter(last["landing_x"], last["landing_y"],
                           color=color, marker="x", s=120, linewidths=3, zorder=10)

    if show_court:
        draw_court(ax, is_3d=False)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("排球轨迹 — 俯视图")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if len(colors) <= 10:
        ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存俯视图: {save_path}")
    else:
        plt.show()
    plt.close(fig)
