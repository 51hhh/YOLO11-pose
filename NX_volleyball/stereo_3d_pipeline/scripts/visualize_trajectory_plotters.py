# -*- coding: utf-8 -*-
"""Plotting helpers for offline volleyball trajectory visualization."""

from __future__ import annotations

import sys

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def animate_trajectory(df, fps=30, show_court=True, save_path=None, trail=20):
    frames_all = sorted(df["frame_id"].unique())
    n_frames = len(frames_all)
    colors = get_track_colors(df["track_id"])
    track_ids = sorted(df["track_id"].unique())

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.12)

    if show_court:
        draw_court(ax, is_3d=True)

    track_data = {tid: df[df["track_id"] == tid].sort_values("frame_id") for tid in track_ids}
    lines = {}
    points = {}
    for tid in track_ids:
        c = colors[tid]
        line, = ax.plot([], [], [], "-", color=c, linewidth=1.5, alpha=0.8)
        point, = ax.plot([], [], [], "o", color=c, markersize=6)
        lines[tid] = line
        points[tid] = point

    title_text = ax.set_title("", fontsize=12)
    margin = 1.0
    ax.set_xlim(df["x"].min() - margin, max(df["x"].max() + margin, COURT_WIDTH + margin))
    ax.set_ylim(df["y"].min() - margin, max(df["y"].max() + margin, COURT_LENGTH + margin))
    ax.set_zlim(0, max(df["z"].max() + margin, NET_HEIGHT + 2))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=25, azim=-60)

    ax_slider = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    slider = Slider(ax_slider, "帧", 0, n_frames - 1, valinit=0, valstep=1)

    def update(frame_idx):
        frame_idx = min(int(frame_idx), n_frames - 1)
        current_fid = frames_all[frame_idx]
        start_idx = max(0, frame_idx - trail)
        visible_fids = set(frames_all[start_idx:frame_idx + 1])

        for tid in track_ids:
            visible = track_data[tid][track_data[tid]["frame_id"].isin(visible_fids)]
            if len(visible) > 0:
                lines[tid].set_data_3d(visible["x"].values, visible["y"].values, visible["z"].values)
                last = visible.iloc[-1]
                points[tid].set_data_3d([last["x"]], [last["y"]], [last["z"]])
            else:
                lines[tid].set_data_3d([], [], [])
                points[tid].set_data_3d([], [], [])

        t = df[df["frame_id"] == current_fid]["timestamp"].iloc[0] if current_fid in df["frame_id"].values else 0
        title_text.set_text(f"帧 {current_fid}  |  t = {t:.3f}s  |  [{frame_idx + 1}/{n_frames}]")
        return list(lines.values()) + list(points.values()) + [title_text]

    if save_path and save_path.endswith(".mp4"):
        print(f"正在生成动画 ({n_frames}帧, {fps}fps)...")
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
        anim.save(save_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2000))
        print(f"已保存动画: {save_path}")
        plt.close(fig)
    else:
        slider.on_changed(update)
        update(0)
        plt.show()
        plt.close(fig)


def export_html(df, show_court=True, save_path="trajectory.html"):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("错误: HTML输出需要 plotly，请运行 pip install plotly")
        sys.exit(1)

    fig = go.Figure()
    colors_list = list(mcolors.TABLEAU_COLORS.values())
    for i, (tid, group) in enumerate(df.groupby("track_id")):
        color = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatter3d(
            x=group["x"], y=group["y"], z=group["z"],
            mode="lines+markers",
            marker=dict(size=2, color=color),
            line=dict(color=color, width=3),
            name=f"Track {tid}",
            hovertemplate="x=%{x:.2f}m<br>y=%{y:.2f}m<br>z=%{z:.2f}m<extra>Track %{text}</extra>",
            text=[str(tid)] * len(group),
        ))
        if all(c in df.columns for c in ["landing_x", "landing_y"]):
            landings = group.dropna(subset=["landing_x", "landing_y"])
            if len(landings) > 0:
                last = landings.iloc[-1]
                fig.add_trace(go.Scatter3d(
                    x=[last["landing_x"]], y=[last["landing_y"]], z=[0],
                    mode="markers", marker=dict(size=8, color=color, symbol="x"),
                    name=f"落点 T{tid}", showlegend=False,
                ))

    if show_court:
        cx = [0, COURT_WIDTH, COURT_WIDTH, 0, 0]
        cy = [0, 0, COURT_LENGTH, COURT_LENGTH, 0]
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=[0] * 5, mode="lines",
            line=dict(color="green", width=3), name="球场", showlegend=True,
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, COURT_WIDTH], y=[NET_Y, NET_Y], z=[NET_HEIGHT, NET_HEIGHT],
            mode="lines", line=dict(color="black", width=4), name="球网",
        ))

    fig.update_layout(
        title="排球3D轨迹 — 交互式",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
        width=1200,
        height=800,
    )
    fig.write_html(save_path)
    print(f"已保存HTML: {save_path}")
