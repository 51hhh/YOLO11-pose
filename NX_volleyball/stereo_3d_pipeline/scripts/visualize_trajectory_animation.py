# -*- coding: utf-8 -*-
"""Animation output for offline volleyball trajectory visualization."""

from __future__ import annotations

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from visualize_trajectory_common import (
    COURT_LENGTH,
    COURT_WIDTH,
    NET_HEIGHT,
    draw_court,
    get_track_colors,
)


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
