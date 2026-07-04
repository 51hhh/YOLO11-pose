# -*- coding: utf-8 -*-
"""HTML output for offline volleyball trajectory visualization."""

from __future__ import annotations

import sys

import matplotlib.colors as mcolors

from visualize_trajectory_common import COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, NET_Y


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
