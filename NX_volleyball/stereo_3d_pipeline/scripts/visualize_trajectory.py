#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排球3D轨迹离线可视化工具
读取C++管线录制的CSV轨迹数据，支持3D静态图、动画回放、BEV俯视图。

CSV格式:
  frame_id,timestamp,track_id,x,y,z,vx,vy,vz,ax,ay,az,confidence,method,landing_x,landing_y,landing_t
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors


# 排球场标准尺寸
COURT_LENGTH = 18.0  # y方向 (m)
COURT_WIDTH = 9.0    # x方向 (m)
NET_Y = 9.0          # 网线y位置 (场地中线)
NET_HEIGHT = 2.43    # 网线高度 (m)


def load_csv(path: str) -> pd.DataFrame:
    """读取CSV轨迹文件"""
    df = pd.read_csv(path)
    # 确保必需列存在
    required = ['frame_id', 'timestamp', 'track_id', 'x', 'y', 'z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必需列: {missing}")
    df.sort_values(['track_id', 'frame_id'], inplace=True)
    return df


def draw_court(ax, is_3d=True):
    """绘制排球场地平面和网线"""
    # 场地边界 (z=0平面)
    corners_x = [0, COURT_WIDTH, COURT_WIDTH, 0, 0]
    corners_y = [0, 0, COURT_LENGTH, COURT_LENGTH, 0]

    if is_3d:
        ax.plot(corners_x, corners_y, [0]*5, 'g-', linewidth=1.5, alpha=0.6)
        # 中线
        ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], [0, 0], 'g--', linewidth=1, alpha=0.4)
        # 网线
        net_x = [0, COURT_WIDTH]
        net_z = [NET_HEIGHT, NET_HEIGHT]
        ax.plot(net_x, [NET_Y, NET_Y], net_z, 'k-', linewidth=2, alpha=0.7, label='球网')
        # 网面
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        net_verts = [[(0, NET_Y, 0), (COURT_WIDTH, NET_Y, 0),
                      (COURT_WIDTH, NET_Y, NET_HEIGHT), (0, NET_Y, NET_HEIGHT)]]
        net_poly = Poly3DCollection(net_verts, alpha=0.1, facecolor='gray')
        ax.add_collection3d(net_poly)
        # 3m攻击线
        for offset in [NET_Y - 3, NET_Y + 3]:
            ax.plot([0, COURT_WIDTH], [offset, offset], [0, 0], 'g:', linewidth=0.8, alpha=0.4)
    else:
        ax.plot(corners_x, corners_y, 'g-', linewidth=1.5, alpha=0.6)
        ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], 'k-', linewidth=2, alpha=0.7)
        for offset in [NET_Y - 3, NET_Y + 3]:
            ax.plot([0, COURT_WIDTH], [offset, offset], 'g:', linewidth=0.8, alpha=0.4)


def get_track_colors(track_ids):
    """为不同track_id分配颜色"""
    unique_ids = sorted(track_ids.unique())
    cmap = plt.cm.get_cmap('tab10', max(len(unique_ids), 1))
    return {tid: cmap(i) for i, tid in enumerate(unique_ids)}


def plot_static_3d(df, show_court=True, save_path=None):
    """静态3D轨迹图"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = get_track_colors(df['track_id'])

    for tid, group in df.groupby('track_id'):
        color = colors[tid]
        ax.plot(group['x'].values, group['y'].values, group['z'].values,
                '-', color=color, linewidth=1.5, label=f'Track {tid}')
        # 起点/终点标记
        ax.scatter(*group.iloc[0][['x','y','z']], color=color, marker='o', s=40)
        ax.scatter(*group.iloc[-1][['x','y','z']], color=color, marker='s', s=40)

        # 速度矢量箭头 (间隔采样)
        if all(c in df.columns for c in ['vx', 'vy', 'vz']):
            step = max(1, len(group) // 10)
            sampled = group.iloc[::step]
            valid = sampled.dropna(subset=['vx', 'vy', 'vz'])
            if len(valid) > 0:
                scale = 0.05  # 箭头缩放
                ax.quiver(valid['x'], valid['y'], valid['z'],
                          valid['vx']*scale, valid['vy']*scale, valid['vz']*scale,
                          color=color, alpha=0.5, arrow_length_ratio=0.3)

        # 落点预测标记
        if all(c in df.columns for c in ['landing_x', 'landing_y']):
            landings = group.dropna(subset=['landing_x', 'landing_y'])
            if len(landings) > 0:
                last = landings.iloc[-1]
                ax.scatter(last['landing_x'], last['landing_y'], 0,
                           color=color, marker='x', s=120, linewidths=3, zorder=10)

    if show_court:
        draw_court(ax, is_3d=True)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('排球3D轨迹')
    if len(colors) <= 10:
        ax.legend(loc='upper left', fontsize=8)

    # 视角
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存静态图: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_bev(df, show_court=True, save_path=None):
    """俯视图 (Bird's Eye View)"""
    fig, ax = plt.subplots(figsize=(8, 16))

    colors = get_track_colors(df['track_id'])

    for tid, group in df.groupby('track_id'):
        color = colors[tid]
        ax.plot(group['x'].values, group['y'].values,
                '-', color=color, linewidth=1.5, label=f'Track {tid}')
        ax.scatter(group.iloc[0]['x'], group.iloc[0]['y'], color=color, marker='o', s=40)
        ax.scatter(group.iloc[-1]['x'], group.iloc[-1]['y'], color=color, marker='s', s=40)

        # 落点
        if all(c in df.columns for c in ['landing_x', 'landing_y']):
            landings = group.dropna(subset=['landing_x', 'landing_y'])
            if len(landings) > 0:
                last = landings.iloc[-1]
                ax.scatter(last['landing_x'], last['landing_y'],
                           color=color, marker='x', s=120, linewidths=3, zorder=10)

    if show_court:
        draw_court(ax, is_3d=False)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('排球轨迹 — 俯视图')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if len(colors) <= 10:
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存俯视图: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def animate_trajectory(df, fps=30, show_court=True, save_path=None, trail=20):
    """动画回放，带拖尾效果和时间轴"""
    frames_all = sorted(df['frame_id'].unique())
    n_frames = len(frames_all)
    frame_map = {fid: i for i, fid in enumerate(frames_all)}

    colors = get_track_colors(df['track_id'])
    track_ids = sorted(df['track_id'].unique())

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(bottom=0.12)

    if show_court:
        draw_court(ax, is_3d=True)

    # 预构建每帧数据
    track_data = {}
    for tid in track_ids:
        g = df[df['track_id'] == tid].sort_values('frame_id')
        track_data[tid] = g

    # 绘图元素
    lines = {}
    points = {}
    for tid in track_ids:
        c = colors[tid]
        line, = ax.plot([], [], [], '-', color=c, linewidth=1.5, alpha=0.8)
        point, = ax.plot([], [], [], 'o', color=c, markersize=6)
        lines[tid] = line
        points[tid] = point

    title_text = ax.set_title('', fontsize=12)

    # 固定坐标范围
    margin = 1.0
    ax.set_xlim(df['x'].min() - margin, max(df['x'].max() + margin, COURT_WIDTH + margin))
    ax.set_ylim(df['y'].min() - margin, max(df['y'].max() + margin, COURT_LENGTH + margin))
    ax.set_zlim(0, max(df['z'].max() + margin, NET_HEIGHT + 2))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=25, azim=-60)

    # 时间轴滑块
    ax_slider = fig.add_axes([0.15, 0.03, 0.7, 0.02])
    slider = Slider(ax_slider, '帧', 0, n_frames - 1, valinit=0, valstep=1)

    def update(frame_idx):
        frame_idx = int(frame_idx)
        if frame_idx >= n_frames:
            frame_idx = n_frames - 1
        current_fid = frames_all[frame_idx]
        start_idx = max(0, frame_idx - trail)
        visible_fids = set(frames_all[start_idx:frame_idx + 1])

        for tid in track_ids:
            g = track_data[tid]
            mask = g['frame_id'].isin(visible_fids)
            visible = g[mask]

            if len(visible) > 0:
                lines[tid].set_data_3d(visible['x'].values, visible['y'].values, visible['z'].values)
                last = visible.iloc[-1]
                points[tid].set_data_3d([last['x']], [last['y']], [last['z']])
            else:
                lines[tid].set_data_3d([], [], [])
                points[tid].set_data_3d([], [], [])

        t = df[df['frame_id'] == current_fid]['timestamp'].iloc[0] if current_fid in df['frame_id'].values else 0
        title_text.set_text(f'帧 {current_fid}  |  t = {t:.3f}s  |  [{frame_idx+1}/{n_frames}]')
        return list(lines.values()) + list(points.values()) + [title_text]

    if save_path and save_path.endswith('.mp4'):
        print(f"正在生成动画 ({n_frames}帧, {fps}fps)...")
        anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                        interval=1000//fps, blit=False)
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer)
        print(f"已保存动画: {save_path}")
        plt.close(fig)
    else:
        # 交互模式: 滑块控制
        slider.on_changed(update)
        update(0)
        plt.show()
        plt.close(fig)


def export_html(df, show_court=True, save_path='trajectory.html'):
    """使用plotly导出交互式HTML"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("错误: HTML输出需要 plotly，请运行 pip install plotly")
        sys.exit(1)

    fig = go.Figure()
    colors_list = list(mcolors.TABLEAU_COLORS.values())

    for i, (tid, group) in enumerate(df.groupby('track_id')):
        color = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatter3d(
            x=group['x'], y=group['y'], z=group['z'],
            mode='lines+markers',
            marker=dict(size=2, color=color),
            line=dict(color=color, width=3),
            name=f'Track {tid}',
            hovertemplate='x=%{x:.2f}m<br>y=%{y:.2f}m<br>z=%{z:.2f}m<extra>Track %{text}</extra>',
            text=[str(tid)] * len(group)
        ))

        # 落点
        if all(c in df.columns for c in ['landing_x', 'landing_y']):
            landings = group.dropna(subset=['landing_x', 'landing_y'])
            if len(landings) > 0:
                last = landings.iloc[-1]
                fig.add_trace(go.Scatter3d(
                    x=[last['landing_x']], y=[last['landing_y']], z=[0],
                    mode='markers', marker=dict(size=8, color=color, symbol='x'),
                    name=f'落点 T{tid}', showlegend=False
                ))

    if show_court:
        # 场地边界
        cx = [0, COURT_WIDTH, COURT_WIDTH, 0, 0]
        cy = [0, 0, COURT_LENGTH, COURT_LENGTH, 0]
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=[0]*5, mode='lines',
            line=dict(color='green', width=3), name='球场', showlegend=True
        ))
        # 网线
        fig.add_trace(go.Scatter3d(
            x=[0, COURT_WIDTH], y=[NET_Y, NET_Y], z=[NET_HEIGHT, NET_HEIGHT],
            mode='lines', line=dict(color='black', width=4), name='球网'
        ))

    fig.update_layout(
        title='排球3D轨迹 — 交互式',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=1200, height=800
    )

    fig.write_html(save_path)
    print(f"已保存HTML: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='排球3D轨迹可视化工具')
    parser.add_argument('--input', '-i', required=True, help='CSV轨迹文件路径')
    parser.add_argument('--output', '-o', default=None, help='输出文件路径 (不指定则交互显示)')
    parser.add_argument('--format', '-f', choices=['png', 'mp4', 'html'], default='png',
                        help='输出格式 (默认: png)')
    parser.add_argument('--fps', type=int, default=30, help='动画帧率 (默认: 30)')
    parser.add_argument('--court', action='store_true', default=True, help='显示球场 (默认开启)')
    parser.add_argument('--no-court', dest='court', action='store_false', help='不显示球场')
    # argparse不支持--3d作为参数名，用--view替代
    parser.add_argument('--view', choices=['3d', 'bev'], default='3d',
                        help='视角: 3d或bev俯视 (默认: 3d)')
    parser.add_argument('--trail', type=int, default=20, help='动画拖尾帧数 (默认: 20)')
    args = parser.parse_args()

    # 读取数据
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"错误: 文件不存在 — {csv_path}")
        sys.exit(1)

    df = load_csv(str(csv_path))
    print(f"已加载 {len(df)} 条记录, {df['track_id'].nunique()} 条轨迹, "
          f"帧范围 [{df['frame_id'].min()}, {df['frame_id'].max()}]")

    # 确定输出路径
    out = args.output
    if out is None and args.format != 'png':
        out = str(csv_path.with_suffix(f'.{args.format}'))

    # 分发
    if args.format == 'html':
        export_html(df, show_court=args.court, save_path=out or str(csv_path.with_suffix('.html')))
    elif args.format == 'mp4':
        animate_trajectory(df, fps=args.fps, show_court=args.court,
                           save_path=out or str(csv_path.with_suffix('.mp4')), trail=args.trail)
    else:
        # png — 静态图
        if args.view == 'bev':
            plot_bev(df, show_court=args.court, save_path=out)
        else:
            plot_static_3d(df, show_court=args.court, save_path=out)

        # 无输出路径时也生成动画交互预览
        if out is None:
            animate_trajectory(df, fps=args.fps, show_court=args.court, trail=args.trail)


if __name__ == '__main__':
    main()
