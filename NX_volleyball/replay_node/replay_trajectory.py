#!/usr/bin/env python3
"""
排球轨迹 CSV 回放节点 — 终端交互式控制 + 滤波预测可视化

用法:
  python3 replay_trajectory.py --csv <path.csv> [--speed 1.0] [--loop]
  python3 replay_trajectory.py --csv <path.csv> --filter imm --ground-height 0.0

键盘控制:
  Space   暂停/继续
  → / d   前进 1 帧
  ← / a   后退 1 帧
  . / w   前进 10 帧
  , / s   后退 10 帧
  +/=     加速 (×2)
  -       减速 (÷2)
  r       回到起点
  q       退出
"""

import csv
import sys
import os
import time
import argparse
import termios
import tty
import select
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    has_detection: bool
    obs_x: float
    obs_y: float
    obs_z: float


def _safe_float(row, key):
    val = row.get(key)
    if val is None or val.strip() == '':
        return None
    return float(val)


def load_csv(path: str) -> List[Frame]:
    frames = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        # 兼容多种CSV格式:
        #   1) obs_x/obs_y/obs_z + has_detection (agx-zed raw_mode=false)
        #   2) x/y/z + has_detection (旧格式)
        #   3) x/y/z + confidence (NX-hik raw_mode=true, 无 has_detection 列)
        for row in reader:
            if 'has_detection' in row:
                has_det = int(row['has_detection']) == 1
            elif 'confidence' in row:
                # NX-hik 格式: confidence > 0 视为有检测
                conf = _safe_float(row, 'confidence')
                has_det = conf is not None and conf > 0.0
            else:
                has_det = True

            if 'obs_x' in row:
                x = _safe_float(row, 'obs_x') or 0.0
                y = _safe_float(row, 'obs_y') or 0.0
                z = _safe_float(row, 'obs_z') or 0.0
            else:
                x = _safe_float(row, 'x') or 0.0
                y = _safe_float(row, 'y') or 0.0
                z = _safe_float(row, 'z') or 0.0
            frames.append(Frame(
                frame_id=int(row['frame_id']),
                timestamp=float(row['timestamp']),
                has_detection=has_det,
                obs_x=x,
                obs_y=y,
                obs_z=z,
            ))
    return frames


class ReplayNode(Node):
    def __init__(self, frames: List[Frame], speed: float, loop: bool,
                 filter_name: str = None, ground_height: float = 0.0,
                 config_path: str = None):
        super().__init__('trajectory_replay')
        self.frames = frames
        self.speed = speed
        self.loop = loop
        self.idx = 0
        self.paused = False
        self.last_publish_time = 0.0

        # Filter setup
        self.filter_name = filter_name
        self.ground_height = ground_height
        self.filt = None
        self.gravity_vec = np.array([0.0, 9.81, 0.0])
        self.filtered_positions = []
        self.landing_point = None
        self._last_filter_ts = None       # 上次滤波器更新的时间戳
        self._init_obs_buffer = []        # 物理反推初始化缓冲
        self._filter_initialized = False  # 滤波器是否完成物理初始化
        self._outlier_gate = 2.0          # 异常点门限 (m)

        if filter_name:
            self._init_filter(filter_name, config_path)

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.pub_realtime = self.create_publisher(PointStamped, '/ball/realtime', best_effort)
        self.pub_path = self.create_publisher(Path, '/ball/actual_path', best_effort)
        self.pub_filtered_pos = self.create_publisher(PointStamped, '/ball/filtered_pos', best_effort)
        self.pub_filtered_path = self.create_publisher(Path, '/ball/filtered_path', best_effort)
        self.pub_predicted_path = self.create_publisher(Path, '/ball/predicted_path', best_effort)
        self.pub_landing = self.create_publisher(PointStamped, '/ball/landing', reliable)
        self.pub_ground_plane = self.create_publisher(Marker, '/ball/ground_plane', reliable)

        # 终端原始模式
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def _init_filter(self, filter_name: str, config_path: str = None):
        """Initialize filter from trajectory_analysis module."""
        # Add trajectory_analysis to path
        ta_dir = os.path.join(os.path.dirname(__file__), '..', 'trajectory_analysis')
        ta_dir = os.path.abspath(ta_dir)
        if ta_dir not in sys.path:
            sys.path.insert(0, ta_dir)

        from filters import create_filter

        # Load config
        if config_path is None:
            config_path = os.path.join(ta_dir, 'config.yaml')

        params = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Get gravity_vec from config
            physics = config.get('physics', {})
            gv = physics.get('gravity_vec', [0.0, 9.81, 0.0])
            self.gravity_vec = np.array(gv)
            # Get filter-specific params
            filters_cfg = config.get('filters', {})
            # Handle robust_ prefix
            lookup_name = filter_name[7:] if filter_name.startswith('robust_') else filter_name
            params = dict(filters_cfg.get(lookup_name, {}))

        self.filt = create_filter(filter_name, **params)
        self.filt.reset()
        print(f"  滤波器: {filter_name} (参数: {params})")
        print(f"  落地平面: y = {self.ground_height:.3f}m")

    def _reset_filter(self):
        """Reset filter state for replay restart."""
        if self.filt:
            self.filt.reset()
            self.filtered_positions.clear()
            self.landing_point = None
            self._last_filter_ts = None
            self._init_obs_buffer.clear()
            self._filter_initialized = False

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()

    def get_key(self) -> Optional[str]:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.01)[0] else ''
                ch3 = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.01)[0] else ''
                if ch2 == '[':
                    if ch3 == 'C': return 'RIGHT'
                    if ch3 == 'D': return 'LEFT'
                return None
            return ch
        return None

    def handle_input(self) -> bool:
        key = self.get_key()
        if key is None:
            return True
        if key == 'q':
            return False
        elif key == ' ':
            self.paused = not self.paused
        elif key in ('RIGHT', 'd'):
            self.seek(1)
        elif key in ('LEFT', 'a'):
            self.seek(-1)
        elif key in ('.', 'w'):
            self.seek(10)
        elif key in (',', 's'):
            self.seek(-10)
        elif key in ('+', '='):
            self.speed = min(self.speed * 2.0, 16.0)
        elif key == '-':
            self.speed = max(self.speed / 2.0, 0.125)
        elif key == 'r':
            self.idx = 0
            self._reset_filter()
        return True

    def seek(self, delta: int):
        self.idx = max(0, min(len(self.frames) - 1, self.idx + delta))
        self.publish_current()

    def make_stamp(self) -> Time:
        now = self.get_clock().now().to_msg()
        return now

    def publish_current(self):
        frame = self.frames[self.idx]
        stamp = self.make_stamp()

        # Realtime point
        # CSV坐标(世界坐标): obs_x=右, obs_y=重力方向(下为正), obs_z=深度(前)
        # RViz坐标: msg.x=obs_x(右), msg.y=obs_z(前), msg.z=-obs_y(上)
        if frame.has_detection:
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'vision_world'
            msg.point.x = frame.obs_x
            msg.point.y = frame.obs_z
            msg.point.z = -frame.obs_y
            self.pub_realtime.publish(msg)

        # Actual path (all detected frames up to current)
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = 'vision_world'
        for i in range(self.idx + 1):
            f = self.frames[i]
            if f.has_detection:
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = 'vision_world'
                pose.pose.position.x = f.obs_x
                pose.pose.position.y = f.obs_z
                pose.pose.position.z = -f.obs_y
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
        self.pub_path.publish(path)

        # Filter processing
        if self.filt and frame.has_detection:
            self._process_filter(frame, stamp)

    def _process_filter(self, frame, stamp):
        """Run filter update with outlier rejection and physics-guided init."""
        obs = np.array([frame.obs_x, frame.obs_y, frame.obs_z])

        # 1) 剔除无效点 — 坐标全为零或接近原点视为丢失
        if np.linalg.norm(obs) < 0.01:
            return

        # 2) 间隔检测 — 长时间无有效更新则重置滤波器(新段落)
        if self._last_filter_ts is not None:
            gap = frame.timestamp - self._last_filter_ts
            if gap > 0.3:
                # 长间隔：认为是新段落，重置滤波器
                self.filt.reset()
                self._init_obs_buffer.clear()
                self._filter_initialized = False
                self.filtered_positions.clear()
                self.landing_point = None
                self._last_filter_ts = None

        # 3) 物理反推初始化阶段 — 收集前几帧拟合抛物线
        if not self._filter_initialized:
            self._init_obs_buffer.append((frame.timestamp, obs.copy()))
            if len(self._init_obs_buffer) >= 3:
                self._physics_init()
            else:
                # 初始化期间只做首次update让滤波器有位置
                self.filt.update(obs[0], obs[1], obs[2])
                self._last_filter_ts = frame.timestamp
                return
            if not self._filter_initialized:
                return

        # 4) Predict step — 使用真实dt(自上次滤波器更新以来)
        dt = frame.timestamp - self._last_filter_ts
        if dt > 0:
            self.filt.predict(dt)

        # 5) 异常点检测 — 与滤波器预测位置比较
        predicted_state = self.filt.get_state()
        pred_pos = np.array([predicted_state.x, predicted_state.y, predicted_state.z])
        innovation_dist = np.linalg.norm(obs - pred_pos)

        if innovation_dist > self._outlier_gate:
            # 异常点：不更新滤波器，不发布滤波结果
            self._last_filter_ts = frame.timestamp
            return

        # 6) Update step
        state = self.filt.update(obs[0], obs[1], obs[2])
        self._last_filter_ts = frame.timestamp

        pos = np.array([state.x, state.y, state.z])
        vel = np.array([state.vx, state.vy, state.vz])
        self.filtered_positions.append(pos)

        # Publish filtered position
        fmsg = PointStamped()
        fmsg.header.stamp = stamp
        fmsg.header.frame_id = 'vision_world'
        fmsg.point.x = pos[0]
        fmsg.point.y = pos[2]   # z -> rviz y (前)
        fmsg.point.z = -pos[1]  # -y -> rviz z (上)
        self.pub_filtered_pos.publish(fmsg)

        # Publish filtered path
        fpath = Path()
        fpath.header.stamp = stamp
        fpath.header.frame_id = 'vision_world'
        for p in self.filtered_positions:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'vision_world'
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[2]
            pose.pose.position.z = -p[1]
            pose.pose.orientation.w = 1.0
            fpath.poses.append(pose)
        self.pub_filtered_path.publish(fpath)

        # Predict landing point
        self._predict_landing(pos, vel, stamp)

    def _physics_init(self):
        """Physics-guided initialization: fit parabola to first observations.
        
        Given ≥3 observations with known gravity, solve for initial velocity:
          obs[i] = obs[0] + v0 * dt_i + 0.5 * g * dt_i^2
        Rearranging:
          (obs[i] - obs[0] - 0.5*g*dt_i^2) / dt_i = v0
        Use least-squares over all pairs for robustness.
        """
        buf = self._init_obs_buffer
        t0 = buf[0][0]
        p0 = buf[0][1]
        g = self.gravity_vec

        # Least-squares: for each observation i>0, estimate v0
        # A * v0 = b  where each row is dt_i * I3, b_i = (obs_i - p0 - 0.5*g*dt_i^2)
        A_rows = []
        b_rows = []
        for i in range(1, len(buf)):
            dt_i = buf[i][0] - t0
            if dt_i < 0.005:
                continue
            obs_i = buf[i][1]
            b_i = obs_i - p0 - 0.5 * g * dt_i**2
            A_rows.append(dt_i * np.eye(3))
            b_rows.append(b_i)

        if len(A_rows) < 2:
            return

        A = np.vstack(A_rows)
        b = np.concatenate(b_rows)
        v0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Verify physical consistency: predict middle point
        mid_idx = len(buf) // 2
        dt_mid = buf[mid_idx][0] - t0
        pred_mid = p0 + v0 * dt_mid + 0.5 * g * dt_mid**2
        if np.linalg.norm(pred_mid - buf[mid_idx][1]) > 0.5:
            # Poor fit — don't use physics init, let filter converge naturally
            self._filter_initialized = True
            self._last_filter_ts = buf[-1][0]
            return

        # Initialize filter with physics-derived state
        # Feed all buffered observations sequentially to build up the filter
        self.filt.reset()
        for i, (ts, obs) in enumerate(buf):
            if i > 0:
                dt = ts - buf[i-1][0]
                if dt > 0:
                    self.filt.predict(dt)
            self.filt.update(obs[0], obs[1], obs[2])

        # Override velocity estimate if filter supports direct state access
        # (Most EKFs: x[3:6] = velocity) — try to inject v0
        if hasattr(self.filt, 'x') and hasattr(self.filt.x, '__len__') and len(self.filt.x) >= 6:
            self.filt.x[3:6] = v0
        elif hasattr(self.filt, '_models'):
            # IMM: inject into all sub-models
            for m_x in self.filt._models if hasattr(self.filt, '_models') else []:
                if hasattr(m_x, '__len__') and len(m_x) >= 6:
                    m_x[3:6] = v0

        self._filter_initialized = True
        self._last_filter_ts = buf[-1][0]

        # Record positions for path visualization
        for _, obs in buf:
            self.filtered_positions.append(obs.copy())

    def _predict_landing(self, pos, vel, stamp):
        """Solve parabolic landing: pos_y + vy*t + 0.5*g_y*t^2 = ground_height."""
        gy = self.gravity_vec[1]  # 正值表示向下
        y0 = pos[1]
        vy = vel[1]

        # 解方程: y0 + vy*t + 0.5*gy*t^2 = ground_height
        # 0.5*gy*t^2 + vy*t + (y0 - ground_height) = 0
        a = 0.5 * gy
        b = vy
        c = y0 - self.ground_height

        discriminant = b * b - 4 * a * c
        if discriminant < 0 or abs(a) < 1e-10:
            return

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b + sqrt_d) / (2 * a)
        t2 = (-b - sqrt_d) / (2 * a)

        # Pick smallest positive time
        candidates = [t for t in [t1, t2] if t > 0.01]
        if not candidates:
            return
        t_land = min(candidates)

        # Cap prediction to 3 seconds
        if t_land > 3.0:
            return

        # Landing position
        landing_x = pos[0] + vel[0] * t_land
        landing_y = self.ground_height
        landing_z = pos[2] + vel[2] * t_land
        self.landing_point = np.array([landing_x, landing_y, landing_z])

        # Publish landing point
        lmsg = PointStamped()
        lmsg.header.stamp = stamp
        lmsg.header.frame_id = 'vision_world'
        lmsg.point.x = landing_x
        lmsg.point.y = landing_z          # z -> rviz y
        lmsg.point.z = -landing_y         # -y -> rviz z
        self.pub_landing.publish(lmsg)

        # Publish predicted parabolic path (20 points from now to landing)
        ppath = Path()
        ppath.header.stamp = stamp
        ppath.header.frame_id = 'vision_world'
        n_points = 20
        for i in range(n_points + 1):
            t = t_land * i / n_points
            px = pos[0] + vel[0] * t
            py = pos[1] + vel[1] * t + 0.5 * gy * t * t
            pz = pos[2] + vel[2] * t
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'vision_world'
            pose.pose.position.x = px
            pose.pose.position.y = pz     # z -> rviz y
            pose.pose.position.z = -py    # -y -> rviz z
            pose.pose.orientation.w = 1.0
            ppath.poses.append(pose)
        self.pub_predicted_path.publish(ppath)

        # Publish ground plane marker
        self._publish_ground_plane(stamp)

    def _publish_ground_plane(self, stamp):
        """Publish a flat plane marker at ground_height."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = 'vision_world'
        marker.ns = 'ground'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # ground_height is world-y (down positive), rviz z = -y
        marker.pose.position.x = 0.0
        marker.pose.position.y = 4.0   # 前方中心
        marker.pose.position.z = -self.ground_height
        marker.pose.orientation.w = 1.0
        marker.scale.x = 6.0   # 宽
        marker.scale.y = 10.0  # 深
        marker.scale.z = 0.005
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 0.3
        self.pub_ground_plane.publish(marker)

    def run(self):
        total = len(self.frames)
        print(f"\n{'='*56}")
        print(f"  轨迹回放: {total} 帧, 速度 {self.speed:.1f}x")
        if self.filter_name:
            print(f"  滤波: {self.filter_name}  落地面: y={self.ground_height:.2f}m")
        print(f"{'='*56}")
        print("  Space=暂停  ←/→=±1帧  ,/.=±10帧  +/-=加减速  r=重头  q=退出")
        print(f"{'='*56}\n")

        try:
            while rclpy.ok():
                if not self.handle_input():
                    break

                if not self.paused:
                    now = time.monotonic()
                    if self.idx < total:
                        # 计算帧间隔
                        if self.idx > 0:
                            dt = (self.frames[self.idx].timestamp
                                  - self.frames[self.idx - 1].timestamp)
                        else:
                            dt = 1.0 / 60.0

                        wait = dt / self.speed
                        elapsed = now - self.last_publish_time

                        if elapsed >= wait:
                            self.publish_current()
                            self.last_publish_time = now
                            self.idx += 1

                            # 状态行
                            f = self.frames[min(self.idx, total - 1)]
                            d = '●' if f.has_detection else '○'
                            landing_str = ''
                            if self.landing_point is not None:
                                lp = self.landing_point
                                landing_str = f' L({lp[0]:.1f},{lp[2]:.1f})'
                            sys.stdout.write(
                                f"\r  ▶ [{self.idx:>5}/{total}] "
                                f"{self.speed:.1f}x "
                                f"{d} ({f.obs_x:.2f},{f.obs_y:.2f},{f.obs_z:.2f})"
                                f"{landing_str}   "
                            )
                            sys.stdout.flush()
                    else:
                        if self.loop:
                            self.idx = 0
                            self._reset_filter()
                            print("\n  ↻ 循环重播...")
                        else:
                            print("\n  ■ 播放完毕")
                            # 暂停等待用户退出或按 r 重播
                            self.paused = True
                else:
                    # 暂停状态，低频刷新
                    f = self.frames[min(self.idx, total - 1)]
                    d = '●' if f.has_detection else '○'
                    sys.stdout.write(
                        f"\r  ⏸ [{self.idx:>5}/{total}] "
                        f"{self.speed:.1f}x "
                        f"{d} ({f.obs_x:.2f},{f.obs_y:.2f},{f.obs_z:.2f})   "
                    )
                    sys.stdout.flush()
                    time.sleep(0.05)

                time.sleep(0.001)  # 避免 CPU 空转

        except KeyboardInterrupt:
            pass
        finally:
            print("\n  退出回放")


def main():
    parser = argparse.ArgumentParser(description='排球轨迹CSV回放 + 滤波预测可视化')
    parser.add_argument('--csv', required=True, help='CSV 文件路径')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍率')
    parser.add_argument('--loop', action='store_true', help='循环播放')
    parser.add_argument('--filter', type=str, default=None,
                        help='滤波器名称 (如 imm, gravity_ekf_6d, fast_gravity_ekf)')
    parser.add_argument('--ground-height', type=float, default=0.0,
                        help='落地平面高度 (world-y坐标, 正值=向下偏移)')
    parser.add_argument('--config', type=str, default=None,
                        help='滤波器配置文件路径 (默认: trajectory_analysis/config.yaml)')
    args = parser.parse_args()

    frames = load_csv(args.csv)
    if not frames:
        print(f"错误: CSV 为空或读取失败: {args.csv}")
        sys.exit(1)

    rclpy.init()
    node = ReplayNode(frames, args.speed, args.loop,
                      filter_name=args.filter,
                      ground_height=args.ground_height,
                      config_path=args.config)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
