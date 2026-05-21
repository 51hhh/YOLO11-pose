#!/usr/bin/env python3
"""
排球轨迹 CSV 回放节点 — 终端交互式控制

用法:
  python3 replay_trajectory.py --csv <path.csv> [--speed 1.0] [--loop]

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
import time
import argparse
import termios
import tty
import select
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time


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
        # 兼容新(obs_x/obs_y/obs_z)和旧(x/y/z)格式
        for row in reader:
            has_det = int(row['has_detection']) == 1
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
    def __init__(self, frames: List[Frame], speed: float, loop: bool):
        super().__init__('trajectory_replay')
        self.frames = frames
        self.speed = speed
        self.loop = loop
        self.idx = 0
        self.paused = False
        self.last_publish_time = 0.0

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.pub_realtime = self.create_publisher(PointStamped, '/ball/realtime', best_effort)
        self.pub_path = self.create_publisher(Path, '/ball/actual_path', best_effort)

        # 终端原始模式
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

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
        # CSV坐标: obs_x=相机右, obs_y=相机下, obs_z=深度
        # RViz坐标: msg.x=右, msg.y=深度(前), msg.z=上(-obs_y)
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

    def run(self):
        total = len(self.frames)
        print(f"\n{'='*56}")
        print(f"  轨迹回放: {total} 帧, 速度 {self.speed:.1f}x")
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
                            sys.stdout.write(
                                f"\r  ▶ [{self.idx:>5}/{total}] "
                                f"{self.speed:.1f}x "
                                f"{d} ({f.obs_x:.2f},{f.obs_y:.2f},{f.obs_z:.2f})   "
                            )
                            sys.stdout.flush()
                    else:
                        if self.loop:
                            self.idx = 0
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
    parser = argparse.ArgumentParser(description='排球轨迹CSV回放')
    parser.add_argument('--csv', required=True, help='CSV 文件路径')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍率')
    parser.add_argument('--loop', action='store_true', help='循环播放')
    args = parser.parse_args()

    frames = load_csv(args.csv)
    if not frames:
        print(f"错误: CSV 为空或读取失败: {args.csv}")
        sys.exit(1)

    rclpy.init()
    node = ReplayNode(frames, args.speed, args.loop)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
