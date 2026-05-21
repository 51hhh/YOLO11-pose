#!/usr/bin/env python3
"""测试 /ball/realtime 的抖动程度 (std, peak-to-peak)"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped
import numpy as np
import sys

class JitterTest(Node):
    def __init__(self, n_samples=200):
        super().__init__('jitter_test')
        self.n_samples = n_samples
        self.data = []
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.sub = self.create_subscription(
            PointStamped, '/ball/realtime', self.cb, qos)
        self.get_logger().info(f'采集 {n_samples} 帧...')

    def cb(self, msg):
        self.data.append([msg.point.x, msg.point.y, msg.point.z])
        if len(self.data) >= self.n_samples:
            self.analyze()
            raise SystemExit

    def analyze(self):
        arr = np.array(self.data)
        labels = ['X(右)', 'Y(深)', 'Z(上)']
        print(f'\n===== 抖动分析 ({len(arr)} samples) =====')
        print(f'{"轴":<8} {"均值":>8} {"标准差":>8} {"峰峰值":>8} {"相对σ":>8}')
        print('-' * 48)
        for i, label in enumerate(labels):
            col = arr[:, i]
            mean = np.mean(col)
            std = np.std(col)
            ptp = np.ptp(col)
            rel = std / abs(mean) * 100 if abs(mean) > 0.01 else 0
            print(f'{label:<8} {mean:>8.4f} {std:>8.4f} {ptp:>8.4f} {rel:>7.2f}%')
        
        # 深度方向详细
        z = arr[:, 1]  # Y=深度
        print(f'\n深度(Y)详细: min={z.min():.4f} max={z.max():.4f} range={z.ptp():.4f}m')
        print(f'  1σ范围: [{z.mean()-z.std():.4f}, {z.mean()+z.std():.4f}]')
        print(f'  3σ范围: [{z.mean()-3*z.std():.4f}, {z.mean()+3*z.std():.4f}]')
        
        # 帧间差分 (高频抖动)
        diffs = np.diff(arr, axis=0)
        print(f'\n帧间跳变 (frame-to-frame):')
        print(f'{"轴":<8} {"均值|Δ|":>10} {"最大|Δ|":>10}')
        print('-' * 32)
        for i, label in enumerate(labels):
            d = np.abs(diffs[:, i])
            print(f'{label:<8} {d.mean():>10.5f} {d.max():>10.5f}')

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rclpy.init()
    node = JitterTest(n)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
