#!/usr/bin/env python3
"""
可视化工具模块
用于绘制检测框、关键点、圆形和追踪轨迹
"""
import cv2
import numpy as np
from typing import List
from tracker import Detection, Track

class Visualizer:
    """可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        # 颜色定义 (BGR)
        self.colors = {
            'bbox': (0, 255, 0),       # 绿色
            'circle': (255, 0, 0),     # 蓝色
            'keypoint': (0, 0, 255),   # 红色
            'center': (255, 255, 0),   # 青色
            'track': (255, 165, 0),    # 橙色
            'velocity': (255, 0, 255), # 紫色
        }
        
        # 追踪轨迹历史 (用于绘制轨迹线)
        self.track_history = {}  # track_id -> [(cx, cy), ...]
        self.max_history = 30    # 最多保留 30 帧历史
    
    def draw(
        self,
        image: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        show_keypoints: bool = True,
        show_circle: bool = True,
        show_velocity: bool = True,
        show_trajectory: bool = True
    ) -> np.ndarray:
        """
        绘制检测和追踪结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            tracks: 追踪轨迹列表
            show_keypoints: 是否显示关键点
            show_circle: 是否显示拟合圆
            show_velocity: 是否显示速度向量
            show_trajectory: 是否显示轨迹线
        
        Returns:
            可视化图像
        """
        vis = image.copy()
        
        # 绘制检测结果
        for det in detections:
            if show_keypoints:
                self._draw_keypoints(vis, det.keypoints, det.keypoints_conf)
            
            if show_circle:
                self._draw_circle(vis, det.cx, det.cy, det.r, det.conf)
        
        # 绘制追踪结果
        for track in tracks:
            # 绘制圆形
            self._draw_track_circle(vis, track)
            
            # 绘制速度向量
            if show_velocity:
                self._draw_velocity(vis, track)
            
            # 绘制轨迹线
            if show_trajectory:
                self._draw_trajectory(vis, track)
            
            # 绘制 ID 和信息
            self._draw_track_info(vis, track)
        
        return vis
    
    def _draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ):
        """绘制关键点"""
        # 关键点名称
        kpt_names = ['Center', 'Top', 'Bottom', 'Left', 'Right']
        
        for i, (kpt, conf) in enumerate(zip(keypoints, confidences)):
            x, y = int(kpt[0]), int(kpt[1])
            
            # 根据置信度调整颜色
            if conf > 0.8:
                color = self.colors['keypoint']
            elif conf > 0.5:
                color = (0, 165, 255)  # 橙色
            else:
                color = (128, 128, 128)  # 灰色
            
            # Center 点用特殊颜色
            if i == 0:
                color = self.colors['center']
                radius = 6
            else:
                radius = 4
            
            # 绘制关键点
            cv2.circle(image, (x, y), radius, color, -1)
            cv2.circle(image, (x, y), radius+2, (255, 255, 255), 1)
            
            # 绘制标签
            label = f"{kpt_names[i]}"
            cv2.putText(
                image, label, (x+10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
    
    def _draw_circle(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        conf: float
    ):
        """绘制拟合圆"""
        center = (int(cx), int(cy))
        radius = int(r)
        
        # 绘制圆
        cv2.circle(image, center, radius, self.colors['circle'], 2)
        
        # 绘制圆心
        cv2.circle(image, center, 3, self.colors['circle'], -1)
        
        # 绘制置信度
        label = f"{conf:.2f}"
        cv2.putText(
            image, label, (int(cx)+int(r)+5, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['circle'], 2
        )
    
    def _draw_track_circle(self, image: np.ndarray, track: Track):
        """绘制追踪圆形"""
        center = (int(track.cx), int(track.cy))
        radius = int(track.r)
        
        # 使用不同颜色区分不同轨迹
        color = self._get_track_color(track.id)
        
        # 绘制圆 (更粗的线条)
        cv2.circle(image, center, radius, color, 3)
        
        # 绘制圆心
        cv2.circle(image, center, 5, color, -1)
        cv2.circle(image, center, 7, (255, 255, 255), 2)
    
    def _draw_velocity(self, image: np.ndarray, track: Track):
        """绘制速度向量"""
        # 计算速度向量的终点
        scale = 10.0  # 速度缩放因子
        end_x = int(track.cx + track.vx * scale)
        end_y = int(track.cy + track.vy * scale)
        
        start = (int(track.cx), int(track.cy))
        end = (end_x, end_y)
        
        # 绘制箭头
        cv2.arrowedLine(
            image, start, end,
            self.colors['velocity'], 2, tipLength=0.3
        )
        
        # 绘制速度值
        speed = np.sqrt(track.vx**2 + track.vy**2)
        label = f"{speed:.1f} px/frame"
        cv2.putText(
            image, label, (end_x+5, end_y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['velocity'], 1
        )
    
    def _draw_trajectory(self, image: np.ndarray, track: Track):
        """绘制轨迹线"""
        # 更新轨迹历史
        if track.id not in self.track_history:
            self.track_history[track.id] = []
        
        self.track_history[track.id].append((int(track.cx), int(track.cy)))
        
        # 限制历史长度
        if len(self.track_history[track.id]) > self.max_history:
            self.track_history[track.id].pop(0)
        
        # 绘制轨迹线
        points = self.track_history[track.id]
        if len(points) > 1:
            color = self._get_track_color(track.id)
            for i in range(1, len(points)):
                # 渐变透明度
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(image, points[i-1], points[i], color, thickness)
    
    def _draw_track_info(self, image: np.ndarray, track: Track):
        """绘制追踪信息"""
        x, y = int(track.cx), int(track.cy)
        r = int(track.r)
        
        # 信息文本
        info_lines = [
            f"ID: {track.id}",
            f"Conf: {track.conf:.2f}",
            f"Age: {track.age}",
        ]
        
        # 绘制背景框
        text_y = y - r - 10
        for i, line in enumerate(info_lines):
            text_size = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            
            # 背景矩形
            cv2.rectangle(
                image,
                (x - 5, text_y - text_size[1] - 5),
                (x + text_size[0] + 5, text_y + 5),
                (0, 0, 0), -1
            )
            
            # 文本
            color = self._get_track_color(track.id)
            cv2.putText(
                image, line, (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            
            text_y -= text_size[1] + 10
    
    def _get_track_color(self, track_id: int) -> tuple:
        """根据轨迹 ID 生成颜色"""
        # 使用 HSV 色彩空间生成不同颜色
        hue = (track_id * 50) % 180
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, color_bgr))
    
    def clear_history(self):
        """清空轨迹历史"""
        self.track_history.clear()
