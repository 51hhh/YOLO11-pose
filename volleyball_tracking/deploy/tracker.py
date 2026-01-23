#!/usr/bin/env python3
"""
ByteTrack + 卡尔曼滤波追踪模块
实现多目标追踪和轨迹平滑
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Detection:
    """检测结果数据类"""
    cx: float  # 圆心 x
    cy: float  # 圆心 y
    r: float   # 半径
    conf: float  # 置信度
    keypoints: np.ndarray  # (5, 2) 关键点坐标
    keypoints_conf: np.ndarray  # (5,) 关键点置信度

@dataclass
class Track:
    """追踪轨迹数据类"""
    id: int  # 轨迹 ID
    cx: float  # 当前圆心 x
    cy: float  # 当前圆心 y
    r: float   # 当前半径
    vx: float  # x 方向速度
    vy: float  # y 方向速度
    conf: float  # 置信度
    age: int   # 轨迹年龄
    hits: int  # 命中次数
    time_since_update: int  # 自上次更新的帧数
    kf: KalmanFilter  # 卡尔曼滤波器

class VolleyballTracker:
    """排球追踪器 (ByteTrack + Kalman)"""
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 150
    ):
        """
        初始化追踪器
        
        Args:
            track_thresh: 高分检测阈值
            track_buffer: 保留低分检测的帧数
            match_thresh: IoU 匹配阈值
            frame_rate: 视频帧率
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracks: List[Track] = []
        self.next_id = 0
        self.dt = 1.0 / frame_rate
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        更新追踪
        
        Args:
            detections: 当前帧的检测结果
        
        Returns:
            活跃的追踪轨迹
        """
        # 分离高分和低分检测 (ByteTrack 核心思想)
        high_dets = [d for d in detections if d.conf >= self.track_thresh]
        low_dets = [d for d in detections if d.conf < self.track_thresh]
        
        # 预测所有轨迹的下一帧位置
        for track in self.tracks:
            track.kf.predict()
            track.cx = track.kf.x[0]
            track.cy = track.kf.x[1]
            track.r = track.kf.x[2]
            track.vx = track.kf.x[3]
            track.vy = track.kf.x[4]
            track.time_since_update += 1
        
        # 第一阶段: 匹配高分检测
        matched, unmatched_tracks, unmatched_dets = self._match(
            self.tracks, high_dets
        )
        
        # 更新匹配的轨迹
        for track_idx, det_idx in matched:
            self._update_track(self.tracks[track_idx], high_dets[det_idx])
        
        # 第二阶段: 用低分检测匹配未匹配的轨迹
        # 这是 ByteTrack 的关键：保留低分检测用于追踪
        unmatched_tracks_list = [self.tracks[i] for i in unmatched_tracks]
        matched_low, unmatched_tracks_low, _ = self._match(
            unmatched_tracks_list, low_dets
        )
        
        for track_idx, det_idx in matched_low:
            self._update_track(unmatched_tracks_list[track_idx], low_dets[det_idx])
        
        # 创建新轨迹 (只用高分检测)
        for det_idx in unmatched_dets:
            self._create_track(high_dets[det_idx])
        
        # 删除长时间未更新的轨迹
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.track_buffer
        ]
        
        # 返回活跃轨迹 (至少命中 3 次)
        return [t for t in self.tracks if t.hits >= 3]
    
    def _match(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匹配轨迹和检测
        使用匈牙利算法进行最优匹配
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # 计算 IoU 矩阵
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._circle_iou(
                    track.cx, track.cy, track.r,
                    det.cx, det.cy, det.r
                )
        
        # 匈牙利匹配 (最大化 IoU)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched = []
        unmatched_tracks = []
        unmatched_dets = list(range(len(detections)))
        
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.match_thresh:
                matched.append((i, j))
                unmatched_dets.remove(j)
            else:
                unmatched_tracks.append(i)
        
        # 添加未匹配的轨迹
        for i in range(len(tracks)):
            if i not in row_ind:
                unmatched_tracks.append(i)
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _circle_iou(
        self,
        cx1: float, cy1: float, r1: float,
        cx2: float, cy2: float, r2: float
    ) -> float:
        """
        计算两个圆的 IoU
        使用精确的圆相交面积公式
        """
        d = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        # 不相交
        if d >= r1 + r2:
            return 0.0
        
        # 一个包含另一个
        if d <= abs(r1 - r2):
            r_min = min(r1, r2)
            r_max = max(r1, r2)
            return (r_min / r_max) ** 2
        
        # 部分相交 - 使用精确公式
        intersection_area = (
            r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2*d*r1)) +
            r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2*d*r2)) -
            0.5 * np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
        )
        union_area = np.pi * (r1**2 + r2**2) - intersection_area
        
        return intersection_area / union_area
    
    def _create_track(self, detection: Detection):
        """创建新轨迹"""
        kf = self._create_kalman_filter()
        kf.x[:3] = [detection.cx, detection.cy, detection.r]
        
        track = Track(
            id=self.next_id,
            cx=detection.cx,
            cy=detection.cy,
            r=detection.r,
            vx=0.0,
            vy=0.0,
            conf=detection.conf,
            age=0,
            hits=1,
            time_since_update=0,
            kf=kf
        )
        
        self.tracks.append(track)
        self.next_id += 1
    
    def _update_track(self, track: Track, detection: Detection):
        """更新轨迹"""
        # 卡尔曼更新
        z = np.array([detection.cx, detection.cy, detection.r])
        track.kf.update(z)
        
        # 更新状态
        track.cx = track.kf.x[0]
        track.cy = track.kf.x[1]
        track.r = track.kf.x[2]
        track.vx = track.kf.x[3]
        track.vy = track.kf.x[4]
        track.conf = detection.conf
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
    
    def _create_kalman_filter(self) -> KalmanFilter:
        """
        创建卡尔曼滤波器
        状态向量: [cx, cy, r, vx, vy, vr, ax, ay]
        观测向量: [cx, cy, r]
        """
        kf = KalmanFilter(dim_x=8, dim_z=3)
        
        # 状态转移矩阵 (匀加速运动模型)
        dt = self.dt
        kf.F = np.array([
            [1, 0, 0, dt, 0,  0, 0.5*dt**2, 0],
            [0, 1, 0, 0,  dt, 0, 0, 0.5*dt**2],
            [0, 0, 1, 0,  0,  dt, 0, 0],
            [0, 0, 0, 1,  0,  0, dt, 0],
            [0, 0, 0, 0,  1,  0, 0, dt],
            [0, 0, 0, 0,  0,  1, 0, 0],
            [0, 0, 0, 0,  0,  0, 1, 0],
            [0, 0, 0, 0,  0,  0, 0, 1]
        ])
        
        # 观测矩阵 (只观测位置和半径)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0]
        ])
        
        # 过程噪声协方差 (根据实际运动调整)
        kf.Q *= 0.1
        
        # 观测噪声协方差 (圆心精度高，半径精度低)
        kf.R = np.diag([0.5, 0.5, 1.0])
        
        # 初始协方差
        kf.P *= 10.0
        
        return kf
