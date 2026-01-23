#!/usr/bin/env python3
"""
几何拟合算法模块
实现加权最小二乘、RANSAC 和代数拟合方法
"""
import numpy as np
from scipy.optimize import least_squares
from typing import Tuple

class CircleFitter:
    """圆形拟合器"""
    
    def __init__(self, method: str = 'weighted_lsq'):
        """
        初始化拟合器
        
        Args:
            method: 拟合方法 ('weighted_lsq', 'ransac', 'algebraic')
        """
        self.method = method
    
    def fit(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        拟合圆形
        
        Args:
            keypoints: (5, 2) 关键点坐标 [x, y]
            confidences: (5,) 关键点置信度
        
        Returns:
            (cx, cy, r, quality): 圆心、半径和拟合质量 (0-1)
        """
        if self.method == 'weighted_lsq':
            return self._fit_weighted_lsq(keypoints, confidences)
        elif self.method == 'ransac':
            return self._fit_ransac(keypoints, confidences)
        else:
            return self._fit_algebraic(keypoints)
    
    def _fit_weighted_lsq(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        加权最小二乘拟合
        Center 点权重翻倍，提高圆心精度
        """
        # Center 点权重翻倍
        weights = confidences.copy()
        weights[0] *= 2.0
        
        def residuals(params):
            cx, cy, r = params
            dx = keypoints[:, 0] - cx
            dy = keypoints[:, 1] - cy
            distances = np.sqrt(dx**2 + dy**2)
            return weights * (distances - r)
        
        # 初始猜测: Center 点作为圆心，极值点平均距离作为半径
        x0 = [
            keypoints[0, 0],
            keypoints[0, 1],
            np.mean(np.linalg.norm(keypoints[1:] - keypoints[0], axis=1))
        ]
        
        try:
            result = least_squares(residuals, x0, method='lm')
            cx, cy, r = result.x
            
            # 计算拟合质量 (基于残差)
            final_residuals = residuals(result.x)
            quality = 1.0 / (1.0 + np.std(final_residuals))
            
            return cx, cy, r, quality
        except:
            # 拟合失败，返回 Center 点和平均距离
            cx, cy = keypoints[0]
            r = np.mean(np.linalg.norm(keypoints[1:] - keypoints[0], axis=1))
            return cx, cy, r, 0.0
    
    def _fit_ransac(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        threshold: float = 2.0,
        iterations: int = 50
    ) -> Tuple[float, float, float, float]:
        """
        RANSAC 鲁棒拟合
        适用于有遮挡或异常点的情况
        """
        best_circle = None
        best_inliers = 0
        best_quality = 0.0
        
        for _ in range(iterations):
            # 随机选 3 个高置信度点
            probs = confidences / confidences.sum()
            idx = np.random.choice(5, 3, replace=False, p=probs)
            sample_pts = keypoints[idx]
            
            # 拟合圆
            try:
                circle = self._fit_algebraic(sample_pts)
                cx, cy, r = circle[:3]
                
                # 计算内点数量
                distances = np.abs(
                    np.linalg.norm(keypoints - np.array([cx, cy]), axis=1) - r
                )
                inliers = np.sum(distances < threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_circle = circle
                    best_quality = inliers / len(keypoints)
            except:
                continue
        
        if best_circle is None:
            # RANSAC 失败，回退到加权最小二乘
            return self._fit_weighted_lsq(keypoints, confidences)
        
        return (*best_circle[:3], best_quality)
    
    def _fit_algebraic(
        self,
        keypoints: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        代数拟合 (快速但精度较低)
        使用最小二乘法求解线性方程组
        """
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        
        # 构建矩阵 A 和向量 b
        # 圆方程: x^2 + y^2 + Dx + Ey + F = 0
        # 转换为: x^2 + y^2 = -Dx - Ey - F
        A = np.column_stack([x, y, np.ones_like(x)])
        b = x**2 + y**2
        
        # 求解最小二乘
        try:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            cx = c[0] / 2
            cy = c[1] / 2
            r = np.sqrt(c[2] + cx**2 + cy**2)
            return cx, cy, r, 1.0
        except:
            # 失败时返回质心和平均距离
            cx, cy = keypoints.mean(axis=0)
            r = np.mean(np.linalg.norm(keypoints - np.array([cx, cy]), axis=1))
            return cx, cy, r, 0.0
    
    def validate_circle(
        self,
        cx: float,
        cy: float,
        r: float,
        img_width: int,
        img_height: int,
        min_radius: float = 5.0,
        max_radius: float = 100.0
    ) -> bool:
        """
        验证圆形的合理性
        
        Args:
            cx, cy, r: 圆心和半径
            img_width, img_height: 图像尺寸
            min_radius, max_radius: 半径范围
        
        Returns:
            是否合理
        """
        # 检查半径范围
        if r < min_radius or r > max_radius:
            return False
        
        # 检查圆心是否在图像内 (允许一定容差)
        margin = r
        if (cx < -margin or cx > img_width + margin or
            cy < -margin or cy > img_height + margin):
            return False
        
        return True
