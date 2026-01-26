#!/usr/bin/env python3
"""
生成测试用的默认标定文件
用于在没有真实标定数据时测试编译和基本功能
"""

import cv2
import numpy as np
import os

def create_default_calibration(output_file="stereo_calib.yaml"):
    """
    创建默认的双目标定参数
    
    假设:
    - 图像分辨率: 1280x720
    - 焦距: 1000 像素
    - 主点: 图像中心
    - 基线: 0.25m (25cm)
    - 无畸变
    """
    
    # 相机内参 (假设两个相机相同)
    K = np.array([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # 畸变系数 (假设无畸变)
    D = np.zeros((1, 5), dtype=np.float64)
    
    # 投影矩阵
    # P1 = [K | 0]
    P1 = np.hstack([K, np.zeros((3, 1))])
    
    # P2 = [K | -K*T], T = [baseline, 0, 0]^T
    baseline = 0.25  # 25cm
    T = np.array([[baseline], [0.0], [0.0]])
    P2 = np.hstack([K, -K @ T])
    
    # 创建 FileStorage
    fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
    
    # 写入参数
    fs.write("K1", K)
    fs.write("D1", D)
    fs.write("K2", K)
    fs.write("D2", D)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("baseline", baseline)
    
    fs.release()
    
    print(f"✅ 默认标定文件已创建: {output_file}")
    print(f"   图像尺寸: 1280x720")
    print(f"   焦距: 1000 px")
    print(f"   基线: {baseline} m")
    print(f"   ⚠️  注意: 这是测试用的默认参数，请使用真实标定数据替换")

def main():
    # 确保 calibration 目录存在
    calib_dir = "calibration"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
        print(f"📁 创建目录: {calib_dir}")
    
    output_file = os.path.join(calib_dir, "stereo_calib.yaml")
    
    # 检查文件是否已存在
    if os.path.exists(output_file):
        response = input(f"⚠️  文件已存在: {output_file}\n是否覆盖? (y/N): ")
        if response.lower() != 'y':
            print("取消操作")
            return
    
    create_default_calibration(output_file)

if __name__ == "__main__":
    main()
