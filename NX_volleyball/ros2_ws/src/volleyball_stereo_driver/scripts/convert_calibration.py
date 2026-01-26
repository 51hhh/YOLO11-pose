#!/usr/bin/env python3
"""
标定文件格式转换工具
将 NumPy .npz 格式转换为 OpenCV YAML 格式
"""

import numpy as np
import cv2
import sys
import os

def convert_npz_to_yaml(npz_file, yaml_file):
    """
    转换 .npz 标定文件为 .yaml 格式
    
    Args:
        npz_file: 输入的 .npz 文件路径
        yaml_file: 输出的 .yaml 文件路径
    """
    if not os.path.exists(npz_file):
        print(f"❌ 错误: 文件不存在: {npz_file}")
        return False
    
    try:
        # 加载 .npz 文件
        data = np.load(npz_file)
        
        print(f"📂 加载标定文件: {npz_file}")
        print(f"   包含的键: {list(data.files)}")
        
        # 创建 YAML 文件
        fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)
        
        # 写入所有参数
        for key in data.files:
            value = data[key]
            print(f"   写入 {key}: shape={value.shape}, dtype={value.dtype}")
            fs.write(key, value)
        
        fs.release()
        
        print(f"✅ 转换成功: {yaml_file}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python3 convert_calibration.py <input.npz> [output.yaml]")
        print("示例: python3 convert_calibration.py stereo_calib.npz stereo_calib.yaml")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        yaml_file = sys.argv[2]
    else:
        # 自动生成输出文件名
        yaml_file = os.path.splitext(npz_file)[0] + ".yaml"
    
    convert_npz_to_yaml(npz_file, yaml_file)

if __name__ == "__main__":
    main()
