#!/usr/bin/env python3
"""
双目相机标定 - 棋盘格采集工具

使用方法:
  1. 打印棋盘格标定板 (推荐 9x6, 方格 30mm)
  2. 运行此脚本
  3. 按空格键采集图像
  4. 采集 20-30 对图像后按 'q' 退出
  5. 运行 stereo_calibrate.py 进行标定
"""

import sys
import os
import time
import threading
import cv2
import numpy as np
import Jetson.GPIO as GPIO

# 导入海康相机类
sys.path.append(os.path.dirname(__file__))
sys.path.append('../scripts')
from hik_camera import HikCamera

# ==================== 配置 ====================
# PWM 配置
PWM_PIN = 32
PWM_FREQ = 10  # 降低到 10 Hz，方便手动采集
PWM_DUTY = 50

# 相机配置
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1
EXPOSURE_TIME = 2000  # 增加曝光时间，确保棋盘格清晰

# 棋盘格配置
CHESSBOARD_SIZE = (9, 6)  # 内角点数量 (列, 行)
SQUARE_SIZE = 30  # 方格尺寸 (mm)

# 输出配置
OUTPUT_DIR = "./data"
MIN_IMAGES = 20  # 最少采集数量
MAX_IMAGES = 50  # 最多采集数量

# ==================== 全局变量 ====================
pwm = None
cam_left = None
cam_right = None
image_count = 0
last_capture_time = 0

# ==================== PWM 控制 ====================
def start_pwm():
    """启动 PWM"""
    global pwm
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PWM_PIN, GPIO.OUT, initial=GPIO.LOW)
    
    pwm = GPIO.PWM(PWM_PIN, PWM_FREQ)
    pwm.start(PWM_DUTY)
    
    print(f"✅ PWM 已启动: {PWM_FREQ} Hz")

def stop_pwm():
    """停止 PWM"""
    global pwm
    
    if pwm:
        pwm.stop()
        GPIO.cleanup()

# ==================== 相机初始化 ====================
def init_cameras():
    """初始化双目相机"""
    global cam_left, cam_right
    
    # 左相机
    cam_left = HikCamera(CAMERA_LEFT_INDEX)
    cam_left.open()
    cam_left.set_trigger_mode('On')
    cam_left.set_trigger_source('Line0')
    cam_left.set_trigger_activation('RisingEdge')
    cam_left.set_exposure_time(EXPOSURE_TIME)
    cam_left.start_grabbing()
    
    # 右相机
    cam_right = HikCamera(CAMERA_RIGHT_INDEX)
    cam_right.open()
    cam_right.set_trigger_mode('On')
    cam_right.set_trigger_source('Line0')
    cam_right.set_trigger_activation('RisingEdge')
    cam_right.set_exposure_time(EXPOSURE_TIME)
    cam_right.start_grabbing()
    
    print("✅ 双目相机已初始化")

def close_cameras():
    """关闭相机"""
    if cam_left:
        cam_left.close()
    if cam_right:
        cam_right.close()

# ==================== 棋盘格检测 ====================
def detect_chessboard(image, draw=True):
    """
    检测棋盘格角点
    
    Returns:
        corners: 角点坐标 (N, 1, 2)
        success: 是否检测成功
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        # 亚像素精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        if draw:
            cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, corners, ret)
    
    return corners, ret

# ==================== 主程序 ====================
def main():
    global image_count, last_capture_time
    
    print("="*60)
    print("双目相机标定 - 棋盘格采集")
    print("="*60)
    print(f"棋盘格尺寸: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} (内角点)")
    print(f"方格尺寸: {SQUARE_SIZE} mm")
    print(f"目标采集: {MIN_IMAGES}-{MAX_IMAGES} 对图像")
    print("="*60)
    print()
    
    # 创建输出目录
    os.makedirs(f"{OUTPUT_DIR}/left", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/right", exist_ok=True)
    
    # 启动 PWM
    print("📡 启动 PWM 触发...")
    start_pwm()
    time.sleep(0.5)
    
    # 初始化相机
    print("📷 初始化相机...")
    init_cameras()
    time.sleep(1)
    
    # 采集循环
    print("\n🎯 开始采集:")
    print("  - 按 [空格] 采集当前图像")
    print("  - 按 [q] 退出")
    print("  - 按 [c] 清除所有已采集图像")
    print()
    
    try:
        while True:
            # 采集图像
            img_left = cam_left.grab_image(timeout_ms=1000)
            img_right = cam_right.grab_image(timeout_ms=1000)
            
            if img_left is None or img_right is None:
                print("\r⚠️  等待触发...", end='', flush=True)
                continue
            
            # 检测棋盘格
            img_left_draw = img_left.copy()
            img_right_draw = img_right.copy()
            
            corners_left, found_left = detect_chessboard(img_left_draw)
            corners_right, found_right = detect_chessboard(img_right_draw)
            
            # 显示状态
            status_left = "✅" if found_left else "❌"
            status_right = "✅" if found_right else "❌"
            
            # 添加文本信息
            cv2.putText(
                img_left_draw,
                f"Left {status_left} | Count: {image_count}/{MIN_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if found_left else (0, 0, 255), 2
            )
            cv2.putText(
                img_right_draw,
                f"Right {status_right} | Count: {image_count}/{MIN_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if found_right else (0, 0, 255), 2
            )
            
            # 拼接显示
            display = np.hstack([img_left_draw, img_right_draw])
            display = cv2.resize(display, (1920, 540))  # 缩小显示
            
            cv2.imshow('Stereo Calibration', display)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # 空格键 - 采集
                # 检查是否检测到棋盘格
                if not (found_left and found_right):
                    print("\n❌ 未检测到棋盘格，请调整位置")
                    continue
                
                # 检查采集间隔 (避免重复采集)
                current_time = time.time()
                if current_time - last_capture_time < 1.0:
                    print("\n⚠️  采集太快，请等待 1 秒")
                    continue
                
                # 保存图像
                filename = f"{image_count:04d}.jpg"
                cv2.imwrite(f"{OUTPUT_DIR}/left/{filename}", img_left)
                cv2.imwrite(f"{OUTPUT_DIR}/right/{filename}", img_right)
                
                image_count += 1
                last_capture_time = current_time
                
                print(f"\n✅ 已采集 {image_count} 对图像")
                
                # 检查是否达到目标
                if image_count >= MAX_IMAGES:
                    print(f"\n✅ 已达到最大采集数量 ({MAX_IMAGES})")
                    break
            
            elif key == ord('q'):  # q - 退出
                if image_count < MIN_IMAGES:
                    print(f"\n⚠️  至少需要 {MIN_IMAGES} 对图像，当前只有 {image_count}")
                    response = input("确定退出? (y/n): ")
                    if response.lower() != 'y':
                        continue
                break
            
            elif key == ord('c'):  # c - 清除
                response = input(f"\n⚠️  确定清除所有 {image_count} 对图像? (y/n): ")
                if response.lower() == 'y':
                    # 删除所有图像
                    for i in range(image_count):
                        filename = f"{i:04d}.jpg"
                        try:
                            os.remove(f"{OUTPUT_DIR}/left/{filename}")
                            os.remove(f"{OUTPUT_DIR}/right/{filename}")
                        except:
                            pass
                    
                    image_count = 0
                    print("✅ 已清除所有图像")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    
    finally:
        # 清理
        cv2.destroyAllWindows()
        close_cameras()
        stop_pwm()
    
    # 总结
    print("\n" + "="*60)
    print("采集完成")
    print("="*60)
    print(f"总计: {image_count} 对图像")
    print(f"保存位置: {OUTPUT_DIR}")
    
    if image_count >= MIN_IMAGES:
        print("\n✅ 可以开始标定:")
        print("   python3 stereo_calibrate.py")
    else:
        print(f"\n⚠️  图像数量不足 (需要至少 {MIN_IMAGES} 对)")
    
    print()

if __name__ == "__main__":
    main()
