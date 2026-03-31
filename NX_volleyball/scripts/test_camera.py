#!/usr/bin/env python3
"""
双目相机同步测试
测试 PWM 触发 + 双目相机采集 + 时间戳同步

测试流程:
  1. 启动 PWM 触发 (100 Hz)
  2. 同时采集左右相机
  3. 验证时间戳同步 (<1ms)
  4. 保存测试图像
"""

import sys
import os
import time
import threading
import collections
import numpy as np
import cv2

try:
    import gpiod
    HAS_GPIOD = True
except ImportError:
    HAS_GPIOD = False

# 导入海康相机类
sys.path.append(os.path.dirname(__file__))
from hik_camera import HikCamera

# ==================== 配置 ====================
GPIOCHIP = "gpiochip2"  # GPIO芯片(与主进程一致)
LINE_OFFSET = 7         # GPIO line偏移(gpiochip2 line 7)
PWM_FREQ = 100  # 100 Hz
PWM_DUTY = 50

# 相机配置
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1
EXPOSURE_TIME = 800  # 800us

# 测试配置
TEST_DURATION = 10  # 测试 10 秒
SAVE_IMAGES = True
OUTPUT_DIR = "../calibration/data/test"

# ==================== 全局变量 ====================
gpio_chip = None
gpio_line = None
pwm_running = False
pwm_thread = None
running = False
frame_count_left = 0
frame_count_right = 0
sync_errors = []

# ==================== 软件 PWM 控制 (libgpiod) ====================
def _pwm_loop():
    """软件PWM循环"""
    global pwm_running
    period = 1.0 / PWM_FREQ
    high_t = period * (PWM_DUTY / 100.0)
    low_t = period - high_t
    BUSY_THRESHOLD = 0.0005
    next_edge = time.perf_counter()
    while pwm_running:
        gpio_line.set_value(1)
        next_edge += high_t
        _accurate_sleep(next_edge - time.perf_counter(), BUSY_THRESHOLD)
        gpio_line.set_value(0)
        next_edge += low_t
        _accurate_sleep(next_edge - time.perf_counter(), BUSY_THRESHOLD)

def _accurate_sleep(duration, busy_threshold):
    if duration <= 0:
        return
    if duration > busy_threshold:
        time.sleep(duration - busy_threshold)
    target = time.perf_counter() + min(duration, busy_threshold)
    while time.perf_counter() < target:
        pass

def start_pwm():
    """启动 PWM 触发 (libgpiod)"""
    global gpio_chip, gpio_line, pwm_running, pwm_thread
    if not HAS_GPIOD:
        print("❌ libgpiod 不可用, 请安装: sudo apt install python3-libgpiod")
        return
    gpio_chip = gpiod.Chip(GPIOCHIP)
    gpio_line = gpio_chip.get_line(LINE_OFFSET)
    gpio_line.request(consumer="test_pwm", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    pwm_running = True
    pwm_thread = threading.Thread(target=_pwm_loop, daemon=True)
    pwm_thread.start()
    print(f"✅ PWM 已启动: {PWM_FREQ} Hz, {PWM_DUTY}% (gpiod {GPIOCHIP} line {LINE_OFFSET})")

def stop_pwm():
    """停止 PWM"""
    global pwm_running, pwm_thread, gpio_line, gpio_chip
    pwm_running = False
    if pwm_thread:
        pwm_thread.join(timeout=1.0)
        pwm_thread = None
    if gpio_line:
        try:
            gpio_line.set_value(0)
            gpio_line.release()
        except Exception:
            pass
        gpio_line = None
    if gpio_chip:
        try:
            gpio_chip.close()
        except Exception:
            pass
        gpio_chip = None
    print("✅ PWM 已停止")

# ==================== 相机采集线程 ====================
def camera_thread(camera_index, camera_name, images_queue):
    """
    相机采集线程
    
    Args:
        camera_index: 相机索引
        camera_name: 相机名称 ('left' or 'right')
        images_queue: 图像队列 (用于同步)
    """
    global running, frame_count_left, frame_count_right
    
    try:
        # 打开相机
        cam = HikCamera(camera_index)
        cam.open()
        
        # 配置触发
        cam.set_trigger_mode('On')
        cam.set_trigger_source('Line0')
        cam.set_trigger_activation('RisingEdge')
        cam.set_exposure_time(EXPOSURE_TIME)
        
        # 开始采集
        cam.start_grabbing()
        
        print(f"✅ {camera_name} 相机已启动")
        
        # 采集循环
        while running:
            image = cam.grab_image(timeout_ms=1000)
            
            if image is not None:
                timestamp = time.time()
                
                # 添加到队列
                images_queue.append({
                    'camera': camera_name,
                    'image': image,
                    'timestamp': timestamp,
                    'frame_id': frame_count_left if camera_name == 'left' else frame_count_right
                })
                
                # 更新计数
                if camera_name == 'left':
                    frame_count_left += 1
                else:
                    frame_count_right += 1
        
        # 清理
        cam.close()
        print(f"✅ {camera_name} 相机已关闭")
    
    except Exception as e:
        print(f"❌ {camera_name} 相机错误: {e}")
        import traceback
        traceback.print_exc()

# ==================== 主程序 ====================
def main():
    global running
    
    print("="*60)
    print("双目相机同步测试")
    print("="*60)
    print(f"PWM 频率: {PWM_FREQ} Hz")
    print(f"曝光时间: {EXPOSURE_TIME} us")
    print(f"测试时长: {TEST_DURATION} 秒")
    print("="*60)
    print()
    
    # 创建输出目录
    if SAVE_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/left", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/right", exist_ok=True)
    
    # 启动 PWM
    print("📡 启动 PWM 触发...")
    start_pwm()
    time.sleep(0.5)
    
    # 创建图像队列 (线程安全)
    images_queue = collections.deque(maxlen=200)
    
    # 启动相机线程
    print("📷 启动相机...")
    running = True
    
    thread_left = threading.Thread(
        target=camera_thread,
        args=(CAMERA_LEFT_INDEX, 'left', images_queue)
    )
    thread_right = threading.Thread(
        target=camera_thread,
        args=(CAMERA_RIGHT_INDEX, 'right', images_queue)
    )
    
    thread_left.start()
    thread_right.start()
    
    time.sleep(1)
    
    # 测试循环
    print(f"\n🔄 开始测试 ({TEST_DURATION} 秒)...")
    print("="*60)
    
    start_time = time.time()
    last_print_time = start_time
    
    try:
        while time.time() - start_time < TEST_DURATION:
            # 每秒打印一次状态
            if time.time() - last_print_time >= 1.0:
                elapsed = time.time() - start_time
                fps_left = frame_count_left / elapsed if elapsed > 0 else 0
                fps_right = frame_count_right / elapsed if elapsed > 0 else 0
                
                print(f"[{int(elapsed)}s] "
                      f"左: {frame_count_left} 帧 ({fps_left:.1f} FPS) | "
                      f"右: {frame_count_right} 帧 ({fps_right:.1f} FPS) | "
                      f"队列: {len(images_queue)}")
                
                last_print_time = time.time()
            
            # 处理图像队列 (检查同步)
            if len(images_queue) >= 2:
                # 查找配对的图像
                left_images = [img for img in images_queue if img['camera'] == 'left']
                right_images = [img for img in images_queue if img['camera'] == 'right']
                
                if left_images and right_images:
                    # 取最早的一对
                    img_left = left_images[0]
                    img_right = right_images[0]
                    
                    # 计算时间戳差异
                    ts_diff = abs(img_left['timestamp'] - img_right['timestamp'])
                    sync_errors.append(ts_diff * 1000)  # 转换为 ms
                    
                    # 保存图像
                    if SAVE_IMAGES and img_left['frame_id'] % 10 == 0:  # 每 10 帧保存一次
                        cv2.imwrite(
                            f"{OUTPUT_DIR}/left/{img_left['frame_id']:06d}.jpg",
                            img_left['image']
                        )
                        cv2.imwrite(
                            f"{OUTPUT_DIR}/right/{img_right['frame_id']:06d}.jpg",
                            img_right['image']
                        )
                    
                    # 从队列移除
                    images_queue.remove(img_left)
                    images_queue.remove(img_right)
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    
    # 停止采集
    print("\n🛑 停止采集...")
    running = False
    
    thread_left.join(timeout=2)
    thread_right.join(timeout=2)
    
    # 停止 PWM
    stop_pwm()
    
    # 统计结果
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    
    elapsed = time.time() - start_time
    print(f"测试时长: {elapsed:.2f} 秒")
    print(f"左相机: {frame_count_left} 帧 ({frame_count_left/elapsed:.1f} FPS)")
    print(f"右相机: {frame_count_right} 帧 ({frame_count_right/elapsed:.1f} FPS)")
    
    if sync_errors:
        sync_errors_np = np.array(sync_errors)
        print(f"\n同步误差统计:")
        print(f"  平均: {sync_errors_np.mean():.3f} ms")
        print(f"  中位数: {np.median(sync_errors_np):.3f} ms")
        print(f"  最大: {sync_errors_np.max():.3f} ms")
        print(f"  最小: {sync_errors_np.min():.3f} ms")
        print(f"  <1ms: {np.sum(sync_errors_np < 1.0)} / {len(sync_errors_np)} "
              f"({np.sum(sync_errors_np < 1.0)/len(sync_errors_np)*100:.1f}%)")
    
    if SAVE_IMAGES:
        print(f"\n✅ 图像已保存到: {OUTPUT_DIR}")
    
    print("\n下一步:")
    print("  1. 检查同步误差是否 <1ms")
    print("  2. 查看保存的图像")
    print("  3. 开始相机标定:")
    print("     cd ../calibration")
    print("     python3 capture_chessboard.py")
    print()

if __name__ == "__main__":
    main()
