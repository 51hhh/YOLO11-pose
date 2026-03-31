#!/usr/bin/env python3
"""
capture_chessboard.py - 海康双目相机棋盘格图像采集工具

用法:
  python3 capture_chessboard.py                      # 默认参数
  python3 capture_chessboard.py --free-run            # 自由运行模式(无PWM触发)

操作:
  空格  - 采集当前帧(需双目均检测到棋盘格)
  q/ESC - 退出
  c     - 清空已采集图像

依赖: MVS SDK, libgpiod (python3-libgpiod)
"""

import sys
import os
import time
import argparse
import threading

import cv2
import numpy as np

# libgpiod (PWM硬件触发，兼容自定义载板)
try:
    import gpiod
    HAS_GPIOD = True
except ImportError:
    HAS_GPIOD = False

# 海康相机封装
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
try:
    from hik_camera import HikCamera
except ImportError:
    HikCamera = None

# ================== 默认配置 ==================
GPIOCHIP = "gpiochip2"  # GPIO芯片(与主进程一致)
LINE_OFFSET = 7         # GPIO line偏移(gpiochip2 line 7)
PWM_FREQ = 10           # Hz
PWM_DUTY = 50           # 占空比%

BOARD_WIDTH = 9         # 内角点列数
BOARD_HEIGHT = 6        # 内角点行数
SQUARE_SIZE = 30.0      # 方格边长(mm)

EXPOSURE_TIME = 2000    # 曝光时间(us)
OUTPUT_DIR = "./calibration_images"
# ==============================================


def parse_args():
    p = argparse.ArgumentParser(description="双目棋盘格标定图像采集")
    p.add_argument('--free-run', action='store_true',
                   help='自由运行模式(不使用PWM触发)')
    p.add_argument('--no-pwm', action='store_true',
                   help='禁用PWM输出')
    p.add_argument('-o', '--output', default=OUTPUT_DIR,
                   help='输出目录(默认: calibration_images)')
    p.add_argument('-e', '--exposure', type=int, default=EXPOSURE_TIME,
                   help='曝光时间us(默认: 2000)')
    return p.parse_args()


class PWMController:
    """基于libgpiod的软件PWM触发控制器，兼容自定义载板"""

    def __init__(self):
        self.chip = None
        self.line = None
        self.running = False
        self.thread = None

    def start(self):
        if not HAS_GPIOD:
            print("[PWM] libgpiod 不可用，跳过")
            print("  安装: sudo apt install python3-libgpiod")
            return
        try:
            self.chip = gpiod.Chip(GPIOCHIP)
            self.line = self.chip.get_line(LINE_OFFSET)
            self.line.request(
                consumer="calib_pwm",
                type=gpiod.LINE_REQ_DIR_OUT,
                default_vals=[0])
        except Exception as e:
            print(f"[PWM] GPIO初始化失败: {e}")
            self._cleanup_gpio()
            return

        # 启动PWM线程
        period = 1.0 / PWM_FREQ
        high_t = period * (PWM_DUTY / 100.0)
        low_t = period - high_t
        self.running = True
        self.thread = threading.Thread(
            target=self._pwm_loop, args=(high_t, low_t), daemon=True)
        self.thread.start()
        print(f"[PWM] 已启动 {PWM_FREQ}Hz (gpiod {GPIOCHIP} line {LINE_OFFSET})")

    def _pwm_loop(self, high_t, low_t):
        """软件PWM循环(高精度perf_counter + 忙等待补偿)"""
        BUSY_THRESHOLD = 0.0005  # 0.5ms以下用忙等待
        next_edge = time.perf_counter()
        while self.running:
            # 高电平
            self.line.set_value(1)
            next_edge += high_t
            self._accurate_sleep(next_edge - time.perf_counter(), BUSY_THRESHOLD)
            # 低电平
            self.line.set_value(0)
            next_edge += low_t
            self._accurate_sleep(next_edge - time.perf_counter(), BUSY_THRESHOLD)

    @staticmethod
    def _accurate_sleep(duration, busy_threshold):
        """高精度睡眠: sleep + perf_counter忙等待补偿"""
        if duration <= 0:
            return
        if duration > busy_threshold:
            time.sleep(duration - busy_threshold)
        target = time.perf_counter() + min(duration, busy_threshold)
        while time.perf_counter() < target:
            pass

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        self._cleanup_gpio()

    def _cleanup_gpio(self):
        if self.line:
            try:
                self.line.set_value(0)
                self.line.release()
            except Exception:
                pass
            self.line = None
        if self.chip:
            try:
                self.chip.close()
            except Exception:
                pass
            self.chip = None


class StereoCaptureSession:
    """双目标定图像采集会话"""

    def __init__(self, args):
        self.board_size = (BOARD_WIDTH, BOARD_HEIGHT)
        self.output_dir = args.output
        self.exposure = args.exposure
        self.free_run = args.free_run
        self.use_pwm = not args.no_pwm

        self.cam_left = None
        self.cam_right = None
        self.pwm_ctrl = PWMController() if self.use_pwm else None
        self.capture_count = 0
        self.last_capture_time = 0.0

        # 角点检测标志: FILTER_QUADS 过滤假四边形
        self.cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                         | cv2.CALIB_CB_NORMALIZE_IMAGE
                         | cv2.CALIB_CB_FILTER_QUADS)

    def setup(self):
        """初始化相机和输出目录"""
        os.makedirs(os.path.join(self.output_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "right"), exist_ok=True)

        if self.use_pwm:
            self.pwm_ctrl.start()
            time.sleep(0.3)

        self._open_cameras()
        time.sleep(0.5)

    def teardown(self):
        """释放资源"""
        cv2.destroyAllWindows()
        if self.cam_left:
            self.cam_left.close()
        if self.cam_right:
            self.cam_right.close()
        if self.pwm_ctrl:
            self.pwm_ctrl.stop()

    def _open_cameras(self):
        """打开左右海康相机并配置触发模式"""
        if HikCamera is None:
            raise RuntimeError("HikCamera 模块不可用")

        self.cam_left = HikCamera(0)
        self.cam_left.open()
        self.cam_right = HikCamera(1)
        self.cam_right.open()

        for cam, name in [(self.cam_left, "左"), (self.cam_right, "右")]:
            if self.free_run:
                cam.set_trigger_mode('Off')
            else:
                cam.set_trigger_mode('On')
                cam.set_trigger_source('Line0')
                cam.set_trigger_activation('RisingEdge')
            cam.set_exposure_time(self.exposure)
            cam.start_grabbing()
            print(f"[{name}相机] 就绪")

    def _to_gray(self, img):
        """转换为灰度图"""
        if img is None:
            return None
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def detect_corners(self, gray):
        """检测棋盘格角点并做亚像素精化"""
        found, corners = cv2.findChessboardCorners(
            gray, self.board_size, self.cb_flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
        return corners, found

    def _save_pair(self, img_left, img_right):
        """保存一对图像为无损PNG"""
        name = f"{self.capture_count:04d}.png"
        lp = os.path.join(self.output_dir, "left", name)
        rp = os.path.join(self.output_dir, "right", name)
        cv2.imwrite(lp, img_left)
        cv2.imwrite(rp, img_right)
        self.capture_count += 1
        self.last_capture_time = time.time()
        print(f"\n[已采集] 第 {self.capture_count} 对")

    def _clear_all(self):
        """清空已采集的所有图像"""
        for sub in ("left", "right"):
            d = os.path.join(self.output_dir, sub)
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        self.capture_count = 0
        print("[已清空] 所有图像")

    def run(self):
        """主循环"""
        print("=" * 50)
        print("双目棋盘格图像采集")
        print("=" * 50)
        print(f"棋盘格: {self.board_size[0]}x{self.board_size[1]} 内角点")
        print(f"输出:   {self.output_dir}/")
        print(f"模式:   {'自由运行' if self.free_run else 'PWM触发'}")
        print("操作: 空格=采集  q/ESC=退出  c=清空")
        print("=" * 50)

        try:
            self._capture_loop()
        except KeyboardInterrupt:
            print("\n[中断]")
        finally:
            self.teardown()

        print(f"\n共采集 {self.capture_count} 对图像")
        if self.capture_count >= 10:
            print(f"\n可以进行标定:")
            print(f"  python3 stereo_calibration.py -s {SQUARE_SIZE}")

    def _capture_loop(self):
        while True:
            img_left = self.cam_left.grab_image(timeout_ms=1000)
            img_right = self.cam_right.grab_image(timeout_ms=1000)

            if img_left is None or img_right is None:
                print("\r等待触发...", end='', flush=True)
                continue

            gray_l = self._to_gray(img_left)
            gray_r = self._to_gray(img_right)

            # 绘制预览
            draw_l = (img_left.copy() if img_left.ndim == 3
                      else cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR))
            draw_r = (img_right.copy() if img_right.ndim == 3
                      else cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR))

            corners_l, found_l = self.detect_corners(gray_l)
            corners_r, found_r = self.detect_corners(gray_r)

            if found_l:
                cv2.drawChessboardCorners(
                    draw_l, self.board_size, corners_l, True)
            if found_r:
                cv2.drawChessboardCorners(
                    draw_r, self.board_size, corners_r, True)

            # 状态信息
            sl = "OK" if found_l else "--"
            sr = "OK" if found_r else "--"
            label = f"L:{sl} R:{sr} | {self.capture_count}"
            cv2.putText(draw_l, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if (found_l and found_r)
                        else (0, 0, 255), 2)

            # 拼接显示
            display = np.hstack([draw_l, draw_r])
            h, w = display.shape[:2]
            if w > 1920:
                scale = 1920 / w
                display = cv2.resize(display, (1920, int(h * scale)))
            cv2.imshow("Stereo Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not (found_l and found_r):
                    print("\n未在双目中同时检测到棋盘格")
                elif time.time() - self.last_capture_time < 1.0:
                    print("\n间隔太短，请等待1秒")
                else:
                    self._save_pair(img_left, img_right)
            elif key in (ord('q'), 27):
                break
            elif key == ord('c'):
                self._clear_all()


def main():
    args = parse_args()
    session = StereoCaptureSession(args)
    session.setup()
    session.run()


if __name__ == "__main__":
    main()
