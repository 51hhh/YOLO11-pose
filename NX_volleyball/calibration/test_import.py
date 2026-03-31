#!/usr/bin/env python3
"""快速导入测试脚本"""
import sys
import os

print("=== 1. 测试 gpiod ===")
try:
    import gpiod
    chip = gpiod.Chip("gpiochip2")
    print(f"  gpiochip2: {chip.name()}, {chip.num_lines()} lines")
    # 测试请求 line 7
    line = chip.get_line(7)
    if line.is_used():
        print(f"  line 7: BUSY (consumer: {line.consumer()})")
    else:
        line.request(consumer="test", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
        line.set_value(1)
        import time
        time.sleep(0.01)
        line.set_value(0)
        line.release()
        print("  line 7: OK (toggle test passed)")
    chip.close()
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== 2. 测试 HikCamera 导入 ===")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
try:
    from hik_camera import HikCamera
    print("  HikCamera: OK")
    cam = HikCamera(0)
    devs = cam.list_devices()
    print(f"  Devices found: {len(devs)}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== 3. 测试 capture_chessboard 导入 ===")
try:
    import capture_chessboard
    print("  capture_chessboard: OK")
    # 测试 PWMController
    pwm = capture_chessboard.PWMController()
    pwm.start()
    import time
    time.sleep(1)
    pwm.stop()
    print("  PWMController: OK (1s test)")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== ALL TESTS DONE ===")
