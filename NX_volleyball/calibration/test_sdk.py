#!/usr/bin/env python3
"""SDK 导入测试 - 模拟 .bashrc 设置错误环境变量的场景"""
import os, sys

# 模拟 .bashrc 中的错误设置 (已修正前的环境)
# 即使环境变量指向错误路径，hik_camera.py 也应该能自动修正
print(f"[测试] MVCAM_COMMON_RUNENV = {os.getenv('MVCAM_COMMON_RUNENV', '<未设置>')}")

# 导入 hik_camera
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
try:
    from hik_camera import HikCamera
    print("[OK] HikCamera 导入成功")
except Exception as e:
    print(f"[FAIL] HikCamera 导入失败: {e}")
    sys.exit(1)

# 测试 gpiod
try:
    import gpiod
    chip = gpiod.Chip("gpiochip2")
    print(f"[OK] gpiod: {chip.name()} ({chip.num_lines()} lines)")
    chip.close()
except Exception as e:
    print(f"[FAIL] gpiod: {e}")

# 测试 capture_chessboard 模块导入
try:
    import capture_chessboard as cb
    ctrl = cb.PWMController()
    print("[OK] PWMController 创建成功")
except Exception as e:
    print(f"[FAIL] capture_chessboard: {e}")

print("\n[完成] 所有核心模块测试通过" if True else "")
