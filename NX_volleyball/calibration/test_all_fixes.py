#!/usr/bin/env python3
"""全面测试脚本 - 验证所有 review 修复项"""
import os, sys, subprocess, importlib

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

print("=" * 60)
print("全面验证测试")
print("=" * 60)

# ============================================================
# 1. hik_camera.py - SDK 导入 (即使环境变量错误)
# ============================================================
print("\n--- 1. HikCamera SDK 导入 ---")

# 先故意设置错误环境变量
os.environ["MVCAM_COMMON_RUNENV"] = "/home/nvidia/NX_volleyball/lib"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    # 强制重新导入
    if 'hik_camera' in sys.modules:
        del sys.modules['hik_camera']
    if 'MvCameraControl_class' in sys.modules:
        del sys.modules['MvCameraControl_class']
    from hik_camera import HikCamera
    check("HikCamera 导入 (错误环境变量)", True)
except Exception as e:
    check("HikCamera 导入 (错误环境变量)", False, str(e))

# 检查环境变量被修正
env_val = os.getenv("MVCAM_COMMON_RUNENV", "")
so_path = os.path.join(env_val, "aarch64", "libMvCameraControl.so")
check("MVCAM 环境变量自动修正", os.path.isfile(so_path), f"当前值: {env_val}")

# ============================================================
# 2. gpiod (libgpiod) 可用性
# ============================================================
print("\n--- 2. libgpiod ---")
try:
    import gpiod
    chip = gpiod.Chip("gpiochip2")
    n_lines = chip.num_lines()
    check("gpiod gpiochip2 可用", n_lines > 0, f"{n_lines} lines")
    chip.close()
except Exception as e:
    check("gpiod gpiochip2", False, str(e))

# ============================================================
# 3. capture_chessboard - PWMController (libgpiod)
# ============================================================
print("\n--- 3. capture_chessboard PWMController ---")
sys.path.insert(0, os.path.dirname(__file__))
try:
    if 'capture_chessboard' in sys.modules:
        del sys.modules['capture_chessboard']
    import capture_chessboard as cb
    
    # 检查不再使用 Jetson.GPIO
    src = open(cb.__file__).read()
    check("无 Jetson.GPIO 依赖", "Jetson.GPIO" not in src)
    check("使用 libgpiod", "import gpiod" in src)
    
    # 测试 PWM 控制器启停
    ctrl = cb.PWMController()
    ctrl.start()
    import time
    time.sleep(0.3)
    check("PWM 启动", ctrl.running)
    ctrl.stop()
    check("PWM 停止", not ctrl.running)
except Exception as e:
    check("capture_chessboard", False, str(e))

# ============================================================
# 4. test_camera.py - 无 Jetson.GPIO
# ============================================================
print("\n--- 4. test_camera.py ---")
tc_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'test_camera.py')
try:
    src = open(tc_path).read()
    check("test_camera 无 Jetson.GPIO", "Jetson.GPIO" not in src)
    check("test_camera 使用 gpiod", "import gpiod" in src)
    check("test_camera 使用 deque", "collections.deque" in src or "collections" in src)
except Exception as e:
    check("test_camera.py", False, str(e))

# ============================================================
# 5. start_system.sh - 节点名
# ============================================================
print("\n--- 5. start_system.sh ---")
ss_path = os.path.join(os.path.dirname(__file__), '..', 
    'ros2_ws/src/volleyball_stereo_driver/scripts/start_system.sh')
try:
    src = open(ss_path).read()
    check("节点名 volleyball_tracker_node", "volleyball_tracker_node" in src)
    check("无错误节点名 stereo_system_node", "stereo_system_node" not in src)
except Exception as e:
    check("start_system.sh", False, str(e))

# ============================================================
# 6. stereo_matcher.cpp - 键名匹配
# ============================================================
print("\n--- 6. stereo_matcher.cpp ---")
sm_path = os.path.join(os.path.dirname(__file__), '..', 
    'ros2_ws/src/volleyball_stereo_driver/src/stereo_matcher.cpp')
try:
    src = open(sm_path).read()
    check("使用 camera_matrix_left 键名", 'fs["camera_matrix_left"]' in src)
    check("使用 projection_left 键名", 'fs["projection_left"]' in src)
    check("无旧键名 K1", 'fs["K1"]' not in src)
    check("baseline mm→m 转换", "baseline_ /= 1000" in src)
except Exception as e:
    check("stereo_matcher.cpp", False, str(e))

# ============================================================
# 7. hik_camera_wrapper.cpp - BayerRG8 通道
# ============================================================
print("\n--- 7. hik_camera_wrapper.cpp ---")
hw_path = os.path.join(os.path.dirname(__file__), '..', 
    'ros2_ws/src/volleyball_stereo_driver/src/hik_camera_wrapper.cpp')
try:
    src = open(hw_path).read()
    check("BayerRG8 单通道判断", "is_bayer" in src and "CV_8UC1" in src)
    check("mat_type 条件分配", "mat_type" in src)
except Exception as e:
    check("hik_camera_wrapper.cpp", False, str(e))

# ============================================================
# 8. optimize_system.sh - 幂等性
# ============================================================
print("\n--- 8. optimize_system.sh ---")
os_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'optimize_system.sh')
try:
    src = open(os_path).read()
    check("grep -q 幂等检查", "grep -q" in src)
except Exception as e:
    check("optimize_system.sh", False, str(e))

# ============================================================
# 9. stereo_calibration.py - 导入测试
# ============================================================
print("\n--- 9. stereo_calibration.py ---")
try:
    import stereo_calibration as sc
    check("stereo_calibration 导入", True)
    check("有 calibrate_stereo 函数", hasattr(sc, 'calibrate_stereo'))
    check("有 save_calibration 函数", hasattr(sc, 'save_calibration'))
    check("有 reject_outliers 函数", hasattr(sc, 'reject_outliers'))
except Exception as e:
    check("stereo_calibration.py", False, str(e))

# ============================================================
# 10. stereo_depth_test.py - 导入测试
# ============================================================
print("\n--- 10. stereo_depth_test.py ---")
try:
    import stereo_depth_test as sd
    check("stereo_depth_test 导入", True)
except Exception as e:
    check("stereo_depth_test.py", False, str(e))

# ============================================================
# 11. 标定参数 YAML 键名一致性验证
# ============================================================
print("\n--- 11. 标定 YAML 键名一致性 ---")
import cv2
calib_yaml = os.path.join(os.path.dirname(__file__), '..',
    'ros2_ws/src/volleyball_stereo_driver/calibration/stereo_calib.yaml')
if os.path.isfile(calib_yaml):
    fs = cv2.FileStorage(calib_yaml, cv2.FILE_STORAGE_READ)
    # 检查 stereo_matcher.cpp 需要的键是否存在
    keys_needed = ["camera_matrix_left", "distortion_coefficients_left",
                   "camera_matrix_right", "distortion_coefficients_right",
                   "projection_left", "projection_right", "baseline"]
    for key in keys_needed:
        node = fs.getNode(key)
        check(f"YAML 键 '{key}' 存在", not node.empty(), 
              "" if not node.empty() else "缺失!")
    
    # 验证基线值
    bl = fs.getNode("baseline").real()
    check(f"baseline={bl:.1f}mm (合理范围)", 50 < bl < 1000)
    fs.release()
else:
    print(f"  [SKIP] 标定文件不存在: {calib_yaml}")

# ============================================================
# 12. create_default_calibration.py - 统一键名
# ============================================================
print("\n--- 12. create_default_calibration.py ---")
cd_path = os.path.join(os.path.dirname(__file__), '..', 
    'ros2_ws/src/volleyball_stereo_driver/scripts/create_default_calibration.py')
try:
    src = open(cd_path).read()
    check("使用 camera_matrix_left", "camera_matrix_left" in src)
    check("无旧键名 K1", '"K1"' not in src)
except Exception as e:
    check("create_default_calibration.py", False, str(e))

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"测试结果: {PASS}/{total} 通过, {FAIL} 失败")
if FAIL == 0:
    print("✅ 全部通过!")
else:
    print("❌ 有失败项，请检查上方输出")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
