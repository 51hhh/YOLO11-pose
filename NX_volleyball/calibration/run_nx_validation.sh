#!/bin/bash
set -e

echo "=========================================="
echo "NX 回归验证开始"
echo "=========================================="

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CALIB_DIR="$ROOT_DIR/calibration"
WS_DIR="$ROOT_DIR/ros2_ws"

echo "ROOT_DIR: $ROOT_DIR"

echo "[1/6] Python 语法检查"
python3 -m py_compile "$CALIB_DIR/capture_chessboard.py"
python3 -m py_compile "$CALIB_DIR/stereo_calibration.py"
python3 -m py_compile "$CALIB_DIR/stereo_depth_test.py"
python3 -m py_compile "$ROOT_DIR/scripts/hik_camera.py"
python3 -m py_compile "$ROOT_DIR/scripts/test_camera.py"

echo "[2/6] 标定工具帮助命令检查"
python3 "$CALIB_DIR/capture_chessboard.py" --help >/dev/null
python3 "$CALIB_DIR/stereo_calibration.py" --help >/dev/null
python3 "$CALIB_DIR/stereo_depth_test.py" --help >/dev/null

echo "[3/6] 核心修复项静态验证"
python3 "$CALIB_DIR/test_all_fixes.py"

echo "[4/6] ROS2 环境检查"
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
else
  echo "[FAIL] /opt/ros/humble/setup.bash 不存在"
  exit 1
fi

echo "[5/6] 编译 volleyball_stereo_driver"
if [ -d "$WS_DIR" ]; then
  cd "$WS_DIR"
  colcon build --packages-select volleyball_stereo_driver
  source "$WS_DIR/install/setup.bash"
else
  echo "[FAIL] 未找到 ros2_ws: $WS_DIR"
  exit 1
fi

echo "[6/6] 可执行文件检查"
ros2 pkg executables volleyball_stereo_driver | grep -E "volleyball_tracker_node|volleyball_visualizer_node" >/dev/null

echo "=========================================="
echo "✅ NX 回归验证通过"
echo "=========================================="
