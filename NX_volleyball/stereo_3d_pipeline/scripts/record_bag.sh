#!/bin/bash
# record_bag.sh - 启动 pipeline 并录制诊断 rosbag
# 用法: ./scripts/record_bag.sh [--config <yaml>] [--duration <sec>] [--output <dir>]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
CONFIG="${CONFIG:-$PROJECT_DIR/config/pipeline_yolo26_gpu.yaml}"
DURATION=""
OUTPUT_DIR="$PROJECT_DIR/bags"
TOPICS="/debug/depth_full /debug/depth_roi /debug/raw_obs /ball/realtime /ball/landing /ball/predicted_path /ball/actual_path"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c) CONFIG="$2"; shift 2;;
        --duration|-d) DURATION="$2"; shift 2;;
        --output|-o) OUTPUT_DIR="$2"; shift 2;;
        --topics|-t) TOPICS="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# 环境
source /opt/ros/humble/setup.bash

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BAG_PATH="$OUTPUT_DIR/diag_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "=== Diagnostic Recording ==="
echo "  Config:   $CONFIG"
echo "  Output:   $BAG_PATH"
echo "  Topics:   $TOPICS"
echo "  Duration: ${DURATION:-unlimited}"
echo ""

# 启动 pipeline (后台)
"$BUILD_DIR/stereo_pipeline" --config "$CONFIG" &
PIPELINE_PID=$!
echo "Pipeline started (PID: $PIPELINE_PID)"

# 等待 topic 就绪
echo "Waiting for topics..."
sleep 3

# 启动 rosbag 录制
BAG_CMD="ros2 bag record -o $BAG_PATH $TOPICS"
if [ -n "$DURATION" ]; then
    BAG_CMD="timeout ${DURATION}s $BAG_CMD"
fi

echo "Recording: $BAG_CMD"
eval $BAG_CMD &
BAG_PID=$!

# 信号处理: Ctrl+C 优雅退出
cleanup() {
    echo ""
    echo "Stopping recording..."
    kill $BAG_PID 2>/dev/null || true
    wait $BAG_PID 2>/dev/null || true
    echo "Stopping pipeline..."
    kill $PIPELINE_PID 2>/dev/null || true
    wait $PIPELINE_PID 2>/dev/null || true
    echo "Done. Bag saved to: $BAG_PATH"
    ros2 bag info "$BAG_PATH" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM

# 等待录制结束 (duration 模式) 或用户中断
wait $BAG_PID 2>/dev/null || true

# duration 模式自动结束时清理 pipeline
kill $PIPELINE_PID 2>/dev/null || true
wait $PIPELINE_PID 2>/dev/null || true
echo "Recording complete. Bag: $BAG_PATH"
ros2 bag info "$BAG_PATH" 2>/dev/null || true
