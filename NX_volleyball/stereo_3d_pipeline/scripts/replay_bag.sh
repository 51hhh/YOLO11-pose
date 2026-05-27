#!/bin/bash
# replay_bag.sh - 重放 rosbag 并启动 rviz 可视化
# 用法: ./scripts/replay_bag.sh <bag_path> [--rate <speed>] [--loop]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BAG_PATH=""
RATE="1.0"
LOOP=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --rate|-r) RATE="$2"; shift 2;;
        --loop|-l) LOOP="--loop"; shift;;
        -*) echo "Unknown option: $1"; exit 1;;
        *) BAG_PATH="$1"; shift;;
    esac
done

if [ -z "$BAG_PATH" ]; then
    echo "Usage: $0 <bag_path> [--rate <speed>] [--loop]"
    echo ""
    echo "Available bags:"
    ls -1d "$PROJECT_DIR"/bags/diag_* 2>/dev/null || echo "  (none found in $PROJECT_DIR/bags/)"
    exit 1
fi

# 环境
source /opt/ros/humble/setup.bash

echo "=== Bag Replay ==="
echo "  Bag:  $BAG_PATH"
echo "  Rate: ${RATE}x"
echo "  Loop: ${LOOP:-no}"
echo ""
echo "Tip: 在另一终端运行 start_rviz.sh 查看可视化"
echo ""

# 显示 bag 信息
ros2 bag info "$BAG_PATH"
echo ""

# 播放
ros2 bag play "$BAG_PATH" --rate "$RATE" $LOOP
