#!/bin/bash
# 双目排球追踪系统 - 启动脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "双目排球追踪系统 - 启动"
echo "=========================================="

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ 错误: 需要 sudo 权限${NC}"
    echo "   请使用: sudo ./start_system.sh"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${GREEN}工作空间: $WS_DIR${NC}"

# Source ROS2 环境
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo -e "${RED}❌ 未找到 ROS2 环境${NC}"
    exit 1
fi

# Source 工作空间
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
else
    echo -e "${RED}❌ 工作空间未编译${NC}"
    echo "   请先运行: colcon build --packages-select volleyball_stereo_driver"
    exit 1
fi

echo -e "${GREEN}✅ 环境已加载${NC}"
echo ""

# 运行节点 (节点会自动查找并加载配置文件)
echo -e "${GREEN}🚀 启动双目排球追踪系统...${NC}"
echo -e "${YELLOW}提示: 节点会自动查找配置文件${NC}"
echo ""

ros2 run volleyball_stereo_driver volleyball_tracker_node
