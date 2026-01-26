#!/bin/bash
# PWM 触发节点启动脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PWM 触发节点启动"
echo "=========================================="

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️  建议使用 sudo 运行以提升线程优先级${NC}"
    echo "   sudo ./start_pwm_node.sh"
    echo ""
fi

# Source ROS2 环境
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}✅ ROS2 环境已加载${NC}"
else
    echo "❌ 未找到 ROS2 环境"
    exit 1
fi

# Source 工作空间
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
    echo -e "${GREEN}✅ 工作空间已加载${NC}"
else
    echo "❌ 工作空间未编译，请先运行:"
    echo "   cd $WS_DIR"
    echo "   colcon build --packages-select volleyball_stereo_driver"
    exit 1
fi

# 解析参数
FREQUENCY=100.0
DUTY_CYCLE=50.0
GPIO_CHIP="gpiochip2"
GPIO_LINE=7

while [[ $# -gt 0 ]]; do
    case $1 in
        --frequency)
            FREQUENCY="$2"
            shift 2
            ;;
        --duty-cycle)
            DUTY_CYCLE="$2"
            shift 2
            ;;
        --gpio-chip)
            GPIO_CHIP="$2"
            shift 2
            ;;
        --gpio-line)
            GPIO_LINE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--frequency 100.0] [--duty-cycle 50.0] [--gpio-chip gpiochip2] [--gpio-line 7]"
            exit 1
            ;;
    esac
done

echo ""
echo "配置:"
echo "  GPIO: $GPIO_CHIP line $GPIO_LINE"
echo "  频率: $FREQUENCY Hz"
echo "  占空比: $DUTY_CYCLE %"
echo ""

# 运行节点
echo -e "${GREEN}🚀 启动 PWM 触发节点...${NC}"
echo ""

ros2 run volleyball_stereo_driver pwm_trigger_node \
    --ros-args \
    -p gpio_chip:=$GPIO_CHIP \
    -p gpio_line:=$GPIO_LINE \
    -p frequency:=$FREQUENCY \
    -p duty_cycle:=$DUTY_CYCLE
