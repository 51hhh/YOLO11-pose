#!/bin/bash
# 排球追踪可视化节点快速启动脚本

echo "🎥 启动可视化节点..."
echo "================================"

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash
source ~/NX_volleyball/ros2_ws/install/setup.bash

# 启动可视化节点
ros2 run volleyball_stereo_driver volleyball_visualizer_node

echo ""
echo "可视化窗口快捷键:"
echo "  Q / ESC - 退出"
echo "  S - 截图"
echo "  T - 切换轨迹显示"
echo "  C - 清除轨迹历史"
