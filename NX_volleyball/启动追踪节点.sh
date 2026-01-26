#!/bin/bash
# 排球追踪主节点快速启动脚本

echo "🏐 启动排球追踪节点..."
echo "================================"

# 加载 ROS2 环境
source /opt/ros/humble/setup.bash
source ~/NX_volleyball/ros2_ws/install/setup.bash

# 启动主节点（需要 sudo 以使用 GPIO）
sudo -E bash -c "source /opt/ros/humble/setup.bash && \
source ~/NX_volleyball/ros2_ws/install/setup.bash && \
ros2 run volleyball_stereo_driver volleyball_tracker_node"
