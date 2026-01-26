#!/bin/bash
# 修复编译错误并重新部署

echo "🔧 修复CUDA编译问题并重新部署..."

# 1. 同步修复后的代码
echo ""
echo "📦 同步代码..."
cd /home/rick/desktop/yolo/yoloProject
scp -r ./NX_volleyball/ros2_ws/src/volleyball_stereo_driver/ nvidia@10.42.0.148:~/NX_volleyball/ros2_ws/src/
echo "✅ 代码同步完成"

# 2. SSH重新编译
echo ""
echo "🔨 重新编译..."
ssh nvidia@10.42.0.148 << 'EOF'
cd ~/NX_volleyball/ros2_ws

echo "🧹 清理旧构建..."
rm -rf build/ install/ log/

echo "🔧 加载ROS2环境..."
source /opt/ros/humble/setup.bash

echo "⚙️ 编译中..."
colcon build --packages-select volleyball_stereo_driver --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 编译成功!"
    echo ""
    echo "📋 运行命令:"
    echo "   source install/setup.bash"
    echo "   sudo -E ros2 run volleyball_stereo_driver volleyball_tracker_node"
else
    echo ""
    echo "❌ 编译失败"
    exit 1
fi
EOF

echo ""
echo "✨ 完成!"
