#!/bin/bash
# 排球追踪系统 - 性能优化版本部署脚本

set -e

echo "🚀 开始部署性能优化版本..."

# 1. 同步代码到NX
echo ""
echo "📦 Step 1: 同步代码到Jetson NX..."
scp -r ./NX_volleyball/ nvidia@10.42.0.148:~
echo "✅ 代码同步完成"

# 2. SSH到NX进行编译
echo ""
echo "🔨 Step 2: 在Jetson NX上重新编译..."
ssh nvidia@10.42.0.148 << 'EOF'
cd ~/NX_volleyball/ros2_ws

# 清理旧构建
echo "🧹 清理旧构建..."
rm -rf build/ install/ log/

# 加载ROS2环境
echo "🔧 加载ROS2环境..."
source /opt/ros/humble/setup.bash

# 编译 (Release模式 + CUDA优化)
echo "⚙️ 编译中 (启用CUDA GPU加速)..."
colcon build --packages-select volleyball_stereo_driver --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
else
    echo "❌ 编译失败，请检查错误信息"
    exit 1
fi
EOF

echo ""
echo "✨ 部署完成!"
echo ""
echo "📋 下一步操作:"
echo "   ssh nvidia@10.42.0.148"
echo "   cd ~/NX_volleyball/ros2_ws"
echo "   source install/setup.bash"
echo "   sudo -E ros2 run volleyball_stereo_driver volleyball_tracker_node"
echo ""
echo "📊 预期性能提升:"
echo "   FPS:    20.4 → 50+ (提升2.5倍)"
echo "   检测:   35ms → 12ms (减少66%)"
echo "   采集:   14ms →  9ms (减少36%)"
