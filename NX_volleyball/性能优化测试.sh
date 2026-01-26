#!/bin/bash
# 性能优化测试脚本

echo "========================================="
echo "🚀 排球追踪系统性能优化测试"
echo "========================================="
echo ""
echo "📋 本次优化内容:"
echo "  1. 相机像素格式设置为BGR8 (免去Bayer转换)"
echo "  2. 双相机并行采集 (线程并行)"
echo "  3. 海康SDK高性能配置 (网络包/缓冲区)"
echo "  4. GPU内存预分配 (避免每帧malloc/free)"
echo "  5. TensorRT异步推理 (enqueueV2)"
echo ""

# 进入工作区
cd ~/NX_volleyball/ros2_ws

# 编译
echo "🔧 正在编译..."
colcon build --packages-select volleyball_stereo_driver --cmake-args -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi
echo "✅ 编译成功"
echo ""

# 加载环境
source install/setup.bash

echo "========================================="
echo "📊 预期性能改进:"
echo "  采集: 16-19ms → 8-10ms (-50%)"
echo "  检测: 25-30ms → 20-25ms (-20%)"
echo "  总FPS: 21-24 → 35-40 (+60%)"
echo "========================================="
echo ""

echo "🏃 启动追踪节点..."
echo "   按 Ctrl+C 停止测试"
echo ""

# 运行节点 (需要sudo权限访问GPIO)
sudo -E bash -c 'source ~/NX_volleyball/ros2_ws/install/setup.bash && ros2 run volleyball_stereo_driver volleyball_tracker_node'
