#!/bin/bash
# ============================================================
# 部署 Bayer RG8 + CUDA去马赛克模式到 Jetson NX
# 功能: 使用 Bayer RG8 格式 + GPU加速去马赛克实现 100fps @ 9867us 曝光
# ============================================================

set -e  # 遇到错误立即退出

NX_IP="10.42.0.247"
NX_USER="nvidia"
PROJECT_DIR="/home/rick/desktop/yolo/yoloProject/NX_volleyball"

echo "=========================================="
echo "📦 部署 Bayer RG8 + CUDA加速模式到 Jetson NX"
echo "=========================================="

# 1. 上传修改后的文件
echo ""
echo "📤 [1/4] 上传配置文件和源代码..."
scp "${PROJECT_DIR}/ros2_ws/src/volleyball_stereo_driver/config/tracker_params.yaml" \
    "${NX_USER}@${NX_IP}:~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/config/"

scp "${PROJECT_DIR}/ros2_ws/src/volleyball_stereo_driver/src/hik_camera_wrapper.cpp" \
    "${PROJECT_DIR}/ros2_ws/src/volleyball_stereo_driver/src/yolo_preprocessor.cu" \
    "${PROJECT_DIR}/ros2_ws/src/volleyball_stereo_driver/src/yolo_detector.cpp" \
    "${NX_USER}@${NX_IP}:~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/src/"

echo "✅ 文件上传完成 (4个文件)"

# 2. 在 NX 上编译
echo ""
echo "🔨 [2/4] 在 NX 上编译..."
ssh "${NX_USER}@${NX_IP}" << 'EOF'
cd ~/NX_volleyball/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select volleyball_stereo_driver \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
EOF

echo "✅ 编译完成"

# 3. 显示修改内容
echo ""
echo "📝 [3/4] 修改内容总结:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎥 相机参数:"
echo "   曝光时间: 6000.0 → 9867.0 us (+64%)"
echo "   增益:     15.0   → 10.9854 dB (-28%)"
echo ""
echo "🎨 像素格式优先级:"
echo "   1️⃣  BayerRG8   (100fps, 1.56MB/帧) ← 新优先"
echo "   2️⃣  BGR8       (76fps, 4.67MB/帧)"
echo "   3️⃣  RGB8       (76fps, 4.67MB/帧)"
echo ""
echo "⚙️  GPU加速处理:"
echo "   ⚡ CUDA去马赛克kernel (preprocessBayerRGKernel)"
echo "      - Bayer RG8 → RGB 去马赛克"
echo "      - Bilinear Resize (1440x1080 → 320x320)"
echo "      - 归一化 [0,255] → [0,1]"
echo "      - HWC → CHW 格式转换"
echo "      - 融合操作，节省中间缓冲"
echo ""
echo "   ⚡ Y字形流水线 (双CUDA流并行)"
echo "      Stream1: [H2D_L] → [Kernel_L] ┐"
echo "      Stream2: [H2D_R] → [Kernel_R] ├→ [Batch=2推理]"
echo "                                     └"
echo ""
echo "   ⚡ CPU零开销"
echo "      - 相机: 直接memcpy Bayer数据"
echo "      - GPU: 全部格式转换+预处理"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 4. 运行测试
echo ""
echo "🚀 [4/4] 准备启动节点..."
echo ""
echo "运行以下命令启动节点:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ssh ${NX_USER}@${NX_IP}"
echo "cd ~/NX_volleyball/ros2_ws"
echo "source /opt/ros/humble/setup.bash"
echo "source install/setup.bash"
echo "sudo -E ros2 run volleyball_stereo_driver volleyball_tracker_node"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 预期性能:"
echo "   ✅ 帧率: 100 FPS (之前76fps, +32%)"
echo "   ✅ 曝光: 9867us (更好的图像质量, +64%)"
echo "   ✅ 增益: 10.9854dB (更低噪声, -28%)"
echo "   ✅ 带宽: 1.56MB/帧 (降低67%)"
echo "   ✅ CPU: 零开销 (全GPU加速)"
echo ""
echo "🔍 验证指标:"
echo "   看到 '✅ 相机像素格式: BayerRG8 (100fps支持)'"
echo "   推理性能应与BGR模式相近 (~2ms预处理)"
echo "   同步率应保持 100%"
echo ""
echo "✅ 部署完成！"
