#!/bin/bash
# 编译并运行海康双目标定采集工具

set -e

echo "🔨 编译 capture_stereo_images..."

cd "$(dirname "$0")"

# 检查海康SDK路径
if [ ! -d "/opt/MVS" ]; then
    echo "❌ 未找到海康SDK，请先安装到 /opt/MVS"
    exit 1
fi

# 编译
g++ -o capture_stereo capture_stereo_images.cpp \
    -I/opt/MVS/include \
    -L/opt/MVS/lib/aarch64 \
    -lMvCameraControl \
    $(pkg-config --cflags --libs opencv4) \
    -std=c++17 \
    -O3

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "运行采集工具:"
    echo "  ./capture_stereo"
    echo ""
    echo "或指定相机索引:"
    echo "  ./capture_stereo 0 1  (左相机索引=0, 右相机索引=1)"
else
    echo "❌ 编译失败"
    exit 1
fi
