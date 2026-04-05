#!/bin/bash
# ============================================================
# deploy_and_test.sh — 部署 stereo_3d_pipeline 到 NX 并编译测试
#
# 使用方法:
#   bash scripts/deploy_and_test.sh [build|test|all]
#
# 前提: ssh-copy-id nvidia@192.168.31.56 (或使用密码)
# ============================================================

set -e

NX_HOST="nvidia@192.168.31.56"
NX_DIR="/home/nvidia/NX_volleyball/stereo_3d_pipeline"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH_CMD="ssh -o StrictHostKeyChecking=no ${NX_HOST}"
SCP_CMD="scp -o StrictHostKeyChecking=no -r"

# ==================== 同步代码 ====================
sync_code() {
    echo "=== Syncing code to NX ==="
    # 创建远端目录结构
    ${SSH_CMD} "mkdir -p ${NX_DIR}/{src/{pipeline,stereo,detect,fusion,calibration,rectify,capture,utils},config,scripts}"

    # 同步源文件
    ${SCP_CMD} "${LOCAL_DIR}/src/" "${NX_HOST}:${NX_DIR}/src/"
    ${SCP_CMD} "${LOCAL_DIR}/config/" "${NX_HOST}:${NX_DIR}/config/"
    ${SCP_CMD} "${LOCAL_DIR}/CMakeLists.txt" "${NX_HOST}:${NX_DIR}/"

    echo "=== Code synced ==="
}

# ==================== 编译 ====================
build() {
    echo "=== Building on NX ==="
    ${SSH_CMD} << 'ENDSSH'
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline

# 创建 build 目录
mkdir -p build && cd build

# CMake 配置 (Orin NX = sm_87)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH="87" \
    2>&1

# 编译 (NX 6核)
make -j6 2>&1

echo "=== Build complete ==="
ls -la stereo_pipeline 2>/dev/null && echo "Binary OK" || echo "Binary NOT found"
ENDSSH
}

# ==================== 测试 (dry-run, 无相机) ====================
test_dry_run() {
    echo "=== Testing (dry-run, ROI mode) ==="
    ${SSH_CMD} << 'ENDSSH'
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline

# 使用 ROI 配置, dry-run 模式 (无相机时自动使用合成帧)
timeout 10 build/stereo_pipeline --config config/pipeline_roi.yaml 2>&1 || true

echo "=== Dry-run test complete ==="
ENDSSH
}

# ==================== 性能测试 ====================
perf_test() {
    echo "=== Performance test (ROI mode, 5 seconds) ==="
    ${SSH_CMD} << 'ENDSSH'
cd /home/nvidia/NX_volleyball/stereo_3d_pipeline

# 设置最大性能模式
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

# 运行 5 秒性能测试
timeout 5 build/stereo_pipeline --config config/pipeline_roi.yaml 2>&1 || true

echo ""
echo "=== Performance test complete ==="
ENDSSH
}

# ==================== 主流程 ====================
case "${1:-all}" in
    sync)
        sync_code
        ;;
    build)
        sync_code
        build
        ;;
    test)
        test_dry_run
        ;;
    perf)
        perf_test
        ;;
    all)
        sync_code
        build
        test_dry_run
        ;;
    *)
        echo "Usage: $0 [sync|build|test|perf|all]"
        exit 1
        ;;
esac
