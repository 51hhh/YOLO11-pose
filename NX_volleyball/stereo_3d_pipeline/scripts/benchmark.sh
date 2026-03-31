#!/bin/bash
# ============================================================
# benchmark.sh - 性能基准测试 (在 NX 上运行)
#
# 用法:
#   ./benchmark.sh [duration_seconds]
# ============================================================

set -e

DURATION="${1:-10}"
PIPELINE_BIN="./build/stereo_pipeline"
CONFIG="config/pipeline.yaml"
NSYS_OUTPUT="benchmark_report"

echo "============================================"
echo "  Pipeline Benchmark"
echo "  Duration: ${DURATION}s"
echo "============================================"

# 检查可执行文件
if [ ! -f "${PIPELINE_BIN}" ]; then
    echo "ERROR: ${PIPELINE_BIN} not found. Build first:"
    echo "  mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 1. 确保最高性能模式
echo "[1/3] Setting performance mode..."
sudo nvpmodel -m 2 2>/dev/null || echo "  (nvpmodel skip)"
sudo jetson_clocks 2>/dev/null || echo "  (jetson_clocks skip)"

# 2. 普通运行 (打印 FPS 统计)
echo "[2/3] Running pipeline for ${DURATION}s..."
timeout "${DURATION}" "${PIPELINE_BIN}" --config "${CONFIG}" 2>&1 | tail -20

# 3. nsys 分析 (可选)
if command -v nsys &> /dev/null; then
    echo "[3/3] Running nsys profiling..."
    nsys profile \
        --output="${NSYS_OUTPUT}" \
        --trace=cuda,nvtx,osrt \
        --duration="${DURATION}" \
        --force-overwrite=true \
        "${PIPELINE_BIN}" --config "${CONFIG}" 2>&1 | tail -5

    echo ""
    echo "nsys report: ${NSYS_OUTPUT}.nsys-rep"
    echo "View with: nsys stats ${NSYS_OUTPUT}.nsys-rep"
else
    echo "[3/3] nsys not found, skipping profiling"
fi

echo ""
echo "Benchmark complete!"
