#!/bin/bash
# ============================================================
# build_engine.sh - 编译 TensorRT Engine (在 NX 上运行)
#
# 用法:
#   ./build_engine.sh <onnx_path> [output_engine]
#
# 说明:
#   - DLA Core 0, INT8 量化, GPU Fallback
#   - 需要先用 trtexec 或 calibration 数据集生成 INT8 校准表
# ============================================================

set -e

ONNX_PATH="${1:?Usage: $0 <onnx_path> [output_engine]}"
ENGINE_PATH="${2:-models/yolov8n_int8.engine}"
CALIB_CACHE="models/int8_calib.cache"

INPUT_SIZE=320
BATCH_SIZE=1
DLA_CORE=0
WORKSPACE_MB=1024

echo "============================================"
echo "  TensorRT Engine Builder"
echo "============================================"
echo "  ONNX:       ${ONNX_PATH}"
echo "  Engine:     ${ENGINE_PATH}"
echo "  Input size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "  DLA Core:   ${DLA_CORE}"
echo "  Precision:  INT8"
echo "============================================"

# 检查 trtexec
if ! command -v trtexec &> /dev/null; then
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
    if [ ! -f "$TRTEXEC" ]; then
        echo "ERROR: trtexec not found"
        exit 1
    fi
else
    TRTEXEC="trtexec"
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

# 构建命令
CMD="${TRTEXEC} \
    --onnx=${ONNX_PATH} \
    --saveEngine=${ENGINE_PATH} \
    --int8 \
    --useDLACore=${DLA_CORE} \
    --allowGPUFallback \
    --workspace=${WORKSPACE_MB} \
    --shapes=images:${BATCH_SIZE}x3x${INPUT_SIZE}x${INPUT_SIZE}"

# 如果有校准缓存, 加载
if [ -f "${CALIB_CACHE}" ]; then
    CMD="${CMD} --calib=${CALIB_CACHE}"
    echo "  Using calibration cache: ${CALIB_CACHE}"
fi

echo ""
echo "Running: ${CMD}"
echo ""

eval "${CMD}"

echo ""
echo "Engine saved to: ${ENGINE_PATH}"
echo "Done!"
