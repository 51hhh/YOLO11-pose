#!/bin/bash
# ============================================================
# build_engine.sh - 编译 TensorRT Engine (在 NX 上运行)
#
# 用法:
#   ./build_engine.sh <onnx_path> [output_engine] [mode]
#
# 说明:
#   mode=gpu: GPU FP16 engine
#   mode=dla: DLA FP16 engine (allowGPUFallback)
#
# 注意:
#   TensorRT 10.x 在部分 JetPack 版本上对 --memPoolSize 的 MiB 后缀解析异常,
#   这里统一使用“字节数”避免 workspace 被错误设置。
# ============================================================

set -euo pipefail

ONNX_PATH="${1:?Usage: $0 <onnx_path> [output_engine] [mode=gpu|dla]}"
ENGINE_PATH="${2:-models/yolo_fp16.engine}"
MODE="${3:-gpu}"

WORKSPACE_BYTES=4294967296   # 4 GiB
DLA_CORE=0

echo "============================================"
echo "  TensorRT Engine Builder"
echo "============================================"
echo "  ONNX:       ${ONNX_PATH}"
echo "  Engine:     ${ENGINE_PATH}"
echo "  Mode:       ${MODE}"
echo "  Workspace:  ${WORKSPACE_BYTES} bytes"
if [ "${MODE}" = "dla" ]; then
    echo "  DLA Core:   ${DLA_CORE}"
fi
echo "  Precision:  FP16"
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

BASE_CMD="${TRTEXEC} \
    --onnx=${ONNX_PATH} \
    --saveEngine=${ENGINE_PATH} \
    --fp16 \
    --memPoolSize=workspace:${WORKSPACE_BYTES}"

if [ "${MODE}" = "dla" ]; then
    CMD="${BASE_CMD} --useDLACore=${DLA_CORE} --allowGPUFallback"
else
    CMD="${BASE_CMD}"
fi

echo ""
echo "Running: ${CMD}"
echo ""

eval "${CMD}"

if [ ! -f "${ENGINE_PATH}" ]; then
    echo "ERROR: Engine build finished but file not found: ${ENGINE_PATH}"
    exit 2
fi

echo ""
echo "Engine saved to: ${ENGINE_PATH}"
echo "Done!"
