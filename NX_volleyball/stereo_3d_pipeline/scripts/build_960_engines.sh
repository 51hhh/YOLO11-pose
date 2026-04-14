#!/bin/bash
# ============================================================
# build_960_engines.sh — 在 NX 上构建 960×960 模型 TensorRT 引擎
#
# 用法: ./scripts/build_960_engines.sh
# 前提: ONNX 文件已同步到 ~/NX_volleyball/model/
# ============================================================

set -euo pipefail

MODEL_DIR="/home/nvidia/NX_volleyball/model"
WORKSPACE_BYTES=4294967296   # 4 GiB

# 检查 trtexec
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC" ]; then
    TRTEXEC="trtexec"
fi

build_engine() {
    local ONNX="$1"
    local ENGINE="$2"
    local NAME="$3"

    if [ -f "$ENGINE" ]; then
        echo "[SKIP] $NAME: Engine already exists at $ENGINE"
        return 0
    fi

    if [ ! -f "$ONNX" ]; then
        echo "[ERROR] $NAME: ONNX not found at $ONNX"
        return 1
    fi

    echo "============================================"
    echo "  Building: $NAME"
    echo "  ONNX:     $ONNX"
    echo "  Engine:   $ENGINE"
    echo "  Mode:     GPU FP16"
    echo "============================================"

    $TRTEXEC \
        --onnx="$ONNX" \
        --saveEngine="$ENGINE" \
        --fp16 \
        --memPoolSize=workspace:${WORKSPACE_BYTES}

    if [ -f "$ENGINE" ]; then
        echo "[OK] $NAME: Engine saved to $ENGINE"
    else
        echo "[ERROR] $NAME: Build failed"
        return 1
    fi
}

echo "=== Building 960×960 TensorRT Engines ==="

# YOLO11-S 960 FP16
build_engine \
    "${MODEL_DIR}/yolo11_s_960.onnx" \
    "${MODEL_DIR}/yolo11_s_960_fp16.engine" \
    "YOLO11-S 960 FP16"

# YOLOv8-M 960 FP16
build_engine \
    "${MODEL_DIR}/yolo8_m_960.onnx" \
    "${MODEL_DIR}/yolo8_m_960_fp16.engine" \
    "YOLOv8-M 960 FP16"

echo ""
echo "=== All builds complete ==="
ls -lh ${MODEL_DIR}/*.engine 2>/dev/null || echo "No engines found"
