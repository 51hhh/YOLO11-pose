#!/bin/bash
# ============================================================
# build_dla_engine.sh - 一键构建 DLA FP16 Engine
#
# 用法:
#   ./scripts/build_dla_engine.sh [onnx_path] [engine_path]
#
# 默认:
#   onnx   = $HOME/NX_volleyball/model/yolo26.onnx
#   engine = $HOME/NX_volleyball/model/yolo26_dla_fp16.engine
# ============================================================

set -euo pipefail

ONNX_PATH="${1:-$HOME/NX_volleyball/model/yolo26.onnx}"
ENGINE_PATH="${2:-$HOME/NX_volleyball/model/yolo26_dla_fp16.engine}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/build_engine.sh" "${ONNX_PATH}" "${ENGINE_PATH}" "dla"

echo "[OK] DLA engine ready: ${ENGINE_PATH}"
