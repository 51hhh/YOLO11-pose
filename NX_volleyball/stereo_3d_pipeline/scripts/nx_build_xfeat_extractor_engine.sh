#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/nvidia/NX_volleyball/stereo_3d_pipeline}"
XFEAT_REPO="${XFEAT_REPO:-/home/nvidia/neural_repos/accelerated_features}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/neural}"
ROI_SIZE="${ROI_SIZE:-224}"
OPSET="${OPSET:-17}"
TRT_FP16="${TRT_FP16:-1}"

mkdir -p "${OUT_DIR}"

ONNX_PATH="${OUT_DIR}/xfeat_extractor_${ROI_SIZE}.onnx"
ENGINE_PATH="${OUT_DIR}/xfeat_extractor_${ROI_SIZE}.engine"

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcudss/12:${LD_LIBRARY_PATH:-}"

python3 "${ROOT_DIR}/scripts/export_xfeat_extractor_onnx.py" \
  --xfeat-repo "${XFEAT_REPO}" \
  --out "${ONNX_PATH}" \
  --roi-size "${ROI_SIZE}" \
  --opset "${OPSET}"

TRT_ARGS=(
  --onnx="${ONNX_PATH}"
  --saveEngine="${ENGINE_PATH}"
  --memPoolSize=workspace:1024
)

if [[ "${TRT_FP16}" == "1" ]]; then
  TRT_ARGS+=(--fp16)
fi

trtexec "${TRT_ARGS[@]}"

echo "${ENGINE_PATH}"
