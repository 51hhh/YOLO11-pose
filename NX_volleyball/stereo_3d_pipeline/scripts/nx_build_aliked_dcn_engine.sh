#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/nvidia/NX_volleyball/stereo_3d_pipeline}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/neural}"
ROI_SIZE="${ROI_SIZE:-128}"
TOP_K="${TOP_K:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OPSET="${OPSET:-19}"
TRT_FP16="${TRT_FP16:-1}"
ALIKED_MODEL="${ALIKED_MODEL:-aliked-t16}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-/home/nvidia/neural_repos/lightglue_src}"
PLUGIN_LIBRARY_PATH="${PLUGIN_LIBRARY_PATH:-${OUT_DIR}/plugins/libdcnv2.so}"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcudss/12:${LD_LIBRARY_PATH:-}"

suffix="${ROI_SIZE}_top${TOP_K}"
if [[ "${BATCH_SIZE}" != "1" ]]; then
  suffix="${suffix}_b${BATCH_SIZE}"
fi
model_tag="${ALIKED_MODEL#aliked-}"
onnx_path="${OUT_DIR}/aliked_${model_tag}_dcn_extractor_${suffix}.onnx"
engine_path="${OUT_DIR}/aliked_${model_tag}_dcn_extractor_${suffix}.engine"

if [[ ! -f "${PLUGIN_LIBRARY_PATH}" ]]; then
  echo "Missing DCNv2 plugin: ${PLUGIN_LIBRARY_PATH}" >&2
  echo "Build it first: scripts/nx_build_dcnv2_plugin.sh" >&2
  exit 1
fi

python3 "${ROOT_DIR}/scripts/export_aliked_dcn_extractor_onnx.py" \
  --out "${onnx_path}" \
  --roi-size "${ROI_SIZE}" \
  --top-k "${TOP_K}" \
  --batch-size "${BATCH_SIZE}" \
  --opset "${OPSET}" \
  --aliked-model "${ALIKED_MODEL}"

trt_args=(
  --onnx="${onnx_path}"
  --saveEngine="${engine_path}"
  --memPoolSize=workspace:1024
  --staticPlugins="${PLUGIN_LIBRARY_PATH}"
)
if [[ "${TRT_FP16}" == "1" ]]; then
  trt_args+=(--fp16)
fi

trtexec "${trt_args[@]}"
echo "${engine_path}"
