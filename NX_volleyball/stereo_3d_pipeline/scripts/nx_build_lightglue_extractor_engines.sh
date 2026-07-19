#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/nvidia/NX_volleyball/stereo_3d_pipeline}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/neural}"
ROI_SIZE="${ROI_SIZE:-224}"
TOP_K="${TOP_K:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OPSET="${OPSET:-17}"
TRT_FP16="${TRT_FP16:-1}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-/home/nvidia/neural_repos/lightglue_src}"
BACKENDS="${BACKENDS:-superpoint}"
ALIKED_MODEL="${ALIKED_MODEL:-aliked-n16}"
ALLOW_FAILURES="${ALLOW_FAILURES:-0}"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcudss/12:${LD_LIBRARY_PATH:-}"

build_one() {
  local backend="$1"
  local suffix="${ROI_SIZE}_top${TOP_K}"
  if [[ "${BATCH_SIZE}" != "1" ]]; then
    suffix="${suffix}_b${BATCH_SIZE}"
  fi
  local onnx_path="${OUT_DIR}/${backend}_extractor_${suffix}.onnx"
  local engine_path="${OUT_DIR}/${backend}_extractor_${suffix}.engine"
  local export_args=(
    --backend "${backend}"
    --out "${onnx_path}"
    --roi-size "${ROI_SIZE}"
    --top-k "${TOP_K}"
    --batch-size "${BATCH_SIZE}"
    --opset "${OPSET}"
  )
  if [[ "${backend}" == "aliked" ]]; then
    export_args+=(--aliked-model "${ALIKED_MODEL}")
  fi

  python3 "${ROOT_DIR}/scripts/export_lightglue_extractor_onnx.py" \
    "${export_args[@]}"

  local trt_args=(
    --onnx="${onnx_path}"
    --saveEngine="${engine_path}"
    --memPoolSize=workspace:1024
  )
  if [[ "${TRT_FP16}" == "1" ]]; then
    trt_args+=(--fp16)
  fi
  trtexec "${trt_args[@]}"
  echo "${engine_path}"
}

failed=()
for backend in ${BACKENDS}; do
  if ! build_one "${backend}"; then
    failed+=("${backend}")
    if [[ "${ALLOW_FAILURES}" != "1" ]]; then
      exit 1
    fi
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "Failed backends: ${failed[*]}" >&2
fi
