#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/nvidia/NX_volleyball/stereo_3d_pipeline}"
PLUGIN_REPO_URL="${PLUGIN_REPO_URL:-https://github.com/flairziv/tensorrt-dcnv2-plugin.git}"
PLUGIN_SRC_DIR="${PLUGIN_SRC_DIR:-${ROOT_DIR}/third_party/tensorrt-dcnv2-plugin}"
PLUGIN_BUILD_DIR="${PLUGIN_BUILD_DIR:-${PLUGIN_SRC_DIR}/build}"
PLUGIN_OUT_DIR="${PLUGIN_OUT_DIR:-${ROOT_DIR}/models/neural/plugins}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

mkdir -p "${PLUGIN_OUT_DIR}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ ! -d "${PLUGIN_SRC_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${PLUGIN_SRC_DIR}")"
  git clone --depth 1 "${PLUGIN_REPO_URL}" "${PLUGIN_SRC_DIR}"
else
  git -C "${PLUGIN_SRC_DIR}" pull --ff-only
fi

cmake -S "${PLUGIN_SRC_DIR}/src" -B "${PLUGIN_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${PLUGIN_BUILD_DIR}" -j"$(nproc)"

plugin_so="$(find "${PLUGIN_BUILD_DIR}" -name 'lib*dcn*.so' -o -name 'lib*DCN*.so' | head -1)"
if [[ -z "${plugin_so}" ]]; then
  echo "Cannot find built DCNv2 plugin .so under ${PLUGIN_BUILD_DIR}" >&2
  exit 1
fi

cp -f "${plugin_so}" "${PLUGIN_OUT_DIR}/libdcnv2.so"
echo "${PLUGIN_OUT_DIR}/libdcnv2.so"
