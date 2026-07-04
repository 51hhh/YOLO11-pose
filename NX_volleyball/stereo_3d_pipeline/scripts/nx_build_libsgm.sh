#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party/libSGM"
SRC_DIR="${THIRD_PARTY_DIR}/src"
BUILD_DIR="${THIRD_PARTY_DIR}/build"
INSTALL_DIR="${THIRD_PARTY_DIR}/install"
LIBSGM_REPO="${LIBSGM_REPO:-https://github.com/fixstars/libSGM.git}"
LIBSGM_REF="${LIBSGM_REF:-master}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDACXX=/usr/local/cuda/bin/nvcc
fi

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone "${LIBSGM_REPO}" "${SRC_DIR}"
fi

git -C "${SRC_DIR}" fetch --tags --depth 1 origin "${LIBSGM_REF}" || true
git -C "${SRC_DIR}" checkout "${LIBSGM_REF}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DENABLE_SAMPLES=OFF \
  -DENABLE_TESTS=OFF

cmake --build "${BUILD_DIR}" --target install -j"$(nproc)"

echo "libSGM installed to ${INSTALL_DIR}"
echo "Reconfigure stereo_pipeline; CMake auto-searches third_party/libSGM/install."
