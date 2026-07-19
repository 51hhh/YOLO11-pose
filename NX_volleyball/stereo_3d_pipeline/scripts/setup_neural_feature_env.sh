#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/../../.venv-stereo-neural}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
XFEAT_DIR="${XFEAT_DIR:-${HOME}/.local/share/stereo_3d_pipeline/neural_repos/accelerated_features}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv not found at ${UV_BIN}. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

"${UV_BIN}" venv --python 3.12 "${VENV_DIR}"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
  opencv-python scipy tqdm matplotlib kornia \
  git+https://github.com/cvg/LightGlue.git

mkdir -p "$(dirname "${XFEAT_DIR}")"
if [[ -d "${XFEAT_DIR}/.git" ]]; then
  git -C "${XFEAT_DIR}" pull --ff-only
else
  git clone --depth 1 https://github.com/verlab/accelerated_features.git "${XFEAT_DIR}"
fi

echo "Neural feature env ready:"
echo "  python: ${VENV_DIR}/bin/python"
echo "  xfeat:  ${XFEAT_DIR}"
