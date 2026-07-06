#!/usr/bin/env bash
# Run the P0/P1 signed-dy regression on Jetson NX and summarize candidate CSVs.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/nvidia/NX_volleyball/stereo_3d_pipeline}"
CONFIG="${CONFIG:-config/pipeline_record_p0p1.yaml}"
OUT_BASE="${OUT_BASE:-/home/nvidia/trajectory_dataset}"
DURATION_SEC=20
BUILD=0
DEBUG_DUMP=0
DEBUG_STRIDE=100
DEBUG_MAX=20

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --duration SEC     Run duration before SIGINT (default: ${DURATION_SEC})
  --out-base DIR     Output parent directory (default: ${OUT_BASE})
  --root DIR         Project root (default: ${ROOT_DIR})
  --config PATH      Config path relative to root, or absolute path
  --build            Build stereo_pipeline before running
  --debug-dump       Enable realtime zoom/debug dump
  --debug-stride N   Debug dump stride (default: ${DEBUG_STRIDE})
  --debug-max N      Max debug dump frames (default: ${DEBUG_MAX})
  -h, --help         Show this help

Outputs:
  <out-dir>/traj.csv
  <out-dir>/traj.frames.csv
  <out-dir>/traj.p2_diagnostic.csv
  <out-dir>/traj.log
  <out-dir>/p1_candidate_report.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      DURATION_SEC="${2:?missing --duration value}"
      shift 2
      ;;
    --out-base)
      OUT_BASE="${2:?missing --out-base value}"
      shift 2
      ;;
    --root)
      ROOT_DIR="${2:?missing --root value}"
      shift 2
      ;;
    --config)
      CONFIG="${2:?missing --config value}"
      shift 2
      ;;
    --build)
      BUILD=1
      shift
      ;;
    --debug-dump)
      DEBUG_DUMP=1
      shift
      ;;
    --debug-stride)
      DEBUG_STRIDE="${2:?missing --debug-stride value}"
      shift 2
      ;;
    --debug-max)
      DEBUG_MAX="${2:?missing --debug-max value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "Project root not found: ${ROOT_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

BUILD_DIR=""
for candidate in build_standalone build; do
  if [[ -x "${candidate}/stereo_pipeline" ]]; then
    BUILD_DIR="${candidate}"
    break
  fi
done
if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in build_standalone build; do
    if [[ -d "${candidate}" ]]; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BUILD_DIR}" ]]; then
  echo "No build directory found under ${ROOT_DIR}" >&2
  exit 1
fi

if [[ "${BUILD}" -eq 1 ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

BIN="${BUILD_DIR}/stereo_pipeline"
if [[ ! -x "${BIN}" ]]; then
  echo "stereo_pipeline binary not found: ${BIN}" >&2
  exit 1
fi

if [[ "${CONFIG}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG}"
else
  CONFIG_PATH="${CONFIG}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

RUN_ID="p1_dy_regression_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_BASE%/}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

cat > "${OUT_DIR}/traj.metadata.yaml" <<EOF
dataset_type: regression
scene: p1_signed_dy_regression
known_z: null
static: null
notes: Validate P1 center_patch multi_point and cuda_template with per-pair signed dy prior.
EOF

cmd=(
  "./${BIN}"
  --config "${CONFIG_PATH}"
  --recording-out "${OUT_DIR}/traj.csv"
)

if [[ "${DEBUG_DUMP}" -eq 1 ]]; then
  cmd+=(
    --debug-realtime-dump
    --debug-realtime-dump-dir "${OUT_DIR}/debug"
    --debug-realtime-dump-stride "${DEBUG_STRIDE}"
    --debug-realtime-dump-max "${DEBUG_MAX}"
  )
fi

echo "NX P1 signed-dy regression"
echo "Root:     ${ROOT_DIR}"
echo "Build:    ${BUILD_DIR}"
echo "Config:   ${CONFIG_PATH}"
echo "Duration: ${DURATION_SEC}s"
echo "Out:      ${OUT_DIR}"
echo "Command:  timeout --signal=INT ${DURATION_SEC}s ${cmd[*]}"

set +e
timeout --signal=INT "${DURATION_SEC}s" "${cmd[@]}" 2>&1 | tee "${OUT_DIR}/traj.log"
run_status=${PIPESTATUS[0]}
set -e

if [[ "${run_status}" -ne 0 && "${run_status}" -ne 124 && "${run_status}" -ne 130 ]]; then
  echo "stereo_pipeline failed with status ${run_status}" >&2
  exit "${run_status}"
fi

if [[ ! -f "${OUT_DIR}/traj.csv" ]]; then
  echo "trajectory CSV not found: ${OUT_DIR}/traj.csv" >&2
  exit 1
fi

python3 scripts/analyze_p1_candidate_csv.py "${OUT_DIR}/traj.csv" \
  | tee "${OUT_DIR}/p1_candidate_report.txt"

echo "Regression output: ${OUT_DIR}"
