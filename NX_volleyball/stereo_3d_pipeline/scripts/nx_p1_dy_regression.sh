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
  <out-dir>/control_gate_report.txt
  <out-dir>/odom_source.txt
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

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS2 Humble not found; control-gate recording requires ROS2." >&2
  exit 1
fi
set +u
source /opt/ros/humble/setup.bash
set -u

# Ensure GPU is locked at max frequency (jetson_clocks)
GPU_FREQ_PATH="/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"
if [[ -f "${GPU_FREQ_PATH}/cur_freq" ]]; then
  cur_freq=$(cat "${GPU_FREQ_PATH}/cur_freq" 2>/dev/null || echo 0)
  max_freq=$(cat "${GPU_FREQ_PATH}/max_freq" 2>/dev/null || echo 0)
  if [[ "${cur_freq}" -gt 0 && "${max_freq}" -gt 0 && "${cur_freq}" -lt "${max_freq}" ]]; then
    echo "[INFO] GPU at ${cur_freq}/${max_freq} Hz, locking with jetson_clocks..."
    if command -v sudo &>/dev/null; then
      echo "nvidia" | sudo -S jetson_clocks 2>/dev/null || true
      sleep 1
      cur_freq=$(cat "${GPU_FREQ_PATH}/cur_freq" 2>/dev/null || echo 0)
      echo "[INFO] GPU now at ${cur_freq} Hz"
    else
      echo "[WARN] sudo not available, cannot auto-lock GPU frequency" >&2
    fi
  fi
fi

BUILD_DIR="${BUILD_DIR:-}"
is_ros2_build() {
  local dir="$1"
  grep -q -- "-DHAS_ROS2" \
    "${dir}/CMakeFiles/stereo_pipeline.dir/flags.make" 2>/dev/null
}
if [[ -n "${BUILD_DIR}" ]] && ! is_ros2_build "${BUILD_DIR}"; then
  echo "Requested BUILD_DIR is not a ROS2 build: ${BUILD_DIR}" >&2
  exit 1
fi
if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in build_ros2 build; do
    if [[ -x "${candidate}/stereo_pipeline" ]] && is_ros2_build "${candidate}"; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BUILD_DIR}" && "${BUILD}" -eq 1 ]]; then
  for candidate in build_ros2 build; do
    if [[ -d "${candidate}" ]] && is_ros2_build "${candidate}"; then
      BUILD_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BUILD_DIR}" ]]; then
  echo "No ROS2 stereo_pipeline build found under ${ROOT_DIR}." >&2
  echo "Configure and build build_ros2 before recording gate decisions." >&2
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
camera_height_m: 0.50
prediction_path: bbox_d0_student_t_ekf_rk4
control_gate_recorded: true
notes: Record raw candidates plus ungated and realtime-gated landing predictions.
EOF
cp "${CONFIG_PATH}" "${OUT_DIR}/pipeline_config.yaml"
if [[ -f "${ROOT_DIR}/config/disparity_offset_fit_20260709.json" ]]; then
  cp "${ROOT_DIR}/config/disparity_offset_fit_20260709.json" \
    "${OUT_DIR}/disparity_offset_fit.json"
fi

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

CONTROL_GOAL_TOPIC="${CONTROL_GOAL_TOPIC:-/nx/debug/auto_goal_pose}"

echo "NX P1 signed-dy regression"
echo "Root:     ${ROOT_DIR}"
echo "Build:    ${BUILD_DIR}"
echo "Config:   ${CONFIG_PATH}"
echo "Duration: ${DURATION_SEC}s"
echo "Out:      ${OUT_DIR}"
echo "Gate:     ${CONTROL_GOAL_TOPIC}"
echo "Command:  timeout --signal=INT ${DURATION_SEC}s ${cmd[*]}"

# The ROS2 build records the same gate logic on a debug topic. Do not
# accidentally run while a motion controller is listening to the real command
# topic.
goal_topic_info="$(ros2 topic info /auto/goal_pose 2>/dev/null || true)"
goal_subscribers="$(awk '/Subscription count:/ {print $3}' \
  <<< "${goal_topic_info}" | tail -1)"
goal_subscribers="${goal_subscribers:-0}"
if [[ "${goal_subscribers}" -gt 0 && "${ALLOW_CONTROL_SUBSCRIBER:-0}" != "1" ]]; then
  echo "[ERROR] /auto/goal_pose has ${goal_subscribers} subscriber(s)." >&2
  echo "Disable vehicle control or explicitly set ALLOW_CONTROL_SUBSCRIBER=1." >&2
  exit 2
fi

odom_pid=""
cleanup() {
  if [[ -n "${odom_pid}" ]] && kill -0 "${odom_pid}" 2>/dev/null; then
    kill "${odom_pid}" 2>/dev/null || true
    wait "${odom_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

odom_topic_info="$(ros2 topic info /odom 2>/dev/null || true)"
odom_publishers="$(awk '/Publisher count:/ {print $3}' \
  <<< "${odom_topic_info}" | tail -1)"
odom_publishers="${odom_publishers:-0}"
if [[ "${odom_publishers}" -gt 0 ]]; then
  echo "existing" > "${OUT_DIR}/odom_source.txt"
  echo "[INFO] Using ${odom_publishers} existing /odom publisher(s)"
else
  echo "zero_20hz" > "${OUT_DIR}/odom_source.txt"
  echo "[INFO] No /odom publisher; starting zero odom at 20 Hz"
  timeout "$((DURATION_SEC + 5))s" ros2 topic pub -r 20 \
    /odom nav_msgs/msg/Odometry \
    "{header: {frame_id: vision_world}, child_frame_id: base_link, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}" \
    > "${OUT_DIR}/odom_zero.log" 2>&1 &
  odom_pid=$!
  sleep 1
fi

set +e
timeout --signal=INT "${DURATION_SEC}s" "${cmd[@]}" 2>&1 | tee "${OUT_DIR}/traj.log"
run_status=${PIPESTATUS[0]}
set -e
cleanup
trap - EXIT INT TERM

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

python3 scripts/analyze_control_gate_csv.py "${OUT_DIR}/traj.csv" \
  | tee "${OUT_DIR}/control_gate_report.txt"

echo "Regression output: ${OUT_DIR}"
