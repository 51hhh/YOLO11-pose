#!/usr/bin/env bash
# 回放轨迹 CSV，发布排球轨迹 + EKF 预测落点到 RViz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV_DEFAULT="$ROOT/stereo_3d_pipeline/test_logs/trajectory_dataset/p1_dy_regression_20260710_042552/traj.csv"
CSV="${1:-$CSV_DEFAULT}"
SPEED="${SPEED:-1.0}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0.50}"
TRACK_ID="${TRACK_ID:-}"

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV"
  echo "Usage: $0 [traj.csv]"
  exit 1
fi

# ROS2 env (adjust if needed)
if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
if [[ -f "$HOME/NX_volleyball/ros2_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/NX_volleyball/ros2_ws/install/setup.bash"
fi

EXTRA=()
if [[ -n "$TRACK_ID" ]]; then
  EXTRA+=(--track-id "$TRACK_ID")
fi

echo "CSV: $CSV"
echo "camera_height=${CAMERA_HEIGHT}m  speed=${SPEED}x"
echo "RViz: Fixed Frame = vision_world"
echo "  Add topics: /ball/actual_path /ball/predicted_path /ball/landing /ball/ground_plane /ball/ball_marker /ball/landing_marker"

python3 "$ROOT/replay_node/replay_trajectory.py" \
  --csv "$CSV" \
  --mode landing_ekf \
  --camera-height "$CAMERA_HEIGHT" \
  --speed "$SPEED" \
  "${EXTRA[@]}"
