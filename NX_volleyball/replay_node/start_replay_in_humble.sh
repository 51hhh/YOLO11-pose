#!/usr/bin/env bash
# 在 humble Docker 容器中启动：RViz + CSV 回放 + EKF 落点
# 兼容你之前的 /home/rick/mid360/start_rviz.sh 用法
set -euo pipefail

CONTAINER="${CONTAINER:-humble}"
CSV_DEFAULT="/home/rick/mid360/YOLO11-pose/NX_volleyball/stereo_3d_pipeline/test_logs/trajectory_dataset/p1_dy_regression_20260710_042552/traj.csv"
CSV="${1:-$CSV_DEFAULT}"
RVIZ_FILE="${RVIZ_FILE:-/home/rick/mid360/config/volleyball_view.rviz}"
REPLAY_PY="/home/rick/mid360/YOLO11-pose/NX_volleyball/replay_node/replay_trajectory.py"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0.50}"
SPEED="${SPEED:-0.5}"
TRACK_ID="${TRACK_ID:-}"
MAX_RANGE_M="${MAX_RANGE_M:-}"

shift $(( $# > 0 ? 1 : 0 )) || true
EXTRA=("$@")

if [[ ! -f "$CSV" ]]; then
  echo "CSV not found: $CSV"
  echo "Usage: $0 [traj.csv] [--speed 0.5]  # 默认不剔除; 可选 --max-range-m 10"
  exit 1
fi

DOCKER=docker
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  else
    echo "Cannot access docker. Run on host with docker permissions."
    exit 1
  fi
fi

if ! $DOCKER ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "Starting container $CONTAINER ..."
    $DOCKER start "$CONTAINER" >/dev/null
  else
    echo "Container not found: $CONTAINER"
    $DOCKER ps -a
    exit 1
  fi
fi

echo "RViz + landing replay"
echo "  container: $CONTAINER"
echo "  csv: $CSV"
echo "  camera_height: ${CAMERA_HEIGHT}m"
echo "  speed: ${SPEED}x"

# cleanup old viewers
$DOCKER exec "$CONTAINER" bash -c 'pkill -9 rviz2 2>/dev/null; pkill -9 static_transform 2>/dev/null; pkill -9 -f replay_trajectory.py 2>/dev/null' || true
sleep 1
xhost +local:docker 2>/dev/null || xhost +local: 2>/dev/null || true

DDS_ENV='export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; export CYCLONEDDS_URI=file:///home/rick/mid360/YOLO11-pose/NX_volleyball/replay_node/cyclonedds_local.xml; export ROS_DOMAIN_ID=42'

# TF + RViz
$DOCKER exec -d "$CONTAINER" bash -c "source /opt/ros/humble/setup.bash && $DDS_ENV && exec /opt/ros/humble/lib/tf2_ros/static_transform_publisher --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 --frame-id vision_world --child-frame-id base_link"
$DOCKER exec -d "$CONTAINER" bash -c "source /opt/ros/humble/setup.bash && $DDS_ENV && export DISPLAY=${DISPLAY:-:0} && exec rviz2 -d $RVIZ_FILE"
sleep 2
echo "RViz started. Fixed Frame = vision_world"

ARGS=(--csv "$CSV" --mode landing_ekf --camera-height "$CAMERA_HEIGHT" --speed "$SPEED" --loop)
if [[ -n "${MAX_RANGE_M}" ]]; then
  ARGS+=(--max-range-m "$MAX_RANGE_M")
fi
if [[ -n "$TRACK_ID" ]]; then
  ARGS+=(--track-id "$TRACK_ID")
fi
ARGS+=("${EXTRA[@]}")

# quote args for remote bash
CMD_ARGS=""
for a in "${ARGS[@]}"; do
  CMD_ARGS+=" $(printf '%q' "$a")"
done

echo "Topics: /ball/actual_path /ball/predicted_path /ball/landing /ball/ground_plane /ball/ball_marker /ball/landing_marker"
echo "Keys: Space pause | arrows step | +/- speed | r restart | q quit"

$DOCKER exec -it "$CONTAINER" bash -c "source /opt/ros/humble/setup.bash && $DDS_ENV && \
  export PYTHONPATH=/home/rick/mid360/YOLO11-pose/NX_volleyball/stereo_3d_pipeline:\${PYTHONPATH:-} && \
  python3 $REPLAY_PY$CMD_ARGS"
