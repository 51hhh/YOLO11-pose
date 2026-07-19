#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
VOLLEYBALL_ROS_SETUP="${VOLLEYBALL_ROS_SETUP:-/home/nvidia/volleyball_ros2_ws/install/setup.bash}"
PIPELINE_EXECUTABLE="${PIPELINE_EXECUTABLE:-$PROJECT_DIR/build/stereo_pipeline}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-$PROJECT_DIR/config/pipeline_rdk_joint.yaml}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
SOURCE_EPOCH_FILE="${SOURCE_EPOCH_FILE:-/run/volleyball/nx_source_epoch}"

for file in "$ROS_SETUP" "$VOLLEYBALL_ROS_SETUP" "$PIPELINE_EXECUTABLE" "$PIPELINE_CONFIG"; do
  if [ ! -e "$file" ]; then
    echo "缺少联合运行文件: $file" >&2
    exit 1
  fi
done
if [ -z "${CYCLONEDDS_URI:-}" ]; then
  NX_DDS_ADDRESS="${NX_DDS_ADDRESS:-10.43.0.10}"
  RDK_DDS_PEER="${RDK_DDS_PEER:-10.43.0.20}"
  DDS_CONFIG="/tmp/robocon_cyclonedds_nx_${ROS_DOMAIN_ID}.xml"
  umask 077
  cat > "$DDS_CONFIG" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="$NX_DDS_ADDRESS" priority="default" multicast="false"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers><Peer address="$RDK_DDS_PEER"/></Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
  CYCLONEDDS_URI="file://$DDS_CONFIG"
fi

export ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI
source "$ROS_SETUP"
source "$VOLLEYBALL_ROS_SETUP"

if ! chronyc waitsync 30 0.002; then
  echo "NX系统时钟未在2 ms内锁定到RDK，拒绝启动联合视觉。" >&2
  exit 1
fi

install -d -m 0755 "$(dirname "$SOURCE_EPOCH_FILE")"
rm -f "$SOURCE_EPOCH_FILE"

PIPELINE_PID=""
cleanup() {
  if [ -n "$PIPELINE_PID" ]; then
    kill "$PIPELINE_PID" 2>/dev/null || true
    wait "$PIPELINE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PIPELINE_EXECUTABLE" --config "$PIPELINE_CONFIG" &
PIPELINE_PID=$!

for _ in $(seq 1 100); do
  if [ -s "$SOURCE_EPOCH_FILE" ]; then
    break
  fi
  if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
    echo "NX双目管线在生成source_epoch前退出。" >&2
    wait "$PIPELINE_PID"
    exit 1
  fi
  sleep 0.1
done
if [ ! -s "$SOURCE_EPOCH_FILE" ]; then
  echo "等待NX source_epoch超时: $SOURCE_EPOCH_FILE" >&2
  exit 1
fi

echo "NX联合视觉已启动: domain=$ROS_DOMAIN_ID dds=$CYCLONEDDS_URI epoch=$(cat "$SOURCE_EPOCH_FILE")"
ros2 launch volleyball_catch_controller nx_time_sync.launch.py
