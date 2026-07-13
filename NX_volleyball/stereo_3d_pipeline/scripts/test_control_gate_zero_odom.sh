#!/usr/bin/env bash
# Zero-odom, motors-off validation for the realtime ROS2 control-goal gate.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build_ros2}"
CONFIG="${CONFIG:-config/pipeline_record_p0p1.yaml}"
DURATION_SEC="${1:-30}"
LOG_DIR="${LOG_DIR:-/tmp/stereo_control_gate_$(date +%Y%m%d_%H%M%S)}"
CONTROL_GOAL_TOPIC="${CONTROL_GOAL_TOPIC:-/nx/debug/auto_goal_pose}"

cd "${ROOT_DIR}"
source /opt/ros/humble/setup.bash

PIPELINE="${BUILD_DIR}/stereo_pipeline"
if [[ ! -x "${PIPELINE}" ]]; then
    echo "ERROR: ROS2 pipeline binary not found: ${PIPELINE}" >&2
    echo "Build with ROS2 enabled before running this test." >&2
    exit 1
fi

if ! [[ "${DURATION_SEC}" =~ ^[0-9]+$ ]] || (( DURATION_SEC < 5 )); then
    echo "ERROR: duration must be an integer >= 5 seconds" >&2
    exit 1
fi

# Refuse to run when a motion controller is already listening to the real
# command topic, unless the operator explicitly confirms that actuation is
# disabled.
subscriber_count="$(ros2 topic info /auto/goal_pose 2>/dev/null \
    | awk '/Subscription count:/ {print $3}' | tail -1)"
subscriber_count="${subscriber_count:-0}"
if (( subscriber_count > 0 )) && [[ "${ALLOW_CONTROL_SUBSCRIBER:-0}" != "1" ]]; then
    echo "ERROR: /auto/goal_pose has ${subscriber_count} subscriber(s)." >&2
    echo "Disable motor control, or set ALLOW_CONTROL_SUBSCRIBER=1 after confirming safety." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
ODOM_LOG="${LOG_DIR}/odom_zero.log"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
GOAL_LOG="${LOG_DIR}/goal_pose.log"

odom_pid=""
pipeline_pid=""
echo_pid=""
cleanup() {
    for pid in "${echo_pid}" "${pipeline_pid}" "${odom_pid}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Logs: ${LOG_DIR}"
echo "Publishing zero odom at 20 Hz for ${DURATION_SEC}s"
timeout "${DURATION_SEC}s" ros2 topic pub -r 20 /odom nav_msgs/msg/Odometry \
    "{header: {frame_id: vision_world}, child_frame_id: base_link, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}" \
    >"${ODOM_LOG}" 2>&1 &
odom_pid=$!

timeout "${DURATION_SEC}s" "${PIPELINE}" --config "${CONFIG}" \
    >"${PIPELINE_LOG}" 2>&1 &
pipeline_pid=$!

# Wait until the pipeline creates the debug topic; ros2 topic echo exits immediately
# if it is launched before the publisher exists.
for _ in $(seq 1 50); do
    if ros2 topic type "${CONTROL_GOAL_TOPIC}" >/dev/null 2>&1; then
        timeout "${DURATION_SEC}s" ros2 topic echo "${CONTROL_GOAL_TOPIC}" \
            >"${GOAL_LOG}" 2>&1 &
        echo_pid=$!
        break
    fi
    if ! kill -0 "${pipeline_pid}" 2>/dev/null; then
        break
    fi
    sleep 0.1
done

wait "${pipeline_pid}" || true
cleanup
trap - EXIT INT TERM

echo
echo "=== Gate summary ==="
grep -E "TrajectoryPredictor|Ball bridge|Hold control goal|Reject control goal|Publish safe control goal" \
    "${PIPELINE_LOG}" | head -80 || true
echo
echo "Published messages: $(grep -c 'frame_id: base_link' "${GOAL_LOG}" 2>/dev/null || true)"
echo "Debug gate topic: ${CONTROL_GOAL_TOPIC}"
echo "Full logs: ${LOG_DIR}"
