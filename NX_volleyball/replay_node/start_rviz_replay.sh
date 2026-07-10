#!/usr/bin/env bash
# 替代 /home/rick/mid360/start_rviz.sh replay
# 使用本机 cyclonedds_local.xml，避免网卡 192.168.31.72 不存在
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"  # YOLO11-pose
CSV="${1:-$ROOT/NX_volleyball/stereo_3d_pipeline/test_logs/trajectory_dataset/p1_dy_regression_20260710_042552/traj.csv}"
shift $(( $# > 0 ? 1 : 0 )) || true
exec "$ROOT/NX_volleyball/replay_node/start_replay_in_humble.sh" "$CSV" "$@"
