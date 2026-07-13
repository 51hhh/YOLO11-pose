#!/usr/bin/env bash
# Sync the local stereo_3d_pipeline workspace to Jetson NX with rsync.
#
# Default mode is a dry-run compare. Use --apply to write changes.
# Password auth is supported through SSHPASS:
#   SSHPASS=nvidia scripts/sync_nx_workspace.sh --apply

set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="nvidia@10.42.0.149"
REMOTE_DIR="/home/nvidia/NX_volleyball/stereo_3d_pipeline"
APPLY=0
DELETE=0
CHECKSUM=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [check|sync] [options]

Modes:
  check              Dry-run compare (default)
  sync               Apply changes

Options:
  --apply            Apply changes, same as mode sync
  --host HOST        Remote SSH host (default: ${REMOTE_HOST})
  --remote-dir DIR   Remote project dir (default: ${REMOTE_DIR})
  --local-dir DIR    Local project dir (default: ${LOCAL_DIR})
  --delete           Delete stale remote files in synced paths
  --no-delete        Do not delete stale remote files (default)
  --checksum         Compare by checksum instead of size/mtime
  -h, --help         Show this help

Set SSHPASS to use sshpass -e for password auth.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    check|dry-run)
      APPLY=0
      shift
      ;;
    sync|apply|--apply)
      APPLY=1
      shift
      ;;
    --host)
      REMOTE_HOST="${2:?missing --host value}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:?missing --remote-dir value}"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="${2:?missing --local-dir value}"
      shift 2
      ;;
    --no-delete)
      DELETE=0
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    --checksum)
      CHECKSUM=1
      shift
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

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "Local dir does not exist: ${LOCAL_DIR}" >&2
  exit 1
fi

ssh_base=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
rsync_rsh=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "SSHPASS is set but sshpass is not installed locally." >&2
    exit 1
  fi
  ssh_base=(sshpass -e "${ssh_base[@]}")
  rsync_rsh=(sshpass -e "${rsync_rsh[@]}")
fi

rsync_opts=(-az --human-readable --itemize-changes --stats)
if [[ "${APPLY}" -eq 0 ]]; then
  rsync_opts+=(--dry-run)
fi
if [[ "${DELETE}" -eq 1 ]]; then
  rsync_opts+=(--delete)
fi
if [[ "${CHECKSUM}" -eq 1 ]]; then
  rsync_opts+=(--checksum)
fi

excludes=(
  "--exclude=.git/"
  "--exclude=.agents/"
  "--exclude=.codex/"
  "--exclude=.cache/"
  "--exclude=.__codex_sync__/"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=.venv*/"
  "--exclude=venv/"
  "--exclude=build/"
  "--exclude=build_*/"
  "--exclude=cmake-build-*/"
  "--exclude=CMakeFiles/"
  "--exclude=CMakeCache.txt"
  "--exclude=cmake_install.cmake"
  "--exclude=compile_commands.json"
  "--exclude=Makefile"
  "--exclude=models/"
  "--exclude=model/"
  "--exclude=weights/"
  "--exclude=third_party/"
  "--exclude=*.engine"
  "--exclude=*.plan"
  "--exclude=*.onnx"
  "--exclude=*.pt"
  "--exclude=*.pth"
  "--exclude=*.wts"
  "--exclude=logs/"
  "--exclude=test_logs/"
  "--exclude=benchmark_results/"
  "--exclude=diagnose_output/"
  "--exclude=outputs/"
  "--exclude=recordings/"
  "--exclude=trajectory_dataset/"
  "--exclude=baseline_clips/"
  "--exclude=calibration/"
  "--exclude=calibration_images/"
  "--exclude=\$(ALLUSERSPROFILE)/"
  "--exclude=*.csv"
  "--exclude=*.metadata.yaml"
  "--exclude=*.frames.csv"
  "--exclude=*.p2_diagnostic.csv"
  "--exclude=*.bag"
  "--exclude=*.db3"
  "--exclude=*.mp4"
  "--exclude=*.avi"
  "--exclude=*.nsys-rep"
  "--exclude=*.qdrep"
  "--exclude=*.tmp"
)

mode="DRY-RUN"
if [[ "${APPLY}" -eq 1 ]]; then
  mode="APPLY"
fi

echo "NX sync mode: ${mode}"
echo "Local:  ${LOCAL_DIR}/"
echo "Remote: ${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Delete stale remote synced files: ${DELETE}"
echo "Checksum compare: ${CHECKSUM}"

"${ssh_base[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"
"${ssh_base[@]}" "${REMOTE_HOST}" \
  "mkdir -p '/home/nvidia/NX_volleyball/calibration'"

rsync "${rsync_opts[@]}" \
  -e "${rsync_rsh[*]}" \
  "${excludes[@]}" \
  "${LOCAL_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

rsync "${rsync_opts[@]}" \
  -e "${rsync_rsh[*]}" \
  "${LOCAL_DIR}/../calibration/stereo_calib.yaml" \
  "${REMOTE_HOST}:/home/nvidia/NX_volleyball/calibration/stereo_calib.yaml"
