#!/bin/bash
# ============================================================
# pipeline_perf_compare.sh
# 在 NX 上对比 GPU engine 与 DLA engine 的流水线性能
# ============================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/NX_volleyball/stereo_3d_pipeline}"
BUILD_BIN="${BUILD_BIN:-$PROJECT_DIR/build/stereo_pipeline}"
GPU_CFG="${GPU_CFG:-$PROJECT_DIR/config/pipeline.yaml}"
DLA_CFG="${DLA_CFG:-$PROJECT_DIR/config/pipeline_dla.yaml}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/benchmark_results}"
DURATION_SEC="${DURATION_SEC:-20}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$OUT_DIR/pipeline_compare_${TS}.md"

run_case() {
  local name="$1"
  local cfg="$2"
  local log="$OUT_DIR/${name}_${TS}.log"
  local tegra="$OUT_DIR/${name}_${TS}_tegra.log"

  echo "[RUN] $name config=$cfg"

  # 后台采样 tegrastats
  timeout $((DURATION_SEC + 5)) tegrastats --interval 200 > "$tegra" 2>&1 &
  local tegra_pid=$!

  # 运行 pipeline
  timeout "$DURATION_SEC" "$BUILD_BIN" --config "$cfg" > "$log" 2>&1 || true

  # 回收 tegrastats
  wait "$tegra_pid" 2>/dev/null || true

  # 提取 fps
  local fps
  fps=$(grep -E "FPS:" "$log" | tail -1 | sed -E 's/.*FPS:\s*([0-9.]+).*/\1/' || true)
  [ -z "${fps:-}" ] && fps="N/A"

  # 提取关键 stage 统计 (兼容新旧 stage 命名)
  local s0 s1 s2w s3w
  s0=$(grep -E "Stage0_GrabRect" "$log" | tail -1 | awk '{print $2}' || true)
  s1=$(grep -E "Stage1_DetectSubmit|Stage1_Detect" "$log" | tail -1 | awk '{print $2}' || true)
  s2w=$(grep -E "Stage3_WaitStereo" "$log" | tail -1 | awk '{print $2}' || true)
  s3w=$(grep -E "Stage3_WaitDetect" "$log" | tail -1 | awk '{print $2}' || true)

  # GPU 平均占用
  local gpu_avg
  gpu_avg=$(grep -oE 'GR3D_FREQ [0-9]+%' "$tegra" | sed -E 's/[^0-9]//g' | awk '{sum+=$1;n++} END{if(n>0) printf "%.1f", sum/n; else print "N/A"}')

  echo "| $name | $fps | ${s0:-N/A} | ${s1:-N/A} | ${s2w:-N/A} | ${s3w:-N/A} | ${gpu_avg} |" >> "$SUMMARY"
}

cat > "$SUMMARY" <<EOF
# Pipeline 并行化对比报告 ($TS)

- Duration: ${DURATION_SEC}s
- Binary: ${BUILD_BIN}

| 场景 | FPS | Stage0 Avg(ms) | Stage1 Submit Avg(ms) | Stage3 WaitStereo Avg(ms) | Stage3 WaitDetect Avg(ms) | GPU Avg(%) |
|---|---:|---:|---:|---:|---:|---:|
EOF

run_case "GPU" "$GPU_CFG"
if [ -f "$DLA_CFG" ]; then
  run_case "DLA" "$DLA_CFG"
fi

echo "" >> "$SUMMARY"
echo "## 原始日志" >> "$SUMMARY"
echo "- GPU log: $(basename "$OUT_DIR/GPU_${TS}.log")" >> "$SUMMARY"
if [ -f "$OUT_DIR/DLA_${TS}.log" ]; then
  echo "- DLA log: $(basename "$OUT_DIR/DLA_${TS}.log")" >> "$SUMMARY"
fi

echo "[OK] summary: $SUMMARY"
