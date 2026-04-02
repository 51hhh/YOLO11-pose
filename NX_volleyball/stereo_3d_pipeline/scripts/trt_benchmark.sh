#!/bin/bash
# =============================================================================
# TRT Detector Standalone Benchmark
# Benchmarks the TRT engine without camera using trtexec
# =============================================================================
set -euo pipefail

MODEL_DIR="$HOME/NX_volleyball/model"
RESULT_DIR="$HOME/NX_volleyball/stereo_3d_pipeline/benchmark_results"
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH

mkdir -p "$RESULT_DIR"
REPORT="$RESULT_DIR/trt_benchmark_$(date +%Y%m%d_%H%M%S).txt"

echo "=== TensorRT Inference Benchmark ===" | tee "$REPORT"
echo "Date: $(date)" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

# Set MAXN mode
sudo nvpmodel -m 0 2>/dev/null
sudo jetson_clocks 2>/dev/null

# Find engine file
ENGINE=""
for f in "$MODEL_DIR"/*.engine; do
    [ -f "$f" ] && ENGINE="$f" && break
done

if [ -z "$ENGINE" ]; then
    echo "ERROR: No .engine file found in $MODEL_DIR" | tee -a "$REPORT"
    exit 1
fi

echo "Engine: $ENGINE" | tee -a "$REPORT"
echo "Size: $(ls -lh $ENGINE | awk '{print $5}')" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

# ==========  GPU-only benchmark ==========
echo "--- GPU-only FP16 Benchmark (200 iterations, warmup 3s) ---" | tee -a "$REPORT"
trtexec --loadEngine="$ENGINE" \
    --iterations=200 \
    --warmUp=3000 \
    --avgRuns=50 \
    --percentile=50,90,95,99 \
    --separateProfileRun 2>&1 | tee /tmp/trt_gpu_bench.txt

# Extract key metrics
echo "" | tee -a "$REPORT"
echo "=== GPU Results ===" | tee -a "$REPORT"
grep -E "mean|median|Throughput|percentile|GPU Compute" /tmp/trt_gpu_bench.txt | tee -a "$REPORT"

# ========== Batch size exploration ==========
echo "" | tee -a "$REPORT"
echo "--- Batch Size Comparison ---" | tee -a "$REPORT"
# Note: This only works if the engine was built with dynamic batch

# ========== Concurrent inference + stereo simulation ==========
echo "" | tee -a "$REPORT"
echo "--- Sustained Load Test (1000 iterations) ---" | tee -a "$REPORT"
trtexec --loadEngine="$ENGINE" \
    --iterations=1000 \
    --warmUp=5000 \
    --avgRuns=100 \
    --percentile=50,90,95,99 2>&1 | grep -E "mean|median|Throughput|percentile|GPU Compute|Latency" | tee -a "$REPORT"

# ========== DLA test (may fail if engine not DLA-compatible) ==========
echo "" | tee -a "$REPORT"
echo "--- DLA Core 0 Test ---" | tee -a "$REPORT"
trtexec --loadEngine="$ENGINE" --useDLACore=0 \
    --iterations=100 --warmUp=2000 2>&1 | grep -E "mean|median|Throughput|ERROR|FAIL|percentile" | tee -a "$REPORT" || echo "DLA not supported for this engine" | tee -a "$REPORT"

echo "" | tee -a "$REPORT"
echo "Full report: $REPORT" | tee -a "$REPORT"
echo "=== Benchmark Complete ===" | tee -a "$REPORT"
