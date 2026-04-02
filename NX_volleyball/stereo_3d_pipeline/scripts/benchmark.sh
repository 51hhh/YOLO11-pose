#!/bin/bash
# ============================================================
# benchmark.sh - Orin NX full benchmark: build + model + components + profiling
#
# Usage:
#   bash ~/NX_volleyball/stereo_3d_pipeline/scripts/benchmark.sh
# ============================================================
set -e

BASE_DIR="$HOME/NX_volleyball"
PIPE_DIR="$BASE_DIR/stereo_3d_pipeline"
BUILD_DIR="$PIPE_DIR/build"
MODEL_PT="$BASE_DIR/model/yolo26n.pt"
RESULTS_DIR="$PIPE_DIR/benchmark_results"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/MVS/lib/aarch64:${LD_LIBRARY_PATH:-}

timestamp=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/benchmark_${timestamp}.txt"
mkdir -p "$RESULTS_DIR"

log() { echo "$@" | tee -a "$RESULT_FILE"; }

log "============================================================"
log " Stereo 3D Pipeline Benchmark - $(date)"
log " Platform: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
log "============================================================"

# ========== Phase 1: System Info ==========
log ""
log "=== [Phase 1] System Info ==="
log "L4T: $(head -1 /etc/nv_tegra_release)"
log "CUDA: $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
log "TensorRT: $(dpkg -l 2>/dev/null | grep libnvinfer10 | awk '{print $3}' | head -1)"
log "VPI: $(dpkg -l 2>/dev/null | grep libnvvpi3 | awk '{print $3}' | head -1)"
log "OpenCV: $(pkg-config --modversion opencv4 2>/dev/null)"
free -h 2>&1 | tee -a "$RESULT_FILE"
log "nvpmodel: $(sudo nvpmodel -q 2>/dev/null | head -3)"

# Performance mode
sudo nvpmodel -m 0 2>/dev/null && log "Set MAXN mode" || true
sudo jetson_clocks 2>/dev/null && log "jetson_clocks locked" || true

# ========== Phase 2: Build ==========
log ""
log "=== [Phase 2] CMake + Build ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

log "Running cmake..."
cmake "$PIPE_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH="87" \
    2>&1 | tee -a "$RESULT_FILE"

log ""
log "Running make -j$(nproc)..."
build_start=$(date +%s)
make -j$(nproc) 2>&1 | tee -a "$RESULT_FILE"
build_end=$(date +%s)
log "Build time: $((build_end - build_start)) seconds"
log "Build artifacts:"
ls -lh "$BUILD_DIR/stereo_pipeline" 2>/dev/null | tee -a "$RESULT_FILE" || log "stereo_pipeline: NOT BUILT"
ls -lh "$BUILD_DIR/stereo_calibrate" 2>/dev/null | tee -a "$RESULT_FILE" || true

# ========== Phase 3: Model Conversion ==========
log ""
log "=== [Phase 3] YOLO Model Conversion ==="

ENGINE_FILE="$BASE_DIR/model/yolo26n.engine"

if [ ! -f "$ENGINE_FILE" ] && [ -f "$MODEL_PT" ]; then
    log "Converting $MODEL_PT -> TensorRT FP16 engine (320x320)..."
    bash "$PIPE_DIR/scripts/convert_model.sh" "$MODEL_PT" 320 2>&1 | tee -a "$RESULT_FILE"
    ENGINE_FILE=$(find "$BASE_DIR/model" -name "*.engine" | head -1)
fi

if [ -f "$ENGINE_FILE" ]; then
    log "Engine ready: $ENGINE_FILE ($(du -h "$ENGINE_FILE" | awk '{print $1}'))"
else
    log "WARNING: No engine file. Will skip TRT benchmarks."
fi

# ========== Phase 4: trtexec Benchmark ==========
log ""
log "=== [Phase 4] TensorRT Inference Benchmark (trtexec) ==="

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC" ]; then
    TRTEXEC=$(which trtexec 2>/dev/null || echo "")
fi

if [ -f "$ENGINE_FILE" ] && [ -n "$TRTEXEC" ]; then
    log "--- GPU-only benchmark (FP16) ---"
    $TRTEXEC \
        --loadEngine="$ENGINE_FILE" \
        --warmUp=3000 \
        --duration=10 \
        --avgRuns=100 \
        2>&1 | grep -iE "mean|median|percentile|throughput|GPU Compute|Host Latency|enqueue" | tee -a "$RESULT_FILE"

    log ""
    log "--- DLA + GPU fallback benchmark ---"
    $TRTEXEC \
        --loadEngine="$ENGINE_FILE" \
        --warmUp=3000 \
        --duration=10 \
        --useDLACore=0 \
        --allowGPUFallback \
        2>&1 | grep -iE "mean|median|percentile|throughput|GPU Compute|DLA|Host Latency|enqueue" | tee -a "$RESULT_FILE" \
        || log "DLA benchmark failed (engine may not support DLA)"
else
    log "SKIP: trtexec=$TRTEXEC engine=$ENGINE_FILE"
fi

# ========== Phase 5: VPI Stereo Benchmark ==========
log ""
log "=== [Phase 5] VPI Stereo Disparity Benchmark ==="

cat > /tmp/vpi_bench.py << 'VPIBENCH'
import vpi
import numpy as np
import time

print("VPI Stereo Disparity Benchmark")
print("=" * 60)

configs = [
    ("Full 1280x720 maxDisp=128", 1280, 720, 128),
    ("Half 640x360 maxDisp=64",   640,  360, 64),
]
backends = [("CUDA", vpi.Backend.CUDA)]
# PVA stereo may not support all configs on VPI 3.x
try:
    backends.append(("PVA", vpi.Backend.PVA))
except:
    pass

for res_name, W, H, maxd in configs:
    for bk_name, bk in backends:
        try:
            left = vpi.asimage(np.random.randint(0, 255, (H, W), dtype=np.uint8))
            right = vpi.asimage(np.random.randint(0, 255, (H, W), dtype=np.uint8))
            # Warmup
            for _ in range(10):
                d = vpi.stereodisp(left, right, backend=bk, window=5, maxdisp=maxd)
            with d.rlock():
                pass
            # Bench
            N = 100
            t0 = time.perf_counter()
            for _ in range(N):
                d = vpi.stereodisp(left, right, backend=bk, window=5, maxdisp=maxd)
            with d.rlock():
                pass
            ms = (time.perf_counter() - t0) / N * 1000
            print(f"  {res_name} / {bk_name}: {ms:.2f} ms  ({1000/ms:.0f} FPS)")
        except Exception as e:
            print(f"  {res_name} / {bk_name}: FAILED ({e})")

print("\nVPI Remap (PVA) 1280x720:")
try:
    img = vpi.asimage(np.random.randint(0, 255, (720, 1280), dtype=np.uint8))
    warp = vpi.WarpMap(vpi.WarpGrid(1280, 720))
    for _ in range(10):
        o = vpi.remap(img, warp, backend=vpi.Backend.PVA, interp=vpi.Interp.LINEAR)
    with o.rlock():
        pass
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        o = vpi.remap(img, warp, backend=vpi.Backend.PVA, interp=vpi.Interp.LINEAR)
    with o.rlock():
        pass
    ms = (time.perf_counter() - t0) / N * 1000
    print(f"  PVA Remap: {ms:.2f} ms  ({1000/ms:.0f} FPS)")
except Exception as e:
    print(f"  PVA Remap: FAILED ({e})")
VPIBENCH

python3 /tmp/vpi_bench.py 2>&1 | tee -a "$RESULT_FILE"

# ========== Phase 6: YOLO Python Inference ==========
log ""
log "=== [Phase 6] YOLO Python Inference Benchmark ==="

cat > /tmp/yolo_bench.py << YOLOBENCH
import sys, os, time
import numpy as np
pt = "$MODEL_PT"
eng = pt.replace('.pt', '.engine')

try:
    from ultralytics import YOLO
    import torch
    print(f"PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    for label, path in [("PT", pt), ("TRT", eng)]:
        if not os.path.exists(path):
            print(f"  {label}: SKIP (file not found: {path})")
            continue
        print(f"\n--- {label}: {os.path.basename(path)} ---")
        model = YOLO(path)
        # Warmup
        for _ in range(10):
            model(dummy, imgsz=320, verbose=False)
        # Bench
        N = 100
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            r = model(dummy, imgsz=320, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        times = sorted(times)
        avg = sum(times) / N
        p50 = times[N//2]
        p95 = times[int(N*0.95)]
        p99 = times[int(N*0.99)]
        print(f"  avg: {avg:.2f} ms  ({1000/avg:.0f} FPS)")
        print(f"  p50: {p50:.2f} ms  p95: {p95:.2f} ms  p99: {p99:.2f} ms")
        print(f"  detections: {len(r[0].boxes)}")

except ImportError as e:
    print(f"SKIP: {e}")
except Exception as e:
    print(f"ERROR: {e}")
YOLOBENCH

python3 /tmp/yolo_bench.py 2>&1 | tee -a "$RESULT_FILE"

# ========== Phase 7: Hardware Utilization Under Load ==========
log ""
log "=== [Phase 7] Hardware Utilization Under TRT Load ==="

if [ -f "$ENGINE_FILE" ] && [ -n "$TRTEXEC" ]; then
    # Start tegrastats in background
    tegrastats --interval 500 > /tmp/tegrastats_load.log 2>&1 &
    TPID=$!

    # Run inference for 15 seconds
    $TRTEXEC --loadEngine="$ENGINE_FILE" --warmUp=2000 --duration=15 --streams=1 \
        > /tmp/trtexec_phase7.log 2>&1

    sleep 1
    kill $TPID 2>/dev/null; wait $TPID 2>/dev/null || true

    log "--- tegrastats (last 10 samples during inference) ---"
    tail -10 /tmp/tegrastats_load.log | tee -a "$RESULT_FILE"
    log ""
    log "--- trtexec summary ---"
    grep -iE "mean|median|throughput|percentile|GPU Compute|Host" /tmp/trtexec_phase7.log | tee -a "$RESULT_FILE"
fi

# ========== Phase 8: Idle baseline ==========
log ""
log "=== [Phase 8] Idle Baseline (tegrastats 3s) ==="
timeout 3 tegrastats --interval 1000 2>/dev/null | tee -a "$RESULT_FILE" || true

# ========== Phase 9: Summary ==========
log ""
log "============================================================"
log " BENCHMARK COMPLETE"
log " Results: $RESULT_FILE"
log " Time: $(date)"
log "============================================================"
log ""
log "Next steps:"
log "  1. Review results: cat $RESULT_FILE"
log "  2. Copy engine: mkdir -p $PIPE_DIR/models && cp $ENGINE_FILE $PIPE_DIR/models/"
log "  3. Update config: sed -i 's|engine_path:.*|engine_path: \"models/yolo26n.engine\"|' $PIPE_DIR/config/pipeline.yaml"
log "  4. Run pipeline: $BUILD_DIR/stereo_pipeline -c $PIPE_DIR/config/pipeline.yaml"
