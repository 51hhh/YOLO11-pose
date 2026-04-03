#!/bin/bash
# =============================================================================
# NX Full Test & Benchmark Suite
# Run on Jetson Orin NX to test all pipeline components
# =============================================================================
set -euo pipefail

PROJECT_DIR="$HOME/NX_volleyball/stereo_3d_pipeline"
MODEL_DIR="$HOME/NX_volleyball/model"
BUILD_DIR="$PROJECT_DIR/build"
CONFIG_DIR="$PROJECT_DIR/config"
RESULT_DIR="$PROJECT_DIR/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/MVS/lib/aarch64:${LD_LIBRARY_PATH:-}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$RESULT_DIR"
REPORT="$RESULT_DIR/test_report_${TIMESTAMP}.txt"

log_section() { echo -e "\n${CYAN}========== $1 ==========${NC}" | tee -a "$REPORT"; }
log_pass() { echo -e "  ${GREEN}[PASS]${NC} $1" | tee -a "$REPORT"; }
log_fail() { echo -e "  ${RED}[FAIL]${NC} $1" | tee -a "$REPORT"; }
log_warn() { echo -e "  ${YELLOW}[WARN]${NC} $1" | tee -a "$REPORT"; }
log_info() { echo -e "  [INFO] $1" | tee -a "$REPORT"; }

echo "=== NX Full Test Suite ===" | tee "$REPORT"
echo "Date: $(date)" | tee -a "$REPORT"
echo "Host: $(hostname)" | tee -a "$REPORT"

# ============================================================
# Section 1: System Environment Check
# ============================================================
log_section "1. System Environment"

# Jetson model
JETSON_MODEL=$(cat /sys/module/tegra_fuse/parameters/tegra_chip_id 2>/dev/null || echo "unknown")
log_info "Chip ID: $JETSON_MODEL"
log_info "Kernel: $(uname -r)"
log_info "Memory: $(free -h | awk '/Mem:/{print $2}')"
log_info "Disk free: $(df -h / | awk 'NR==2{print $4}')"

# Power mode
if command -v nvpmodel &>/dev/null; then
    PM=$(sudo nvpmodel -q 2>/dev/null | head -1 || echo "unknown")
    log_info "Power mode: $PM"
    # Set to MAXN for benchmarking
    sudo nvpmodel -m 0 2>/dev/null && log_pass "Set to MAXN mode" || log_warn "Could not set MAXN"
    sudo jetson_clocks 2>/dev/null && log_pass "jetson_clocks enabled" || log_warn "jetson_clocks failed"
fi

# CUDA
if nvcc --version &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $6}' | tr -d ',')
    log_pass "CUDA: $CUDA_VER"
else
    log_fail "CUDA not found"
fi

# TensorRT
if command -v trtexec &>/dev/null; then
    TRT_VER=$(trtexec --help 2>&1 | head -1 | grep -oP 'v\K[0-9]+')
    log_pass "TensorRT: v${TRT_VER}+"
else
    log_fail "trtexec not found"
fi

# VPI
if dpkg -l libnvvpi3 &>/dev/null; then
    VPI_VER=$(dpkg -l libnvvpi3 | awk '/libnvvpi3/{print $3}')
    log_pass "VPI: $VPI_VER"
else
    log_fail "VPI not found"
fi

# OpenCV
CV_VER=$(python3 -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo "not found")
log_info "OpenCV: $CV_VER"

# MVS SDK
if [ -f /opt/MVS/include/MvCameraControl.h ]; then
    log_pass "MVS SDK: /opt/MVS (headers present)"
else
    log_fail "MVS SDK headers not found"
fi

# ============================================================
# Section 2: Build Verification
# ============================================================
log_section "2. Build Verification"

for bin in stereo_pipeline stereo_calibrate capture_chessboard; do
    if [ -x "$BUILD_DIR/$bin" ]; then
        SIZE=$(ls -lh "$BUILD_DIR/$bin" | awk '{print $5}')
        ARCH=$(file "$BUILD_DIR/$bin" | grep -o 'aarch64\|ARM' || echo "?")
        log_pass "$bin ($SIZE, $ARCH)"
    else
        log_fail "$bin not found or not executable"
    fi
done

# ============================================================
# Section 3: Model Check
# ============================================================
log_section "3. Model Files"

for f in yolo26n.pt yolo26n.onnx; do
    if [ -f "$MODEL_DIR/$f" ]; then
        SIZE=$(ls -lh "$MODEL_DIR/$f" | awk '{print $5}')
        log_pass "$f ($SIZE)"
    else
        log_warn "$f not found"
    fi
done

# Check for engine files
for f in "$MODEL_DIR"/*.engine; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        log_pass "$(basename $f) ($SIZE)"
    fi
done

# ============================================================
# Section 4: Camera Test (Hikvision)
# ============================================================
log_section "4. Camera Test (Hikvision MVS)"

# Check if cameras are connected
log_info "Enumerating cameras..."
python3 - <<'CAMTEST' 2>&1 | tee -a "$REPORT"
import sys
sys.path.insert(0, '/opt/MVS/Samples/aarch64/Python/MvImport')
try:
    from MvCameraControl_class import *
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0:
        print(f"  [FAIL] EnumDevices failed: 0x{ret:08X}")
    else:
        n = deviceList.nDeviceNum
        print(f"  [INFO] Found {n} camera(s)")
        for i in range(n):
            dev = deviceList.pDeviceInfo[i]
            if dev.nTLayerType == MV_GIGE_DEVICE:
                info = dev.SpecialInfo.stGigEInfo
                ip = f"{(info.nCurrentIp>>24)&0xFF}.{(info.nCurrentIp>>16)&0xFF}.{(info.nCurrentIp>>8)&0xFF}.{info.nCurrentIp&0xFF}"
                sn = bytes(info.chSerialNumber).decode('ascii','ignore').strip('\0')
                model = bytes(info.chModelName).decode('ascii','ignore').strip('\0')
                print(f"  [PASS] Camera {i}: GigE {model} SN={sn} IP={ip}")
            elif dev.nTLayerType == MV_USB_DEVICE:
                info = dev.SpecialInfo.stUsb3VInfo
                sn = bytes(info.chSerialNumber).decode('ascii','ignore').strip('\0')
                model = bytes(info.chModelName).decode('ascii','ignore').strip('\0')
                print(f"  [PASS] Camera {i}: USB3 {model} SN={sn}")
except ImportError:
    # Try C API through ctypes
    print("  [WARN] Python MVS bindings not found, trying ctypes...")
    import ctypes
    try:
        mvs = ctypes.cdll.LoadLibrary('/opt/MVS/lib/aarch64/libMvCameraControl.so')
        print("  [PASS] libMvCameraControl.so loaded successfully")
    except:
        print("  [FAIL] Could not load MVS library")
except Exception as e:
    print(f"  [FAIL] Camera test error: {e}")
CAMTEST

# Quick grab test using the C++ binary (capture_chessboard with --test)
if [ -x "$BUILD_DIR/capture_chessboard" ]; then
    log_info "Testing camera grab (5 second timeout)..."
    timeout 5 "$BUILD_DIR/capture_chessboard" --test 2>&1 | tail -5 | tee -a "$REPORT" || log_warn "capture_chessboard --test timed out or failed"
fi

# ============================================================
# Section 5: TensorRT Engine Benchmark
# ============================================================
log_section "5. TensorRT Inference Benchmark"

ENGINE_FILE=""
if [ -f "$MODEL_DIR/yolo26_fp16.engine" ]; then
    ENGINE_FILE="$MODEL_DIR/yolo26_fp16.engine"
elif [ -f "$MODEL_DIR/yolo26_dla_fp16.engine" ]; then
    ENGINE_FILE="$MODEL_DIR/yolo26_dla_fp16.engine"
elif [ -f "$MODEL_DIR/yolo26n_fp16.engine" ]; then
    ENGINE_FILE="$MODEL_DIR/yolo26n_fp16.engine"
elif [ -f "$MODEL_DIR/yolo26n_int8.engine" ]; then
    ENGINE_FILE="$MODEL_DIR/yolo26n_int8.engine"
fi

if [ -n "$ENGINE_FILE" ]; then
    log_info "Benchmarking engine: $(basename $ENGINE_FILE)"
    USE_DLA_FLAG="false"
    if [[ "$ENGINE_FILE" == *"dla"* ]]; then
        USE_DLA_FLAG="true"
    fi
    
    # GPU benchmark
    log_info "--- GPU Only ---"
    trtexec --loadEngine="$ENGINE_FILE" \
        --iterations=200 --warmUp=3000 --avgRuns=50 \
        --percentile=50,90,95,99 2>&1 | grep -E "mean|median|Throughput|GPU Compute|percentile" | tee -a "$REPORT"
    
    # DLA benchmark (if engine supports it)
    log_info "--- DLA Core 0 (if available) ---"
    trtexec --loadEngine="$ENGINE_FILE" --useDLACore=0 \
        --iterations=100 --warmUp=2000 --avgRuns=50 2>&1 | grep -E "mean|median|Throughput|percentile" | tee -a "$REPORT" || log_warn "DLA benchmark failed (engine may not support DLA)"
    
elif [ -f "$MODEL_DIR/yolo26.onnx" ]; then
    log_warn "No engine file found, benchmarking from ONNX directly"
    trtexec --onnx="$MODEL_DIR/yolo26.onnx" --fp16 \
        --memPoolSize=workspace:4294967296 \
        --iterations=100 --warmUp=3000 2>&1 | grep -E "mean|median|Throughput|percentile" | tee -a "$REPORT"
else
    log_fail "No model file available for benchmark"
fi

# ============================================================
# Section 6: VPI Component Benchmark
# ============================================================
log_section "6. VPI Component Benchmark"

python3 - <<'VPITEST' 2>&1 | tee -a "$REPORT"
import time
try:
    import vpi
    import numpy as np
    
    print(f"  [INFO] VPI version: {vpi.__version__ if hasattr(vpi, '__version__') else 'unknown'}")
    
    # Test Stereo Disparity
    W, H = 1280, 720
    left = vpi.asimage(np.random.randint(0, 255, (H, W), dtype=np.uint16), format=vpi.Format.U16)
    right = vpi.asimage(np.random.randint(0, 255, (H, W), dtype=np.uint16), format=vpi.Format.U16)
    
    # Warm up
    disparity = vpi.stereo_disparity(left, right, window=5, maxdisp=256, backends=vpi.Backend.CUDA)
    
    # Benchmark
    N = 50
    t0 = time.monotonic()
    for _ in range(N):
        disparity = vpi.stereo_disparity(left, right, window=5, maxdisp=256, backends=vpi.Backend.CUDA)
    t1 = time.monotonic()
    avg_ms = (t1 - t0) / N * 1000
    print(f"  [INFO] VPI Stereo Disparity (CUDA, 1280x720, maxdisp=128): {avg_ms:.2f} ms/frame")
    
    # Half-res test
    left_half = vpi.asimage(np.random.randint(0, 255, (360, 640), dtype=np.uint16), format=vpi.Format.U16)
    right_half = vpi.asimage(np.random.randint(0, 255, (360, 640), dtype=np.uint16), format=vpi.Format.U16)
    
    disparity_half = vpi.stereo_disparity(left_half, right_half, window=5, maxdisp=64, backends=vpi.Backend.CUDA)
    t0 = time.monotonic()
    for _ in range(N):
        disparity_half = vpi.stereo_disparity(left_half, right_half, window=5, maxdisp=64, backends=vpi.Backend.CUDA)
    t1 = time.monotonic()
    avg_half = (t1 - t0) / N * 1000
    print(f"  [INFO] VPI Stereo Disparity (CUDA, 640x360, maxdisp=64):   {avg_half:.2f} ms/frame")
    print(f"  [INFO] Half-res speedup: {avg_ms/avg_half:.1f}x")
    
    # PVA Remap test (if available)
    try:
        import cv2
        map_x = np.random.rand(H, W).astype(np.float32) * (W - 1)
        map_y = np.random.rand(H, W).astype(np.float32) * (H - 1)
        src = vpi.asimage(np.random.randint(0, 255, (H, W), dtype=np.uint8))
        warp = vpi.WarpMap(vpi.WarpGrid((W, H)))
        
        t0 = time.monotonic()
        for _ in range(N):
            out = vpi.remap(src, warp, backends=vpi.Backend.CUDA)
        t1 = time.monotonic()
        avg_remap = (t1 - t0) / N * 1000
        print(f"  [INFO] VPI Remap (CUDA, 1280x720): {avg_remap:.2f} ms/frame")
    except Exception as e:
        print(f"  [WARN] VPI Remap test failed: {e}")
        
except ImportError:
    print("  [WARN] VPI Python bindings not available, skipping")
except Exception as e:
    print(f"  [FAIL] VPI test error: {e}")
VPITEST

# ============================================================
# Section 7: System Resource Monitoring
# ============================================================
log_section "7. Hardware Utilization Snapshot"

# tegrastats snapshot
if command -v tegrastats &>/dev/null; then
    log_info "tegrastats (5 second sample):"
    timeout 5 tegrastats --interval 1000 2>/dev/null | head -5 | tee -a "$REPORT" || log_warn "tegrastats failed"
fi

# jtop info
if command -v jtop &>/dev/null; then
    log_info "jtop available for interactive monitoring: run 'jtop'"
fi

# GPU utilization
if [ -f /sys/devices/gpu.0/load ]; then
    GPU_LOAD=$(cat /sys/devices/gpu.0/load)
    log_info "GPU load: $GPU_LOAD"
fi

# DLA status
for i in 0 1; do
    DLA_DEV="/dev/nvhost-nvdla${i}"
    if [ -e "$DLA_DEV" ]; then
        log_pass "DLA Core $i: $DLA_DEV exists"
    fi
done

# ============================================================
# Section 8: Pipeline Integration Test
# ============================================================
log_section "8. Pipeline Integration Test"

if [ -x "$BUILD_DIR/stereo_pipeline" ] && [ -n "$ENGINE_FILE" ]; then
    # Create a test config with the actual engine path
    TEST_CONFIG="$RESULT_DIR/test_pipeline.yaml"
    cat > "$TEST_CONFIG" <<YAMLEOF
camera:
  left_index: 0
  right_index: 1
  serial_left: ""
  serial_right: ""
  exposure_us: 3000.0
  gain_db: 0.0
  use_trigger: false
  trigger_source: "Software"
  trigger_activation: "RisingEdge"
  width: 1440
  height: 1080

calibration:
  file: "calibration/stereo_calib.yaml"

rectify:
  output_width: 1280
  output_height: 720

detector:
  engine_path: "$ENGINE_FILE"
  input_size: 640
  use_dla: $USE_DLA_FLAG
  dla_core: 0
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 10

stereo:
  max_disparity: 256
  window_size: 5
  quality: 6
  use_half_resolution: false

fusion:
  min_depth: 0.3
  max_depth: 15.0

performance:
  pwm_frequency: 100.0
  log_interval: 50
YAMLEOF

    log_info "Starting pipeline (10 second test)..."
    timeout 10 "$BUILD_DIR/stereo_pipeline" --config "$TEST_CONFIG" 2>&1 | tee -a "$REPORT" || true
    log_info "Pipeline test completed"
else
    log_warn "Pipeline binary or engine not available, skipping integration test"
fi

# ============================================================
# Section 9: Summary
# ============================================================
log_section "9. Test Summary"

PASS_COUNT=$(grep -c '\[PASS\]' "$REPORT" || echo 0)
FAIL_COUNT=$(grep -c '\[FAIL\]' "$REPORT" || echo 0)
WARN_COUNT=$(grep -c '\[WARN\]' "$REPORT" || echo 0)

echo -e "${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  ${YELLOW}WARN: $WARN_COUNT${NC}" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"
echo "Full report saved to: $REPORT" | tee -a "$REPORT"
echo "=== Test Complete ===" | tee -a "$REPORT"
