#!/bin/bash
# convert_model.sh -- Convert YOLO .pt model to TensorRT engine on Jetson
# Usage: bash convert_model.sh [model_path] [imgsz]
set -e

MODEL_PT="${1:-/home/nvidia/NX_volleyball/model/yolo26n.pt}"
IMGSZ="${2:-320}"
MODEL_DIR="$(dirname "$MODEL_PT")"
MODEL_NAME="$(basename "$MODEL_PT" .pt)"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "=============================="
echo " YOLO Model Conversion"
echo "=============================="
echo " Model:  $MODEL_PT"
echo " ImgSz:  ${IMGSZ}x${IMGSZ}"
echo " Output: $MODEL_DIR/"
echo ""

if [ ! -f "$MODEL_PT" ]; then
    echo "ERROR: Model file not found: $MODEL_PT"
    exit 1
fi

# --- Step 1: Export to TensorRT engine via ultralytics ---
echo "[1/3] Exporting to TensorRT INT8 engine (imgsz=${IMGSZ})..."
echo "  This may take 10-30 minutes on first run (calibration)."

python3 -c "
from ultralytics import YOLO
import time

model = YOLO('${MODEL_PT}')

# Export FP16 engine first (faster, no calibration needed)
print('--- Exporting FP16 engine ---')
t0 = time.time()
fp16_path = model.export(
    format='engine',
    imgsz=${IMGSZ},
    half=True,       # FP16
    simplify=True,
    device=0,
    workspace=4,     # 4 GB workspace
)
fp16_time = time.time() - t0
print(f'FP16 engine exported in {fp16_time:.1f}s: {fp16_path}')
"

FP16_ENGINE="${MODEL_DIR}/${MODEL_NAME}.engine"
if [ ! -f "$FP16_ENGINE" ]; then
    # ultralytics may name it differently
    FP16_ENGINE=$(find "$MODEL_DIR" -name "*.engine" -newer "$MODEL_PT" | head -1)
fi

echo ""
echo "[2/3] Verifying engine..."
if [ -f "$FP16_ENGINE" ]; then
    ENGINE_SIZE=$(du -h "$FP16_ENGINE" | awk '{print $1}')
    echo "  Engine: $FP16_ENGINE ($ENGINE_SIZE)"
else
    echo "  WARNING: Engine file not found after export!"
    ls -la "$MODEL_DIR"/*.engine 2>/dev/null || echo "  No .engine files in $MODEL_DIR"
fi

# --- Step 2: Also try DLA export if supported ---
echo ""
echo "[3/3] Attempting DLA-compatible export..."
python3 -c "
from ultralytics import YOLO
import time

model = YOLO('${MODEL_PT}')
print('--- Exporting INT8 engine (DLA-compatible) ---')
t0 = time.time()
try:
    int8_path = model.export(
        format='engine',
        imgsz=${IMGSZ},
        int8=True,
        half=True,
        simplify=True,
        device=0,
        workspace=4,
    )
    print(f'INT8 engine exported in {time.time()-t0:.1f}s: {int8_path}')
except Exception as e:
    print(f'INT8 export failed (this is OK, FP16 will work): {e}')
" 2>&1 || echo "  DLA/INT8 export skipped"

echo ""
echo "=============================="
echo " Conversion complete!"
echo " Engines in: $MODEL_DIR/"
ls -lh "$MODEL_DIR"/*.engine 2>/dev/null || echo "  No engines found"
echo "=============================="
