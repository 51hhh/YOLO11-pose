#!/bin/bash
set -e
cd /home/nvidia/sot_export
EXPORT_DIR=/home/nvidia/sot_export/exported

echo "=========================================="
echo "=== SOT Model Export Pipeline (NX)     ==="
echo "=========================================="

# ------ 1. NanoTrack ------
echo ""
echo "[1/4] NanoTrack: clone repo + download weights + export"
if [ ! -d "nanotrack" ]; then
    git clone --depth 1 https://github.com/nicedaddy/NanoTrack.git nanotrack 2>&1 || {
        echo "[WARN] nicedaddy/NanoTrack not available, trying HonglinChu..."
        git clone --depth 1 https://github.com/nicedaddy/SiamTrackers.git nanotrack_alt 2>&1
        # fallback: use pysot-based nanotrack
    }
fi

# Try multiple NanoTrack weight sources
NANO_CKPT=""
if [ -f "nanotrack_weights/nanotrackv2.pth" ]; then
    NANO_CKPT="nanotrack_weights/nanotrackv2.pth"
elif [ -f "nanotrack/models/nanotrackv2/nanotrackv2.pth" ]; then
    NANO_CKPT="nanotrack/models/nanotrackv2/nanotrackv2.pth"
fi

if [ -n "$NANO_CKPT" ]; then
    echo "  Using checkpoint: $NANO_CKPT"
    python3 export_nanotrack_onnx.py \
        --repo_dir nanotrack \
        --checkpoint "$NANO_CKPT" \
        --out_dir "$EXPORT_DIR" 2>&1
else
    echo "  [INFO] No NanoTrack weights found. Manual download needed:"
    echo "  wget -P nanotrack_weights/ <nanotrackv2.pth URL>"
    echo "  Skipping NanoTrack export for now."
fi

# ------ 2. SiamFC (self-contained, no repo needed) ------
echo ""
echo "[2/4] SiamFC: export (self-contained, random weights for validation)"
python3 export_siamfc.py --out_dir "$EXPORT_DIR" 2>&1
echo "[OK] SiamFC exported (architecture validation)"

# ------ 3. LightTrack ------
echo ""
echo "[3/4] LightTrack: clone repo + export"
if [ ! -d "LightTrack" ]; then
    git clone --depth 1 https://github.com/researchmm/LightTrack.git LightTrack 2>&1 || {
        echo "[WARN] LightTrack clone failed"
    }
fi

LIGHT_CKPT=""
if [ -f "lighttrack_weights/LightTrackM.pth" ]; then
    LIGHT_CKPT="lighttrack_weights/LightTrackM.pth"
fi

if [ -n "$LIGHT_CKPT" ]; then
    echo "  Using checkpoint: $LIGHT_CKPT"
    python3 export_lighttrack.py \
        --repo_dir LightTrack \
        --checkpoint "$LIGHT_CKPT" \
        --out_dir "$EXPORT_DIR" 2>&1
else
    echo "  [INFO] No LightTrack weights found."
    echo "  Manual download: wget -P lighttrack_weights/ <LightTrackM.pth>"
    echo "  Skipping LightTrack export."
fi

# ------ 4. MixFormerV2 ------
echo ""
echo "[4/4] MixFormerV2: clone repo + export"
if [ ! -d "MixFormerV2" ]; then
    pip3 install timm 2>&1 | tail -1
    git clone --depth 1 https://github.com/MCG-NJU/MixFormerV2.git MixFormerV2 2>&1 || {
        echo "[WARN] MixFormerV2 clone failed, trying MixViT..."
        git clone --depth 1 https://github.com/MCG-NJU/MixViT.git MixFormerV2 2>&1
    }
fi

MIX_CKPT=""
if [ -f "mixformer_weights/MixFormerV2_S.pth" ]; then
    MIX_CKPT="mixformer_weights/MixFormerV2_S.pth"
elif [ -f "MixFormerV2/checkpoints/MixFormerV2_S.pth" ]; then
    MIX_CKPT="MixFormerV2/checkpoints/MixFormerV2_S.pth"
fi

if [ -n "$MIX_CKPT" ]; then
    echo "  Using checkpoint: $MIX_CKPT"
    python3 export_mixformerv2_onnx.py \
        --repo_dir MixFormerV2 \
        --checkpoint "$MIX_CKPT" \
        --out_dir "$EXPORT_DIR" 2>&1
else
    echo "  [INFO] No MixFormerV2 weights found."
    echo "  Manual download: wget -P mixformer_weights/ <MixFormerV2_S.pth>"
    echo "  Skipping MixFormerV2 export."
fi

echo ""
echo "=========================================="
echo "=== Export Results                      ==="
echo "=========================================="
ls -lh "$EXPORT_DIR"/*.onnx 2>/dev/null || echo "No ONNX files found"

echo ""
echo "=== Next: trtexec conversion ==="
echo "Run for each ONNX file in $EXPORT_DIR"
