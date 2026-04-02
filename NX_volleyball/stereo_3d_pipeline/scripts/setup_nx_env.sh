#!/bin/bash
# setup_nx_env.sh -- Setup Jetson Orin NX environment for volleyball tracking
# Run on NX: bash ~/NX_volleyball/stereo_3d_pipeline/scripts/setup_nx_env.sh
set -e

echo "=============================="
echo " Jetson Orin NX Environment Setup"
echo "=============================="

# --- 1. System info ---
echo ""
echo "[1/6] System Info"
cat /etc/nv_tegra_release | head -1
echo "CUDA: $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "TensorRT: $(dpkg -l | grep libnvinfer10 | awk '{print $3}' | head -1)"
echo "OpenCV: $(pkg-config --modversion opencv4 2>/dev/null || echo 'N/A')"
echo "VPI: $(dpkg -l | grep libnvvpi3 | awk '{print $3}' | head -1)"

# --- 2. Add CUDA to PATH ---
echo ""
echo "[2/6] Configuring CUDA PATH"
if ! grep -q 'cuda' ~/.bashrc 2>/dev/null; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "  Added CUDA to ~/.bashrc"
else
    echo "  CUDA PATH already configured"
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# --- 3. Install pip packages ---
echo ""
echo "[3/6] Installing Python packages"
pip3 install --upgrade pip 2>/dev/null || true

# Check if torch is installed; if not, install Jetson-compatible version
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  Installing PyTorch for JetPack 6..."
    # JetPack 6.x / L4T R36 uses torch 2.5+ for CUDA 12.6
    # Official NVIDIA wheel for Jetson:
    pip3 install --no-cache-dir \
        torch torchvision \
        --index-url https://pypi.nvidia.com \
        2>/dev/null || \
    pip3 install --no-cache-dir \
        torch torchvision \
        --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v61 \
        2>/dev/null || \
    echo "  WARNING: Auto-install failed. Install torch manually from https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
else
    echo "  PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
fi

# Install ultralytics for YOLO model export
pip3 install --no-cache-dir ultralytics 2>/dev/null || echo "  WARNING: ultralytics install failed"

# Verify
echo "  Verifying packages..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch: NOT AVAILABLE"
python3 -c "import ultralytics; print(f'  Ultralytics {ultralytics.__version__}')" 2>/dev/null || echo "  Ultralytics: NOT AVAILABLE"

# --- 4. Install build dependencies ---
echo ""
echo "[4/6] Installing build dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    libyaml-cpp-dev \
    libgpiod-dev \
    2>/dev/null

# --- 5. Set max performance mode ---
echo ""
echo "[5/6] Setting performance mode"
sudo nvpmodel -m 0 2>/dev/null && echo "  nvpmodel: MAXN (mode 0)" || echo "  nvpmodel set failed (may need reboot)"
sudo jetson_clocks 2>/dev/null && echo "  jetson_clocks: locked to max frequency" || echo "  jetson_clocks failed"

# --- 6. Verify critical libs ---
echo ""
echo "[6/6] Verifying libraries"
# VPI
ldconfig -p 2>/dev/null | grep -q libnvvpi && echo "  VPI: OK" || echo "  VPI: NOT FOUND"
# TensorRT
ldconfig -p 2>/dev/null | grep -q libnvinfer && echo "  TensorRT: OK" || echo "  TensorRT: NOT FOUND"
# OpenCV
ldconfig -p 2>/dev/null | grep -q libopencv_core && echo "  OpenCV: OK" || echo "  OpenCV: NOT FOUND"
# CUDA runtime
ldconfig -p 2>/dev/null | grep -q libcudart && echo "  CUDA Runtime: OK" || echo "  CUDA Runtime: NOT FOUND"
# yaml-cpp
ldconfig -p 2>/dev/null | grep -q libyaml-cpp && echo "  yaml-cpp: OK" || echo "  yaml-cpp: NOT FOUND"

echo ""
echo "=============================="
echo " Setup complete!"
echo "=============================="
