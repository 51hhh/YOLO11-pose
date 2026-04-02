#!/bin/bash
# Fix missing libcusparseLt.so.0 for PyTorch on Jetson
# Create a stub library so torch can load without cusparseLt functionality

set -e

TORCH_LIB=$(python3 -c "import pathlib, torch; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null || true)

if [ -z "$TORCH_LIB" ]; then
    # torch cannot import yet, find it manually
    TORCH_LIB=$(find /home/nvidia/.local/lib -path "*/torch/lib" -type d 2>/dev/null | head -1)
fi

if [ -z "$TORCH_LIB" ]; then
    echo "ERROR: Cannot find torch lib directory"
    exit 1
fi

echo "Torch lib dir: $TORCH_LIB"

if [ -f "$TORCH_LIB/libcusparseLt.so.0" ]; then
    echo "libcusparseLt.so.0 already exists"
    exit 0
fi

# Create a minimal stub .so
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/stub.c" << 'EOF'
// Stub for libcusparseLt.so.0 on Jetson (cusparseLt not available)
void cusparseLtInit(void) {}
EOF

gcc -shared -o "$TMPDIR/libcusparseLt.so.0" "$TMPDIR/stub.c"
cp "$TMPDIR/libcusparseLt.so.0" "$TORCH_LIB/"
rm -rf "$TMPDIR"

echo "Created stub libcusparseLt.so.0 in $TORCH_LIB"

# Verify torch imports now
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
