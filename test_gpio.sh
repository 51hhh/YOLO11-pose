#!/bin/bash
# Test GPIO access on Orin NX
echo "=== GPIO test ==="

# Test gpiochip0 (Tegra native) toggle on an unused line
echo "Testing gpiochip0 line 105..."
gpioset -m time -u 100000 gpiochip0 105=1 &
GPT_PID=$!
sleep 0.2
kill $GPT_PID 2>/dev/null
echo "gpiochip0 toggle: OK"

# Attempt to release gpiochip2 line 7 from kernel hog
echo "Testing gpiochip2 line 7 (camera trigger)..."
gpioset gpiochip2 7=1 2>&1
if [ $? -ne 0 ]; then
    echo "Direct access failed (kernel hog). Creating device tree overlay..."
    # Try using raw ioctl via python
    python3 << 'PYEOF'
import struct, fcntl, os, time

GPIO_GET_LINEHANDLE_IOCTL = 0xC16CB403
GPIOHANDLE_REQUEST_OUTPUT = 0x02
GPIOHANDLE_SET_LINE_VALUES_IOCTL = 0xC040B409

fd = os.open("/dev/gpiochip2", os.O_RDWR)
print(f"Opened /dev/gpiochip2, fd={fd}")

# Request line 7 as output
# struct gpiohandle_request: u32[64] lineoffsets, u32[64] default_values, 
#   char[32] consumer_label, u32 lines, u32 flags, s32 fd
req = struct.pack('=64I64I32s2If', *([7]+[0]*63), *([1]+[0]*63), b'pwm_test', 1, GPIOHANDLE_REQUEST_OUTPUT, 0)
try:
    result = fcntl.ioctl(fd, GPIO_GET_LINEHANDLE_IOCTL, req)
    line_fd = struct.unpack_from('i', result, 64*4+64*4+32+4+4)[0]
    print(f"Got line handle fd={line_fd}")
    
    # Set high
    vals = struct.pack('=64I', *([1]+[0]*63))
    fcntl.ioctl(line_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, vals)
    print("Set GPIO HIGH")
    
    time.sleep(0.1)
    
    # Set low
    vals = struct.pack('=64I', *([0]*64))
    fcntl.ioctl(line_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, vals)
    print("Set GPIO LOW")
    
    os.close(line_fd)
    print("GPIO toggle: SUCCESS")
except OSError as e:
    print(f"GPIO ioctl failed: {e}")

os.close(fd)
PYEOF
fi

echo "=== GPIO test done ==="
