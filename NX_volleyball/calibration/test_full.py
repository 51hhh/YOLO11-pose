#!/usr/bin/env python3
"""test_camera import + PWM test"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

print("=== test_camera.py libgpiod test ===")

# Test 1: Can import without Jetson.GPIO error
try:
    # Manually run the imports that test_camera.py does
    import gpiod
    print("  gpiod import: OK")
except ImportError as e:
    print(f"  gpiod import: FAIL - {e}")

# Test 2: Test the PWM functions from test_camera
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    
    # Simulate what test_camera does
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_camera", 
        os.path.join(os.path.dirname(__file__), "..", "scripts", "test_camera.py"))
    mod = importlib.util.module_from_spec(spec)
    
    # Just check the source has gpiod not Jetson.GPIO
    with open(os.path.join(os.path.dirname(__file__), "..", "scripts", "test_camera.py")) as f:
        src = f.read()
    
    has_jetson_gpio = "import Jetson.GPIO" in src
    has_gpiod = "import gpiod" in src
    has_gpiochip2 = "gpiochip2" in src
    
    print(f"  Jetson.GPIO reference: {'FAIL - still present!' if has_jetson_gpio else 'OK - removed'}")
    print(f"  gpiod reference: {'OK - present' if has_gpiod else 'FAIL - missing'}")
    print(f"  gpiochip2 reference: {'OK - present' if has_gpiochip2 else 'FAIL - missing'}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: Check capture_chessboard.py also clean
try:
    with open(os.path.join(os.path.dirname(__file__), "capture_chessboard.py")) as f:
        src = f.read()
    
    has_jetson_gpio = "import Jetson.GPIO" in src
    has_gpiod = "import gpiod" in src
    
    print(f"\n  capture_chessboard Jetson.GPIO: {'FAIL!' if has_jetson_gpio else 'OK - removed'}")
    print(f"  capture_chessboard gpiod: {'OK - present' if has_gpiod else 'FAIL - missing'}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: Verify capture_chessboard --help works
print("\n=== capture_chessboard --help ===")
import subprocess
result = subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), "capture_chessboard.py"), "--help"],
    capture_output=True, text=True, timeout=10)
print(result.stdout[:500] if result.stdout else result.stderr[:500])

# Test 5: Check gpiod PWM start/stop
print("\n=== PWM start/stop test ===")
try:
    chip = gpiod.Chip("gpiochip2")
    line = chip.get_line(7)
    if line.is_used():
        print(f"  line 7 busy (consumer: {line.consumer()}), skip toggle")
    else:
        line.request(consumer="test_pwm", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
        import time
        for i in range(5):
            line.set_value(1)
            time.sleep(0.001)
            line.set_value(0)
            time.sleep(0.001)
        line.release()
        print("  PWM toggle (5 cycles): OK")
    chip.close()
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== ALL DONE ===")
