import vpi, numpy as np, time

W, H = 1280, 720
src_np = np.random.randint(0, 255, (H, W), dtype=np.uint8)

# Check available remap-like functions
funcs = [x for x in dir(vpi) if 'emap' in x.lower() or 'warp' in x.lower()]
print('VPI remap/warp functions:', funcs)

img = vpi.asimage(src_np)
methods = [x for x in dir(img) if 'emap' in x.lower() or 'warp' in x.lower()]
print('Image remap/warp methods:', methods)

# Try Image.remap method
warp = vpi.WarpMap(vpi.WarpGrid((W, H)))

N = 200
for backend_name, backend in [('CUDA', vpi.Backend.CUDA), ('VIC', vpi.Backend.VIC)]:
    try:
        src = vpi.asimage(src_np, format=vpi.Format.U8)
        # warmup
        for _ in range(30):
            with backend:
                out = src.remap(warp)
            out.cpu()

        # single remap
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            with backend:
                out = src.remap(warp)
            out.cpu()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        print(f"  Single {backend_name}: avg={sum(times)/len(times):.3f}ms min={times[0]:.3f}ms p50={times[N//2]:.3f}ms p99={times[int(N*0.99)]:.3f}ms")

        # dual remap (L+R)
        times2 = []
        for _ in range(N):
            t0 = time.perf_counter()
            with backend:
                outL = src.remap(warp)
                outR = src.remap(warp)
            outR.cpu()
            times2.append((time.perf_counter() - t0) * 1000)
        times2.sort()
        print(f"  Dual   {backend_name}: avg={sum(times2)/len(times2):.3f}ms min={times2[0]:.3f}ms p50={times2[N//2]:.3f}ms p99={times2[int(N*0.99)]:.3f}ms")
    except Exception as e:
        print(f"  {backend_name}: FAILED - {e}")
