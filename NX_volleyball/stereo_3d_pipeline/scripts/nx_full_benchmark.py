#!/usr/bin/env python3
"""NX benchmark: VPI backends + TRT engines + DLA-GPU hybrid"""
import json, os, time, subprocess

MODEL = "/home/nvidia/NX_volleyball/model/yolo26.onnx"
RESULT = "/home/nvidia/NX_volleyball/benchmark_results.json"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
results = {}

# ============================================================
# 1. VPI Backend Comparison (CUDA vs VIC for Remap 1280x720)
# ============================================================
print("=" * 60)
print("[1/4] VPI Backend Comparison (Remap 1280x720)")
print("=" * 60)

import vpi
import numpy as np

W, H = 1280, 720
src_np = np.random.randint(0, 255, (H, W), dtype=np.uint8)

warp = vpi.WarpMap(vpi.WarpGrid((W, H)))

N_WARMUP = 30
N_ITER = 300

backends_map = {
    'CUDA': vpi.Backend.CUDA,
    'VIC':  vpi.Backend.VIC,
}

for bname, backend in backends_map.items():
    try:
        src = vpi.asimage(src_np, format=vpi.Format.U8)
        # Warmup
        for _ in range(N_WARMUP):
            with backend:
                out = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            out.cpu()

        # Single remap benchmark
        times = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            with backend:
                out = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            out.cpu()
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        p50 = sorted(times)[len(times)//2]
        p99 = sorted(times)[int(0.99 * len(times))]
        print(f"  Single {bname:6s}: avg={avg:.3f}ms  min={mn:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms  max={mx:.3f}ms")
        results[f'vpi_single_{bname}'] = dict(avg=avg, min=mn, p50=p50, p99=p99, max=mx)

        # Dual remap (L+R)
        times2 = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            with backend:
                outL = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                outR = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            outR.cpu()
            times2.append((time.perf_counter() - t0) * 1000)

        avg2 = sum(times2) / len(times2)
        mn2 = min(times2)
        p502 = sorted(times2)[len(times2)//2]
        p992 = sorted(times2)[int(0.99 * len(times2))]
        print(f"  Dual   {bname:6s}: avg={avg2:.3f}ms  min={mn2:.3f}ms  p50={p502:.3f}ms  p99={p992:.3f}ms")
        results[f'vpi_dual_{bname}'] = dict(avg=avg2, min=mn2, p50=p502, p99=p992)
    except Exception as e:
        print(f"  {bname:6s}: FAILED - {e}")
        results[f'vpi_single_{bname}'] = dict(error=str(e))

# ============================================================
# 2. TRT Engine Benchmarks (trtexec)
# ============================================================
print("\n" + "=" * 60)
print("[2/4] TRT Engine Benchmarks")
print("=" * 60)

def run_trtexec(engine_path, label):
    cmd = f"{TRTEXEC} --loadEngine={engine_path} --iterations=500 --warmUp=3000 --avgRuns=10 2>&1"
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=120)
    info = {}
    for line in out.split('\n'):
        if 'Throughput' in line:
            val = float(line.split(':')[1].strip().split()[0])
            info['throughput_fps'] = val
            print(f"  [{label}] Throughput: {val:.1f} qps")
        if 'GPU Compute Time' in line and 'mean' in line:
            for p in line.split(','):
                p = p.strip()
                if 'mean' in p:
                    val = float(p.split('=')[1].strip().replace('ms',''))
                    info['gpu_compute_mean'] = val
                if 'min' in p and '=' in p:
                    val = float(p.split('=')[1].strip().replace('ms',''))
                    info['gpu_compute_min'] = val
                if 'max' in p and '=' in p:
                    val = float(p.split('=')[1].strip().replace('ms',''))
                    info['gpu_compute_max'] = val
            print(f"  [{label}] GPU Compute: mean={info.get('gpu_compute_mean',0):.3f}ms min={info.get('gpu_compute_min',0):.3f}ms max={info.get('gpu_compute_max',0):.3f}ms")
        if 'Total Host' in line and 'mean' in line:
            for p in line.split(','):
                p = p.strip()
                if 'mean' in p:
                    val = float(p.split('=')[1].strip().replace('ms',''))
                    info['total_host_mean'] = val
            print(f"  [{label}] Total Host: mean={info.get('total_host_mean',0):.3f}ms")
    results[label] = info
    return info

engines = [
    ('GPU_INT8_640',  '/home/nvidia/NX_volleyball/model/yolo26_gpu_int8_640.engine'),
    ('GPU_FP16_640',  '/home/nvidia/NX_volleyball/model/yolo26_gpu_fp16_640.engine'),
    ('DLA0_INT8_640', '/home/nvidia/NX_volleyball/model/yolo26_dla0_int8_640.engine'),
    ('DLA0_FP16_640', '/home/nvidia/NX_volleyball/model/yolo26_dla0_fp16_640.engine'),
]

for label, engine in engines:
    if os.path.exists(engine):
        try:
            run_trtexec(engine, label)
        except Exception as e:
            print(f"  [{label}] FAILED: {e}")
    else:
        print(f"  [{label}] Not found")

# ============================================================
# 3. DLA-GPU Hybrid Engine Build & Test
# ============================================================
print("\n" + "=" * 60)
print("[3/4] DLA-GPU Hybrid Engine Build & Benchmark")
print("=" * 60)

import onnx
model_onnx = onnx.load(MODEL)
nodes = model_onnx.graph.node
all_names = [n.name for n in nodes if n.name]
attn_names = [n for n in all_names if 'attn' in n.lower()]
print(f"  Total ONNX nodes: {len(all_names)}, Attention nodes: {len(attn_names)}")

# Hybrid A: all attention nodes -> GPU, rest -> DLA
gpu_spec_a = ','.join([f'{n}:GPU' for n in attn_names])

# Hybrid B: attention + head (model.23) -> GPU
head_names = [n for n in all_names if 'model.23' in n]
gpu_spec_b = ','.join([f'{n}:GPU' for n in attn_names + head_names])

# Hybrid C: only backbone (model.0-model.9) on DLA, everything else GPU
backbone_names = [n for n in all_names if any(f'/model.{i}/' in n for i in range(10))]
non_backbone = [n for n in all_names if n not in backbone_names]
gpu_spec_c = ','.join([f'{n}:GPU' for n in non_backbone])

hybrid_configs = [
    ('Hybrid_A', 'attn->GPU rest->DLA', gpu_spec_a,
     '/home/nvidia/NX_volleyball/model/yolo26_hybrid_a_int8.engine'),
    ('Hybrid_B', 'attn+head->GPU backbone+neck->DLA', gpu_spec_b,
     '/home/nvidia/NX_volleyball/model/yolo26_hybrid_b_int8.engine'),
    ('Hybrid_C', 'backbone->DLA rest->GPU', gpu_spec_c,
     '/home/nvidia/NX_volleyball/model/yolo26_hybrid_c_int8.engine'),
]

for name, desc, spec, engine_path in hybrid_configs:
    print(f"\n  --- {name}: {desc} ---")
    cmd = (f"{TRTEXEC} --onnx={MODEL} --useDLACore=0 --allowGPUFallback --int8 --fp16 "
           f"--memPoolSize=workspace:4096MiB "
           f"--saveEngine={engine_path} "
           f"--layerDeviceTypes={spec} "
           f"--skipInference 2>&1")
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, timeout=300)
        if 'PASSED' in out:
            print(f"  Build: PASSED")
            run_trtexec(engine_path, name)
        else:
            print(f"  Build: FAILED")
            # Print last few relevant lines
            for line in out.split('\n'):
                if 'error' in line.lower() or 'warning' in line.lower():
                    print(f"    {line.strip()[:120]}")
            results[name] = dict(error='build failed')
    except subprocess.TimeoutExpired:
        print(f"  Build: TIMEOUT (>300s)")
        results[name] = dict(error='timeout')
    except Exception as e:
        print(f"  Build exception: {e}")
        results[name] = dict(error=str(e))

# ============================================================
# 4. Summary
# ============================================================
print("\n" + "=" * 60)
print("[4/4] Full Pipeline Latency Budget (10ms target)")
print("=" * 60)

vpi_cuda = results.get('vpi_dual_CUDA', {}).get('avg', 99)
vpi_vic  = results.get('vpi_dual_VIC', {}).get('avg', 99)
best_vpi = min(vpi_cuda, vpi_vic)
best_hw  = 'VIC' if vpi_vic < vpi_cuda else 'CUDA'

overhead = 0.5 + best_vpi + 1.3 + 0.1
budget = 10.0 - overhead

print(f"\n  Fixed overhead: grab=0.5 + remap({best_hw})={best_vpi:.2f} + pre/post=1.3 + roi=0.1 = {overhead:.2f}ms")
print(f"  Detection budget: {budget:.2f}ms")
print()
print(f"  {'Config':<16s} {'Infer(ms)':>10s} {'Total(ms)':>10s} {'Fits?':>6s} {'GPU%':>6s}")
print(f"  {'='*16} {'='*10} {'='*10} {'='*6} {'='*6}")

rows = [
    ('GPU INT8',  results.get('GPU_INT8_640',{}).get('gpu_compute_mean', -1), '100%'),
    ('GPU FP16',  results.get('GPU_FP16_640',{}).get('gpu_compute_mean', -1), '100%'),
    ('DLA0 INT8', results.get('DLA0_INT8_640',{}).get('gpu_compute_mean', -1), '~5%'),
    ('DLA0 FP16', results.get('DLA0_FP16_640',{}).get('gpu_compute_mean', -1), '~5%'),
    ('Hybrid A',  results.get('Hybrid_A',{}).get('gpu_compute_mean', -1), '~50%'),
    ('Hybrid B',  results.get('Hybrid_B',{}).get('gpu_compute_mean', -1), '~30%'),
    ('Hybrid C',  results.get('Hybrid_C',{}).get('gpu_compute_mean', -1), '~20%'),
]

for nm, infer, gpu in rows:
    if infer < 0:
        print(f"  {nm:<16s} {'N/A':>10s} {'N/A':>10s} {'N/A':>6s} {gpu:>6s}")
    else:
        total = overhead + infer
        fits = 'YES' if total < 10.0 else 'NO'
        mark = ' <<<' if total < 10.0 else ''
        print(f"  {nm:<16s} {infer:10.2f} {total:10.2f} {fits:>6s} {gpu:>6s}{mark}")

with open(RESULT, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  All results saved to {RESULT}")
print("\nBenchmark complete!")
