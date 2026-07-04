"""TensorRT engine benchmark section for NX full benchmark."""

from __future__ import annotations

import os
import subprocess
from typing import Dict

from nx_full_benchmark_common import print_section


def run_trtexec(trtexec: str, results: Dict[str, dict], engine_path: str, label: str) -> dict:
    cmd = f"{trtexec} --loadEngine={engine_path} --iterations=500 --warmUp=3000 --avgRuns=10 2>&1"
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=120)
    info = {}
    for line in out.split("\n"):
        if "Throughput" in line:
            val = float(line.split(":")[1].strip().split()[0])
            info["throughput_fps"] = val
            print(f"  [{label}] Throughput: {val:.1f} qps")
        if "GPU Compute Time" in line and "mean" in line:
            for part in line.split(","):
                part = part.strip()
                if "mean" in part:
                    info["gpu_compute_mean"] = float(part.split("=")[1].strip().replace("ms", ""))
                if "min" in part and "=" in part:
                    info["gpu_compute_min"] = float(part.split("=")[1].strip().replace("ms", ""))
                if "max" in part and "=" in part:
                    info["gpu_compute_max"] = float(part.split("=")[1].strip().replace("ms", ""))
            print(
                f"  [{label}] GPU Compute: mean={info.get('gpu_compute_mean', 0):.3f}ms "
                f"min={info.get('gpu_compute_min', 0):.3f}ms max={info.get('gpu_compute_max', 0):.3f}ms"
            )
        if "Total Host" in line and "mean" in line:
            for part in line.split(","):
                part = part.strip()
                if "mean" in part:
                    info["total_host_mean"] = float(part.split("=")[1].strip().replace("ms", ""))
            print(f"  [{label}] Total Host: mean={info.get('total_host_mean', 0):.3f}ms")
    results[label] = info
    return info


def run_engine_benchmarks(trtexec: str, results: Dict[str, dict]) -> None:
    print_section("[2/4] TRT Engine Benchmarks")
    engines = [
        ("GPU_INT8_640", "/home/nvidia/NX_volleyball/model/yolo26_gpu_int8_640.engine"),
        ("GPU_FP16_640", "/home/nvidia/NX_volleyball/model/yolo26_gpu_fp16_640.engine"),
        ("DLA0_INT8_640", "/home/nvidia/NX_volleyball/model/yolo26_dla0_int8_640.engine"),
        ("DLA0_FP16_640", "/home/nvidia/NX_volleyball/model/yolo26_dla0_fp16_640.engine"),
    ]

    for label, engine in engines:
        if os.path.exists(engine):
            try:
                run_trtexec(trtexec, results, engine, label)
            except Exception as exc:
                print(f"  [{label}] FAILED: {exc}")
        else:
            print(f"  [{label}] Not found")
