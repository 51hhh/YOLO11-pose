"""DLA/GPU hybrid build benchmark section for NX full benchmark."""

from __future__ import annotations

import subprocess
from typing import Dict

from nx_full_benchmark_common import print_section
from nx_full_benchmark_trt import run_trtexec


def _hybrid_configs(model_path: str) -> list[tuple[str, str, str, str]]:
    import onnx

    model_onnx = onnx.load(model_path)
    nodes = model_onnx.graph.node
    all_names = [node.name for node in nodes if node.name]
    attn_names = [name for name in all_names if "attn" in name.lower()]
    print(f"  Total ONNX nodes: {len(all_names)}, Attention nodes: {len(attn_names)}")

    gpu_spec_a = ",".join([f"{name}:GPU" for name in attn_names])
    head_names = [name for name in all_names if "model.23" in name]
    gpu_spec_b = ",".join([f"{name}:GPU" for name in attn_names + head_names])
    backbone_names = [name for name in all_names if any(f"/model.{idx}/" in name for idx in range(10))]
    non_backbone = [name for name in all_names if name not in backbone_names]
    gpu_spec_c = ",".join([f"{name}:GPU" for name in non_backbone])

    return [
        ("Hybrid_A", "attn->GPU rest->DLA", gpu_spec_a, "/home/nvidia/NX_volleyball/model/yolo26_hybrid_a_int8.engine"),
        (
            "Hybrid_B",
            "attn+head->GPU backbone+neck->DLA",
            gpu_spec_b,
            "/home/nvidia/NX_volleyball/model/yolo26_hybrid_b_int8.engine",
        ),
        ("Hybrid_C", "backbone->DLA rest->GPU", gpu_spec_c, "/home/nvidia/NX_volleyball/model/yolo26_hybrid_c_int8.engine"),
    ]


def run_hybrid_benchmarks(model_path: str, trtexec: str, results: Dict[str, dict]) -> None:
    print_section("[3/4] DLA-GPU Hybrid Engine Build & Benchmark")

    for name, desc, spec, engine_path in _hybrid_configs(model_path):
        print(f"\n  --- {name}: {desc} ---")
        cmd = (
            f"{trtexec} --onnx={model_path} --useDLACore=0 --allowGPUFallback --int8 --fp16 "
            f"--memPoolSize=workspace:4096MiB "
            f"--saveEngine={engine_path} "
            f"--layerDeviceTypes={spec} "
            f"--skipInference 2>&1"
        )
        try:
            out = subprocess.check_output(cmd, shell=True, text=True, timeout=300)
            if "PASSED" in out:
                print("  Build: PASSED")
                run_trtexec(trtexec, results, engine_path, name)
            else:
                print("  Build: FAILED")
                for line in out.split("\n"):
                    if "error" in line.lower() or "warning" in line.lower():
                        print(f"    {line.strip()[:120]}")
                results[name] = {"error": "build failed"}
        except subprocess.TimeoutExpired:
            print("  Build: TIMEOUT (>300s)")
            results[name] = {"error": "timeout"}
        except Exception as exc:
            print(f"  Build exception: {exc}")
            results[name] = {"error": str(exc)}
