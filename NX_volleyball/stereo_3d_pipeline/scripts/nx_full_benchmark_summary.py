"""Latency budget summary section for NX full benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from nx_full_benchmark_common import print_section


def write_latency_budget_summary(results: Dict[str, dict], result_path: str | Path) -> None:
    print_section("[4/4] Full Pipeline Latency Budget (10ms target)")

    vpi_cuda = results.get("vpi_dual_CUDA", {}).get("avg", 99)
    vpi_vic = results.get("vpi_dual_VIC", {}).get("avg", 99)
    best_vpi = min(vpi_cuda, vpi_vic)
    best_hw = "VIC" if vpi_vic < vpi_cuda else "CUDA"

    overhead = 0.5 + best_vpi + 1.3 + 0.1
    budget = 10.0 - overhead

    print(f"\n  Fixed overhead: grab=0.5 + remap({best_hw})={best_vpi:.2f} + pre/post=1.3 + roi=0.1 = {overhead:.2f}ms")
    print(f"  Detection budget: {budget:.2f}ms")
    print()
    print(f"  {'Config':<16s} {'Infer(ms)':>10s} {'Total(ms)':>10s} {'Fits?':>6s} {'GPU%':>6s}")
    print(f"  {'='*16} {'='*10} {'='*10} {'='*6} {'='*6}")

    rows = [
        ("GPU INT8", results.get("GPU_INT8_640", {}).get("gpu_compute_mean", -1), "100%"),
        ("GPU FP16", results.get("GPU_FP16_640", {}).get("gpu_compute_mean", -1), "100%"),
        ("DLA0 INT8", results.get("DLA0_INT8_640", {}).get("gpu_compute_mean", -1), "~5%"),
        ("DLA0 FP16", results.get("DLA0_FP16_640", {}).get("gpu_compute_mean", -1), "~5%"),
        ("Hybrid A", results.get("Hybrid_A", {}).get("gpu_compute_mean", -1), "~50%"),
        ("Hybrid B", results.get("Hybrid_B", {}).get("gpu_compute_mean", -1), "~30%"),
        ("Hybrid C", results.get("Hybrid_C", {}).get("gpu_compute_mean", -1), "~20%"),
    ]

    for name, infer, gpu in rows:
        if infer < 0:
            print(f"  {name:<16s} {'N/A':>10s} {'N/A':>10s} {'N/A':>6s} {gpu:>6s}")
        else:
            total = overhead + infer
            fits = "YES" if total < 10.0 else "NO"
            mark = " <<<" if total < 10.0 else ""
            print(f"  {name:<16s} {infer:10.2f} {total:10.2f} {fits:>6s} {gpu:>6s}{mark}")

    with Path(result_path).open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  All results saved to {result_path}")
    print("\nBenchmark complete!")
