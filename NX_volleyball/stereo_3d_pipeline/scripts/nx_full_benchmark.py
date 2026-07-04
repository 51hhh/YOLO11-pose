#!/usr/bin/env python3
"""NX benchmark: VPI backends + TRT engines + DLA-GPU hybrid."""

from __future__ import annotations

from nx_full_benchmark_sections import (
    run_engine_benchmarks,
    run_hybrid_benchmarks,
    run_vpi_backend_benchmark,
    write_latency_budget_summary,
)


MODEL = "/home/nvidia/NX_volleyball/model/yolo26.onnx"
RESULT = "/home/nvidia/NX_volleyball/benchmark_results.json"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def main() -> int:
    results: dict[str, dict] = {}
    run_vpi_backend_benchmark(results)
    run_engine_benchmarks(TRTEXEC, results)
    run_hybrid_benchmarks(MODEL, TRTEXEC, results)
    write_latency_budget_summary(results, RESULT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
