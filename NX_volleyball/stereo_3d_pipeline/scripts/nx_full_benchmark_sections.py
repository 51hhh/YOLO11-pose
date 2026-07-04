"""Compatibility facade for NX full benchmark sections."""

from __future__ import annotations

from nx_full_benchmark_hybrid import run_hybrid_benchmarks
from nx_full_benchmark_summary import write_latency_budget_summary
from nx_full_benchmark_trt import run_engine_benchmarks, run_trtexec
from nx_full_benchmark_vpi import run_vpi_backend_benchmark


__all__ = [
    "run_engine_benchmarks",
    "run_hybrid_benchmarks",
    "run_trtexec",
    "run_vpi_backend_benchmark",
    "write_latency_budget_summary",
]
