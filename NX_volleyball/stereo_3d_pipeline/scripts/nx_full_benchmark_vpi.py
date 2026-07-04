"""VPI remap benchmark section for NX full benchmark."""

from __future__ import annotations

import time
from typing import Dict

from nx_full_benchmark_common import print_section, summary_stats


def run_vpi_backend_benchmark(results: Dict[str, dict]) -> None:
    print_section("[1/4] VPI Backend Comparison (Remap 1280x720)")

    import numpy as np
    import vpi

    width, height = 1280, 720
    warmup_iters = 30
    benchmark_iters = 300
    src_np = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    warp = vpi.WarpMap(vpi.WarpGrid((width, height)))
    backends_map = {
        "CUDA": vpi.Backend.CUDA,
        "VIC": vpi.Backend.VIC,
    }

    for bname, backend in backends_map.items():
        try:
            src = vpi.asimage(src_np, format=vpi.Format.U8)
            for _ in range(warmup_iters):
                with backend:
                    out = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                out.cpu()

            times = []
            for _ in range(benchmark_iters):
                t0 = time.perf_counter()
                with backend:
                    out = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                out.cpu()
                times.append((time.perf_counter() - t0) * 1000)

            stats = summary_stats(times)
            print(
                f"  Single {bname:6s}: avg={stats['avg']:.3f}ms  min={stats['min']:.3f}ms  "
                f"p50={stats['p50']:.3f}ms  p99={stats['p99']:.3f}ms  max={stats['max']:.3f}ms"
            )
            results[f"vpi_single_{bname}"] = stats

            times2 = []
            for _ in range(benchmark_iters):
                t0 = time.perf_counter()
                with backend:
                    out_l = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                    out_r = vpi.remap(src, warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
                out_r.cpu()
                _ = out_l
                times2.append((time.perf_counter() - t0) * 1000)

            stats2 = summary_stats(times2)
            print(
                f"  Dual   {bname:6s}: avg={stats2['avg']:.3f}ms  min={stats2['min']:.3f}ms  "
                f"p50={stats2['p50']:.3f}ms  p99={stats2['p99']:.3f}ms"
            )
            results[f"vpi_dual_{bname}"] = {
                "avg": stats2["avg"],
                "min": stats2["min"],
                "p50": stats2["p50"],
                "p99": stats2["p99"],
            }
        except Exception as exc:
            print(f"  {bname:6s}: FAILED - {exc}")
            results[f"vpi_single_{bname}"] = {"error": str(exc)}
