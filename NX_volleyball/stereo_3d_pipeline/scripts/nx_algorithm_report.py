#!/usr/bin/env python3
"""Report writers for NX algorithm matrix tests."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from nx_algorithm_results import row_int


def write_reports(out_dir: Path, rows: list[dict[str, str]], duration_sec: int, project: Path) -> None:
    summary_csv = out_dir / "summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# NX algorithm matrix report",
        "",
        f"- project: {project}",
        f"- duration per case: {duration_sec}s",
        "- mode: isolated true algorithms; all dual-YOLO depth modes are disabled before each case enables its own candidate",
        "- `algo_*` is the profiler stage selected for this case; `async_worker_*` is full async Stage2. `p95` is the main tail-latency gate.",
        "- `late_or_deadline_dropped` means the worker finished late or the result was discarded after the next-frame deadline; CUDA/CPU work is not killed mid-kernel.",
        "- `realtime_path_used_cpu_or_host_gray` means the supposedly realtime P2 run triggered CPU fallback or host gray D2H and must be rerun with those paths disabled.",
        "- `debug_dirs` is populated only when `--debug-on-failure` is used.",
        "",
        "| case | diagnosis | status | fps | algo_stage | algo avg/p95/max | worker avg/p95/max | over_deadline | stale/expired | queue_drop | candidate_valid/frames | rate | median/MAD | support | accepted | frame_cb_skip | host_gray | debug_dirs | note | last error/warn |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        stale_expired = (
            row_int(row, "async_drop_stale_count") +
            row_int(row, "async_drop_stale_ready_count") +
            row_int(row, "async_drop_expired_pending_count") +
            row_int(row, "stage2_drop_stale_roi_count")
        )
        queue_drop = (
            row_int(row, "async_drop_pending_count") +
            row_int(row, "async_drop_no_buffer_count") +
            row_int(row, "async_submit_drop_count")
        )
        debug_dirs = ""
        if row.get("debug_feature_dir") or row.get("debug_realtime_dir"):
            debug_dirs = (
                f'feature={row.get("debug_feature_dir", "")} '
                f'realtime={row.get("debug_realtime_dir", "")}'
            ).strip()
        values = [
            row.get("case", ""),
            row.get("diagnosis", ""),
            row.get("status", ""),
            row.get("fps_last", ""),
            row.get("algo_stage", ""),
            f'{row.get("algo_avg_ms", "")}/{row.get("algo_p95_ms", "")}/{row.get("algo_max_ms", "")}',
            f'{row.get("async_worker_avg_ms", "")}/{row.get("async_worker_p95_ms", "")}/{row.get("async_worker_max_ms", "")}',
            row.get("async_over_deadline_count", ""),
            str(stale_expired) if stale_expired else "",
            str(queue_drop) if queue_drop else "",
            f'{row.get("candidate_valid", "")}/{row.get("candidate_rows", "")}'
            if row.get("candidate_rows") else "",
            row.get("candidate_rate", ""),
            f'{row.get("candidate_median_m", "")}/{row.get("candidate_mad_m", "")}',
            row.get("support_median", ""),
            row.get("async_accepted_count", ""),
            row.get("async_frame_callback_skipped_count", ""),
            f'{row.get("async_need_host_gray_count", "")}/{row.get("async_host_gray_submit_count", "")}',
            debug_dirs,
            row.get("note", ""),
            row.get("last_error_or_warn", "").replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def write_static_findings(project: Path, out_dir: Path) -> None:
    files = [
        path
        for path in (project / "src").rglob("*")
        if path.suffix in {".cpp", ".cu", ".h", ".hpp"}
    ]
    combined = "\n".join(path.read_text(errors="ignore") for path in files)
    findings = {
        "neural_matchGpuRoi_stub": "tensor_binding_not_implemented" in combined,
        "realtime_iou_region_color_patch_symbol": "iou_region_color_patch" in combined,
        "realtime_patch_iou_color_edge_symbol": "patch_iou_color_edge" in combined,
        "realtime_cuda_template_match_symbol": "roi_cuda_template_match" in combined,
        "realtime_cuda_stereo_bm_symbol": "roi_cuda_stereo_bm" in combined,
        "realtime_cuda_stereo_sgm_symbol": "roi_cuda_stereo_sgm" in combined,
        "realtime_sift_symbol": re.search(r"\bsift\b", combined, re.I) is not None,
    }
    with (out_dir / "static_findings.txt").open("w") as f:
        for key, value in findings.items():
            f.write(f"{key}={value}\n")
