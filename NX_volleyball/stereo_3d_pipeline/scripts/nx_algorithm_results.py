#!/usr/bin/env python3
"""Result parsing and reporting for NX algorithm matrix tests."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import median

from nx_algorithm_cases import Case


SKIPPED_ROW_FIELDS = """
case status return_code fps_last stale_drop_last
gpu_candidates_avg_ms gpu_candidates_max_ms dual_yolo_match_avg_ms dual_yolo_match_max_ms
subpixel_avg_ms subpixel_max_ms subpixel_p95_ms subpixel_p99_ms
neural_avg_ms neural_max_ms neural_p95_ms neural_p99_ms
algo_stage algo_avg_ms algo_max_ms algo_p95_ms algo_p99_ms algo_count
opencv_cuda_orb_avg_ms opencv_cuda_orb_max_ms opencv_cuda_orb_p95_ms opencv_cuda_orb_p99_ms
opencv_cuda_template_avg_ms opencv_cuda_template_max_ms opencv_cuda_template_p95_ms opencv_cuda_template_p99_ms
opencv_cuda_stereo_bm_avg_ms opencv_cuda_stereo_bm_max_ms opencv_cuda_stereo_bm_p95_ms opencv_cuda_stereo_bm_p99_ms
opencv_cuda_stereo_sgm_avg_ms opencv_cuda_stereo_sgm_max_ms opencv_cuda_stereo_sgm_p95_ms opencv_cuda_stereo_sgm_p99_ms
cpu_opencv_avg_ms cpu_opencv_max_ms cpu_opencv_orb_avg_ms cpu_opencv_brisk_avg_ms cpu_opencv_akaze_avg_ms cpu_opencv_sift_avg_ms
cpu_fallback_avg_ms cpu_fallback_max_ms cpu_fallback_count
async_worker_avg_ms async_worker_max_ms async_worker_p95_ms async_worker_p99_ms async_worker_count
async_over_deadline_count async_drop_stale_count async_drop_stale_ready_count async_drop_expired_pending_count
async_drop_pending_count async_drop_no_buffer_count async_submit_drop_count async_no_detections_count
async_submitted_count async_accepted_count async_accepted_reused_count async_frame_callback_skipped_count
async_need_host_gray_count async_need_bgr_count async_host_gray_submit_avg_ms async_host_gray_submit_count
async_gray_submit_avg_ms async_bgr_submit_avg_ms async_copy_wait_avg_ms async_slot_copy_wait_avg_ms
async_worker_busy_count stage2_drop_stale_roi_count
candidate_rows candidate_attempted candidate_not_attempted candidate_valid candidate_rate candidate_median_m candidate_mad_m
support_median field_valids candidate_reject_reason target_rows diagnosis
debug_feature_dir debug_realtime_dir debug_feature_rc debug_realtime_rc
neural_stub_or_unbound log note last_error_or_warn
""".split()


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return v


def count_frame_rows(frames_path: Path) -> int:
    if not frames_path.exists():
        return 0
    with frames_path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def profile_stage_for_case(case: Case) -> str:
    if case.neural_backend:
        return "Stage2_NeuralFeatureMatch"
    if case.modes.get("roi_orb_points"):
        return "Stage2_OpenCVCudaORB"
    if case.modes.get("roi_brisk_points"):
        return "Stage2_CPUFeatureOpenCVBRISK"
    if case.modes.get("roi_akaze_points"):
        return "Stage2_CPUFeatureOpenCVAKAZE"
    if case.modes.get("roi_sift_points"):
        return "Stage2_CPUFeatureOpenCVSIFT"
    if case.modes.get("roi_cuda_template_match"):
        return "Stage2_OpenCVCudaTemplateMatch"
    if case.modes.get("roi_cuda_stereo_bm"):
        return "Stage2_OpenCVCudaStereoBM"
    if case.modes.get("roi_cuda_stereo_sgm"):
        return "Stage2_OpenCVCudaStereoSGM"
    if case.modes.get("roi_subpixel"):
        return "Stage2_SubpixelMatch"
    return "Stage2_DualYoloGpuCandidates"


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def row_int(row: dict[str, str], key: str) -> int:
    return parse_int(row.get(key))


def classify_case_result(row: dict[str, str]) -> str:
    status = row.get("status", "")
    if status.startswith("skipped"):
        return "skipped_missing_dependency"
    if status == "failed":
        return "pipeline_failed"

    stale_or_expired = (
        row_int(row, "async_drop_stale_count") +
        row_int(row, "async_drop_stale_ready_count") +
        row_int(row, "async_drop_expired_pending_count") +
        row_int(row, "stage2_drop_stale_roi_count")
    )
    infrastructure_drop = (
        row_int(row, "async_drop_pending_count") +
        row_int(row, "async_drop_no_buffer_count") +
        row_int(row, "async_submit_drop_count")
    )
    if row_int(row, "async_over_deadline_count") > 0 or stale_or_expired > 0:
        return "late_or_deadline_dropped"
    if infrastructure_drop > 0:
        return "async_queue_or_buffer_drop"
    if (
        row_int(row, "cpu_fallback_count") > 0
        or row_int(row, "async_need_host_gray_count") > 0
        or row_int(row, "async_host_gray_submit_count") > 0
    ):
        return "realtime_path_used_cpu_or_host_gray"
    if row_int(row, "async_no_detections_count") > 0 and row_int(row, "target_rows") == 0:
        return "no_detections"
    if row_int(row, "target_rows") == 0 and row_int(row, "async_accepted_count") == 0:
        return "no_accepted_results"
    if row_int(row, "candidate_rows") > 0 and row_int(row, "candidate_valid") == 0:
        return "ran_but_no_valid_candidate"
    if row_int(row, "candidate_valid") > 0:
        return "ok"
    return "needs_log_review"


def should_debug_case(row: dict[str, str]) -> bool:
    return classify_case_result(row) not in {
        "ok",
        "skipped_missing_dependency",
    }


def summarize_candidate_csv(case: Case, csv_path: Path, frames_path: Path) -> dict[str, str]:
    empty = {
        "candidate_rows": "",
        "candidate_attempted": "",
        "candidate_not_attempted": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "candidate_reject_reason": "",
        "target_rows": "",
    }
    frame_total = count_frame_rows(frames_path)
    if not case.candidate_fields or not csv_path.exists():
        return empty

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    target_total = len(rows)
    total = frame_total if frame_total > 0 else target_total
    if target_total == 0 and total == 0:
        return {**empty, "candidate_rows": "0", "target_rows": "0"}

    valid_depths: list[float] = []
    field_counts: dict[str, int] = {field: 0 for field in case.candidate_fields}
    supports: list[float] = []
    attempted = 0
    for row in rows:
        row_values = []
        for field in case.candidate_fields:
            value = parse_float(row.get(field))
            if value is not None and value > 0.0:
                field_counts[field] += 1
                row_values.append(value)
        row_attempted = False
        if case.support_field:
            support = parse_float(row.get(case.support_field))
            if support is not None and support >= 0.0:
                supports.append(support)
                row_attempted = True
        else:
            row_attempted = any(row.get(field) not in (None, "") for field in case.candidate_fields)
        if row_attempted:
            attempted += 1
        if row_values:
            valid_depths.append(row_values[0])

    valid = len(valid_depths)
    not_attempted = max(0, total - attempted)
    if target_total == 0:
        reject_reason = "no_candidate_rows"
    elif attempted == 0:
        reject_reason = "not_attempted"
    elif valid == 0:
        reject_reason = "attempted_no_valid_depth"
    elif valid < attempted:
        reject_reason = "partial_valid"
    else:
        reject_reason = "ok"
    if valid_depths:
        med = median(valid_depths)
        mad = median(abs(v - med) for v in valid_depths)
        med_s = f"{med:.4f}"
        mad_s = f"{mad:.4f}"
    else:
        med_s = ""
        mad_s = ""
    support_s = f"{median(supports):.1f}" if supports else ""
    field_valids = ";".join(f"{k}={v}/{total}" for k, v in field_counts.items())
    return {
        "candidate_rows": str(total),
        "candidate_attempted": str(attempted),
        "candidate_not_attempted": str(not_attempted),
        "candidate_valid": str(valid),
        "candidate_rate": f"{valid / total:.3f}" if total else "",
        "candidate_median_m": med_s,
        "candidate_mad_m": mad_s,
        "support_median": support_s,
        "field_valids": field_valids,
        "candidate_reject_reason": reject_reason,
        "target_rows": str(target_total),
    }


def parse_log(case: Case, log: str, rc: int, log_path: Path) -> dict[str, str]:
    fps_matches = re.findall(r"\[ROI\] FPS:\s*([0-9.]+).*?stale_drop=([0-9]+)", log)

    def stage(name: str) -> tuple[str, str, str, str, str, str, str, str]:
        matches = re.findall(
            rf"^{re.escape(name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)"
            rf"(?:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+))?",
            log,
            re.M,
        )
        if not matches:
            return ("", "", "", "", "", "", "", "")
        match = matches[-1]
        return (
            match[0], match[1], match[2], match[3],
            match[4] or "", match[5] or "", match[6] or "", match[7] or "",
        )

    stage_gpu = stage("Stage2_DualYoloGpuCandidates")
    stage_match = stage("Stage2_DualYoloMatch")
    subpixel = stage("Stage2_SubpixelMatch")
    neural = stage("Stage2_NeuralFeatureMatch")
    opencv_cuda_orb = stage("Stage2_OpenCVCudaORB")
    opencv_cuda_template = stage("Stage2_OpenCVCudaTemplateMatch")
    opencv_cuda_bm = stage("Stage2_OpenCVCudaStereoBM")
    opencv_cuda_sgm = stage("Stage2_OpenCVCudaStereoSGM")
    cpu_opencv = stage("Stage2_CPUFeatureOpenCV")
    cpu_opencv_orb = stage("Stage2_CPUFeatureOpenCVORB")
    cpu_opencv_brisk = stage("Stage2_CPUFeatureOpenCVBRISK")
    cpu_opencv_akaze = stage("Stage2_CPUFeatureOpenCVAKAZE")
    cpu_opencv_sift = stage("Stage2_CPUFeatureOpenCVSIFT")
    cpu_fallback = stage("Stage2_CPUFallbackSearch")
    algo_stage_name = profile_stage_for_case(case)
    algo_stage = stage(algo_stage_name)
    async_worker = stage("Stage2_AsyncRoiWorker")
    async_over_deadline = stage("Stage2_AsyncRoiOverDeadline")
    async_drop_stale = stage("Stage2_AsyncRoiDropStaleResult")
    async_drop_stale_ready = stage("Stage2_AsyncRoiDropStaleReady")
    async_drop_expired_pending = stage("Stage2_AsyncRoiDropExpiredPending")
    async_drop_pending = stage("Stage2_AsyncRoiDropPending")
    async_drop_no_buffer = stage("Stage2_AsyncRoiDropNoBuffer")
    async_submit_drop = stage("Stage2_AsyncRoiSubmitDrop")
    async_no_detections = stage("Stage2_AsyncRoiNoDetections")
    async_submitted = stage("Stage2_AsyncRoiSubmitted")
    async_accepted = stage("Stage2_AsyncRoiAccepted")
    async_accepted_reused = stage("Stage2_AsyncRoiAcceptedReusedSlot")
    async_frame_cb_skipped = stage("Stage2_AsyncRoiFrameCallbackSkippedReusedSlot")
    async_need_host_gray = stage("Stage2_AsyncRoiNeedHostGray")
    async_need_bgr = stage("Stage2_AsyncRoiNeedBgr")
    async_host_gray_submit = stage("Stage2_AsyncRoiHostGrayD2HSubmit")
    async_gray_submit = stage("Stage2_AsyncRoiGrayD2DSubmit")
    async_bgr_submit = stage("Stage2_AsyncRoiBgrD2DSubmit")
    async_copy_wait = stage("Stage2_AsyncRoiCopyWait")
    async_slot_copy_wait = stage("Stage2_AsyncRoiSlotCopyWait")
    async_worker_busy = stage("Stage2_AsyncRoiWorkerBusy")
    stage2_drop_stale_roi = stage("Stage2_DropStaleROI")
    error_lines = [line for line in log.splitlines() if "[ERROR]" in line or "[WARN ]" in line]
    neural_unbound = (
        "does not bind/use NeuralFeatureMatcher outputs yet" in log
        or "tensor_binding_not_implemented" in log
    )
    pipeline_failed = "Pipeline init failed" in log or rc not in (0, 124, 143)
    status = "failed" if pipeline_failed else ("ran_timeout" if rc == 124 else "ran")
    return {
        "case": case.name,
        "status": status,
        "return_code": str(rc),
        "fps_last": fps_matches[-1][0] if fps_matches else "",
        "stale_drop_last": fps_matches[-1][1] if fps_matches else "",
        "gpu_candidates_avg_ms": stage_gpu[0],
        "gpu_candidates_max_ms": stage_gpu[2],
        "dual_yolo_match_avg_ms": stage_match[0],
        "dual_yolo_match_max_ms": stage_match[2],
        "subpixel_avg_ms": subpixel[0],
        "subpixel_max_ms": subpixel[2],
        "subpixel_p95_ms": subpixel[6],
        "subpixel_p99_ms": subpixel[7],
        "neural_avg_ms": neural[0],
        "neural_max_ms": neural[2],
        "neural_p95_ms": neural[6],
        "neural_p99_ms": neural[7],
        "algo_stage": algo_stage_name,
        "algo_avg_ms": algo_stage[0],
        "algo_max_ms": algo_stage[2],
        "algo_p95_ms": algo_stage[6],
        "algo_p99_ms": algo_stage[7],
        "algo_count": algo_stage[3],
        "opencv_cuda_orb_avg_ms": opencv_cuda_orb[0],
        "opencv_cuda_orb_max_ms": opencv_cuda_orb[2],
        "opencv_cuda_orb_p95_ms": opencv_cuda_orb[6],
        "opencv_cuda_orb_p99_ms": opencv_cuda_orb[7],
        "opencv_cuda_template_avg_ms": opencv_cuda_template[0],
        "opencv_cuda_template_max_ms": opencv_cuda_template[2],
        "opencv_cuda_template_p95_ms": opencv_cuda_template[6],
        "opencv_cuda_template_p99_ms": opencv_cuda_template[7],
        "opencv_cuda_stereo_bm_avg_ms": opencv_cuda_bm[0],
        "opencv_cuda_stereo_bm_max_ms": opencv_cuda_bm[2],
        "opencv_cuda_stereo_bm_p95_ms": opencv_cuda_bm[6],
        "opencv_cuda_stereo_bm_p99_ms": opencv_cuda_bm[7],
        "opencv_cuda_stereo_sgm_avg_ms": opencv_cuda_sgm[0],
        "opencv_cuda_stereo_sgm_max_ms": opencv_cuda_sgm[2],
        "opencv_cuda_stereo_sgm_p95_ms": opencv_cuda_sgm[6],
        "opencv_cuda_stereo_sgm_p99_ms": opencv_cuda_sgm[7],
        "cpu_opencv_avg_ms": cpu_opencv[0],
        "cpu_opencv_max_ms": cpu_opencv[2],
        "cpu_opencv_orb_avg_ms": cpu_opencv_orb[0],
        "cpu_opencv_brisk_avg_ms": cpu_opencv_brisk[0],
        "cpu_opencv_akaze_avg_ms": cpu_opencv_akaze[0],
        "cpu_opencv_sift_avg_ms": cpu_opencv_sift[0],
        "cpu_fallback_avg_ms": cpu_fallback[0],
        "cpu_fallback_max_ms": cpu_fallback[2],
        "cpu_fallback_count": cpu_fallback[3],
        "async_worker_avg_ms": async_worker[0],
        "async_worker_max_ms": async_worker[2],
        "async_worker_p95_ms": async_worker[6],
        "async_worker_p99_ms": async_worker[7],
        "async_worker_count": async_worker[3],
        "async_over_deadline_count": async_over_deadline[3],
        "async_drop_stale_count": async_drop_stale[3],
        "async_drop_stale_ready_count": async_drop_stale_ready[3],
        "async_drop_expired_pending_count": async_drop_expired_pending[3],
        "async_drop_pending_count": async_drop_pending[3],
        "async_drop_no_buffer_count": async_drop_no_buffer[3],
        "async_submit_drop_count": async_submit_drop[3],
        "async_no_detections_count": async_no_detections[3],
        "async_submitted_count": async_submitted[3],
        "async_accepted_count": async_accepted[3],
        "async_accepted_reused_count": async_accepted_reused[3],
        "async_frame_callback_skipped_count": async_frame_cb_skipped[3],
        "async_need_host_gray_count": async_need_host_gray[3],
        "async_need_bgr_count": async_need_bgr[3],
        "async_host_gray_submit_avg_ms": async_host_gray_submit[0],
        "async_host_gray_submit_count": async_host_gray_submit[3],
        "async_gray_submit_avg_ms": async_gray_submit[0],
        "async_bgr_submit_avg_ms": async_bgr_submit[0],
        "async_copy_wait_avg_ms": async_copy_wait[0],
        "async_slot_copy_wait_avg_ms": async_slot_copy_wait[0],
        "async_worker_busy_count": async_worker_busy[3],
        "stage2_drop_stale_roi_count": stage2_drop_stale_roi[3],
        "candidate_rows": "",
        "candidate_attempted": "",
        "candidate_not_attempted": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "candidate_reject_reason": "",
        "target_rows": "",
        "diagnosis": "",
        "debug_feature_dir": "",
        "debug_realtime_dir": "",
        "debug_feature_rc": "",
        "debug_realtime_rc": "",
        "neural_stub_or_unbound": "yes" if neural_unbound else "no",
        "log": str(log_path),
        "note": case.note,
        "last_error_or_warn": error_lines[-1][-220:] if error_lines else "",
    }


def skipped_row(case: Case, reason: str, log_path: Path) -> dict[str, str]:
    log_path.write_text(reason + "\n")
    row = {field: "" for field in SKIPPED_ROW_FIELDS}
    row.update({
        "case": case.name,
        "status": "skipped_missing_engine",
        "algo_stage": profile_stage_for_case(case),
        "candidate_reject_reason": reason,
        "diagnosis": "skipped_missing_dependency",
        "neural_stub_or_unbound": "no",
        "log": str(log_path),
        "note": case.note,
        "last_error_or_warn": reason,
    })
    return row
