#!/usr/bin/env python3
"""Result parsing and reporting for NX algorithm matrix tests."""

from __future__ import annotations

import re
from pathlib import Path

from nx_algorithm_candidate_summary import count_frame_rows, parse_float, summarize_candidate_csv
from nx_algorithm_cases import Case
from nx_algorithm_diagnosis import classify_case_result, parse_int, row_int, should_debug_case
from nx_algorithm_profile import parse_profile_stage, profile_stage_for_case


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


def parse_log(case: Case, log: str, rc: int, log_path: Path) -> dict[str, str]:
    fps_matches = re.findall(r"\[ROI\] FPS:\s*([0-9.]+).*?stale_drop=([0-9]+)", log)

    stage_gpu = parse_profile_stage(log, "Stage2_DualYoloGpuCandidates")
    stage_match = parse_profile_stage(log, "Stage2_DualYoloMatch")
    subpixel = parse_profile_stage(log, "Stage2_SubpixelMatch")
    neural = parse_profile_stage(log, "Stage2_NeuralFeatureMatch")
    opencv_cuda_orb = parse_profile_stage(log, "Stage2_OpenCVCudaORB")
    opencv_cuda_template = parse_profile_stage(log, "Stage2_OpenCVCudaTemplateMatch")
    opencv_cuda_bm = parse_profile_stage(log, "Stage2_OpenCVCudaStereoBM")
    opencv_cuda_sgm = parse_profile_stage(log, "Stage2_OpenCVCudaStereoSGM")
    cpu_opencv = parse_profile_stage(log, "Stage2_CPUFeatureOpenCV")
    cpu_opencv_orb = parse_profile_stage(log, "Stage2_CPUFeatureOpenCVORB")
    cpu_opencv_brisk = parse_profile_stage(log, "Stage2_CPUFeatureOpenCVBRISK")
    cpu_opencv_akaze = parse_profile_stage(log, "Stage2_CPUFeatureOpenCVAKAZE")
    cpu_opencv_sift = parse_profile_stage(log, "Stage2_CPUFeatureOpenCVSIFT")
    cpu_fallback = parse_profile_stage(log, "Stage2_CPUFallbackSearch")
    algo_stage_name = profile_stage_for_case(case)
    algo_stage = parse_profile_stage(log, algo_stage_name)
    async_worker = parse_profile_stage(log, "Stage2_AsyncRoiWorker")
    async_over_deadline = parse_profile_stage(log, "Stage2_AsyncRoiOverDeadline")
    async_drop_stale = parse_profile_stage(log, "Stage2_AsyncRoiDropStaleResult")
    async_drop_stale_ready = parse_profile_stage(log, "Stage2_AsyncRoiDropStaleReady")
    async_drop_expired_pending = parse_profile_stage(log, "Stage2_AsyncRoiDropExpiredPending")
    async_drop_pending = parse_profile_stage(log, "Stage2_AsyncRoiDropPending")
    async_drop_no_buffer = parse_profile_stage(log, "Stage2_AsyncRoiDropNoBuffer")
    async_submit_drop = parse_profile_stage(log, "Stage2_AsyncRoiSubmitDrop")
    async_no_detections = parse_profile_stage(log, "Stage2_AsyncRoiNoDetections")
    async_submitted = parse_profile_stage(log, "Stage2_AsyncRoiSubmitted")
    async_accepted = parse_profile_stage(log, "Stage2_AsyncRoiAccepted")
    async_accepted_reused = parse_profile_stage(log, "Stage2_AsyncRoiAcceptedReusedSlot")
    async_frame_cb_skipped = parse_profile_stage(log, "Stage2_AsyncRoiFrameCallbackSkippedReusedSlot")
    async_need_host_gray = parse_profile_stage(log, "Stage2_AsyncRoiNeedHostGray")
    async_need_bgr = parse_profile_stage(log, "Stage2_AsyncRoiNeedBgr")
    async_host_gray_submit = parse_profile_stage(log, "Stage2_AsyncRoiHostGrayD2HSubmit")
    async_gray_submit = parse_profile_stage(log, "Stage2_AsyncRoiGrayD2DSubmit")
    async_bgr_submit = parse_profile_stage(log, "Stage2_AsyncRoiBgrD2DSubmit")
    async_copy_wait = parse_profile_stage(log, "Stage2_AsyncRoiCopyWait")
    async_slot_copy_wait = parse_profile_stage(log, "Stage2_AsyncRoiSlotCopyWait")
    async_worker_busy = parse_profile_stage(log, "Stage2_AsyncRoiWorkerBusy")
    stage2_drop_stale_roi = parse_profile_stage(log, "Stage2_DropStaleROI")
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
