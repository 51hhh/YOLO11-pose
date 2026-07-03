#!/usr/bin/env python3
"""Run isolated NX realtime algorithm cases from pipeline_dual_yolo_roi.yaml.

Each case disables every dual-YOLO depth mode first, then enables only the
candidate under test. Geometry needed as an internal seed may still run inside
that algorithm, but unrelated depth candidates stay disabled.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from nx_algorithm_cases import (
    APPROX_CASES,
    CASES,
    MODE_KEYS,
    RELAXED_CASES,
    Case,
)

def set_yaml_bool(text: str, key: str, value: bool) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)(true|false)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{'true' if value else 'false'}{match.group(3)}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing bool key: {key}")
    return new


def set_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        suffix = match.group(3)
        if suffix.startswith("#"):
            suffix = " " + suffix
        return f"{match.group(1)}{value}{suffix}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing scalar key: {key}")
    return new


def set_depth_mode(text: str, key: str, value: bool) -> str:
    if key not in MODE_KEYS:
        raise RuntimeError(f"unknown depth mode: {key}")
    return set_yaml_bool(text, key, value)


def disable_all_depth_modes(text: str) -> str:
    for key in sorted(MODE_KEYS):
        text = set_depth_mode(text, key, False)
    return text


def set_neural_enabled(text: str, value: bool) -> str:
    pattern = re.compile(
        r"(^neural_feature_matching:\n(?:^[ \t].*\n)*?^[ \t]*enabled:\s*)(true|false)(.*)$",
        re.M,
    )
    replacement = rf"\g<1>{'true' if value else 'false'}\3"
    text, _ = pattern.subn(replacement, text, count=1)
    return text


def render_neural_block(case: Case, neural_model_dir: Path) -> str:
    use_lightglue = str(case.neural_backend == "superpoint_lightglue").lower()
    extractor_engine_path = ""
    if case.neural_engine:
        extractor_engine_path = str(neural_model_dir / case.neural_engine)
    return f"""neural_feature_matching:
  enabled: true
  backend: "{case.neural_backend}"
  extractor_engine_path: "{extractor_engine_path}"
  matcher_engine_path: ""
  fused_engine_path: ""
  roi_size: {case.roi_size}
  top_k: {case.top_k}
  descriptor_dim: {case.descriptor_dim}
  min_matches: {case.neural_min_matches}
  max_y_error_px: {case.neural_max_y_error_px}
  max_disp_delta_px: {case.neural_max_disp_delta_px}
  final_disp_gate_px: {case.neural_final_disp_gate_px}
  min_score: {case.neural_min_score}
  use_lightglue: {use_lightglue}
"""


def upsert_neural_block(text: str, block: str) -> str:
    pattern = re.compile(r"^neural_feature_matching:\n(?:^[ \t].*\n?)*", re.M)
    new, count = pattern.subn(block.rstrip() + "\n", text, count=1)
    if count:
        return new
    return text.rstrip() + "\n\n" + block


def prepare_config(
    base: str,
    case: Case,
    out_dir: Path,
    config_dir: Path,
    neural_model_dir: Path,
) -> Path:
    text = base
    text = re.sub(r"(\nros2:\n\s*)enable:\s*true", r"\1enable: false", text, count=1)
    text = disable_all_depth_modes(text)
    text = set_yaml_bool(text, "subpixel_enabled", False)
    text = set_yaml_bool(text, "fallback_epipolar_search", False)
    text = set_neural_enabled(text, False)
    text = re.sub(
        r'output_path:\s*"dual_yolo_observation_data\.csv"',
        f'output_path: "{out_dir / (case.name + ".csv")}"',
        text,
        count=1,
    )
    for mode, value in case.modes.items():
        text = set_depth_mode(text, mode, value)
    if case.subpixel_enabled is not None:
        text = set_yaml_bool(text, "subpixel_enabled", case.subpixel_enabled)
    for key, value in case.yaml_scalars.items():
        text = set_yaml_scalar(text, key, value)
    if case.neural_backend:
        text = upsert_neural_block(text, render_neural_block(case, neural_model_dir))
    cfg = config_dir / f"{case.name}.yaml"
    cfg.write_text(text)
    return cfg


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
    return {
        "case": case.name,
        "status": "skipped_missing_engine",
        "return_code": "",
        "fps_last": "",
        "stale_drop_last": "",
        "gpu_candidates_avg_ms": "",
        "gpu_candidates_max_ms": "",
        "dual_yolo_match_avg_ms": "",
        "dual_yolo_match_max_ms": "",
        "subpixel_avg_ms": "",
        "subpixel_max_ms": "",
        "subpixel_p95_ms": "",
        "subpixel_p99_ms": "",
        "neural_avg_ms": "",
        "neural_max_ms": "",
        "neural_p95_ms": "",
        "neural_p99_ms": "",
        "algo_stage": profile_stage_for_case(case),
        "algo_avg_ms": "",
        "algo_max_ms": "",
        "algo_p95_ms": "",
        "algo_p99_ms": "",
        "algo_count": "",
        "opencv_cuda_orb_avg_ms": "",
        "opencv_cuda_orb_max_ms": "",
        "opencv_cuda_orb_p95_ms": "",
        "opencv_cuda_orb_p99_ms": "",
        "opencv_cuda_template_avg_ms": "",
        "opencv_cuda_template_max_ms": "",
        "opencv_cuda_template_p95_ms": "",
        "opencv_cuda_template_p99_ms": "",
        "opencv_cuda_stereo_bm_avg_ms": "",
        "opencv_cuda_stereo_bm_max_ms": "",
        "opencv_cuda_stereo_bm_p95_ms": "",
        "opencv_cuda_stereo_bm_p99_ms": "",
        "opencv_cuda_stereo_sgm_avg_ms": "",
        "opencv_cuda_stereo_sgm_max_ms": "",
        "opencv_cuda_stereo_sgm_p95_ms": "",
        "opencv_cuda_stereo_sgm_p99_ms": "",
        "cpu_opencv_avg_ms": "",
        "cpu_opencv_max_ms": "",
        "cpu_opencv_orb_avg_ms": "",
        "cpu_opencv_brisk_avg_ms": "",
        "cpu_opencv_akaze_avg_ms": "",
        "cpu_opencv_sift_avg_ms": "",
        "cpu_fallback_avg_ms": "",
        "cpu_fallback_max_ms": "",
        "cpu_fallback_count": "",
        "async_worker_avg_ms": "",
        "async_worker_max_ms": "",
        "async_worker_p95_ms": "",
        "async_worker_p99_ms": "",
        "async_worker_count": "",
        "async_over_deadline_count": "",
        "async_drop_stale_count": "",
        "async_drop_stale_ready_count": "",
        "async_drop_expired_pending_count": "",
        "async_drop_pending_count": "",
        "async_drop_no_buffer_count": "",
        "async_submit_drop_count": "",
        "async_no_detections_count": "",
        "async_submitted_count": "",
        "async_accepted_count": "",
        "async_accepted_reused_count": "",
        "async_frame_callback_skipped_count": "",
        "async_need_host_gray_count": "",
        "async_need_bgr_count": "",
        "async_host_gray_submit_avg_ms": "",
        "async_host_gray_submit_count": "",
        "async_gray_submit_avg_ms": "",
        "async_bgr_submit_avg_ms": "",
        "async_copy_wait_avg_ms": "",
        "async_slot_copy_wait_avg_ms": "",
        "async_worker_busy_count": "",
        "stage2_drop_stale_roi_count": "",
        "candidate_rows": "",
        "candidate_attempted": "",
        "candidate_not_attempted": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "candidate_reject_reason": reason,
        "target_rows": "",
        "diagnosis": "skipped_missing_dependency",
        "debug_feature_dir": "",
        "debug_realtime_dir": "",
        "debug_feature_rc": "",
        "debug_realtime_rc": "",
        "neural_stub_or_unbound": "no",
        "log": str(log_path),
        "note": case.note,
        "last_error_or_warn": reason,
    }


def run_case(project: Path, binary: Path, duration_sec: int, case: Case, cfg: Path, log_dir: Path) -> dict[str, str]:
    log_path = log_dir / f"{case.name}.log"
    cmd = ["timeout", str(duration_sec), str(binary), "--config", str(cfg)]
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=str(project), stdout=log_file, stderr=subprocess.STDOUT)
    row = parse_log(case, log_path.read_text(errors="replace"), proc.returncode, log_path)
    out_dir = cfg.parent.parent
    row.update(summarize_candidate_csv(
        case,
        out_dir / f"{case.name}.csv",
        out_dir / f"{case.name}.frames.csv",
    ))
    row["diagnosis"] = classify_case_result(row)
    return row


def run_debug_captures(
    project: Path,
    binary: Path,
    case: Case,
    cfg: Path,
    out_dir: Path,
    log_dir: Path,
    realtime_sec: int,
) -> dict[str, str]:
    debug_root = out_dir / "debug" / case.name
    feature_dir = debug_root / "feature_matches"
    realtime_dir = debug_root / "realtime_zoom"
    feature_dir.mkdir(parents=True, exist_ok=True)
    realtime_dir.mkdir(parents=True, exist_ok=True)

    feature_log = log_dir / f"{case.name}.debug_feature_matches.log"
    feature_cmd = [
        "timeout", "12", str(binary),
        "--config", str(cfg),
        "--debug-feature-matches",
        "--debug-feature-matches-dir", str(feature_dir),
    ]
    with feature_log.open("w") as log_file:
        feature_proc = subprocess.run(
            feature_cmd, cwd=str(project),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    realtime_log = log_dir / f"{case.name}.debug_realtime_dump.log"
    realtime_cmd = [
        "timeout", str(realtime_sec), str(binary),
        "--config", str(cfg),
        "--debug-realtime-dump",
        "--debug-realtime-dump-dir", str(realtime_dir),
        "--debug-realtime-dump-stride", "1",
        "--debug-realtime-dump-max", "20",
    ]
    with realtime_log.open("w") as log_file:
        realtime_proc = subprocess.run(
            realtime_cmd, cwd=str(project),
            stdout=log_file, stderr=subprocess.STDOUT,
        )

    return {
        "debug_feature_dir": str(feature_dir),
        "debug_realtime_dir": str(realtime_dir),
        "debug_feature_rc": str(feature_proc.returncode),
        "debug_realtime_rc": str(realtime_proc.returncode),
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/nvidia/NX_volleyball/stereo_3d_pipeline"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/codex_algo_tests"))
    parser.add_argument("--duration-sec", type=int, default=8)
    parser.add_argument(
        "--include-approx",
        action="store_true",
        help="also run custom sparse-lite diagnostic cases; these are not true OpenCV feature algorithms",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="also run relaxed-gate diagnostic cases; not production-quality results",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case names to run after filtering; default runs all formal cases",
    )
    parser.add_argument(
        "--neural-model-dir",
        type=Path,
        default=Path("/home/nvidia/NX_volleyball/stereo_3d_pipeline/models/neural"),
    )
    parser.add_argument(
        "--debug-on-failure",
        action="store_true",
        help="after a failed/no-valid/deadline case, capture feature debug images and short realtime zoom dump",
    )
    parser.add_argument(
        "--debug-realtime-sec",
        type=int,
        default=5,
        help="seconds for --debug-on-failure realtime zoom dump",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project
    out_dir = args.out.expanduser().resolve()
    base_config = project / "config/pipeline_dual_yolo_roi.yaml"
    binary = project / "build/stereo_pipeline"
    config_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = base_config.read_text()

    rows: list[dict[str, str]] = []
    cases = list(CASES)
    if args.include_approx:
        cases.extend(APPROX_CASES)
    if args.include_experimental:
        cases.extend(RELAXED_CASES)
    if args.cases.strip():
        requested = {
            name.strip()
            for name in args.cases.split(",")
            if name.strip()
        }
        known = {case.name for case in cases}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"unknown case(s): {', '.join(missing)}")
        cases = [case for case in cases if case.name in requested]

    for case in cases:
        if case.neural_engine:
            engine_path = args.neural_model_dir / case.neural_engine
            if not engine_path.exists():
                reason = f"missing neural TensorRT engine: {engine_path}"
                print(f"[SKIP] {case.name}: {reason}", flush=True)
                rows.append(skipped_row(case, reason, log_dir / f"{case.name}.log"))
                continue
        cfg = prepare_config(base, case, out_dir, config_dir, args.neural_model_dir)
        print(f"[RUN] {case.name}", flush=True)
        row = run_case(project, binary, args.duration_sec, case, cfg, log_dir)
        if args.debug_on_failure and should_debug_case(row):
            print(f"[DEBUG] {case.name}: {row['diagnosis']}", flush=True)
            row.update(run_debug_captures(
                project, binary, case, cfg, out_dir, log_dir,
                args.debug_realtime_sec,
            ))
        rows.append(row)

    write_reports(out_dir, rows, args.duration_sec, project)
    write_static_findings(project, out_dir)
    print(out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
