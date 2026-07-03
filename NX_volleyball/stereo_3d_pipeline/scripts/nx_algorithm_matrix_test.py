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
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


MODE_KEYS = {
    "bbox_pair",
    "bbox_edges",
    "circle_center",
    "circle_edges",
    "roi_edge_centroid",
    "roi_radial_center",
    "roi_edge_pair_center",
    "roi_corner_points",
    "roi_texture_points",
    "roi_binary_points",
    "roi_orb_points",
    "roi_brisk_points",
    "roi_akaze_points",
    "roi_sift_points",
    "roi_iou_region_color_patch",
    "roi_patch_iou_color_edge",
    "roi_center_patch",
    "roi_subpixel",
    "epipolar_fallback",
    "fallback_template",
    "fallback_feature_points",
}


@dataclass
class Case:
    name: str
    modes: dict[str, bool] = field(default_factory=dict)
    candidate_fields: tuple[str, ...] = ()
    support_field: str | None = None
    subpixel_enabled: bool | None = None
    yaml_scalars: dict[str, str] = field(default_factory=dict)
    neural_backend: str | None = None
    neural_engine: str | None = None
    roi_size: int = 224
    top_k: int = 128
    descriptor_dim: int = 64
    neural_min_matches: int = 8
    neural_max_y_error_px: float = 2.0
    neural_max_disp_delta_px: float = 32.0
    neural_final_disp_gate_px: float = 2.0
    neural_min_score: float = 0.0
    note: str = ""


CASES = (
    Case(
        "bbox_pair",
        {"bbox_pair": True},
        ("z_bbox_center",),
        note="YOLO bbox-center disparity only; no ROI keypoints",
    ),
    Case(
        "circle_center",
        {"circle_center": True},
        ("z_circle_center",),
        note="circle-fit center disparity only",
    ),
    Case(
        "roi_edge_centroid",
        {"roi_edge_centroid": True},
        ("z_roi_edge_centroid",),
        note="CUDA ROI edge centroid only",
    ),
    Case(
        "roi_radial_center",
        {"roi_radial_center": True},
        ("z_roi_radial_center",),
        note="CUDA radial center only",
    ),
    Case(
        "roi_edge_pair_center",
        {"roi_edge_pair_center": True},
        ("z_roi_edge_pair_center",),
        note="CUDA paired-edge center only",
    ),
    Case(
        "roi_center_patch",
        {"roi_center_patch": True},
        ("z_roi_center_patch",),
        support_field="subpixel_support",
        note="CUDA center-patch ZNCC only",
    ),
    Case(
        "roi_subpixel",
        {"roi_subpixel": True},
        ("z_roi_multi_point",),
        support_field="subpixel_support",
        subpixel_enabled=True,
        note="CUDA multi-point subpixel only",
    ),
    Case(
        "opencv_cuda_orb",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        note="true OpenCV CUDA ORB + CUDA matcher",
    ),
    Case(
        "opencv_cpu_brisk",
        {"roi_brisk_points": True},
        ("z_roi_brisk_points",),
        support_field="roi_brisk_points_support",
        note="true OpenCV CPU BRISK",
    ),
    Case(
        "opencv_cpu_akaze",
        {"roi_akaze_points": True},
        ("z_roi_akaze_points",),
        support_field="roi_akaze_points_support",
        note="true OpenCV CPU AKAZE",
    ),
    Case(
        "opencv_cpu_sift",
        {"roi_sift_points": True},
        ("z_roi_sift_points",),
        support_field="roi_sift_points_support",
        note="true OpenCV CPU SIFT",
    ),
    Case(
        "iou_region_color_patch",
        {"roi_iou_region_color_patch": True},
        ("z_roi_iou_region_color_patch",),
        support_field="roi_iou_region_color_patch_support",
        note="CUDA BGR color IoU/patch candidate",
    ),
    Case(
        "patch_iou_color_edge",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        note="CUDA BGR color-edge patch candidate",
    ),
    Case(
        "neural_xfeat",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_128.engine",
        roi_size=128,
        top_k=64,
        descriptor_dim=64,
        note="TensorRT XFeat 128 extractor; C++ postprocess/mutual-NN",
    ),
    Case(
        "neural_aliked",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="aliked",
        neural_engine="aliked_extractor_224_top128.engine",
        descriptor_dim=128,
    ),
    Case(
        "neural_superpoint_lightglue",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top128.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=256,
        note="TensorRT SuperPoint extractor; direct descriptor matching fallback",
    ),
)

APPROX_CASES = (
    Case(
        "approx_corner_points",
        {"roi_corner_points": True},
        ("z_roi_corner_points",),
        support_field="roi_corner_points_support",
        note="custom CUDA sparse-lite corner; not OpenCV ORB/BRISK/AKAZE/SIFT",
    ),
    Case(
        "approx_texture_points",
        {"roi_texture_points": True},
        ("z_roi_texture_points",),
        support_field="roi_texture_points_support",
        note="custom CUDA sparse-lite texture; diagnostic only",
    ),
    Case(
        "approx_binary_points",
        {"roi_binary_points": True},
        ("z_roi_binary_points",),
        support_field="roi_binary_points_support",
        note="custom CUDA sparse-lite binary; diagnostic only",
    ),
)


RELAXED_CASES = (
    Case(
        "realtime_gpu_bundle",
        {
            "bbox_pair": True,
            "circle_center": True,
            "roi_edge_centroid": True,
            "roi_radial_center": True,
            "roi_edge_pair_center": True,
            "roi_center_patch": True,
            "roi_subpixel": True,
        },
        (
            "z_bbox_center",
            "z_circle_center",
            "z_roi_edge_centroid",
            "z_roi_radial_center",
            "z_roi_edge_pair_center",
            "z_roi_center_patch",
            "z_roi_multi_point",
        ),
        support_field="subpixel_support",
        subpixel_enabled=True,
        note="diagnostic only: all 100fps-capable CUDA depth candidates enabled together",
    ),
    Case(
        "opencv_cuda_orb_relaxed",
        {"roi_orb_points": True},
        ("z_roi_orb_points",),
        support_field="roi_orb_points_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "8.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "6.0",
            "feature_reverse_check_px": "-1.0",
            "feature_overlap_scale": "0.90",
            "feature_mad_scale": "4.0",
            "feature_ransac_gate_px": "3.0",
        },
        note="diagnostic only: true OpenCV CUDA ORB with relaxed gates",
    ),
    Case(
        "iou_region_color_patch_relaxed",
        {"roi_iou_region_color_patch": True},
        ("z_roi_iou_region_color_patch",),
        support_field="roi_iou_region_color_patch_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "8.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "6.0",
            "feature_reverse_check_px": "-1.0",
            "feature_overlap_scale": "0.90",
            "feature_mad_scale": "4.0",
            "feature_ransac_gate_px": "3.0",
        },
        note="diagnostic only: CUDA color IoU/patch with relaxed gates",
    ),
    Case(
        "patch_iou_color_edge_relaxed",
        {"roi_patch_iou_color_edge": True},
        ("z_roi_patch_iou_color_edge",),
        support_field="roi_patch_iou_color_edge_support",
        yaml_scalars={
            "subpixel_min_points": "3",
            "subpixel_min_confidence": "0.10",
            "subpixel_max_disp_delta_px": "8.0",
            "subpixel_max_stddev_px": "3.0",
            "feature_y_tolerance_px": "6.0",
            "feature_reverse_check_px": "-1.0",
            "feature_overlap_scale": "0.90",
            "feature_mad_scale": "4.0",
            "feature_ransac_gate_px": "3.0",
        },
        note="diagnostic only: CUDA color-edge patch with relaxed gates",
    ),
    Case(
        "neural_xfeat_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_128.engine",
        roi_size=128,
        top_k=64,
        descriptor_dim=64,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT XFeat with relaxed gates",
    ),
    Case(
        "neural_xfeat_160",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_160.engine",
        roi_size=160,
        top_k=96,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 160 extractor",
    ),
    Case(
        "neural_xfeat_224",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 224 extractor",
    ),
    Case(
        "neural_xfeat_224_top64",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=64,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 224 extractor top64",
    ),
    Case(
        "neural_xfeat_224_top32",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=32,
        descriptor_dim=64,
        note="diagnostic only: TensorRT XFeat 224 extractor top32",
    ),
    Case(
        "neural_xfeat_224_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="xfeat",
        neural_engine="xfeat_extractor_224.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=64,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT XFeat 224 extractor with relaxed gates",
    ),
    Case(
        "neural_superpoint_lightglue_relaxed",
        candidate_fields=("z_roi_neural_feature",),
        support_field="roi_neural_feature_support",
        neural_backend="superpoint_lightglue",
        neural_engine="superpoint_extractor_224_top128.engine",
        roi_size=224,
        top_k=128,
        descriptor_dim=256,
        neural_min_matches=4,
        neural_max_y_error_px=6.0,
        neural_max_disp_delta_px=96.0,
        neural_final_disp_gate_px=6.0,
        note="diagnostic only: TensorRT SuperPoint extractor with relaxed gates",
    ),
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


def summarize_candidate_csv(case: Case, csv_path: Path, frames_path: Path) -> dict[str, str]:
    empty = {
        "candidate_rows": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
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
    for row in rows:
        row_values = []
        for field in case.candidate_fields:
            value = parse_float(row.get(field))
            if value is not None and value > 0.0:
                field_counts[field] += 1
                row_values.append(value)
        if row_values:
            valid_depths.append(row_values[0])
            if case.support_field:
                support = parse_float(row.get(case.support_field))
                if support is not None and support >= 0.0:
                    supports.append(support)

    valid = len(valid_depths)
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
        "candidate_valid": str(valid),
        "candidate_rate": f"{valid / total:.3f}" if total else "",
        "candidate_median_m": med_s,
        "candidate_mad_m": mad_s,
        "support_median": support_s,
        "field_valids": field_valids,
        "target_rows": str(target_total),
    }


def parse_log(case: Case, log: str, rc: int, log_path: Path) -> dict[str, str]:
    fps_matches = re.findall(r"\[ROI\] FPS:\s*([0-9.]+).*?stale_drop=([0-9]+)", log)

    def stage(name: str) -> tuple[str, str, str, str]:
        matches = re.findall(
            rf"^{re.escape(name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
            log,
            re.M,
        )
        return matches[-1] if matches else ("", "", "", "")

    stage_gpu = stage("Stage2_DualYoloGpuCandidates")
    stage_match = stage("Stage2_DualYoloMatch")
    subpixel = stage("Stage2_SubpixelMatch")
    neural = stage("Stage2_NeuralFeatureMatch")
    async_worker = stage("Stage2_AsyncRoiWorker")
    async_over_deadline = stage("Stage2_AsyncRoiOverDeadline")
    async_drop_stale = stage("Stage2_AsyncRoiDropStaleResult")
    async_accepted = stage("Stage2_AsyncRoiAccepted")
    async_accepted_reused = stage("Stage2_AsyncRoiAcceptedReusedSlot")
    async_frame_cb_skipped = stage("Stage2_AsyncRoiFrameCallbackSkippedReusedSlot")
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
        "neural_avg_ms": neural[0],
        "neural_max_ms": neural[2],
        "async_worker_avg_ms": async_worker[0],
        "async_worker_max_ms": async_worker[2],
        "async_over_deadline_count": async_over_deadline[3],
        "async_drop_stale_count": async_drop_stale[3],
        "async_accepted_count": async_accepted[3],
        "async_accepted_reused_count": async_accepted_reused[3],
        "async_frame_callback_skipped_count": async_frame_cb_skipped[3],
        "candidate_rows": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "target_rows": "",
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
        "neural_avg_ms": "",
        "neural_max_ms": "",
        "async_worker_avg_ms": "",
        "async_worker_max_ms": "",
        "async_over_deadline_count": "",
        "async_drop_stale_count": "",
        "async_accepted_count": "",
        "async_accepted_reused_count": "",
        "async_frame_callback_skipped_count": "",
        "candidate_rows": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "target_rows": "",
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
    return row


def write_reports(out_dir: Path, rows: list[dict[str, str]], duration_sec: int, project: Path) -> None:
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# NX algorithm matrix report",
        "",
        f"- project: {project}",
        f"- duration per case: {duration_sec}s",
        "- mode: isolated true algorithms; all dual-YOLO depth modes are disabled before each case enables its own candidate",
        "",
        "| case | status | rc | fps_last | stale | gpu_avg_ms | match_avg_ms | subpixel_avg_ms | neural_avg_ms | async_worker_avg_ms | over_deadline | candidate_valid/frames | candidate_rate | target_rows | median_m | mad_m | support_med | accepted | frame_cb_skipped | neural_unbound | note | last error/warn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        values = [
            row["case"],
            row["status"],
            row["return_code"],
            row["fps_last"],
            row["stale_drop_last"],
            row["gpu_candidates_avg_ms"],
            row["dual_yolo_match_avg_ms"],
            row["subpixel_avg_ms"],
            row["neural_avg_ms"],
            row["async_worker_avg_ms"],
            row["async_over_deadline_count"],
            f'{row["candidate_valid"]}/{row["candidate_rows"]}'
            if row["candidate_rows"] else "",
            row["candidate_rate"],
            row["target_rows"],
            row["candidate_median_m"],
            row["candidate_mad_m"],
            row["support_median"],
            row["async_accepted_count"],
            row["async_frame_callback_skipped_count"],
            row["neural_stub_or_unbound"],
            row["note"],
            row["last_error_or_warn"].replace("|", "\\|"),
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
        rows.append(run_case(project, binary, args.duration_sec, case, cfg, log_dir))

    write_reports(out_dir, rows, args.duration_sec, project)
    write_static_findings(project, out_dir)
    print(out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
