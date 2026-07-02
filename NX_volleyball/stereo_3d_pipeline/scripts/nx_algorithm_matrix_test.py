#!/usr/bin/env python3
"""Run a short NX realtime algorithm matrix from pipeline_dual_yolo_roi.yaml."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


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
    subpixel_enabled: bool | None = None
    neural_backend: str | None = None
    note: str = ""


CASES = (
    Case("default_geometry", note="bbox/circle/edge/radial/edge_pair geometry candidates"),
    Case("roi_center_patch", {"roi_center_patch": True}),
    Case("roi_subpixel", {"roi_subpixel": True}, subpixel_enabled=True),
    Case("roi_corner_points", {"roi_corner_points": True}),
    Case("roi_texture_points", {"roi_texture_points": True}),
    Case("roi_binary_points", {"roi_binary_points": True}),
    Case("all_sparse_gpu", {"roi_corner_points": True, "roi_texture_points": True, "roi_binary_points": True}),
    Case("roi_orb_points", {"roi_orb_points": True}),
    Case("roi_brisk_points", {"roi_brisk_points": True}),
    Case("roi_akaze_points", {"roi_akaze_points": True}),
    Case("roi_sift_points", {"roi_sift_points": True}),
    Case("roi_iou_region_color_patch", {"roi_iou_region_color_patch": True}),
    Case("roi_patch_iou_color_edge", {"roi_patch_iou_color_edge": True}),
    Case("neural_xfeat", neural_backend="xfeat"),
    Case("neural_aliked", neural_backend="aliked"),
    Case("neural_superpoint_lightglue", neural_backend="superpoint_lightglue"),
)


def set_yaml_bool(text: str, key: str, value: bool) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)(true|false)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{'true' if value else 'false'}{match.group(3)}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing bool key: {key}")
    return new


def set_depth_mode(text: str, key: str, value: bool) -> str:
    if key not in MODE_KEYS:
        raise RuntimeError(f"unknown depth mode: {key}")
    return set_yaml_bool(text, key, value)


def prepare_config(base: str, case: Case, out_dir: Path, config_dir: Path) -> Path:
    text = base
    text = re.sub(r"(\nros2:\n\s*)enable:\s*true", r"\1enable: false", text, count=1)
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
    if case.neural_backend:
        use_lightglue = str(case.neural_backend == "superpoint_lightglue").lower()
        text += f"""

neural_feature_matching:
  enabled: true
  backend: "{case.neural_backend}"
  extractor_engine_path: ""
  matcher_engine_path: ""
  fused_engine_path: ""
  roi_size: 224
  top_k: 128
  descriptor_dim: 64
  min_matches: 8
  max_y_error_px: 2.0
  max_disp_delta_px: 32.0
  final_disp_gate_px: 2.0
  min_score: 0.0
  use_lightglue: {use_lightglue}
"""
    cfg = config_dir / f"{case.name}.yaml"
    cfg.write_text(text)
    return cfg


def parse_log(case: Case, log: str, rc: int, log_path: Path) -> dict[str, str]:
    fps_matches = re.findall(r"\[ROI\] FPS:\s*([0-9.]+).*?stale_drop=([0-9]+)", log)
    stage_gpu = re.findall(
        r"^Stage2_DualYoloGpuCandidates\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
        log,
        re.M,
    )
    stage_match = re.findall(
        r"^Stage2_DualYoloMatch\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
        log,
        re.M,
    )
    subpixel = re.findall(
        r"^Stage2_SubpixelMatch\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
        log,
        re.M,
    )
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
        "gpu_candidates_avg_ms": stage_gpu[-1][0] if stage_gpu else "",
        "gpu_candidates_max_ms": stage_gpu[-1][2] if stage_gpu else "",
        "dual_yolo_match_avg_ms": stage_match[-1][0] if stage_match else "",
        "dual_yolo_match_max_ms": stage_match[-1][2] if stage_match else "",
        "subpixel_avg_ms": subpixel[-1][0] if subpixel else "",
        "subpixel_max_ms": subpixel[-1][2] if subpixel else "",
        "neural_stub_or_unbound": "yes" if neural_unbound else "no",
        "log": str(log_path),
        "note": case.note,
        "last_error_or_warn": error_lines[-1][-220:] if error_lines else "",
    }


def run_case(project: Path, binary: Path, duration_sec: int, case: Case, cfg: Path, log_dir: Path) -> dict[str, str]:
    log_path = log_dir / f"{case.name}.log"
    cmd = ["timeout", str(duration_sec), str(binary), "--config", str(cfg)]
    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=str(project), stdout=log_file, stderr=subprocess.STDOUT)
    return parse_log(case, log_path.read_text(errors="replace"), proc.returncode, log_path)


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
        "",
        "| case | status | rc | fps_last | stale | gpu_avg_ms | gpu_max_ms | match_avg_ms | subpixel_avg_ms | neural_unbound | note | last error/warn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        values = [
            row["case"],
            row["status"],
            row["return_code"],
            row["fps_last"],
            row["stale_drop_last"],
            row["gpu_candidates_avg_ms"],
            row["gpu_candidates_max_ms"],
            row["dual_yolo_match_avg_ms"],
            row["subpixel_avg_ms"],
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project
    base_config = project / "config/pipeline_dual_yolo_roi.yaml"
    binary = project / "build/stereo_pipeline"
    config_dir = args.out / "configs"
    log_dir = args.out / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = base_config.read_text()

    rows: list[dict[str, str]] = []
    for case in CASES:
        cfg = prepare_config(base, case, args.out, config_dir)
        print(f"[RUN] {case.name}", flush=True)
        rows.append(run_case(project, binary, args.duration_sec, case, cfg, log_dir))

    write_reports(args.out, rows, args.duration_sec, project)
    write_static_findings(project, args.out)
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
