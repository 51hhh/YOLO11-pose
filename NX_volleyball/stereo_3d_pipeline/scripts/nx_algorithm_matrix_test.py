#!/usr/bin/env python3
"""Run isolated NX realtime algorithm cases from pipeline_dual_yolo_roi.yaml.

Each case disables every dual-YOLO depth mode first, then enables only the
candidate under test. Geometry needed as an internal seed may still run inside
that algorithm, but unrelated depth candidates stay disabled.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from nx_algorithm_config import prepare_config
from nx_algorithm_cases import (
    APPROX_CASES,
    CASES,
    RELAXED_CASES,
    Case,
)
from nx_algorithm_results import (
    classify_case_result,
    parse_log,
    should_debug_case,
    skipped_row,
    summarize_candidate_csv,
)
from nx_algorithm_report import (
    write_reports,
    write_static_findings,
)


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
