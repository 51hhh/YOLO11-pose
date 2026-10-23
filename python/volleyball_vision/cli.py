"""统一的 volleyball 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from .capture import finalize_capture_stage, run_capture
from .config import ConfigError, flatten, load_profile
from .dataset import DatasetError, validate_dataset, validate_trial
from .doctor import checks_as_dicts, run_doctor
from .mainline_config import build_mainline_pipeline_config
from .replay import run_replay
from .runtime import run_runtime


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="configs/profiles 下的 profile 名")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="仅覆盖已存在的配置键，可重复",
    )


def _config_validate(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    print(f"OK profile={config.profile} sha256={config.fingerprint_sha256}")
    for layer in config.layers:
        print(f"  {layer}")
    return 0


def _config_render(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    if args.format == "json":
        print(json.dumps(config.data, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(yaml.safe_dump(dict(config.data), allow_unicode=True, sort_keys=False), end="")
    print(f"# config_sha256: {config.fingerprint_sha256}", file=sys.stderr)
    return 0


def _config_explain(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    value = config.value(args.key)
    print(yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip())
    print(f"source: {config.source(args.key)}")
    return 0


def _config_diff(args: argparse.Namespace) -> int:
    left = load_profile(args.left)
    right = load_profile(args.right)
    left_values = flatten(left.data)
    right_values = flatten(right.data)
    changed = False
    for key in sorted(set(left_values) | set(right_values)):
        lhs = left_values.get(key, "<missing>")
        rhs = right_values.get(key, "<missing>")
        if lhs != rhs:
            changed = True
            print(f"{key}: {lhs!r} -> {rhs!r}")
    if not changed:
        print("无差异")
    return 0


def _config_render_mainline(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    payload = build_mainline_pipeline_config(
        config,
        capture_stage=args.capture_stage,
        frame_limit=args.frame_limit,
    )
    print(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), end="")
    print(f"# config_sha256: {config.fingerprint_sha256}", file=sys.stderr)
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    checks = run_doctor(config)
    if args.json:
        print(json.dumps(checks_as_dicts(checks), ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(f"{check.status.upper():7s} {check.name}: {check.message}")
    return 0 if all(check.ok for check in checks) else 3


def _run(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    # dry-run 只生成可审计配置和命令，不接触硬件；开发机也应能使用。
    # 真正启动时仍必须通过完整的模型、NX、GPIO、MVS、CUDA 与 ROS2 门禁。
    if not args.dry_run:
        checks = run_doctor(config)
        errors = [check for check in checks if not check.ok]
        if errors:
            details = "\n".join(f"  - {check.name}: {check.message}" for check in errors)
            raise DatasetError(f"启动前 doctor 未通过:\n{details}")
    result = run_runtime(
        config,
        generated_config=args.generated_config,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"DRY_RUN config={result.generated_config}")
        print(result.command_text)
    else:
        print(f"OK runtime returncode={result.returncode}")
    return 0


def _load_conditions(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"conditions JSON 无效: {source}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetError("conditions JSON 顶层必须是对象")
    return value


def _capture(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    # 预演只生成不可发布的 staging 与主线配置，不接触相机、GPIO 或模型文件。
    # 真采集仍必须通过完整 doctor 门禁。
    if not args.dry_run:
        checks = run_doctor(config)
        errors = [check for check in checks if not check.ok]
        if errors:
            details = "\n".join(
                f"  - {check.name}: {check.message}" for check in errors
            )
            raise DatasetError(f"采集前 doctor 未通过:\n{details}")
    output = run_capture(
        config,
        trial_id=args.trial,
        session_id=args.session,
        ball_id=args.ball,
        duration_s=args.duration,
        conditions=_load_conditions(args.conditions),
        notes=args.notes,
        dataset_root=args.dataset_root,
        keep_stage=args.keep_stage,
        dry_run=args.dry_run,
    )
    label = "DRY_RUN stage" if args.dry_run else "OK trial"
    print(f"{label}={output}")
    return 0


def _capture_finalize(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    output = finalize_capture_stage(args.stage, args.dataset_root, config)
    print(f"OK trial={output}")
    return 0


def _dataset_validate(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    verify_hashes = not args.no_verify_hashes
    if (path / "metadata.json").is_file():
        summary = validate_trial(path, verify_hashes=verify_hashes)
        print(
            f"OK trial={summary.trial_id} "
            f"rows={sum(summary.table_rows.values())} raw_files={summary.raw_files}"
        )
        return 0
    summary = validate_dataset(path, verify_hashes=verify_hashes)
    print(
        f"OK dataset={summary.root} calibrations={summary.calibrations} "
        f"trials={len(summary.trials)}"
    )
    for trial in summary.trials:
        print(
            f"  {trial.trial_id}: rows={sum(trial.table_rows.values())} "
            f"raw_files={trial.raw_files}"
        )
    if summary.partial_directories:
        print(
            "  WARNING unpublished partial directories: "
            + ", ".join(summary.partial_directories),
            file=sys.stderr,
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    config = load_profile(args.profile, overrides=args.set)
    result = run_replay(
        args.trial,
        config,
        shadow_trajectory=args.shadow,
        output_root=args.output_root,
        max_time_delta_ms=args.max_time_delta_ms,
        verify_hashes=not args.no_verify_hashes,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "output_directory": str(result.output_directory),
                    "fingerprint_sha256": result.fingerprint_sha256,
                    "reused": result.reused,
                    "metrics": result.metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        quality = result.metrics["quality"]
        print(
            f"OK replay={result.output_directory} "
            f"observations={quality['direct_observations']} "
            f"pairs={quality['accepted_pairs']}/{quality['pair_attempts']} "
            f"reused={str(result.reused).lower()}"
        )
        comparison = result.metrics.get("shadow_comparison")
        if comparison is not None:
            error = comparison["position_error_m"]
            print(
                f"  shadow matched={comparison['matched_samples']}/"
                f"{comparison['reference_samples']} "
                f"rmse_m={error['rmse']} p95_m={error['p95']}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volleyball")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("config", help="校验和解析分层配置")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)

    validate = config_commands.add_parser("validate", help="校验配置")
    _add_profile_arguments(validate)
    validate.set_defaults(handler=_config_validate)

    render = config_commands.add_parser("render", help="输出最终合并配置")
    _add_profile_arguments(render)
    render.add_argument("--format", choices=("yaml", "json"), default="yaml")
    render.set_defaults(handler=_config_render)

    explain = config_commands.add_parser("explain", help="解释某键的值和来源")
    _add_profile_arguments(explain)
    explain.add_argument("key", help="点分隔配置键")
    explain.set_defaults(handler=_config_explain)

    diff = config_commands.add_parser("diff", help="对比两个 profile")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.set_defaults(handler=_config_diff)

    render_mainline = config_commands.add_parser(
        "render-mainline", help="查看投影到原生 NX 主线后的最小 YAML"
    )
    _add_profile_arguments(render_mainline)
    render_mainline.add_argument(
        "--capture-stage",
        help="capture profile 的绝对 staging 路径",
    )
    render_mainline.add_argument(
        "--frame-limit",
        type=int,
        help="capture profile 的有限双目帧数",
    )
    render_mainline.set_defaults(handler=_config_render_mainline)

    doctor = subparsers.add_parser("doctor", help="只读检查配置、模型、标定和运行环境")
    _add_profile_arguments(doctor)
    doctor.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    doctor.set_defaults(handler=_doctor)

    runtime = subparsers.add_parser("run", help="自检后启动 NX 实时管线")
    _add_profile_arguments(runtime)
    runtime.add_argument(
        "--generated-config",
        help="保留投影后的管线 YAML；为防误覆盖，路径必须不存在",
    )
    runtime.add_argument("--dry-run", action="store_true", help="只生成配置并显示命令")
    runtime.set_defaults(handler=_run)

    capture = subparsers.add_parser("capture", help="执行无丢帧 v2.1 双目采集")
    _add_profile_arguments(capture)
    capture.add_argument("--trial", required=True, help="不可复用的 trial ID")
    capture.add_argument("--session", required=True, help="采集 session ID")
    capture.add_argument("--ball", required=True, help="实体排球 ID")
    capture.add_argument("--duration", type=float, default=3.0, help="采集秒数，默认 3")
    capture.add_argument("--conditions", help="球压、温湿度等 JSON 文件")
    capture.add_argument("--notes", default="")
    capture.add_argument("--dataset-root", help="覆盖 profile 中的数据集根目录")
    capture.add_argument("--keep-stage", action="store_true", help="成功后仍保留中间文件")
    capture.add_argument("--dry-run", action="store_true", help="只生成 staging 与管线配置")
    capture.set_defaults(handler=_capture)

    capture_finalize = subparsers.add_parser(
        "capture-finalize", help="恢复并发布一次已完整结束的 staging"
    )
    _add_profile_arguments(capture_finalize)
    capture_finalize.add_argument("--stage", required=True)
    capture_finalize.add_argument("--dataset-root", required=True)
    capture_finalize.set_defaults(handler=_capture_finalize)

    replay = subparsers.add_parser(
        "replay",
        help="确定性读取真实 trial，并可比较影子仿真轨迹",
    )
    _add_profile_arguments(replay)
    replay.add_argument("--trial", required=True, help="v2.1 trial 目录")
    replay.add_argument(
        "--shadow",
        help="待评估轨迹 CSV，至少含 timestamp_s,x_m,y_m,z_m",
    )
    replay.add_argument(
        "--output-root",
        help="派生结果根目录；默认写入 trial/derived",
    )
    replay.add_argument(
        "--max-time-delta-ms",
        type=float,
        default=20.0,
        help="最近时间戳配对上限，默认 20 ms",
    )
    replay.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="仅快速诊断时跳过原始文件哈希检查",
    )
    replay.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    replay.set_defaults(handler=_replay)

    dataset_parser = subparsers.add_parser("dataset", help="校验真实采集数据包")
    dataset_commands = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_commands.add_parser("validate", help="校验数据集或单个 trial")
    dataset_validate.add_argument("path")
    dataset_validate.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="只在快速诊断时跳过原始文件 SHA-256",
    )
    dataset_validate.set_defaults(handler=_dataset_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except (ConfigError, DatasetError) as error:
        prefix = "CONFIG_ERROR" if isinstance(error, ConfigError) else "DATASET_ERROR"
        print(f"{prefix}: {error}", file=sys.stderr)
        return 2

