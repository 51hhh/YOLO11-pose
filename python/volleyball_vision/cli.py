"""统一的 volleyball 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from .config import ConfigError, flatten, load_profile
from .dataset import DatasetError, validate_dataset, validate_trial


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

