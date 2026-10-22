"""选择并投影当前启用的 NX 可执行管线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ConfigError, ResolvedConfig
from .legacy_config import write_legacy_pipeline_config
from .mainline_config import write_mainline_pipeline_config


PIPELINE_MAINLINE = "mainline"
PIPELINE_LEGACY = "legacy"


def active_pipeline(config: ResolvedConfig) -> str:
    value = str(config.data["adapters"]["active_pipeline"])
    if value not in {PIPELINE_MAINLINE, PIPELINE_LEGACY}:
        raise ConfigError(f"不支持的 adapters.active_pipeline: {value}")
    return value


def pipeline_executable(
    root: Path,
    config: ResolvedConfig,
) -> Path:
    pipeline = active_pipeline(config)
    adapter = config.data["adapters"][f"{pipeline}_pipeline"]
    return (root / adapter["executable"]).resolve()


def write_pipeline_config(
    destination: str | Path,
    config: ResolvedConfig,
    *,
    capture_stage: str | Path | None = None,
    frame_limit: int | None = None,
    project_root: str | Path | None = None,
) -> Path:
    arguments: dict[str, Any] = {"project_root": project_root}
    if capture_stage is not None or frame_limit is not None:
        arguments.update(
            capture_stage=capture_stage,
            frame_limit=frame_limit,
        )
    if active_pipeline(config) == PIPELINE_MAINLINE:
        return write_mainline_pipeline_config(destination, config, **arguments)
    return write_legacy_pipeline_config(destination, config, **arguments)


def pipeline_command(
    root: Path,
    config: ResolvedConfig,
    generated_config: str | Path,
    *,
    visualize: bool = False,
) -> tuple[str, ...]:
    pipeline = active_pipeline(config)
    if visualize and pipeline != PIPELINE_LEGACY:
        raise ConfigError(
            "原生 mainline 不提供 GUI；请使用 ROS2/录制数据可视化，"
            "或显式选择 legacy 回归管线"
        )
    command = (
        str(pipeline_executable(root, config)),
        "--config",
        str(Path(generated_config).resolve()),
    )
    return command + (("--visualize",) if visualize else ())
