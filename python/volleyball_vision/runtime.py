"""实时 NX 管线的单一启动入口。

主线配置始终先投影为一份可审计的旧 NX YAML，再启动适配器。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import tempfile

from .config import ResolvedConfig, find_project_root
from .dataset import DatasetError
from .legacy_config import write_legacy_pipeline_config


@dataclass(frozen=True, slots=True)
class RuntimeLaunchResult:
    command: tuple[str, ...]
    generated_config: Path
    dry_run: bool
    returncode: int | None

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)


def _validate_runtime(config: ResolvedConfig) -> None:
    if config.data["runtime"]["mode"] != "runtime":
        raise DatasetError("volleyball run 只接受 runtime profile")
    if config.data["recording"]["enabled"]:
        raise DatasetError("runtime profile 不得隐式启用真实数据录制")


def _write_generated_config(
    destination: Path,
    config: ResolvedConfig,
    root: Path,
) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return write_legacy_pipeline_config(
            destination,
            config,
            project_root=root,
        )
    except OSError as error:
        raise DatasetError(f"无法写入生成配置 {destination}: {error}") from error


def _command(
    root: Path,
    config: ResolvedConfig,
    generated_config: Path,
) -> tuple[str, ...]:
    executable = (
        root / config.data["adapters"]["legacy_pipeline"]["executable"]
    ).resolve()
    return (
        str(executable),
        "--config",
        str(generated_config),
    )


def run_runtime(
    config: ResolvedConfig,
    *,
    generated_config: str | Path | None = None,
    dry_run: bool = False,
    project_root: str | Path | None = None,
) -> RuntimeLaunchResult:
    root = find_project_root(project_root)
    _validate_runtime(config)

    if generated_config is not None:
        output = Path(generated_config).resolve()
        written = _write_generated_config(output, config, root)
        command = _command(root, config, written)
        if dry_run:
            return RuntimeLaunchResult(command, written, True, None)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise DatasetError(
                f"NX 实时管线退出码为 {completed.returncode}，"
                f"生成配置保留在 {written}"
            )
        return RuntimeLaunchResult(command, written, False, completed.returncode)

    if dry_run:
        directory = Path(tempfile.mkdtemp(prefix="volleyball-runtime-"))
        output = directory / f"{config.profile}-{config.fingerprint_sha256[:12]}.yaml"
        written = _write_generated_config(output, config, root)
        command = _command(root, config, written)
        return RuntimeLaunchResult(command, written, True, None)

    with tempfile.TemporaryDirectory(prefix="volleyball-runtime-") as directory:
        output = Path(directory) / "pipeline.yaml"
        written = _write_generated_config(output, config, root)
        command = _command(root, config, written)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise DatasetError(f"NX 实时管线退出码为 {completed.returncode}")
        return RuntimeLaunchResult(command, written, False, completed.returncode)
