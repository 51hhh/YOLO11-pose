"""原生 NX v6 staging 的采集编排与 v2.1 原子发布入口。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from .config import ResolvedConfig, find_project_root
from .dataset import DatasetError, require_identifier
from .git_state import GitStateError, inspect_git_state
from .hardware_manifest import (
    HardwareManifestSnapshot,
    load_hardware_manifest,
    write_stage_hardware_snapshot,
)
from .hashing import sha256_file
from .mainline_capture import (
    build_mainline_capture_context,
    finalize_mainline_capture_stage,
)
from .mainline_config import write_mainline_pipeline_config
from .runtime import mainline_command


def prepare_capture_stage(
    stage: str | Path,
    context: Mapping[str, Any],
    *,
    hardware_manifest: HardwareManifestSnapshot | None = None,
) -> Path:
    """独占创建 staging，并先冻结本次采集身份。"""

    directory = Path(stage).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "context.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(context), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    if hardware_manifest is not None:
        write_stage_hardware_snapshot(directory, hardware_manifest)
    return directory


def finalize_capture_stage(
    stage: str | Path,
    dataset_root: str | Path,
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """严格校验唯一受支持的 v6 staging，并发布冻结 v2.1 trial。"""

    return finalize_mainline_capture_stage(
        stage,
        dataset_root,
        config,
        project_root=project_root,
    )


def run_capture(
    config: ResolvedConfig,
    *,
    trial_id: str,
    session_id: str,
    ball_id: str,
    duration_s: float,
    conditions: Mapping[str, Any] | None = None,
    notes: str = "",
    dataset_root: str | Path | None = None,
    keep_stage: bool = False,
    dry_run: bool = False,
    hardware_manifest: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """运行一次有限时长采集；失败时保留 staging 供审计。"""

    root = find_project_root(project_root)
    if config.data["runtime"]["mode"] != "capture":
        raise DatasetError("volleyball capture 只接受 capture profile")
    if not config.data["recording"]["enabled"]:
        raise DatasetError("capture profile 必须启用 recording")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise DatasetError("duration_s 必须是正的有限秒数")
    require_identifier(trial_id, field_name="trial_id")
    require_identifier(session_id, field_name="session_id")
    require_identifier(ball_id, field_name="ball_id")
    manifest_snapshot = None
    if hardware_manifest is not None:
        manifest_snapshot = load_hardware_manifest(
            hardware_manifest,
            config=config,
            require_verified=not dry_run,
            project_root=root,
        )
    elif not dry_run:
        raise DatasetError("真实采集必须通过 --hardware-manifest 固定硬件与软件证据")

    capture_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else (root / config.data["recording"]["dataset_root"]).resolve()
    )
    staging_root = capture_root / ".capture-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{trial_id}-", dir=staging_root))
    # mkdtemp 已原子占用唯一名称；prepare 以独占创建写入身份文件。
    stage.rmdir()
    context = build_mainline_capture_context(
        config,
        trial_id=trial_id,
        session_id=session_id,
        ball_id=ball_id,
        conditions=conditions,
        notes=notes,
        hardware_manifest_sha256=(
            manifest_snapshot.sha256 if manifest_snapshot is not None else None
        ),
        verify_assets=not dry_run,
        project_root=root,
    )
    prepare_capture_stage(stage, context, hardware_manifest=manifest_snapshot)
    frame_limit = max(1, round(duration_s * config.data["runtime"]["nominal_fps"]))
    generated_config = write_mainline_pipeline_config(
        stage / "pipeline.yaml",
        config,
        capture_stage=stage,
        frame_limit=frame_limit,
        expected_camera_identity=(
            manifest_snapshot.camera_identity
            if manifest_snapshot is not None
            else None
        ),
        project_root=root,
    )
    command = mainline_command(root, config, generated_config)
    if dry_run:
        return stage

    try:
        try:
            launch_git_state = inspect_git_state(root)
        except GitStateError as error:
            raise DatasetError(str(error)) from error
        if (
            not launch_git_state.clean
            or launch_git_state.revision_label
            != context["perception_pipeline"]["code_revision"]
        ):
            raise DatasetError("启动前 Git 状态已变化，拒绝执行真实采集")
        executable = Path(command[0])
        if sha256_file(executable) != context["runtime_executable_sha256"]:
            raise DatasetError("启动前原生主线可执行文件哈希已变化")
        staged_manifest = load_hardware_manifest(
            stage / "hardware" / "manifest.json",
            config=config,
            require_verified=True,
            project_root=root,
        )
        if staged_manifest.sha256 != context["hardware_manifest_sha256"]:
            raise DatasetError("启动前硬件清单哈希已变化")
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise DatasetError(
                f"NX 采集管线退出码为 {completed.returncode}，staging={stage}"
            )
        published = finalize_capture_stage(
            stage,
            capture_root,
            config,
            project_root=root,
        )
    except Exception:
        keep_stage = True
        raise
    finally:
        if not keep_stage and stage.exists():
            shutil.rmtree(stage)
    return published
