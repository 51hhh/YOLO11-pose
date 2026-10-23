"""正式采集使用的双目硬件与软件版本证据。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import jsonschema

from .config import ResolvedConfig, find_project_root
from .dataset import DatasetError


_MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HardwareAttachment:
    """读取后冻结的单个相机参数导出文件。"""

    role: str
    relative_path: PurePosixPath
    sha256: str
    data: bytes
    source_path: Path

    @property
    def raw_label(self) -> str:
        return f"camera_parameters_{self.role}"


@dataclass(frozen=True, slots=True)
class HardwareManifestSnapshot:
    """一次读取中冻结的清单正文和全部小型附件。"""

    source_path: Path
    payload: Mapping[str, Any]
    data: bytes
    sha256: str
    attachments: tuple[HardwareAttachment, ...]

    @property
    def camera_models(self) -> dict[str, str]:
        cameras = self.payload["cameras"]
        return {
            role: str(cameras[role]["model"])
            for role in ("left", "right")
        }

    @property
    def camera_firmware_versions(self) -> dict[str, str]:
        cameras = self.payload["cameras"]
        return {
            role: str(cameras[role]["firmware_version"])
            for role in ("left", "right")
        }

    @property
    def camera_identity(self) -> dict[str, dict[str, str]]:
        return {
            role: {
                "model": self.camera_models[role],
                "firmware_version": self.camera_firmware_versions[role],
            }
            for role in ("left", "right")
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_schema(project_root: str | Path | None) -> Mapping[str, Any]:
    path = (
        find_project_root(project_root)
        / "contracts"
        / "hardware"
        / "rig-manifest-v1.schema.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"硬件清单 schema 无法读取: {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetError("硬件清单 schema 顶层必须是对象")
    return value


def _validate_schema(
    payload: Mapping[str, Any], *, project_root: str | Path | None
) -> None:
    validator = jsonschema.Draft202012Validator(
        _load_schema(project_root),
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if not errors:
        return
    details = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise DatasetError(
        "硬件清单 schema 校验失败:\n"
        + "\n".join(f"  - {detail}" for detail in details)
    )


def _safe_attachment_path(base: Path, value: str, *, role: str) -> tuple[PurePosixPath, Path]:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DatasetError(f"{role} 相机参数导出路径必须是清单目录内的相对路径")
    if relative == PurePosixPath("manifest.json"):
        raise DatasetError(f"{role} 相机参数导出文件不得覆盖硬件清单")
    source = (base / Path(*relative.parts)).resolve()
    if source == base or base not in source.parents:
        raise DatasetError(f"{role} 相机参数导出路径逃逸清单目录")
    return relative, source


def load_hardware_manifest(
    source: str | Path,
    *,
    config: ResolvedConfig | None = None,
    require_verified: bool = True,
    project_root: str | Path | None = None,
) -> HardwareManifestSnapshot:
    """一次性读取、校验并冻结硬件清单及相机参数导出文件。"""

    path = Path(source).resolve()
    try:
        data = path.read_bytes()
    except OSError as error:
        raise DatasetError(f"硬件清单无法读取: {path}: {error}") from error
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"硬件清单不是有效 UTF-8 JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DatasetError("硬件清单顶层必须是对象")
    _validate_schema(payload, project_root=project_root)
    if require_verified and payload["status"] != "verified":
        raise DatasetError("真实采集要求 status=verified 的硬件清单")
    if payload["status"] == "verified" and (
        payload["verified_at_utc"] is None or payload["verified_by"] is None
    ):
        raise DatasetError("verified 硬件清单必须记录核验时间和核验人")
    if payload["status"] == "verified":
        declared_values = [
            payload["cameras"][role][key]
            for role in ("left", "right")
            for key in ("model", "firmware_version")
        ]
        declared_values.extend(
            payload["cameras"][role]["lens"][key]
            for role in ("left", "right")
            for key in ("model", "focus_setting")
        )
        declared_values.extend(
            component["version"] for component in payload["software"].values()
        )
        if any("REPLACE" in str(value).upper() for value in declared_values):
            raise DatasetError("verified 硬件清单仍包含未替换的模板值")

    if config is not None:
        expected_shutter = config.data["cameras"]["sync"]["shutter_type"]
        for role in ("left", "right"):
            declared = payload["cameras"][role]
            expected = config.data["cameras"][role]
            if declared["serial"] != expected["serial"]:
                raise DatasetError(f"{role} 硬件清单序列号与 profile 不一致")
            if declared["shutter_type"] != expected_shutter:
                raise DatasetError(f"{role} 硬件清单快门类型与 profile 不一致")

    attachments: list[HardwareAttachment] = []
    seen_paths: set[PurePosixPath] = set()
    base = path.parent.resolve()
    for role in ("left", "right"):
        export = payload["cameras"][role]["parameter_export"]
        relative, attachment_path = _safe_attachment_path(
            base, str(export["path"]), role=role
        )
        if relative in seen_paths:
            raise DatasetError("左右相机参数导出文件不得引用同一路径")
        seen_paths.add(relative)
        try:
            size = attachment_path.stat().st_size
            if size == 0:
                raise DatasetError(f"{role} 相机参数导出文件为空")
            if size > _MAX_ATTACHMENT_BYTES:
                raise DatasetError(
                    f"{role} 相机参数导出文件超过 {_MAX_ATTACHMENT_BYTES} 字节上限"
                )
            attachment_data = attachment_path.read_bytes()
        except OSError as error:
            raise DatasetError(
                f"{role} 相机参数导出文件无法读取: {attachment_path}: {error}"
            ) from error
        actual_sha256 = _sha256_bytes(attachment_data)
        if actual_sha256 != export["sha256"]:
            raise DatasetError(f"{role} 相机参数导出文件 SHA-256 不匹配")
        attachments.append(
            HardwareAttachment(
                role=role,
                relative_path=relative,
                sha256=actual_sha256,
                data=attachment_data,
                source_path=attachment_path,
            )
        )
    return HardwareManifestSnapshot(
        source_path=path,
        payload=payload,
        data=data,
        sha256=_sha256_bytes(data),
        attachments=tuple(attachments),
    )


def write_stage_hardware_snapshot(
    stage: str | Path, snapshot: HardwareManifestSnapshot
) -> Path:
    """把同一次读取冻结的清单和附件独占写入 staging。"""

    root = Path(stage).resolve() / "hardware"
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "manifest.json"
    with manifest.open("xb") as handle:
        handle.write(snapshot.data)
    for attachment in snapshot.attachments:
        destination = root.joinpath(*attachment.relative_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(attachment.data)
    return manifest
