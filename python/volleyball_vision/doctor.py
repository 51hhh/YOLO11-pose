"""运行前环境自检；只读，不隐式安装或修改系统。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Iterable

from .calibration import load_opencv_stereo_calibration
from .config import ResolvedConfig, find_project_root
from .model_manifest import resolve_model, sha256_file


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: str
    message: str

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "warning", "skipped"}


def _check_path(name: str, path: Path, *, executable: bool = False) -> DoctorCheck:
    if not path.is_file():
        return DoctorCheck(name, "error", f"文件不存在: {path}")
    if executable and not path.stat().st_mode & 0o111:
        return DoctorCheck(name, "error", f"文件不可执行: {path}")
    return DoctorCheck(name, "ok", str(path))


def run_doctor(
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> tuple[DoctorCheck, ...]:
    root = find_project_root(project_root)
    checks: list[DoctorCheck] = [DoctorCheck("project_root", "ok", str(root))]
    calibration = (root / config.data["calibration"]["file"]).resolve()
    calibration_check = _check_path("calibration", calibration)
    checks.append(calibration_check)
    if calibration_check.status == "ok":
        try:
            loaded = load_opencv_stereo_calibration(calibration)
            checks.append(
                DoctorCheck(
                    "calibration_semantics",
                    "ok",
                    f"{loaded.image_width}x{loaded.image_height}, baseline={loaded.baseline_m:.6f}m",
                )
            )
        except Exception as error:
            checks.append(DoctorCheck("calibration_semantics", "error", str(error)))

    mode = config.data["runtime"]["mode"]
    if config.data["detector"]["backend"] == "recorded":
        checks.append(DoctorCheck("model", "skipped", "回放使用已录制检测"))
    else:
        entry, artifact = resolve_model(config, project_root=root)
        model_check = _check_path("model", artifact)
        checks.append(model_check)
        if model_check.status == "ok":
            actual = sha256_file(artifact)
            expected = entry.get("sha256")
            if not expected:
                checks.append(DoctorCheck("model_sha256", "error", f"模型清单尚未固定 SHA-256: {actual}"))
            elif expected != actual:
                checks.append(DoctorCheck("model_sha256", "error", f"期望 {expected}，实际 {actual}"))
            else:
                checks.append(DoctorCheck("model_sha256", "ok", actual))

    adapter = config.data["adapters"]["legacy_pipeline"]
    if mode == "replay":
        checks.append(DoctorCheck("legacy_pipeline", "skipped", "确定性回放不启动硬件管线"))
    else:
        checks.append(_check_path("legacy_pipeline", (root / adapter["executable"]).resolve(), executable=True))
        checks.append(_check_path("legacy_base_config", (root / adapter["base_config"]).resolve()))

    dataset_root = (root / config.data["recording"]["dataset_root"]).resolve()
    probe = dataset_root if dataset_root.exists() else dataset_root.parent
    if not probe.exists():
        checks.append(DoctorCheck("dataset_root", "warning", f"运行时将创建: {dataset_root}"))
    elif not probe.is_dir():
        checks.append(DoctorCheck("dataset_root", "error", f"父路径不是目录: {probe}"))
    else:
        checks.append(DoctorCheck("dataset_root", "ok", str(dataset_root)))
    for command in ("cmake", "git"):
        location = shutil.which(command)
        checks.append(
            DoctorCheck(
                f"command:{command}",
                "ok" if location else "error",
                location or "命令不可用",
            )
        )
    return tuple(checks)


def checks_as_dicts(checks: Iterable[DoctorCheck]) -> list[dict[str, object]]:
    return [asdict(check) | {"ok": check.ok} for check in checks]

