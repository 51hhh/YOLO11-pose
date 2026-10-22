"""运行前环境自检；只读，不隐式安装或修改系统。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import shutil
import stat
from typing import Callable, Iterable, Mapping

from .calibration import load_opencv_stereo_calibration
from .config import ResolvedConfig, find_project_root
from .model_manifest import resolve_model, sha256_file
from .runtime import mainline_executable
from .runtime_dependencies import (
    DependencyProbe,
    probe_dynamic_linkage,
    probe_live_sdk_dependencies,
)


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


def _dependency_check(probe: DependencyProbe) -> DoctorCheck:
    return DoctorCheck(
        probe.name,
        "ok" if probe.available else "error",
        probe.message,
    )


def _system_path(system_root: Path, absolute_path: str | Path) -> Path:
    """把绝对系统路径映射到可注入的只读测试根目录。"""

    path = Path(absolute_path)
    if not path.is_absolute():
        raise ValueError(f"系统路径必须是绝对路径: {path}")
    return system_root / path.relative_to("/")


def _first_existing(system_root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        rooted = _system_path(system_root, pattern)
        if any(character in pattern for character in "*?["):
            relative_pattern = str(Path(pattern).relative_to("/"))
            matches = sorted(system_root.glob(relative_pattern))
            if matches:
                return matches[0]
        elif rooted.exists():
            return rooted
    return None


def _hardware_checks(
    config: ResolvedConfig,
    *,
    system_root: Path,
    machine: str,
    environment: Mapping[str, str],
    command_lookup: Callable[[str], str | None],
) -> list[DoctorCheck]:
    """检查启动硬件管线所需的主机事实，不枚举或改写相机。"""

    if config.data["cameras"]["backend"] != "hik_mvs":
        return []

    checks: list[DoctorCheck] = []
    platform_id = str(config.data["platform"]["id"])
    expected_jetson = platform_id.startswith("jetson-")
    architecture_ok = not expected_jetson or machine in {"aarch64", "arm64"}
    checks.append(
        DoctorCheck(
            "platform_architecture",
            "ok" if architecture_ok else "error",
            f"configured={platform_id}, machine={machine}",
        )
    )

    model_path = _system_path(system_root, "/proc/device-tree/model")
    release_path = _system_path(system_root, "/etc/nv_tegra_release")
    jetson_description = ""
    for path in (model_path, release_path):
        try:
            jetson_description = path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
        except OSError:
            continue
        if jetson_description:
            break
    jetson_ok = not expected_jetson or bool(jetson_description)
    checks.append(
        DoctorCheck(
            "jetson_platform",
            "ok" if jetson_ok else "error",
            jetson_description or "未找到 /proc/device-tree/model 或 /etc/nv_tegra_release",
        )
    )

    sync = config.data["cameras"]["sync"]
    if sync["method"] == "hardware_trigger":
        gpio_path = _system_path(system_root, f"/dev/{sync['gpio_chip']}")
        if not gpio_path.exists():
            gpio_check = DoctorCheck("gpio_device", "error", f"设备不存在: {gpio_path}")
        else:
            try:
                is_character_device = stat.S_ISCHR(gpio_path.stat().st_mode)
            except OSError:
                is_character_device = False
            if not is_character_device:
                gpio_check = DoctorCheck("gpio_device", "error", f"不是字符设备: {gpio_path}")
            elif system_root == Path("/") and not os.access(gpio_path, os.R_OK | os.W_OK):
                gpio_check = DoctorCheck("gpio_device", "error", f"当前用户不能读写: {gpio_path}")
            else:
                gpio_check = DoctorCheck(
                    "gpio_device",
                    "ok",
                    f"{gpio_path} line={sync['gpio_line']}",
                )
        checks.append(gpio_check)

    mvs_header = _first_existing(
        system_root,
        ("/opt/MVS/include/MvCameraControl.h", "/opt/MVS/include/MvCameraControl/MvCameraControl.h"),
    )
    mvs_library = _first_existing(
        system_root,
        (
            "/opt/MVS/lib/*/libMvCameraControl.so",
            "/opt/MVS/lib/libMvCameraControl.so",
        ),
    )
    mvs_ok = mvs_header is not None and mvs_library is not None
    checks.append(
        DoctorCheck(
            "hik_mvs_runtime",
            "ok" if mvs_ok else "error",
            (
                f"header={mvs_header}, library={mvs_library}"
                if mvs_ok
                else "缺少 /opt/MVS 的 MvCameraControl 头文件或运行库"
            ),
        )
    )

    if config.data["platform"]["accelerator"] in {"cuda", "cuda_dla"}:
        cuda_root = _system_path(system_root, "/usr/local/cuda")
        gpu_device = _first_existing(
            system_root,
            ("/dev/nvhost-gpu", "/dev/nvidia0"),
        )
        cuda_ok = cuda_root.exists() and gpu_device is not None
        checks.append(
            DoctorCheck(
                "cuda_runtime",
                "ok" if cuda_ok else "error",
                (
                    f"toolkit={cuda_root}, device={gpu_device}"
                    if cuda_ok
                    else "缺少 /usr/local/cuda 或 GPU 设备节点"
                ),
            )
        )

    checks.extend(
        _dependency_check(probe)
        for probe in probe_live_sdk_dependencies(system_root)
    )

    if config.data["publisher"]["enabled"] and config.data["publisher"]["backend"] == "ros2":
        ros2 = command_lookup("ros2")
        ros_environment = environment.get("AMENT_PREFIX_PATH", "")
        ros_ok = bool(ros2 and ros_environment)
        checks.append(
            DoctorCheck(
                "ros2_environment",
                "ok" if ros_ok else "error",
                (
                    f"ros2={ros2}, AMENT_PREFIX_PATH 已设置"
                    if ros_ok
                    else "请 source ROS2 与工作区，使 ros2 和 AMENT_PREFIX_PATH 可用"
                ),
            )
        )
    return checks


def _backend_plan(config: ResolvedConfig) -> DoctorCheck:
    publisher = config.data["publisher"]
    publish_backend = publisher["backend"] if publisher["enabled"] else "disabled"
    return DoctorCheck(
        "pipeline_backend_config",
        "ok",
        ", ".join(
            (
                f"camera={config.data['cameras']['backend']}",
                f"accelerator={config.data['platform']['accelerator']}",
                f"rectification={config.data['rectification']['backend']}",
                f"detector={config.data['detector']['backend']}",
                f"triangulation={config.data['stereo']['triangulation']['backend']}",
                f"publisher={publish_backend}",
            )
        ),
    )


def _expected_linked_libraries(
    config: ResolvedConfig,
) -> tuple[tuple[str, str], ...]:
    expected: list[tuple[str, str]] = [
        ("MVS", "libMvCameraControl"),
        ("OpenCV", "libopencv_core"),
        ("OpenCV calib3d", "libopencv_calib3d"),
    ]
    if config.data["platform"]["accelerator"] in {"cuda", "cuda_dla"}:
        expected.append(("CUDA runtime", "libcudart"))
    if config.data["detector"]["backend"] == "tensorrt":
        expected.append(("TensorRT", "libnvinfer"))
    publisher = config.data["publisher"]
    if publisher["enabled"] and publisher["backend"] == "ros2":
        expected.append(("ROS2", "librclcpp"))
    return tuple(expected)


def run_doctor(
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
    system_root: str | Path = "/",
    machine: str | None = None,
    environment: Mapping[str, str] | None = None,
    command_lookup: Callable[[str], str | None] = shutil.which,
) -> tuple[DoctorCheck, ...]:
    root = find_project_root(project_root)
    resolved_system_root = Path(system_root).resolve()
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

    if mode == "replay":
        checks.append(
            DoctorCheck(
                "pipeline_executable",
                "skipped",
                "确定性回放不启动硬件管线",
            )
        )
    else:
        checks.append(_backend_plan(config))
        executable = mainline_executable(root, config)
        executable_check = _check_path(
            "pipeline_executable",
            executable,
            executable=True,
        )
        checks.append(executable_check)
        checks.extend(
            _hardware_checks(
                config,
                system_root=resolved_system_root,
                machine=machine or platform.machine(),
                environment=environment if environment is not None else os.environ,
                command_lookup=command_lookup,
            )
        )
        if executable_check.status == "ok":
            ldd = command_lookup("ldd")
            if ldd is None:
                checks.append(
                    DoctorCheck(
                        "pipeline_linkage",
                        "error",
                        "ldd 命令不可用，无法验证运行时动态库",
                    )
                )
            else:
                checks.append(
                    _dependency_check(
                        probe_dynamic_linkage(
                            executable,
                            ldd,
                            _expected_linked_libraries(config),
                            probe_name="pipeline_linkage",
                        )
                    )
                )
        else:
            checks.append(
                DoctorCheck(
                    "pipeline_linkage",
                    "skipped",
                    "所选 NX 可执行文件不可用，无法检查动态链接",
                )
            )

    dataset_root = (root / config.data["recording"]["dataset_root"]).resolve()
    probe = dataset_root if dataset_root.exists() else dataset_root.parent
    if not probe.exists():
        checks.append(DoctorCheck("dataset_root", "warning", f"运行时将创建: {dataset_root}"))
    elif not probe.is_dir():
        checks.append(DoctorCheck("dataset_root", "error", f"父路径不是目录: {probe}"))
    elif config.data["recording"]["enabled"] and not os.access(probe, os.W_OK | os.X_OK):
        checks.append(DoctorCheck("dataset_root", "error", f"当前用户不能写入: {probe}"))
    else:
        checks.append(DoctorCheck("dataset_root", "ok", str(dataset_root)))
    for command in ("cmake", "git"):
        location = command_lookup(command)
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

