"""NX 运行依赖的只读探针。

这里仅判断开发头文件、动态库与已构建二进制的链接事实；实际算子是否在
CUDA/VPI/TensorRT 上提交，仍应由启动日志与 Nsight 时间线验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    name: str
    available: bool
    message: str


@dataclass(frozen=True, slots=True)
class _SdkRequirement:
    name: str
    headers: tuple[str, ...]
    libraries: tuple[tuple[str, tuple[str, ...]], ...]


_LIVE_SDK_REQUIREMENTS = (
    _SdkRequirement(
        name="tensorrt_sdk",
        headers=(
            "/usr/include/aarch64-linux-gnu/NvInfer.h",
            "/usr/include/x86_64-linux-gnu/NvInfer.h",
            "/usr/include/NvInfer.h",
            "/usr/local/include/NvInfer.h",
        ),
        libraries=(
            (
                "nvinfer",
                (
                    "/usr/lib/aarch64-linux-gnu/libnvinfer.so*",
                    "/usr/lib/x86_64-linux-gnu/libnvinfer.so*",
                    "/usr/lib/libnvinfer.so*",
                    "/usr/local/lib/libnvinfer.so*",
                ),
            ),
            (
                "nvinfer_plugin",
                (
                    "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so*",
                    "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so*",
                    "/usr/lib/libnvinfer_plugin.so*",
                    "/usr/local/lib/libnvinfer_plugin.so*",
                ),
            ),
        ),
    ),
    _SdkRequirement(
        name="vpi_sdk",
        headers=(
            "/opt/nvidia/vpi*/include/vpi/VPI.h",
            "/usr/include/vpi/VPI.h",
            "/usr/local/include/vpi/VPI.h",
        ),
        libraries=(
            (
                "nvvpi",
                (
                    "/opt/nvidia/vpi*/lib/aarch64-linux-gnu/libnvvpi.so*",
                    "/opt/nvidia/vpi*/lib/x86_64-linux-gnu/libnvvpi.so*",
                    "/opt/nvidia/vpi*/lib64/libnvvpi.so*",
                    "/usr/lib/aarch64-linux-gnu/libnvvpi.so*",
                    "/usr/lib/x86_64-linux-gnu/libnvvpi.so*",
                    "/usr/lib/libnvvpi.so*",
                    "/usr/local/lib/libnvvpi.so*",
                ),
            ),
        ),
    ),
    _SdkRequirement(
        name="opencv_sdk",
        headers=(
            "/usr/include/opencv4/opencv2/core.hpp",
            "/usr/local/include/opencv4/opencv2/core.hpp",
        ),
        libraries=tuple(
            (
                component,
                (
                    f"/usr/lib/aarch64-linux-gnu/libopencv_{component}.so*",
                    f"/usr/lib/x86_64-linux-gnu/libopencv_{component}.so*",
                    f"/usr/lib/libopencv_{component}.so*",
                    f"/usr/local/lib/libopencv_{component}.so*",
                ),
            )
            for component in (
                "core",
                "imgproc",
                "features2d",
                "calib3d",
                "highgui",
                "imgcodecs",
                "cudastereo",
                "cudaimgproc",
                "cudafilters",
            )
        ),
    ),
)

_MAINLINE_SDK_REQUIREMENTS = (
    _SdkRequirement(
        name="tensorrt_sdk",
        headers=(
            "/usr/include/aarch64-linux-gnu/NvInfer.h",
            "/usr/include/x86_64-linux-gnu/NvInfer.h",
            "/usr/include/NvInfer.h",
            "/usr/local/include/NvInfer.h",
        ),
        libraries=(
            (
                "nvinfer",
                (
                    "/usr/lib/aarch64-linux-gnu/libnvinfer.so*",
                    "/usr/lib/x86_64-linux-gnu/libnvinfer.so*",
                    "/usr/lib/libnvinfer.so*",
                    "/usr/local/lib/libnvinfer.so*",
                ),
            ),
        ),
    ),
    _SdkRequirement(
        name="opencv_sdk",
        headers=(
            "/usr/include/opencv4/opencv2/core.hpp",
            "/usr/local/include/opencv4/opencv2/core.hpp",
        ),
        libraries=tuple(
            (
                component,
                (
                    f"/usr/lib/aarch64-linux-gnu/libopencv_{component}.so*",
                    f"/usr/lib/x86_64-linux-gnu/libopencv_{component}.so*",
                    f"/usr/lib/libopencv_{component}.so*",
                    f"/usr/local/lib/libopencv_{component}.so*",
                ),
            )
            for component in ("core", "calib3d")
        ),
    ),
)


def _system_path(system_root: Path, absolute_path: str) -> Path:
    path = Path(absolute_path)
    if not path.is_absolute():
        raise ValueError(f"系统路径必须是绝对路径: {path}")
    return system_root / path.relative_to("/")


def _first_existing(system_root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        if any(character in pattern for character in "*?["):
            relative_pattern = str(Path(pattern).relative_to("/"))
            matches = sorted(
                candidate
                for candidate in system_root.glob(relative_pattern)
                if candidate.is_file()
            )
            if matches:
                return matches[0]
        else:
            candidate = _system_path(system_root, pattern)
            if candidate.is_file():
                return candidate
    return None


def probe_live_sdk_dependencies(
    system_root: str | Path,
    *,
    pipeline: str = "legacy",
) -> tuple[DependencyProbe, ...]:
    """检查所选 NX 管线构建和运行真正需要的 SDK 组件。"""

    root = Path(system_root).resolve()
    if pipeline == "legacy":
        requirements = _LIVE_SDK_REQUIREMENTS
    elif pipeline == "mainline":
        requirements = _MAINLINE_SDK_REQUIREMENTS
    else:
        raise ValueError(f"不支持的 NX pipeline: {pipeline}")
    probes: list[DependencyProbe] = []
    for requirement in requirements:
        header = _first_existing(root, requirement.headers)
        libraries = {
            component: _first_existing(root, patterns)
            for component, patterns in requirement.libraries
        }
        missing: list[str] = []
        if header is None:
            missing.append("header")
        missing.extend(
            f"lib:{component}"
            for component, path in libraries.items()
            if path is None
        )
        if missing:
            probes.append(
                DependencyProbe(
                    requirement.name,
                    False,
                    "缺少 " + ", ".join(missing),
                )
            )
            continue
        library_text = ", ".join(
            f"{component}={path}" for component, path in libraries.items()
        )
        probes.append(
            DependencyProbe(
                requirement.name,
                True,
                f"header={header}; {library_text}",
            )
        )
    return tuple(probes)


CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def _run_command(command: Sequence[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    return completed.returncode, completed.stdout


def probe_dynamic_linkage(
    executable: str | Path,
    ldd_command: str,
    expected_libraries: Iterable[tuple[str, str]],
    *,
    command_runner: CommandRunner = _run_command,
    probe_name: str = "legacy_linkage",
) -> DependencyProbe:
    """用 ldd 证明二进制没有缺库，且确实链接 profile 所需框架。"""

    command = (ldd_command, str(Path(executable).resolve()))
    returncode, output = command_runner(command)
    if returncode != 0:
        detail = output.strip() or f"ldd 返回 {returncode}"
        return DependencyProbe(probe_name, False, detail)

    unresolved = sorted(
        set(
            re.findall(
                r"^\s*([^\s]+)\s+=>\s+not found\s*$",
                output,
                flags=re.MULTILINE,
            )
        )
    )
    if unresolved:
        return DependencyProbe(
            probe_name,
            False,
            "动态库未解析: " + ", ".join(unresolved),
        )

    absent = [
        f"{label}({token})"
        for label, token in expected_libraries
        if token not in output
    ]
    if absent:
        return DependencyProbe(
            probe_name,
            False,
            "二进制未链接 profile 必需框架: " + ", ".join(absent),
        )

    resolved_lines = sum(bool(line.strip()) for line in output.splitlines())
    return DependencyProbe(
        probe_name,
        True,
        f"动态依赖全部解析，已核对 {resolved_lines} 行",
    )
