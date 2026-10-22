#!/usr/bin/env python3
"""无网络的仓库元数据门禁。"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    "CMakeLists.txt",
    "CMakePresets.json",
    "CONTRIBUTING.md",
    "README.md",
    "adapters",
    "algorithms",
    "configs",
    "contracts",
    "docs",
    "libs",
    "models",
    "ports",
    "pyproject.toml",
    "python",
    "services",
    "tests",
    "tools",
}
FORBIDDEN_TRACKED_PREFIXES = (
    "NX_volleyball/",
    "Stereo/",
    "volleyball_tracking/",
)
ALLOWED_TOOL_FILES = {"tools/check_repository.py"}

PURE_CPP_LAYERS = (
    "libs/core",
    "ports",
    "algorithms",
    "services",
)
FORBIDDEN_HARDWARE_INCLUDES = re.compile(
    r"^\s*#\s*include\s*[<\"](?:cuda|vpi|NvInfer|opencv|rclcpp|"
    r"MvCameraControl|gpiod|yaml-cpp)",
    re.IGNORECASE | re.MULTILINE,
)
FORBIDDEN_DIRECT_VPI_PLANE_ADDRESS = re.compile(
    r"(?:"
    r"(?:buffer\.pitch\.)?planes?\s*\[[^\]]+\]"
    r"|\b[a-zA-Z_]\w*plane\w*"
    r")\s*\.\s*(?:data|pBase|offsetBytes)\b",
    re.IGNORECASE,
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _local_markdown_targets(
    source: Path,
) -> tuple[list[Path], list[str]]:
    targets: list[Path] = []
    errors: list[str] = []
    text = source.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK.finditer(text):
        raw = match.group(1).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        elif " " in raw:
            raw = raw.split(" ", 1)[0]
        if not raw or raw.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", raw):
            continue
        destination = unquote(raw.split("#", 1)[0].split("?", 1)[0])
        if not destination:
            continue
        target = (source.parent / destination).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            errors.append(
                f"{source.relative_to(ROOT)}: 链接越出仓库: {raw}"
            )
            continue
        if not target.exists():
            errors.append(
                f"{source.relative_to(ROOT)}: 目标不存在: {raw}"
            )
            continue
        targets.append(target)
    return targets, errors


def _check_documentation(tracked: list[str]) -> list[str]:
    markdown = {
        (ROOT / path).resolve()
        for path in tracked
        if Path(path).suffix.lower() == ".md"
    }
    errors: list[str] = []
    links: dict[Path, list[Path]] = {}
    for source in sorted(markdown):
        targets, link_errors = _local_markdown_targets(source)
        links[source] = [target for target in targets if target in markdown]
        errors.extend(link_errors)

    entry = (ROOT / "README.md").resolve()
    reachable: set[Path] = set()
    pending = [entry]
    while pending:
        source = pending.pop()
        if source in reachable:
            continue
        reachable.add(source)
        pending.extend(links.get(source, ()))

    orphaned = sorted(
        path.relative_to(ROOT).as_posix()
        for path in markdown - reachable
        if path.is_relative_to(ROOT / "docs")
    )
    if orphaned:
        errors.append(
            "docs 中的文档必须能从根 README 导航到:\n    "
            + "\n    ".join(orphaned)
        )
    return errors


def main() -> int:
    errors: list[str] = []
    staged_entries = _git("ls-files", "-s").splitlines()
    gitlinks = [line for line in staged_entries if line.startswith("160000 ")]
    if gitlinks:
        errors.append("禁止无明确声明的嵌套 gitlink:\n    " + "\n    ".join(gitlinks))

    tracked = _git("ls-files").splitlines()
    top_level = {Path(path).parts[0] for path in tracked}
    unexpected_top_level = sorted(top_level - ALLOWED_TOP_LEVEL)
    if unexpected_top_level:
        errors.append(
            "禁止新增未声明的顶层入口:\n    "
            + "\n    ".join(unexpected_top_level)
        )

    retired_paths = [
        path for path in tracked
        if path.startswith(FORBIDDEN_TRACKED_PREFIXES)
    ]
    if retired_paths:
        errors.append(
            "已退役目录不得重新进入主线:\n    "
            + "\n    ".join(retired_paths)
        )

    generated = [
        path for path in tracked
        if path.endswith((".pyc", ".pyo"))
        or "/__pycache__/" in f"/{path}/"
        or "/.pytest_cache/" in f"/{path}/"
    ]
    if generated:
        errors.append("禁止跟踪 Python 缓存:\n    " + "\n    ".join(generated))

    documentation_errors = _check_documentation(tracked)
    if documentation_errors:
        errors.append(
            "文档导航或本地链接无效:\n    "
            + "\n    ".join(documentation_errors)
        )

    unexpected_tools = sorted(
        path for path in tracked
        if path.startswith("tools/") and path not in ALLOWED_TOOL_FILES
    )
    if unexpected_tools:
        errors.append(
            "tools 只保留仓库维护门禁，用户功能应进入统一 CLI:\n    "
            + "\n    ".join(unexpected_tools)
        )

    flat_cpp_tests = sorted(
        path for path in tracked
        if Path(path).parent == Path("tests/cpp")
        and Path(path).suffix in {".cc", ".cpp", ".cxx"}
    )
    if flat_cpp_tests:
        errors.append(
            "C++ 测试必须归入 mainline 或 legacy_nx:\n    "
            + "\n    ".join(flat_cpp_tests)
        )

    required = [
        "contracts/capture/v2.1.0/tables.json",
        "configs/schema/runtime.schema.json",
        "configs/profiles/nx-rdk.yaml",
        "configs/profiles/free-flight.yaml",
        "models/manifest.yaml",
        "CMakePresets.json",
        "adapters/legacy_nx/CMakeLists.txt",
        "adapters/legacy_nx/config/base_pipeline.yaml",
        "configs/calibration/stereo_calib.yaml",
        "algorithms/stereo/CMakeLists.txt",
        "services/CMakeLists.txt",
        "docs/hardware-tools.md",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        errors.append("缺少主线必需文件:\n    " + "\n    ".join(missing))

    dependency_violations: list[str] = []
    for layer in PURE_CPP_LAYERS:
        layer_root = ROOT / layer
        if not layer_root.exists():
            continue
        for path in layer_root.rglob("*"):
            if path.suffix not in {".h", ".hpp", ".cc", ".cpp", ".cxx"}:
                continue
            text = path.read_text(encoding="utf-8")
            match = FORBIDDEN_HARDWARE_INCLUDES.search(text)
            if match:
                dependency_violations.append(
                    f"{path.relative_to(ROOT)}: {match.group(0).strip()}"
                )
    if dependency_violations:
        errors.append(
            "纯核心/算法/服务层禁止直接依赖硬件 SDK:\n    "
            + "\n    ".join(dependency_violations)
        )

    vpi_plane_access_violations: list[str] = []
    legacy_source_root = ROOT / "adapters/legacy_nx/src"
    vpi_adapter = legacy_source_root / "utils/vpi_plane_access.h"
    if legacy_source_root.exists():
        for path in legacy_source_root.rglob("*"):
            if path.suffix not in {
                ".h", ".hpp", ".cc", ".cpp", ".cxx", ".cu"
            }:
                continue
            if path == vpi_adapter:
                continue
            text = path.read_text(encoding="utf-8")
            match = FORBIDDEN_DIRECT_VPI_PLANE_ADDRESS.search(text)
            if match:
                vpi_plane_access_violations.append(
                    f"{path.relative_to(ROOT)}: {match.group(0)}"
                )
    if vpi_plane_access_violations:
        errors.append(
            "VPI pitch-linear 地址只能经 vpi_plane_access.h 访问:\n    "
            + "\n    ".join(vpi_plane_access_violations)
        )

    if errors:
        print("REPOSITORY_AUDIT_FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("repository audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

