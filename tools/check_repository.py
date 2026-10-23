#!/usr/bin/env python3
"""无网络的仓库元数据门禁。"""

from __future__ import annotations

import ast
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
    "apps",
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
    "adapters/legacy_nx/",
    "tests/cpp/legacy_nx/",
)
ALLOWED_TOOL_FILES = {
    "tools/check_repository.py",
    "tools/hardware/CMakeLists.txt",
    "tools/hardware/common/atomic_file.cpp",
    "tools/hardware/common/atomic_file.h",
    "tools/hardware/common/parallel_tasks.cpp",
    "tools/hardware/common/parallel_tasks.h",
    "tools/hardware/capture_chessboard/args.cpp",
    "tools/hardware/capture_chessboard/args.h",
    "tools/hardware/capture_chessboard/main.cpp",
    "tools/hardware/capture_chessboard/output.cpp",
    "tools/hardware/capture_chessboard/output.h",
    "tools/hardware/hik_frame_metadata_probe/args.cpp",
    "tools/hardware/hik_frame_metadata_probe/args.h",
    "tools/hardware/hik_frame_metadata_probe/camera_session.cpp",
    "tools/hardware/hik_frame_metadata_probe/camera_session.h",
    "tools/hardware/hik_frame_metadata_probe/main.cpp",
    "tools/hardware/hik_frame_metadata_probe/pair_audit.cpp",
    "tools/hardware/hik_frame_metadata_probe/pair_audit.h",
    "tools/hardware/stereo_calibrate/args.cpp",
    "tools/hardware/stereo_calibrate/args.h",
    "tools/hardware/stereo_calibrate/images.cpp",
    "tools/hardware/stereo_calibrate/images.h",
    "tools/hardware/stereo_calibrate/main.cpp",
}
ALLOWED_ROOT_MARKDOWN = {
    "CONTRIBUTING.md",
    "README.md",
}
ALLOWED_DOC_ROOT_FILES = {
    "docs/acceptance.md",
    "docs/architecture.md",
    "docs/capture.md",
    "docs/configuration.md",
    "docs/hardware-tools.md",
    "docs/pipeline.md",
}
ALLOWED_DOC_GROUPS = {
    "adr",
}
ALLOWED_TEST_ROOT_FILES = {
    "tests/__init__.py",
}
ALLOWED_TEST_GROUPS = {
    "contract",
    "cpp",
    "support",
    "unit",
}
ALLOWED_CPP_TEST_GROUPS = {
    "adapters",
    "fakes",
    "hardware_tools",
    "mainline",
}

PURE_CPP_LAYERS = (
    "libs/core",
    "libs/logging",
    "ports",
    "algorithms",
    "services",
    "adapters/capture_stage",
)
FORBIDDEN_HARDWARE_INCLUDES = re.compile(
    r"^\s*#\s*include\s*[<\"](?:cuda|vpi|NvInfer|opencv|rclcpp|"
    r"MvCameraControl|gpiod|yaml-cpp)",
    re.IGNORECASE | re.MULTILINE,
)
FORBIDDEN_TEST_MODULE_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:tests\.)?(?:unit|contract)\.test_[A-Za-z0-9_]+",
    re.MULTILINE,
)
FORBIDDEN_MAINLINE_ADAPTER_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]volleyball/adapters/(?!fake/)',
    re.MULTILINE,
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CMAKE_SOURCE_GROUPS = (
    ("apps/nx_mainline", "apps/nx_mainline/CMakeLists.txt"),
    ("libs/core/src", "libs/core/CMakeLists.txt"),
    ("algorithms/stereo/src", "algorithms/stereo/CMakeLists.txt"),
    ("services/src", "services/CMakeLists.txt"),
    ("adapters/capture_stage/src", "adapters/capture_stage/CMakeLists.txt"),
    ("adapters/gpio/src", "adapters/gpio/CMakeLists.txt"),
    ("adapters/hik_mvs/src", "adapters/hik_mvs/CMakeLists.txt"),
    ("adapters/cuda_preprocess/src", "adapters/cuda_preprocess/CMakeLists.txt"),
    ("adapters/cuda_depth/src", "adapters/cuda_depth/CMakeLists.txt"),
    ("adapters/tensorrt/src", "adapters/tensorrt/CMakeLists.txt"),
    ("adapters/ros2/src", "adapters/ros2/CMakeLists.txt"),
    ("adapters/fake/src", "adapters/fake/CMakeLists.txt"),
    ("tools/hardware", "tools/hardware/CMakeLists.txt"),
    ("tests/cpp/mainline", "tests/cpp/mainline/CMakeLists.txt"),
    ("tests/cpp/adapters", "tests/cpp/adapters/CMakeLists.txt"),
    (
        "tests/cpp/hardware_tools",
        "tests/cpp/hardware_tools/CMakeLists.txt",
    ),
)


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
        if not path.is_relative_to(ROOT / ".github")
    )
    if orphaned:
        errors.append(
            "用户文档必须能从根 README 导航到:\n    "
            + "\n    ".join(orphaned)
        )
    return errors


def _check_cmake_sources(tracked: list[str]) -> list[str]:
    """拒绝未加入对应 CMake 目标的 C++/CUDA 实现或测试。"""

    errors: list[str] = []
    tracked_set = set(tracked)
    for source_root, cmake_file in CMAKE_SOURCE_GROUPS:
        cmake_path = ROOT / cmake_file
        if cmake_file not in tracked_set or not cmake_path.is_file():
            continue
        cmake_text = cmake_path.read_text(encoding="utf-8")
        cmake_directory = Path(cmake_file).parent
        for source in sorted(tracked_set):
            path = Path(source)
            if not source.startswith(f"{source_root}/"):
                continue
            if path.suffix not in {".cc", ".cpp", ".cxx", ".cu"}:
                continue
            token = path.relative_to(cmake_directory).as_posix()
            # 只匹配 basename 会让 tools/*/main.cpp 等同名文件互相代为
            # 通过，导致未接入任何目标的源码逃过仓库审计。
            if token not in cmake_text:
                errors.append(f"{source}: 未列入 {cmake_file}")
    return errors


def _python_module_name(path: Path) -> str:
    relative = path.relative_to(ROOT / "python")
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts)
    return ".".join(relative.with_suffix("").parts)


def _check_python_module_reachability(tracked: list[str]) -> list[str]:
    """拒绝无法从公开包或命令入口到达的生产 Python 模块。"""

    package_prefix = "python/volleyball_vision/"
    paths = {
        _python_module_name(ROOT / path): ROOT / path
        for path in tracked
        if path.startswith(package_prefix) and path.endswith(".py")
    }
    imports: dict[str, set[str]] = {module: set() for module in paths}
    errors: list[str] = []

    for module, path in sorted(paths.items()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path.relative_to(ROOT)}: 无法解析: {error.msg}")
            continue

        is_package = path.name == "__init__.py"
        package_parts = module.split(".") if is_package else module.split(".")[:-1]
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    ascend = node.level - 1
                    if ascend > len(package_parts):
                        continue
                    base_parts = package_parts[: len(package_parts) - ascend]
                    if node.module:
                        base_parts.extend(node.module.split("."))
                        base = ".".join(base_parts)
                        candidates.append(base)
                        candidates.extend(
                            f"{base}.{alias.name}" for alias in node.names
                        )
                    else:
                        candidates.extend(
                            ".".join([*base_parts, alias.name])
                            for alias in node.names
                        )
                elif node.module:
                    candidates.append(node.module)
                    candidates.extend(
                        f"{node.module}.{alias.name}" for alias in node.names
                    )

            imports[module].update(
                candidate for candidate in candidates if candidate in paths
            )

    roots = {
        module
        for module in ("volleyball_vision", "volleyball_vision.__main__")
        if module in paths
    }
    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(imports[module] - reachable)

    orphaned = sorted(
        paths[module].relative_to(ROOT).as_posix()
        for module in paths.keys() - reachable
    )
    if orphaned:
        errors.append(
            "生产 Python 模块必须从包入口或 CLI 可达:\n    "
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

    scattered_markdown: list[str] = []
    for path in tracked:
        if Path(path).suffix.lower() != ".md":
            continue
        if path in ALLOWED_ROOT_MARKDOWN or path.startswith(".github/"):
            continue
        if path == "contracts/capture/v2.1.0/README.md":
            continue
        if path in ALLOWED_DOC_ROOT_FILES:
            continue
        if path.startswith("docs/"):
            relative = Path(path).relative_to("docs")
            if len(relative.parts) > 1 and relative.parts[0] in ALLOWED_DOC_GROUPS:
                continue
        scattered_markdown.append(path)
    if scattered_markdown:
        errors.append(
            "说明文档只能位于根导航、docs 已声明分组或冻结契约目录:\n    "
            + "\n    ".join(sorted(scattered_markdown))
        )

    cmake_source_errors = _check_cmake_sources(tracked)
    if cmake_source_errors:
        errors.append(
            "C++/CUDA 源码或测试未连接到构建目标:\n    "
            + "\n    ".join(cmake_source_errors)
        )

    python_module_errors = _check_python_module_reachability(tracked)
    if python_module_errors:
        errors.append(
            "Python 生产模块存在孤立入口:\n    "
            + "\n    ".join(python_module_errors)
        )

    unexpected_tools = sorted(
        path for path in tracked
        if path.startswith("tools/") and path not in ALLOWED_TOOL_FILES
    )
    if unexpected_tools:
        errors.append(
            "tools 只允许仓库门禁和已审核的硬件验收程序:\n    "
            + "\n    ".join(unexpected_tools)
        )

    flat_cpp_tests = sorted(
        path for path in tracked
        if Path(path).parent == Path("tests/cpp")
        and Path(path).suffix in {".cc", ".cpp", ".cxx"}
    )
    if flat_cpp_tests:
        errors.append(
            "C++ 测试必须归入已声明的职责目录:\n    "
            + "\n    ".join(flat_cpp_tests)
        )

    unexpected_cpp_test_groups = sorted(
        path for path in tracked
        if path.startswith("tests/cpp/")
        and len(Path(path).relative_to("tests/cpp").parts) > 1
        and Path(path).relative_to("tests/cpp").parts[0]
        not in ALLOWED_CPP_TEST_GROUPS
    )
    if unexpected_cpp_test_groups:
        errors.append(
            "C++ 测试只能进入 mainline、adapters、hardware_tools 或 fakes:\n    "
            + "\n    ".join(unexpected_cpp_test_groups)
        )

    misplaced_adapter_tests: list[str] = []
    for path in tracked:
        if not path.startswith("tests/cpp/mainline/") or Path(path).suffix not in {
            ".h",
            ".hpp",
            ".cc",
            ".cpp",
            ".cxx",
        }:
            continue
        match = FORBIDDEN_MAINLINE_ADAPTER_INCLUDE.search(
            (ROOT / path).read_text(encoding="utf-8")
        )
        if match:
            misplaced_adapter_tests.append(
                f"{path}: {match.group(0).strip()}"
            )
    if misplaced_adapter_tests:
        errors.append(
            "mainline 测试不得引用非 fake 适配器，请移入 tests/cpp/adapters:\n    "
            + "\n    ".join(misplaced_adapter_tests)
        )

    flat_python_tests = sorted(
        path for path in tracked
        if Path(path).parent == Path("tests")
        and Path(path).name.startswith("test_")
        and Path(path).suffix == ".py"
    )
    if flat_python_tests:
        errors.append(
            "Python 测试必须归入 unit 或 contract:\n    "
            + "\n    ".join(flat_python_tests)
        )

    unexpected_test_root_files = sorted(
        path for path in tracked
        if Path(path).parent == Path("tests")
        and path not in ALLOWED_TEST_ROOT_FILES
    )
    if unexpected_test_root_files:
        errors.append(
            "tests 根目录只保留包入口，公共夹具进入 tests/support/:\n    "
            + "\n    ".join(unexpected_test_root_files)
        )

    unexpected_test_groups = sorted(
        path for path in tracked
        if path.startswith("tests/")
        and len(Path(path).relative_to("tests").parts) > 1
        and Path(path).relative_to("tests").parts[0]
        not in ALLOWED_TEST_GROUPS
    )
    if unexpected_test_groups:
        errors.append(
            "tests 只能进入 unit、contract、support 或 cpp 职责目录:\n    "
            + "\n    ".join(unexpected_test_groups)
        )

    coupled_test_modules: list[str] = []
    for path in tracked:
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        match = FORBIDDEN_TEST_MODULE_IMPORT.search(
            (ROOT / path).read_text(encoding="utf-8")
        )
        if match:
            coupled_test_modules.append(f"{path}: {match.group(0).strip()}")
    if coupled_test_modules:
        errors.append(
            "测试模块不得相互导入，公共造数逻辑应进入 tests/support/:\n    "
            + "\n    ".join(coupled_test_modules)
        )

    required = [
        "contracts/capture/v2.1.0/tables.json",
        "configs/schema/runtime.schema.json",
        "configs/profiles/nx-rdk.yaml",
        "configs/profiles/free-flight.yaml",
        "models/manifest.yaml",
        "CMakePresets.json",
        "adapters/hik_mvs/CMakeLists.txt",
        "configs/calibration/stereo_calib.yaml",
        "algorithms/stereo/CMakeLists.txt",
        "services/CMakeLists.txt",
        "adapters/capture_stage/CMakeLists.txt",
        "apps/nx_mainline/CMakeLists.txt",
        "apps/nx_mainline/main.cpp",
        "docs/hardware-tools.md",
        "docs/pipeline.md",
        "tests/cpp/adapters/CMakeLists.txt",
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

    if errors:
        print("REPOSITORY_AUDIT_FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("repository audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

