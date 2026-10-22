from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import yaml

from volleyball_vision.capture import (
    STAGE_DETECTION_COLUMNS,
    STAGE_FRAME_COLUMNS,
    STAGE_PAIR_COLUMNS,
    STAGE_SCHEMA_VERSION,
)
from volleyball_vision.mainline_stage_contract import (
    MAINLINE_ASSOCIATION_COLUMNS,
    MAINLINE_DETECTION_COLUMNS,
    MAINLINE_FRAME_COLUMNS,
    MAINLINE_OBSERVATION_COLUMNS,
    MAINLINE_STAGE_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]


class BuildContractTests(unittest.TestCase):
    def test_native_mainline_is_the_default_runtime_and_build_target(self) -> None:
        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        nx_preset = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "nx-mainline-release"
        )
        runtime = yaml.safe_load(
            (ROOT / "configs" / "defaults" / "runtime.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            nx_preset["cacheVariables"]["VOLLEYBALL_BUILD_NX_MAINLINE"],
            "ON",
        )
        self.assertEqual(runtime["adapters"]["active_pipeline"], "mainline")
        self.assertEqual(
            runtime["adapters"]["mainline_pipeline"]["executable"],
            "build/nx-mainline-release/apps/nx_mainline/volleyball_nx_mainline",
        )

    def test_nx_preset_requires_hardware_dependencies(self) -> None:
        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        nx_preset = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "nx-legacy-release"
        )
        cache = nx_preset["cacheVariables"]
        self.assertEqual(cache["VOLLEYBALL_BUILD_LEGACY"], "ON")
        self.assertEqual(cache["VOLLEYBALL_REQUIRE_HIK"], "ON")
        self.assertEqual(cache["VOLLEYBALL_REQUIRE_ROS2"], "ON")

    def test_hik_requirement_reaches_adapter_configure(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        adapter_cmake = (
            ROOT / "adapters" / "legacy_nx" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"-DREQUIRE_HIK=${VOLLEYBALL_REQUIRE_HIK}"', root_cmake
        )
        self.assertIn('option(REQUIRE_HIK ', adapter_cmake)
        self.assertIn("if(REQUIRE_HIK)", adapter_cmake)
        self.assertIn("message(FATAL_ERROR", adapter_cmake)

    def test_legacy_build_and_runtime_share_the_preset_build_tree(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        runtime = yaml.safe_load(
            (ROOT / "configs" / "defaults" / "runtime.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertIn(
            'set(VOLLEYBALL_LEGACY_NX_BINARY_DIR "${CMAKE_BINARY_DIR}/legacy_nx")',
            root_cmake,
        )
        self.assertIn(
            'BINARY_DIR "${VOLLEYBALL_LEGACY_NX_BINARY_DIR}"', root_cmake
        )
        self.assertEqual(
            runtime["adapters"]["legacy_pipeline"]["executable"],
            "build/nx-legacy-superbuild/legacy_nx/stereo_pipeline",
        )
        self.assertNotIn("adapters/legacy_nx/build", root_cmake)

    def test_legacy_pipeline_consumes_the_standalone_hik_adapter(self) -> None:
        legacy_cmake = (
            ROOT / "adapters" / "legacy_nx" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        hik_cmake = (
            ROOT / "adapters" / "hik_mvs" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

        self.assertNotIn("src/capture/hikvision_camera.cpp", legacy_cmake)
        self.assertIn("volleyball::hik_mvs", legacy_cmake)
        self.assertIn("VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR", hik_cmake)
        self.assertIn("VOLLEYBALL_HIK_MVS_SDK_LIBRARY", hik_cmake)
        self.assertIn("src/hikvision_camera.cpp", hik_cmake)

    def test_legacy_pipeline_consumes_the_standalone_tensorrt_cuda_adapter(self) -> None:
        legacy_cmake = (
            ROOT / "adapters" / "legacy_nx" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        adapter_cmake = (
            ROOT / "adapters" / "tensorrt" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

        self.assertNotIn("src/detect/detect_preprocess.cu", legacy_cmake)
        self.assertNotIn("src/detect/dfl_postprocess.cu", legacy_cmake)
        self.assertNotIn("src/detect/trt_detector.cpp", legacy_cmake)
        self.assertIn("volleyball::tensorrt_runtime", legacy_cmake)
        self.assertIn("src/cuda/preprocess.cu", adapter_cmake)
        self.assertIn("src/cuda/dfl_decode.cu", adapter_cmake)
        self.assertIn("src/runtime.cpp", adapter_cmake)
        self.assertIn("src/detector_adapter.cpp", adapter_cmake)
        self.assertIn("volleyball::tensorrt_detector", adapter_cmake)

    def test_nx_build_compiles_the_standalone_cuda_preprocessor(self) -> None:
        legacy_cmake = (
            ROOT / "adapters" / "legacy_nx" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        adapter_cmake = (
            ROOT / "adapters" / "cuda_preprocess" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("volleyball::cuda_stereo_preprocessor", legacy_cmake)
        self.assertIn("src/cuda_stereo_preprocessor.cpp", adapter_cmake)
        self.assertIn("src/cuda/preprocess_kernels.cu", adapter_cmake)

    def test_nx_build_compiles_the_standalone_ros2_publisher(self) -> None:
        legacy_cmake = (
            ROOT / "adapters" / "legacy_nx" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        adapter_cmake = (
            ROOT / "adapters" / "ros2" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("VOLLEYBALL_BUILD_ROS2_ADAPTER ON", legacy_cmake)
        self.assertIn("volleyball::ros2_publisher", legacy_cmake)
        self.assertIn("src/observation_publisher.cpp", adapter_cmake)
        self.assertIn("ament_target_dependencies", adapter_cmake)

    def test_gpio_layout_switch_is_part_of_the_public_contract(self) -> None:
        gpio_cmake = (
            ROOT / "adapters" / "gpio" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "target_compile_definitions(volleyball_gpio PUBLIC NO_GPIOD=1)",
            gpio_cmake,
        )
        self.assertNotIn(
            "target_compile_definitions(volleyball_gpio PRIVATE NO_GPIOD=1)",
            gpio_cmake,
        )

    def test_cpp_and_python_capture_stage_contracts_match(self) -> None:
        header = (
            ROOT
            / "adapters"
            / "capture_v21"
            / "include"
            / "volleyball"
            / "adapters"
            / "capture_v21"
            / "stage_schema.hpp"
        ).read_text(encoding="utf-8")

        version = re.search(
            r"kCaptureV21StageSchemaVersion\s*=\s*(\d+)\s*;",
            header,
        )
        self.assertIsNotNone(version)
        assert version is not None
        self.assertEqual(int(version.group(1)), STAGE_SCHEMA_VERSION)

        def columns(name: str) -> list[str]:
            declaration = re.search(
                rf"\b{name}\s*=\s*\{{(?P<body>.*?)\}};",
                header,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(declaration, name)
            assert declaration is not None
            return re.findall(r'"([^"\\]+)"', declaration.group("body"))

        self.assertEqual(columns("kCaptureV21FrameColumns"), STAGE_FRAME_COLUMNS)
        self.assertEqual(
            columns("kCaptureV21DetectionColumns"),
            STAGE_DETECTION_COLUMNS,
        )
        self.assertEqual(columns("kCaptureV21PairColumns"), STAGE_PAIR_COLUMNS)

    def test_cpp_and_python_mainline_stage_contracts_match(self) -> None:
        header = (
            ROOT
            / "adapters"
            / "capture_v21"
            / "include"
            / "volleyball"
            / "adapters"
            / "capture_v21"
            / "mainline_stage_schema.hpp"
        ).read_text(encoding="utf-8")
        version = re.search(
            r"kCaptureV21MainlineStageSchemaVersion\s*=\s*(\d+)\s*;",
            header,
        )
        self.assertIsNotNone(version)
        assert version is not None
        self.assertEqual(int(version.group(1)), MAINLINE_STAGE_SCHEMA_VERSION)

        def columns(name: str) -> list[str]:
            declaration = re.search(
                rf"\b{name}\s*=\s*\{{(?P<body>.*?)\}};",
                header,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(declaration, name)
            assert declaration is not None
            return re.findall(r'"([^"\\]+)"', declaration.group("body"))

        self.assertEqual(
            columns("kCaptureV21MainlineFrameColumns"), MAINLINE_FRAME_COLUMNS
        )
        self.assertEqual(
            columns("kCaptureV21MainlineDetectionColumns"),
            MAINLINE_DETECTION_COLUMNS,
        )
        self.assertEqual(
            columns("kCaptureV21MainlineAssociationColumns"),
            MAINLINE_ASSOCIATION_COLUMNS,
        )
        self.assertEqual(
            columns("kCaptureV21MainlineObservationColumns"),
            MAINLINE_OBSERVATION_COLUMNS,
        )


if __name__ == "__main__":
    unittest.main()
