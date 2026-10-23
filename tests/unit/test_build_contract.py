from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

import yaml

from volleyball_vision.mainline_stage_contract import (
    MAINLINE_ASSOCIATION_COLUMNS,
    MAINLINE_DETECTION_COLUMNS,
    MAINLINE_FRAME_COLUMNS,
    MAINLINE_OBSERVATION_COLUMNS,
    MAINLINE_STAGE_SCHEMA_VERSION,
)
from volleyball_vision.stereo_measurements import (
    STEREO_MEASUREMENT_COLUMNS,
    STEREO_MEASUREMENT_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]


class BuildContractTests(unittest.TestCase):
    def test_native_mainline_is_the_only_nx_build_and_runtime_target(self) -> None:
        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        names = {preset["name"] for preset in presets["configurePresets"]}
        self.assertIn("nx-mainline-release", names)
        self.assertFalse(any("legacy" in name for name in names))

        nx_preset = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "nx-mainline-release"
        )
        self.assertEqual(
            nx_preset["cacheVariables"]["VOLLEYBALL_BUILD_NX_MAINLINE"],
            "ON",
        )
        self.assertEqual(
            nx_preset["cacheVariables"]["VOLLEYBALL_BUILD_HARDWARE_TOOLS"],
            "ON",
        )

        calibration_preset = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "host-calibration-release"
        )
        self.assertEqual(
            calibration_preset["cacheVariables"][
                "VOLLEYBALL_BUILD_STEREO_CALIBRATE"
            ],
            "ON",
        )
        self.assertEqual(
            calibration_preset["cacheVariables"]["VOLLEYBALL_BUILD_TESTS"],
            "OFF",
        )
        self.assertNotIn(
            "VOLLEYBALL_BUILD_HARDWARE_TOOLS",
            calibration_preset["cacheVariables"],
        )

        runtime = yaml.safe_load(
            (ROOT / "configs" / "defaults" / "runtime.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            runtime["adapters"]["mainline_pipeline"]["executable"],
            "build/nx-mainline-release/apps/nx_mainline/volleyball_nx_mainline",
        )
        self.assertNotIn("active_pipeline", runtime["adapters"])
        self.assertNotIn("legacy_pipeline", runtime["adapters"])

    def test_native_app_links_each_hardware_boundary(self) -> None:
        app_cmake = (ROOT / "apps" / "nx_mainline" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        for target in (
            "volleyball::capture_stage",
            "volleyball::cuda_direct_observation_estimator",
            "volleyball::cuda_stereo_preprocessor",
            "volleyball::gpio",
            "volleyball::hik_mvs",
            "volleyball::tensorrt_detector",
            "volleyball::ros2_publisher",
            "volleyball::services",
        ):
            with self.subTest(target=target):
                self.assertIn(target, app_cmake)

    def test_hardware_tools_are_owned_by_the_root_build(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        tools_cmake = (
            ROOT / "tools" / "hardware" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("add_subdirectory(tools/hardware)", root_cmake)
        self.assertIn("volleyball::hik_mvs", tools_cmake)
        self.assertIn("volleyball::gpio", tools_cmake)

    def test_cuda_probe_failure_reports_toolchain_incompatibility(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("could not compile a ", root_cmake)
        self.assertIn("CUDA compiler probe", root_cmake)
        self.assertIn("host compiler/C library", root_cmake)
        self.assertIn("CMakeConfigureLog.yaml", root_cmake)

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

    def test_hardware_builds_require_gpiod_without_sysfs_fallback(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        gpio_source = (
            ROOT / "adapters" / "gpio" / "src" / "pwm_trigger.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "VOLLEYBALL_BUILD_NX_MAINLINE OR VOLLEYBALL_BUILD_HARDWARE_TOOLS",
            root_cmake,
        )
        self.assertIn(
            "set(VOLLEYBALL_GPIO_WITH_GPIOD ON CACHE BOOL \"\" FORCE)",
            root_cmake,
        )
        self.assertNotIn("/sys/class/gpio", gpio_source)
        self.assertNotIn("sysfs", gpio_source.lower())

    def test_cpp_and_python_mainline_stage_contracts_match(self) -> None:
        header = (
            ROOT
            / "adapters"
            / "capture_stage"
            / "include"
            / "volleyball"
            / "adapters"
            / "capture_stage"
            / "mainline_stage_schema.hpp"
        ).read_text(encoding="utf-8")
        version = re.search(
            r"kMainlineStageSchemaVersion\s*=\s*(\d+)\s*;",
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
            columns("kMainlineStageFrameColumns"), MAINLINE_FRAME_COLUMNS
        )
        self.assertEqual(
            columns("kMainlineStageDetectionColumns"),
            MAINLINE_DETECTION_COLUMNS,
        )
        self.assertEqual(
            columns("kMainlineStageAssociationColumns"),
            MAINLINE_ASSOCIATION_COLUMNS,
        )
        self.assertEqual(
            columns("kMainlineStageObservationColumns"),
            MAINLINE_OBSERVATION_COLUMNS,
        )
        measurement_version = re.search(
            r'kStereoMeasurementSchemaVersion\s*=\s*"([^"]+)"\s*;',
            header,
        )
        self.assertIsNotNone(measurement_version)
        assert measurement_version is not None
        self.assertEqual(
            measurement_version.group(1),
            STEREO_MEASUREMENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            columns("kStereoMeasurementColumns"),
            STEREO_MEASUREMENT_COLUMNS,
        )


if __name__ == "__main__":
    unittest.main()
