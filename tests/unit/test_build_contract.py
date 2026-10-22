from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from volleyball_vision.capture import (
    STAGE_DETECTION_COLUMNS,
    STAGE_FRAME_COLUMNS,
    STAGE_PAIR_COLUMNS,
    STAGE_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]


class BuildContractTests(unittest.TestCase):
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

    def test_cpp_and_python_capture_stage_contracts_match(self) -> None:
        header = (
            ROOT
            / "adapters"
            / "legacy_nx"
            / "src"
            / "capture"
            / "capture_v21_stage_schema.h"
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


if __name__ == "__main__":
    unittest.main()
