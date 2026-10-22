from __future__ import annotations

import json
from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
