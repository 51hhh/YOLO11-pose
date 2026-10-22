from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from volleyball_vision.runtime_dependencies import (
    probe_dynamic_linkage,
    probe_live_sdk_dependencies,
)


class RuntimeDependencyTests(unittest.TestCase):
    def test_missing_live_sdk_components_are_reported_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probes = probe_live_sdk_dependencies(directory)

        by_name = {probe.name: probe for probe in probes}
        self.assertEqual(
            set(by_name),
            {"tensorrt_sdk", "vpi_sdk", "opencv_sdk"},
        )
        for probe in probes:
            with self.subTest(name=probe.name):
                self.assertFalse(probe.available)
                self.assertIn("缺少", probe.message)

    def test_complete_fake_live_sdk_tree_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                "usr/include/aarch64-linux-gnu/NvInfer.h",
                "usr/lib/aarch64-linux-gnu/libnvinfer.so",
                "usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so",
                "opt/nvidia/vpi3/include/vpi/VPI.h",
                "opt/nvidia/vpi3/lib/aarch64-linux-gnu/libnvvpi.so",
                "usr/include/opencv4/opencv2/core.hpp",
            ]
            files.extend(
                f"usr/lib/aarch64-linux-gnu/libopencv_{component}.so"
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
            )
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            probes = probe_live_sdk_dependencies(root)

        self.assertTrue(all(probe.available for probe in probes))

    def test_mainline_sdk_contract_excludes_vpi_and_legacy_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = (
                "usr/include/aarch64-linux-gnu/NvInfer.h",
                "usr/lib/aarch64-linux-gnu/libnvinfer.so",
                "usr/include/opencv4/opencv2/core.hpp",
                "usr/lib/aarch64-linux-gnu/libopencv_core.so",
                "usr/lib/aarch64-linux-gnu/libopencv_calib3d.so",
            )
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            probes = probe_live_sdk_dependencies(root, pipeline="mainline")

        by_name = {probe.name: probe for probe in probes}
        self.assertEqual(set(by_name), {"tensorrt_sdk", "opencv_sdk"})
        self.assertTrue(all(probe.available for probe in probes))

    def test_dynamic_linkage_rejects_unresolved_library(self) -> None:
        def runner(_command: object) -> tuple[int, str]:
            return 0, "libnvinfer.so.8 => not found\n"

        probe = probe_dynamic_linkage(
            "/tmp/stereo_pipeline",
            "/usr/bin/ldd",
            (("TensorRT", "libnvinfer"),),
            command_runner=runner,
        )
        self.assertFalse(probe.available)
        self.assertIn("libnvinfer.so.8", probe.message)

    def test_dynamic_linkage_requires_profile_frameworks(self) -> None:
        def runner(_command: object) -> tuple[int, str]:
            return 0, "libnvvpi.so.3 => /opt/nvidia/vpi3/lib/libnvvpi.so.3\n"

        probe = probe_dynamic_linkage(
            "/tmp/stereo_pipeline",
            "/usr/bin/ldd",
            (("VPI", "libnvvpi"), ("TensorRT", "libnvinfer")),
            command_runner=runner,
        )
        self.assertFalse(probe.available)
        self.assertIn("TensorRT(libnvinfer)", probe.message)

    def test_dynamic_linkage_accepts_resolved_profile_frameworks(self) -> None:
        def runner(_command: object) -> tuple[int, str]:
            return (
                0,
                "libnvvpi.so.3 => /opt/nvidia/vpi3/lib/libnvvpi.so.3\n"
                "libnvinfer.so.8 => /usr/lib/libnvinfer.so.8\n",
            )

        probe = probe_dynamic_linkage(
            "/tmp/stereo_pipeline",
            "/usr/bin/ldd",
            (("VPI", "libnvvpi"), ("TensorRT", "libnvinfer")),
            command_runner=runner,
        )
        self.assertTrue(probe.available)


if __name__ == "__main__":
    unittest.main()
