from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from volleyball_vision.config import load_profile
from volleyball_vision.doctor import run_doctor


ROOT = Path(__file__).resolve().parents[2]


class DoctorTests(unittest.TestCase):
    def test_replay_does_not_require_hardware_environment(self) -> None:
        profile = load_profile("replay", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            checks = run_doctor(
                profile,
                project_root=ROOT,
                system_root=directory,
                machine="x86_64",
                environment={},
                command_lookup=lambda command: f"/usr/bin/{command}",
            )
        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["model"].status, "skipped")
        self.assertEqual(by_name["pipeline_executable"].status, "skipped")
        self.assertNotIn("gpio_device", by_name)
        self.assertNotIn("hik_mvs_runtime", by_name)
        self.assertNotIn("cuda_runtime", by_name)
        self.assertNotIn("tensorrt_sdk", by_name)
        self.assertNotIn("vpi_sdk", by_name)
        self.assertNotIn("opencv_sdk", by_name)
        self.assertNotIn("pipeline_backend_config", by_name)

    def test_nx_profile_rejects_non_jetson_host_and_missing_devices(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            checks = run_doctor(
                profile,
                project_root=ROOT,
                system_root=directory,
                machine="x86_64",
                environment={},
                command_lookup=lambda _command: None,
            )
        by_name = {check.name: check for check in checks}
        for name in (
            "platform_architecture",
            "jetson_platform",
            "gpio_device",
            "hik_mvs_runtime",
            "cuda_runtime",
            "tensorrt_sdk",
            "opencv_sdk",
            "ros2_environment",
        ):
            with self.subTest(name=name):
                self.assertEqual(by_name[name].status, "error")
        self.assertNotIn("vpi_sdk", by_name)
        self.assertEqual(by_name["pipeline_backend_config"].status, "ok")
        self.assertEqual(by_name["pipeline_linkage"].status, "skipped")

    def test_free_flight_does_not_require_ros2_shell(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            checks = run_doctor(
                profile,
                project_root=ROOT,
                system_root=directory,
                machine="aarch64",
                environment={},
                command_lookup=lambda _command: None,
            )
        self.assertNotIn("ros2_environment", {check.name for check in checks})


if __name__ == "__main__":
    unittest.main()
