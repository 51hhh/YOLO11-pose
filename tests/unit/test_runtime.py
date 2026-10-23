from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
import tempfile
import unittest

import yaml

from volleyball_vision.config import load_profile
from volleyball_vision.cli import main
from volleyball_vision.dataset import DatasetError
from volleyball_vision.runtime import run_runtime


ROOT = Path(__file__).resolve().parents[2]


class RuntimeTests(unittest.TestCase):
    def test_cli_dry_run_does_not_require_target_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pipeline.yaml"
            output = io.StringIO()
            with redirect_stdout(output):
                returncode = main(
                    (
                        "run",
                        "--profile",
                        "nx-rdk",
                        "--generated-config",
                        str(destination),
                        "--dry-run",
                    )
                )
            self.assertEqual(returncode, 0)
            self.assertTrue(destination.is_file())
            self.assertIn("DRY_RUN", output.getvalue())

    def test_dry_run_writes_auditable_mainline_projection(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pipeline.yaml"
            result = run_runtime(
                profile,
                generated_config=destination,
                dry_run=True,
                project_root=ROOT,
            )
            self.assertTrue(result.dry_run)
            self.assertIsNone(result.returncode)
            self.assertEqual(result.generated_config, destination)
            self.assertEqual(
                result.command[1:], ("--config", str(destination))
            )
            rendered = yaml.safe_load(destination.read_text(encoding="utf-8"))
            self.assertNotIn("recording", rendered)
            self.assertEqual(rendered["runtime"]["mode"], "runtime")
            self.assertIsNone(rendered["runtime"]["capture_stage"])
            self.assertTrue(rendered["publisher"]["enabled"])
            self.assertEqual(
                Path(result.command[0]),
                ROOT
                / "build"
                / "nx-mainline-release"
                / "apps"
                / "nx_mainline"
                / "volleyball_nx_mainline",
            )

    def test_generated_config_never_overwrites_existing_file(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pipeline.yaml"
            destination.write_text("owned-by-user\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "无法写入"):
                run_runtime(
                    profile,
                    generated_config=destination,
                    dry_run=True,
                    project_root=ROOT,
                )
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "owned-by-user\n",
            )

    def test_replay_profile_cannot_start_hardware_runtime(self) -> None:
        profile = load_profile("replay", project_root=ROOT)
        with self.assertRaisesRegex(DatasetError, "runtime profile"):
            run_runtime(profile, dry_run=True, project_root=ROOT)

    def test_capture_dry_run_does_not_require_hardware_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                returncode = main(
                    (
                        "capture",
                        "--profile",
                        "free-flight",
                        "--trial",
                        "dry-run-001",
                        "--session",
                        "dry-run-session",
                        "--ball",
                        "dry-run-ball",
                        "--duration",
                        "3.0",
                        "--dataset-root",
                        directory,
                        "--dry-run",
                    )
                )
            self.assertEqual(returncode, 0)
            self.assertIn("DRY_RUN stage=", output.getvalue())
            stages = list((Path(directory) / ".capture-staging").iterdir())
            self.assertEqual(len(stages), 1)
            context = yaml.safe_load(
                (stages[0] / "context.json").read_text(encoding="utf-8")
            )
            rendered = yaml.safe_load(
                (stages[0] / "pipeline.yaml").read_text(encoding="utf-8")
            )
            self.assertTrue(context["planning_only"])
            self.assertEqual(context["stage_schema_version"], 6)
            self.assertIsNone(
                context["perception_pipeline"]["weights_sha256"]
            )
            self.assertEqual(rendered["runtime"]["mode"], "capture")
            self.assertEqual(rendered["runtime"]["frame_limit"], 270)
            self.assertEqual(rendered["runtime"]["capture_stage"], str(stages[0]))
            self.assertFalse(rendered["publisher"]["enabled"])

    def test_capture_rejects_non_capture_profile_before_writing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            error = io.StringIO()
            with redirect_stderr(error):
                returncode = main(
                    (
                        "capture",
                        "--profile",
                        "replay",
                        "--trial",
                        "wrong-profile",
                        "--session",
                        "dry-run-session",
                        "--ball",
                        "dry-run-ball",
                        "--dataset-root",
                        directory,
                        "--dry-run",
                    )
                )
            self.assertEqual(returncode, 2)
            self.assertIn("只接受 capture profile", error.getvalue())
            self.assertFalse((Path(directory) / ".capture-staging").exists())

    def test_capture_rejects_invalid_identity_before_writing_stage(self) -> None:
        invalid_arguments = (
            ("--trial", "../outside"),
            ("--session", "/tmp/outside"),
            ("--ball", "invalid/ball"),
        )
        for flag, invalid_value in invalid_arguments:
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as directory:
                    values = {
                        "--trial": "trial-001",
                        "--session": "session-001",
                        "--ball": "ball-001",
                    }
                    values[flag] = invalid_value
                    error = io.StringIO()
                    with redirect_stderr(error):
                        returncode = main(
                            (
                                "capture",
                                "--profile",
                                "free-flight",
                                "--trial",
                                values["--trial"],
                                "--session",
                                values["--session"],
                                "--ball",
                                values["--ball"],
                                "--dataset-root",
                                directory,
                                "--dry-run",
                            )
                        )
                    self.assertEqual(returncode, 2)
                    self.assertIn("有效 identifier", error.getvalue())
                    self.assertFalse(
                        (Path(directory) / ".capture-staging").exists()
                    )


if __name__ == "__main__":
    unittest.main()
