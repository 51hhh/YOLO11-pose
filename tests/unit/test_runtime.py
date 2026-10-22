from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
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

    def test_dry_run_writes_auditable_legacy_projection(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pipeline.yaml"
            result = run_runtime(
                profile,
                generated_config=destination,
                extra_arguments=("--no-display",),
                dry_run=True,
                project_root=ROOT,
            )
            self.assertTrue(result.dry_run)
            self.assertIsNone(result.returncode)
            self.assertEqual(result.generated_config, destination)
            self.assertEqual(result.command[1:3], ("--config", str(destination)))
            self.assertEqual(result.command[-1], "--no-display")
            rendered = yaml.safe_load(destination.read_text(encoding="utf-8"))
            self.assertFalse(rendered["recording"]["enabled"])
            self.assertFalse(rendered["capture_v21"]["enabled"])
            self.assertTrue(rendered["ros2"]["enable"])

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


if __name__ == "__main__":
    unittest.main()
