from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from volleyball_vision.config import ConfigError, load_profile
from volleyball_vision.mainline_config import (
    MAINLINE_CONFIG_SCHEMA_VERSION,
    build_mainline_pipeline_config,
    write_mainline_pipeline_config,
)


ROOT = Path(__file__).resolve().parents[2]


class MainlineConfigTests(unittest.TestCase):
    def test_runtime_projection_contains_only_resolved_launch_facts(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        payload = build_mainline_pipeline_config(profile, project_root=ROOT)

        self.assertEqual(payload["schema_version"], MAINLINE_CONFIG_SCHEMA_VERSION)
        self.assertEqual(payload["config_sha256"], profile.fingerprint_sha256)
        self.assertEqual(payload["runtime"]["mode"], "runtime")
        self.assertIsNone(payload["runtime"]["frame_limit"])
        self.assertIsNone(payload["runtime"]["capture_stage"])
        self.assertTrue(payload["publisher"]["enabled"])
        self.assertTrue(Path(payload["calibration"]["file"]).is_absolute())
        self.assertTrue(Path(payload["detector"]["engine_file"]).is_absolute())
        self.assertEqual(payload["detector"]["output_contract"], "yolo_pre_nms")
        self.assertEqual(payload["detector"]["class_count"], 1)
        self.assertEqual(payload["detector"]["class_id"], 0)
        self.assertIsInstance(payload["association"]["max_capture_delta_ns"], int)
        self.assertEqual(payload["association"]["max_capture_delta_ns"], 100_000)
        self.assertEqual(
            payload["depth"]["method_order"],
            [
                "circle_center",
                "radial_center",
                "edge_pair_center",
                "edge_centroid",
                "bbox_center",
                "bbox_edges",
            ],
        )

    def test_capture_projection_requires_finite_stage_and_disables_publish(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage"
            payload = build_mainline_pipeline_config(
                profile,
                capture_stage=stage,
                frame_limit=270,
                project_root=ROOT,
            )
        self.assertEqual(payload["runtime"]["mode"], "capture")
        self.assertEqual(payload["runtime"]["frame_limit"], 270)
        self.assertEqual(payload["runtime"]["capture_stage"], str(stage))
        self.assertFalse(payload["publisher"]["enabled"])
        self.assertEqual(payload["runtime"]["max_consecutive_timeouts"], 0)

    def test_capture_projection_rejects_missing_stage(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with self.assertRaisesRegex(ConfigError, "staging"):
            build_mainline_pipeline_config(profile, project_root=ROOT)

    def test_profile_rejects_computed_candidate_without_selection_path(self) -> None:
        with self.assertRaisesRegex(ConfigError, "必须进入"):
            load_profile(
                "nx-rdk",
                project_root=ROOT,
                overrides=(
                    "stereo.triangulation.candidate_methods="
                    "[circle_center,bbox_center]",
                    "stereo.triangulation.fallback_methods=[]",
                ),
            )

    def test_writer_is_exclusive_and_round_trips_yaml(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "mainline.yaml"
            written = write_mainline_pipeline_config(
                destination, profile, project_root=ROOT
            )
            payload = yaml.safe_load(written.read_text(encoding="utf-8"))
            self.assertEqual(payload["profile"], "nx-rdk")
            with self.assertRaises(FileExistsError):
                write_mainline_pipeline_config(
                    destination, profile, project_root=ROOT
                )


if __name__ == "__main__":
    unittest.main()
