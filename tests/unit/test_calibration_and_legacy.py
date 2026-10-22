from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from volleyball_vision.calibration import (
    build_capture_calibration,
    load_opencv_stereo_calibration,
)
from volleyball_vision.config import load_profile
from volleyball_vision.legacy_config import build_legacy_pipeline_config


ROOT = Path(__file__).resolve().parents[2]


class CalibrationAndLegacyConfigTests(unittest.TestCase):
    def test_opencv_calibration_is_converted_as_rectified_rig(self) -> None:
        source = ROOT / "NX_volleyball" / "calibration" / "stereo_calib.yaml"
        loaded = load_opencv_stereo_calibration(source)
        self.assertEqual((loaded.image_width, loaded.image_height), (1440, 1080))
        self.assertAlmostEqual(loaded.baseline_m, 0.8531543824782482)
        profile = load_profile("free-flight", project_root=ROOT)
        converted = build_capture_calibration(
            profile,
            project_root=ROOT,
            created_at_utc="2026-10-22T00:00:00Z",
        )
        self.assertEqual(converted["cameras"][0]["distortion_model"], "rectified_pinhole")
        self.assertEqual(
            converted["cameras"][1]["transform_body_from_camera"][0][3],
            converted["baseline_m"],
        )
        self.assertEqual(
            converted["cameras"][0]["clock_domain"],
            "hik-shared-device-clock",
        )

    def test_legacy_projection_disables_lossy_async_capture_path(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            rendered = build_legacy_pipeline_config(
                profile,
                project_root=ROOT,
                capture_stage=directory,
                frame_limit=270,
            )
        self.assertFalse(rendered["performance"]["drop_stale_roi_frames"])
        self.assertFalse(rendered["performance"]["async_roi_stage2"])
        self.assertFalse(rendered["recording"]["enabled"])
        self.assertFalse(rendered["ros2"]["enable"])
        self.assertEqual(rendered["capture_v21"]["overflow_policy"], "block")
        self.assertEqual(rendered["capture_v21"]["frame_limit"], 270)
        self.assertEqual(
            rendered["capture_v21"]["exposure_timestamp_source"],
            "nx_gpio_rising_edge",
        )
        self.assertEqual(
            rendered["detector"]["dual_yolo"]["epipolar_y_tolerance"],
            profile.data["stereo"]["association"]["epipolar_tolerance_px"],
        )

    def test_runtime_projection_has_no_hidden_legacy_side_effects(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        rendered = build_legacy_pipeline_config(profile, project_root=ROOT)
        self.assertFalse(rendered["recording"]["enabled"])
        self.assertFalse(rendered["baseline_recording"]["enabled"])
        self.assertFalse(rendered["prediction"]["enable"])
        self.assertTrue(rendered["ros2"]["enable"])
        self.assertTrue(rendered["ros2"]["nx_observation"]["enable"])
        self.assertEqual(
            rendered["ros2"]["nx_observation"]["topic"],
            "/nx/ball/observation",
        )
        self.assertEqual(
            rendered["ros2"]["nx_observation"]["frame_id"],
            "nx_left_rectified_optical_frame",
        )
        self.assertFalse(rendered["ros2"]["control_goal"]["enable"])
        self.assertFalse(rendered["ros2"]["diagnostic"]["enable"])
        self.assertFalse(rendered["debug_realtime_dump"]["enabled"])


if __name__ == "__main__":
    unittest.main()

