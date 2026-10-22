from __future__ import annotations

from pathlib import Path
import unittest

from volleyball_vision.calibration import (
    build_capture_calibration,
    load_opencv_stereo_calibration,
)
from volleyball_vision.config import load_profile


ROOT = Path(__file__).resolve().parents[2]


class CalibrationTests(unittest.TestCase):
    def test_opencv_calibration_is_converted_as_rectified_rig(self) -> None:
        source = ROOT / "configs" / "calibration" / "stereo_calib.yaml"
        loaded = load_opencv_stereo_calibration(source)
        self.assertEqual((loaded.image_width, loaded.image_height), (1440, 1080))
        self.assertAlmostEqual(loaded.baseline_m, 0.8531543824782482)

        profile = load_profile("free-flight", project_root=ROOT)
        converted = build_capture_calibration(profile, project_root=ROOT)
        self.assertEqual(
            converted["created_at_utc"],
            profile.data["calibration"]["created_at_utc"],
        )
        self.assertEqual(
            converted["cameras"][0]["distortion_model"],
            "rectified_pinhole",
        )
        self.assertEqual(
            converted["cameras"][1]["transform_body_from_camera"][0][3],
            converted["baseline_m"],
        )
        self.assertEqual(
            converted["cameras"][0]["clock_domain"],
            "hik-shared-device-clock",
        )
        self.assertEqual(
            converted["body_frame_id"],
            "nx_left_rectified_optical_frame",
        )


if __name__ == "__main__":
    unittest.main()
