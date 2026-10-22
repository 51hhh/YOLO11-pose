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
        self.assertEqual(rendered["camera"]["trigger_chip"], "gpiochip2")
        self.assertEqual(rendered["camera"]["trigger_line"], 7)
        self.assertEqual(rendered["camera"]["image_node_num"], 3)
        self.assertEqual(rendered["camera"]["embedded_info_clear_rows"], 2)
        self.assertEqual(rendered["camera"]["shutter_type"], "global")
        self.assertEqual(
            rendered["camera"]["exposure_timestamp_source"],
            "nx_gpio_rising_edge",
        )
        self.assertEqual(
            rendered["camera"]["exposure_delay_status"],
            "assumed_zero_unmeasured",
        )
        self.assertEqual(rendered["camera"]["exposure_delay_ns"], 0)
        self.assertIsNone(
            rendered["camera"]["exposure_delay_uncertainty_ns"]
        )
        self.assertEqual(
            rendered["camera"]["exposure_midpoint_offset_ns"],
            4_933_500,
        )
        self.assertEqual(rendered["performance"]["pwm_duty_cycle"], 50.0)
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
        self.assertEqual(rendered["performance"]["log_interval"], 900)
        self.assertFalse(rendered["capture_v21"]["enabled"])
        self.assertFalse(rendered["disparity_offset"]["enabled"])
        self.assertNotIn("file", rendered["disparity_offset"])
        self.assertFalse(rendered["tnr"]["enabled"])
        self.assertFalse(rendered["tracker"]["enabled"])
        self.assertFalse(rendered["display"]["enable"])
        self.assertEqual(rendered["stereo"]["strategy"], "roi_only")
        self.assertTrue(rendered["detector"]["dual_yolo"]["enabled"])
        self.assertTrue(rendered["detector"]["dual_yolo"]["use_for_depth"])
        self.assertFalse(rendered["neural_feature_matching_xfeat"]["enabled"])
        self.assertFalse(
            rendered["neural_feature_matching_superpoint"]["enabled"]
        )
        self.assertFalse(rendered["neural_feature_matching_aliked"]["enabled"])

        enabled_depth_modes = {
            name
            for name, enabled in rendered["detector"]["dual_yolo"][
                "depth_modes"
            ].items()
            if enabled
        }
        self.assertEqual(
            enabled_depth_modes,
            {
                "bbox_pair",
                "bbox_edges",
                "circle_center",
                "roi_edge_centroid",
                "roi_radial_center",
                "roi_edge_pair_center",
                "roi_cuda_template_match",
                "roi_center_patch",
                "roi_subpixel",
                "epipolar_fallback",
            },
        )
        self.assertEqual(
            rendered["detector"]["dual_yolo"]["depth_solver"],
            "roi_subpixel_match",
        )

    def test_candidate_override_fully_replaces_legacy_template_modes(self) -> None:
        profile = load_profile(
            "nx-rdk",
            project_root=ROOT,
            overrides=(
                "stereo.triangulation.primary_method=circle_center",
                "stereo.triangulation.candidate_methods=[circle_center,bbox_center]",
                "stereo.triangulation.fallback_methods=[bbox_center]",
            ),
        )
        rendered = build_legacy_pipeline_config(profile, project_root=ROOT)
        dual = rendered["detector"]["dual_yolo"]
        enabled_depth_modes = {
            name for name, enabled in dual["depth_modes"].items() if enabled
        }
        self.assertEqual(enabled_depth_modes, {"circle_center", "bbox_pair"})
        self.assertEqual(dual["depth_solver"], "circle_center")
        self.assertFalse(dual["subpixel_enabled"])
        self.assertFalse(dual["fallback_epipolar_search"])

    def test_active_geometry_tuning_is_projected_from_main_config(self) -> None:
        profile = load_profile(
            "nx-rdk",
            project_root=ROOT,
            overrides=(
                "stereo.association.bbox_disparity_consistency_ratio=0.22",
                "stereo.association.bbox_disparity_consistency_min_px=31.0",
                "stereo.association.bbox_disparity_penalty_scale=0.61",
                "stereo.triangulation.subpixel.max_disparity_delta_px=1.7",
                "stereo.triangulation.subpixel.max_disparity_delta_ratio=0.04",
                "stereo.triangulation.subpixel.max_depth_delta_m=0.35",
                "stereo.triangulation.fallback_search.margin_px=36",
                "stereo.triangulation.fallback_search.max_width_px=180",
                "stereo.triangulation.circle_fit.max_roi_pixels=12000",
            ),
        )
        rendered = build_legacy_pipeline_config(profile, project_root=ROOT)
        dual = rendered["detector"]["dual_yolo"]
        self.assertEqual(dual["bbox_disparity_consistency_ratio"], 0.22)
        self.assertEqual(dual["bbox_disparity_consistency_min_px"], 31.0)
        self.assertEqual(dual["bbox_disparity_penalty_scale"], 0.61)
        self.assertEqual(dual["subpixel_max_disp_delta_px"], 1.7)
        self.assertEqual(dual["subpixel_max_disp_delta_ratio"], 0.04)
        self.assertEqual(dual["subpixel_max_depth_delta_m"], 0.35)
        self.assertEqual(dual["fallback_search_margin_px"], 36)
        self.assertEqual(dual["fallback_max_width_px"], 180)
        self.assertEqual(dual["circle_max_roi_pixels"], 12000)


if __name__ == "__main__":
    unittest.main()

