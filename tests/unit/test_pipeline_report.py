from __future__ import annotations

from pathlib import Path
import unittest

from volleyball_vision.config import load_profile
from volleyball_vision.pipeline_report import build_pipeline_report


ROOT = Path(__file__).resolve().parents[2]


class PipelineReportTests(unittest.TestCase):
    def _report(self, profile: str) -> dict[str, object]:
        config = load_profile(profile, project_root=ROOT)
        return build_pipeline_report(config, project_root=ROOT)

    def test_nx_report_distinguishes_compute_selection_and_stream_work(self) -> None:
        report = self._report("nx-rdk")
        self.assertEqual(report["evidence"]["level"], "L1_configuration_projection")
        self.assertFalse(report["evidence"]["runtime_observed"])
        self.assertTrue(report["runtime"]["live_pipeline"])
        self.assertFalse(report["image_products"]["black_white_iou_image"])
        self.assertFalse(report["association"]["shifted_iou_is_image"])
        self.assertEqual(report["frame_identity"]["shutter_type"], "global")
        self.assertEqual(
            report["frame_identity"]["exposure_delay_status"],
            "assumed_zero_unmeasured",
        )
        self.assertIsNone(
            report["frame_identity"]["exposure_delay_uncertainty_ns"]
        )
        self.assertEqual(
            report["frame_identity"]["observation_timestamp_semantics"],
            "exposure_midpoint",
        )

        candidates = {
            item["method"]: item for item in report["triangulation"]["candidates"]
        }
        self.assertEqual(candidates["roi_multi_point"]["selection_role"], "primary")
        self.assertEqual(candidates["circle_center"]["selection_role"], "fallback")
        self.assertEqual(candidates["center_patch"]["selection_role"], "computed_only")
        self.assertEqual(
            candidates["cuda_template_match"]["selection_role"],
            "computed_only",
        )
        self.assertTrue(candidates["cuda_template_match"]["compute_enabled"])

        streams = {
            item["id"]: item for item in report["concurrency"]["streams"]
        }
        self.assertTrue(streams["vpi_rectification"]["per_frame_work"])
        self.assertFalse(streams["vpi_full_frame_disparity"]["allocated"])
        self.assertFalse(streams["vpi_full_frame_disparity"]["per_frame_work"])
        self.assertTrue(streams["cuda_p2_ncc"]["per_frame_work"])
        self.assertTrue(streams["cuda_pipeline_gpu"]["per_frame_work"])
        for stream_id in (
            "cuda_p2_xfeat",
            "cuda_p2_superpoint",
            "cuda_p2_aliked",
        ):
            self.assertFalse(streams[stream_id]["allocated"])
            self.assertFalse(streams[stream_id]["per_frame_work"])
        self.assertTrue(streams["async_roi_copy"]["allocated"])
        self.assertFalse(streams["p2_diagnostic_core"]["allocated"])

    def test_capture_report_is_synchronous_and_not_armed_without_stage(self) -> None:
        report = self._report("free-flight")
        self.assertFalse(report["concurrency"]["async_roi_stage2"])
        self.assertFalse(report["concurrency"]["drop_stale_roi_frames"])
        self.assertTrue(report["outputs"]["capture_profile_intent"])
        self.assertFalse(report["outputs"]["capture_v21_armed"])
        self.assertIn("staging", report["outputs"]["capture_v21_note"])

    def test_replay_report_does_not_claim_live_gpu_execution(self) -> None:
        report = self._report("replay")
        self.assertFalse(report["runtime"]["live_pipeline"])
        self.assertFalse(report["evidence"]["requires_target_runtime_confirmation"])
        self.assertFalse(report["detector"]["enabled"])
        self.assertFalse(report["triangulation"]["enabled"])
        self.assertEqual(report["triangulation"]["candidates"], [])
        self.assertTrue(
            next(
                stage for stage in report["stages"]
                if stage["id"] == "recorded_observation_replay"
            )["enabled"]
        )
        self.assertTrue(
            all(
                not stream["allocated"]
                for stream in report["concurrency"]["streams"]
            )
        )


if __name__ == "__main__":
    unittest.main()
