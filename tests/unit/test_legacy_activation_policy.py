from __future__ import annotations

from pathlib import Path
import unittest

from volleyball_vision.config import load_profile
from volleyball_vision.legacy_activation_policy import (
    apply_legacy_activation_policy,
)


ROOT = Path(__file__).resolve().parents[2]


def _hostile_template() -> dict[str, object]:
    return {
        "disparity_offset": {"enabled": True},
        "detector": {
            "use_dla": True,
            "dual_yolo": {
                "enabled": False,
                "right_use_dla": True,
                "use_for_depth": False,
                "fallback_to_roi_match": True,
                "gpu_candidate_refine": False,
            },
        },
        "stereo": {"strategy": "full_frame"},
        "neural_feature_matching": {"enabled": True},
        "neural_feature_matching_xfeat": {"enabled": True},
        "neural_feature_matching_superpoint": {"enabled": True},
        "neural_feature_matching_aliked": {"enabled": True},
        "performance": {
            "drop_stale_roi_frames": False,
            "async_roi_stage2": False,
            "p2_diagnostic_lane_decision_enabled": True,
            "p2_diagnostic_results_enabled": True,
            "p2_diagnostic_artifacts_enabled": True,
        },
        "display": {"enable": True},
    }


class LegacyActivationPolicyTests(unittest.TestCase):
    def test_runtime_profile_overrides_hidden_template_switches(self) -> None:
        profile = load_profile("nx-rdk", project_root=ROOT)
        payload = _hostile_template()

        apply_legacy_activation_policy(payload, profile)

        self.assertFalse(payload["disparity_offset"]["enabled"])
        detector = payload["detector"]
        self.assertFalse(detector["use_dla"])
        dual = detector["dual_yolo"]
        self.assertTrue(dual["enabled"])
        self.assertTrue(dual["use_for_depth"])
        self.assertTrue(dual["gpu_candidate_refine"])
        self.assertFalse(dual["fallback_to_roi_match"])
        self.assertEqual(payload["stereo"]["strategy"], "roi_only")
        for name in (
            "neural_feature_matching",
            "neural_feature_matching_xfeat",
            "neural_feature_matching_superpoint",
            "neural_feature_matching_aliked",
        ):
            self.assertFalse(payload[name]["enabled"])
        performance = payload["performance"]
        self.assertTrue(performance["drop_stale_roi_frames"])
        self.assertTrue(performance["async_roi_stage2"])
        self.assertEqual(performance["async_roi_buffers"], 3)
        self.assertTrue(performance["p2_realtime_lane_decision_enabled"])
        self.assertFalse(performance["p2_diagnostic_lane_decision_enabled"])
        self.assertFalse(performance["p2_diagnostic_results_enabled"])
        self.assertFalse(performance["p2_diagnostic_artifacts_enabled"])
        self.assertFalse(payload["display"]["enable"])

    def test_capture_profile_disables_lossy_execution_without_stage(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        payload = _hostile_template()

        apply_legacy_activation_policy(payload, profile)

        performance = payload["performance"]
        self.assertFalse(performance["drop_stale_roi_frames"])
        self.assertFalse(performance["async_roi_stage2"])

    def test_removing_p2_candidate_disables_p2_lane(self) -> None:
        profile = load_profile(
            "nx-rdk",
            project_root=ROOT,
            overrides=(
                "stereo.triangulation.primary_method=circle_center",
                "stereo.triangulation.candidate_methods=[circle_center]",
                "stereo.triangulation.fallback_methods=[]",
            ),
        )
        payload = _hostile_template()

        apply_legacy_activation_policy(payload, profile)

        performance = payload["performance"]
        self.assertFalse(performance["p2_feature_job_scaffold_enabled"])
        self.assertFalse(performance["p2_realtime_lane_decision_enabled"])


if __name__ == "__main__":
    unittest.main()
