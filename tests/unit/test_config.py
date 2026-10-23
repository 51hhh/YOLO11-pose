from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.config import (
    ConfigError,
    _load_yaml,
    _merge,
    _validate_schema,
    _validate_semantics,
    find_project_root,
    load_profile,
)


ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_all_checked_in_profiles_are_valid(self) -> None:
        for profile in ("nx-rdk", "free-flight", "replay"):
            with self.subTest(profile=profile):
                resolved = load_profile(profile, project_root=ROOT)
                self.assertEqual(len(resolved.fingerprint_sha256), 64)
                self.assertGreaterEqual(len(resolved.layers), 4)

    def test_override_changes_existing_key_and_provenance(self) -> None:
        baseline = load_profile("nx-rdk", project_root=ROOT)
        changed = load_profile(
            "nx-rdk",
            project_root=ROOT,
            overrides=("detector.confidence_threshold=0.42",),
        )
        self.assertEqual(changed.value("detector.confidence_threshold"), 0.42)
        self.assertIn("CLI --set", changed.source("detector.confidence_threshold"))
        self.assertNotEqual(baseline.fingerprint_sha256, changed.fingerprint_sha256)

    def test_override_cannot_create_unknown_key(self) -> None:
        with self.assertRaisesRegex(ConfigError, "只能覆盖已有键"):
            load_profile(
                "nx-rdk",
                project_root=ROOT,
                overrides=("detector.typo=1",),
            )

    def test_duplicate_yaml_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("camera:\n  id: left\n  id: right\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "重复配置键"):
                _load_yaml(path)

    def test_lists_replace_instead_of_append(self) -> None:
        target = {"stereo": {"fallback": ["first", "second"]}}
        provenance: dict[str, str] = {}
        _merge(target, {"stereo": {"fallback": ["third"]}}, "override", provenance)
        self.assertEqual(target["stereo"]["fallback"], ["third"])
        self.assertEqual(provenance["stereo.fallback"], "override")

    def test_schema_rejects_unknown_keys(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["detector"]["confidence_typo"] = 0.1
        with self.assertRaisesRegex(ConfigError, "schema 校验失败"):
            _validate_schema(ROOT, payload)

    def test_schema_rejects_retired_nx_prediction_and_control(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        mutations = (
            lambda value: value.update(
                prediction={
                    "enabled": True,
                    "owner": "nx",
                    "model": "ballistic_drag",
                }
            ),
            lambda value: value["publisher"].update(
                control_goal_enabled=True
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                payload = copy.deepcopy(dict(resolved.data))
                mutate(payload)
                with self.assertRaisesRegex(ConfigError, "schema 校验失败"):
                    _validate_schema(ROOT, payload)

    def test_schema_rejects_retired_ros2_depth_diagnostic(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["diagnostics"]["ros2_depth_enabled"] = True
        with self.assertRaisesRegex(ConfigError, "schema 校验失败"):
            _validate_schema(ROOT, payload)

    def test_schema_rejects_unimplemented_calibration_format(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["calibration"]["format"] = "capture_v2_json"
        with self.assertRaisesRegex(ConfigError, "schema 校验失败"):
            _validate_schema(ROOT, payload)

    def test_semantics_reject_same_camera_serial(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["right"]["serial"] = payload["cameras"]["left"]["serial"]
        with self.assertRaisesRegex(ConfigError, "serial"):
            _validate_semantics(ROOT, payload)

    def test_semantics_reject_enabled_publisher_without_backend(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["publisher"]["backend"] = "none"
        with self.assertRaisesRegex(ConfigError, "backend"):
            _validate_semantics(ROOT, payload)

    def test_publisher_frame_must_match_direct_observation_frame(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["publisher"]["frame_id"] = "renamed_without_transform"
        with self.assertRaisesRegex(ConfigError, "calibration.rig_frame"):
            _validate_semantics(ROOT, payload)

    def test_hardware_trigger_requires_frame_watermark_identity(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["sync"]["frame_identity_source"] = "recorded"
        with self.assertRaisesRegex(ConfigError, "FrameSpecInfo"):
            _validate_semantics(ROOT, payload)

    def test_gpio_exposure_midpoint_requires_global_shutter(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["sync"]["shutter_type"] = "rolling"
        with self.assertRaisesRegex(ConfigError, "shutter_type=global"):
            _validate_semantics(ROOT, payload)

    def test_unmeasured_exposure_delay_is_not_zero_uncertainty(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["sync"]["exposure_delay_uncertainty_ns"] = 0
        with self.assertRaisesRegex(ConfigError, "必须为 null"):
            _validate_semantics(ROOT, payload)

    def test_hik_transport_rejects_latest_frame_strategy(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["transport"]["grab_strategy"] = "recorded"
        with self.assertRaisesRegex(ConfigError, "one_by_one"):
            _validate_semantics(ROOT, payload)

    def test_live_stereo_requires_per_camera_detector(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["detector"]["per_camera"] = False
        with self.assertRaisesRegex(ConfigError, "per_camera"):
            _validate_semantics(ROOT, payload)

    def test_hik_adapter_rejects_unrepresentable_per_camera_settings(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        mutations = (
            ("exposure_us", lambda value: value["cameras"]["right"].update(exposure_us=1000.0)),
            ("gain_db", lambda value: value["cameras"]["right"].update(gain_db=1.0)),
            ("pixel_format", lambda value: value["cameras"]["right"].update(pixel_format="gray8")),
            ("全幅 ROI", lambda value: value["cameras"]["right"]["roi"].update(x=2)),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                payload = copy.deepcopy(dict(resolved.data))
                mutate(payload)
                with self.assertRaisesRegex(ConfigError, expected):
                    _validate_semantics(ROOT, payload)

    def test_live_detector_rejects_unsupported_native_settings(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        mutations = (
            ("正方形", lambda value: value["detector"].update(input_resolution=[640, 384])),
            ("letterbox", lambda value: value["detector"].update(resize_mode="stretch")),
            ("rgb8", lambda value: value["detector"].update(input_format="rgb8")),
            ("bbox_center", lambda value: value["detector"].update(center_method="circle_fit")),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                payload = copy.deepcopy(dict(resolved.data))
                mutate(payload)
                with self.assertRaisesRegex(ConfigError, expected):
                    _validate_semantics(ROOT, payload)

    def test_live_model_manifest_must_match_runtime_contract(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        mutations = (
            (
                "input_resolution",
                lambda value: value["detector"].update(input_resolution=[320, 320]),
            ),
            (
                "input_format",
                lambda value: value["detector"].update(input_format="gray8"),
            ),
            (
                "target",
                lambda value: value["platform"].update(id="another-jetson"),
            ),
            (
                "accelerator",
                lambda value: value["platform"].update(accelerator="cuda_dla"),
            ),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                payload = copy.deepcopy(dict(resolved.data))
                mutate(payload)
                with self.assertRaisesRegex(ConfigError, expected):
                    _validate_semantics(ROOT, payload)

    def test_project_root_can_be_found_from_nested_path(self) -> None:
        self.assertEqual(
            find_project_root(ROOT / "python" / "volleyball_vision"),
            ROOT,
        )

    def test_fallback_method_must_be_an_enabled_candidate(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["stereo"]["triangulation"]["candidate_methods"].remove(
            "bbox_edges"
        )
        with self.assertRaisesRegex(ConfigError, "fallback_methods 必须是"):
            _validate_semantics(ROOT, payload)

    def test_native_fallback_order_is_explicit_and_configurable(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        triangulation = payload["stereo"]["triangulation"]
        triangulation["candidate_methods"] = [
            "roi_multi_point",
            "bbox_center",
            "circle_center",
        ]
        triangulation["fallback_methods"] = ["bbox_center", "circle_center"]
        _validate_semantics(ROOT, payload)

    def test_any_native_candidate_can_be_primary(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        triangulation = payload["stereo"]["triangulation"]
        triangulation["primary_method"] = "bbox_center"
        triangulation["candidate_methods"] = ["bbox_center", "circle_center"]
        triangulation["fallback_methods"] = ["circle_center"]
        _validate_semantics(ROOT, payload)

    def test_computed_candidate_must_be_in_selection_order(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["stereo"]["triangulation"]["fallback_methods"].remove(
            "bbox_edges"
        )
        with self.assertRaisesRegex(ConfigError, "必须进入"):
            _validate_semantics(ROOT, payload)

    def test_recorded_profile_has_no_live_depth_candidates(self) -> None:
        resolved = load_profile("replay", project_root=ROOT)
        triangulation = resolved.data["stereo"]["triangulation"]
        self.assertEqual(triangulation["primary_method"], "recorded")
        self.assertEqual(triangulation["candidate_methods"], [])
        self.assertEqual(triangulation["fallback_methods"], [])

    def test_subpixel_min_points_cannot_exceed_max_points(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["stereo"]["triangulation"]["subpixel"].update(
            min_points=13,
            max_points=12,
        )
        with self.assertRaisesRegex(ConfigError, "min_points"):
            _validate_semantics(ROOT, payload)


if __name__ == "__main__":
    unittest.main()

