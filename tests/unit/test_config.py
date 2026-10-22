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

    def test_hardware_trigger_requires_frame_watermark_identity(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["sync"]["frame_identity_source"] = "recorded"
        with self.assertRaisesRegex(ConfigError, "FrameSpecInfo"):
            _validate_semantics(ROOT, payload)

    def test_hik_transport_rejects_latest_frame_strategy(self) -> None:
        resolved = load_profile("nx-rdk", project_root=ROOT)
        payload = copy.deepcopy(dict(resolved.data))
        payload["cameras"]["transport"]["grab_strategy"] = "recorded"
        with self.assertRaisesRegex(ConfigError, "one_by_one"):
            _validate_semantics(ROOT, payload)

    def test_project_root_can_be_found_from_nested_path(self) -> None:
        self.assertEqual(
            find_project_root(ROOT / "python" / "volleyball_vision"),
            ROOT,
        )


if __name__ == "__main__":
    unittest.main()

