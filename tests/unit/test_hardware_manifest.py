from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.config import load_profile
from volleyball_vision.dataset import DatasetError
from volleyball_vision.hardware_manifest import (
    load_hardware_manifest,
    write_stage_hardware_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(directory: Path, *, status: str = "verified") -> Path:
    exports = {
        "left": b"left-camera-parameters\n",
        "right": b"right-camera-parameters\n",
    }
    for role, data in exports.items():
        path = directory / "camera-parameters" / f"{role}.mfs"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    payload = {
        "schema_version": 1,
        "status": status,
        "rig_id": "hik-stereo-main",
        "verified_at_utc": "2026-10-23T08:00:00Z" if status == "verified" else None,
        "verified_by": "fixture" if status == "verified" else None,
        "cameras": {},
        "software": {
            name: {"version": "fixture", "source": "unit-test"}
            for name in ("jetpack", "cuda", "tensorrt", "mvs", "opencv")
        },
        "notes": "unit fixture",
    }
    serials = {"left": "00D39342665", "right": "00219471413"}
    for role in ("left", "right"):
        payload["cameras"][role] = {
            "serial": serials[role],
            "model": f"fixture-{role}",
            "model_source": "mvs_device_enumeration",
            "firmware_version": "1.0.0",
            "firmware_source": "fixture",
            "shutter_type": "global",
            "parameter_export": {
                "path": f"camera-parameters/{role}.mfs",
                "sha256": _sha256(exports[role]),
                "format": "hik_mvs_feature_export",
            },
            "lens": {
                "model": "fixture-lens",
                "focal_length_mm": 8.0,
                "aperture_f_number": 4.0,
                "focus_mode": "manual",
                "focus_setting": "locked-mark-a",
                "locked": True,
            },
        }
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


class HardwareManifestTests(unittest.TestCase):
    def test_verified_manifest_freezes_exact_manifest_and_exports(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            snapshot = load_hardware_manifest(
                manifest,
                config=profile,
                project_root=ROOT,
            )
            stage = root / "stage"
            stage.mkdir()
            staged = write_stage_hardware_snapshot(stage, snapshot)

            self.assertEqual(staged.read_bytes(), manifest.read_bytes())
            self.assertEqual(snapshot.camera_models["left"], "fixture-left")
            self.assertEqual(
                (stage / "hardware/camera-parameters/right.mfs").read_bytes(),
                b"right-camera-parameters\n",
            )

    def test_pending_manifest_is_only_allowed_for_planning(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_manifest(Path(directory), status="pending")
            with self.assertRaisesRegex(DatasetError, "status=verified"):
                load_hardware_manifest(
                    manifest,
                    config=profile,
                    project_root=ROOT,
                )
            snapshot = load_hardware_manifest(
                manifest,
                config=profile,
                require_verified=False,
                project_root=ROOT,
            )
            self.assertEqual(snapshot.payload["status"], "pending")

    def test_verified_manifest_rejects_unedited_template_values(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_manifest(Path(directory))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["software"]["mvs"]["version"] = "REPLACE"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "模板值"):
                load_hardware_manifest(
                    manifest,
                    config=profile,
                    project_root=ROOT,
                )

    def test_manifest_rejects_profile_identity_and_export_hash_mismatch(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["cameras"]["left"]["serial"] = "wrong-camera"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "序列号"):
                load_hardware_manifest(
                    manifest,
                    config=profile,
                    project_root=ROOT,
                )

            manifest = _write_manifest(root)
            (root / "camera-parameters/left.mfs").write_bytes(b"changed")
            with self.assertRaisesRegex(DatasetError, "SHA-256"):
                load_hardware_manifest(
                    manifest,
                    config=profile,
                    project_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
