from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.capture import (
    STAGE_DETECTION_COLUMNS,
    STAGE_FRAME_COLUMNS,
    STAGE_PAIR_COLUMNS,
    STAGE_SCHEMA_VERSION,
    finalize_capture_stage,
    prepare_capture_stage,
)
from volleyball_vision.config import load_profile
from volleyball_vision.dataset import validate_dataset, with_pipeline_fingerprint


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class CaptureStageTests(unittest.TestCase):
    def _stage(self, root: Path) -> tuple[Path, object]:
        profile = load_profile("free-flight", project_root=ROOT)
        pipeline = with_pipeline_fingerprint(
            {
                "pipeline_id": profile.data["detector"]["model_id"],
                "detector_family": "yolo26",
                "detector_version": "fixture",
                "code_revision": "0123456789abcdef",
                "config_sha256": profile.fingerprint_sha256,
                "weights_sha256": "2" * 64,
                "input_size_px": [640, 640],
                "resize_mode": "letterbox",
                "score_threshold": 0.3,
                "nms_iou_threshold": 0.4,
                "center_method": "bbox_center",
            }
        )
        context = {
            "stage_schema_version": STAGE_SCHEMA_VERSION,
            "trial_id": "flight-stage-0001",
            "session_id": "session-a",
            "ball_id": "ball-01",
            "calibration_id": "vehicle-stereo-main",
            "captured_at_utc": "2026-10-22T00:00:00Z",
            "trajectory_kind": "free_flight",
            "conditions": {
                "environment": {
                    "air_temperature_c": 24.0,
                    "relative_humidity": 0.45,
                    "source": "fixture",
                }
            },
            "notes": "fixture",
            "profile": "free-flight",
            "config_sha256": profile.fingerprint_sha256,
            "perception_pipeline": pipeline,
        }
        stage = root / "stage"
        prepare_capture_stage(stage, context)
        (stage / "pipeline.yaml").write_text("capture_v21: {enabled: true}\n", encoding="utf-8")
        raw = stage / "raw"
        raw.mkdir()
        frame_rows = []
        detection_rows = []
        pair_rows = []
        for frame_id in range(2):
            left_device = 1_000_000_000 + frame_id * 11_111_111
            host = 2_000_000_000 + frame_id * 11_111_111
            left_rel = f"raw/left-{frame_id}.pgm"
            right_rel = f"raw/right-{frame_id}.pgm"
            (stage / left_rel).write_bytes(b"P5\n1 1\n255\n\x00")
            (stage / right_rel).write_bytes(b"P5\n1 1\n255\n\x00")
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "host_capture_timestamp_ns": host,
                    "callback_timestamp_ns": host + 1_000_000,
                    "left_device_timestamp_ns": left_device,
                    "right_device_timestamp_ns": left_device - 50_000,
                    "stereo_timestamp_residual_ns": 100,
                    "left_frame_number": 10 + frame_id,
                    "right_frame_number": 20 + frame_id,
                    "left_frame_counter": 100 + frame_id,
                    "right_frame_counter": 100 + frame_id,
                    "left_trigger_index": 200 + frame_id,
                    "right_trigger_index": 202 + frame_id,
                    "raw_left": left_rel,
                    "raw_right": right_rel,
                }
            )
            for role, cx in (("left", 700.0), ("right", 650.0)):
                detection_rows.append(
                    {
                        "frame_id": frame_id,
                        "camera_role": role,
                        "candidate_id": 0,
                        "cx_px": cx,
                        "cy_px": 500.0,
                        "width_px": 40.0,
                        "height_px": 40.0,
                        "confidence": 0.95,
                        "class_id": 0,
                        "selected": True,
                    }
                )
            pair_rows.append(
                {
                    "pair_id": frame_id,
                    "frame_id": frame_id,
                    "left_candidate_id": 0,
                    "right_candidate_id": 0,
                    "disparity_px": 50.0,
                    "epipolar_error_px": 0.2,
                    "pair_score": 0.1,
                    "semantic_confidence": 0.95,
                    "x_rig_m": 0.1 + frame_id * 0.01,
                    "y_rig_m": 0.2,
                    "z_rig_m": 5.0 - frame_id * 0.1,
                    "accepted": True,
                    "reject_reason": "",
                }
            )
        _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, frame_rows)
        _write_csv(stage / "detections.csv", STAGE_DETECTION_COLUMNS, detection_rows)
        _write_csv(stage / "pairs.csv", STAGE_PAIR_COLUMNS, pair_rows)
        (stage / "status.json").write_text(
            json.dumps(
                {
                    "stage_schema_version": STAGE_SCHEMA_VERSION,
                    "complete": True,
                    "fatal_error": "",
                    "frames_captured": 2,
                    "frames_written": 2,
                    "dropped_frames": 0,
                    "copy_failures": 0,
                    "write_failures": 0,
                }
            ),
            encoding="utf-8",
        )
        return stage, profile

    def test_stage_is_normalized_and_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            dataset = root / "dataset"
            trial = finalize_capture_stage(
                stage,
                dataset,
                profile,
                project_root=ROOT,
            )
            self.assertEqual(trial.name, "flight-stage-0001")
            summary = validate_dataset(dataset, project_root=ROOT)
            self.assertEqual(summary.trials[0].table_rows["camera_frames"], 4)
            self.assertEqual(summary.trials[0].table_rows["detections_2d"], 4)
            self.assertEqual(summary.trials[0].table_rows["stereo_pairs"], 2)
            self.assertEqual(summary.trials[0].table_rows["trajectory"], 2)
            metadata = json.loads((trial / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["capture"]["trajectory_timebase"]["clock_domain"], "hik-shared-device-clock")
            self.assertGreaterEqual(len(metadata["raw_manifest"]), 9)

    def test_stage_with_frame_gap_is_not_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            with (stage / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["frame_id"] = "2"
            (stage / "frames.csv").unlink()
            _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, rows)
            with self.assertRaisesRegex(Exception, "帧缺口"):
                finalize_capture_stage(stage, root / "dataset", profile, project_root=ROOT)

    def test_status_frame_count_mismatch_is_not_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            status_path = stage / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["frames_written"] = 1
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "帧计数"):
                finalize_capture_stage(stage, root / "dataset", profile, project_root=ROOT)

    def test_sync_residual_over_configured_limit_is_not_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            with (stage / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["stereo_timestamp_residual_ns"] = "100001"
            (stage / "frames.csv").unlink()
            _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, rows)
            with self.assertRaisesRegex(Exception, "同步残差"):
                finalize_capture_stage(stage, root / "dataset", profile, project_root=ROOT)

    def test_uint32_frame_counter_wrap_is_continuous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            with (stage / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for role in ("left", "right"):
                rows[0][f"{role}_frame_counter"] = str(0xFFFFFFFF)
                rows[1][f"{role}_frame_counter"] = "0"
            (stage / "frames.csv").unlink()
            _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, rows)
            trial = finalize_capture_stage(
                stage,
                root / "dataset",
                profile,
                project_root=ROOT,
            )
            self.assertTrue((trial / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()

