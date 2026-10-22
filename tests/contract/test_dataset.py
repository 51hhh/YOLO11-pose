from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.dataset import (
    DatasetError,
    TrialPackageWriter,
    validate_dataset,
    validate_trial,
    with_pipeline_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]


def _identity_with_x(x_m: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x_m],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _calibration() -> dict[str, object]:
    cameras = []
    for camera_id, x_m in (("left-camera", 0.0), ("right-camera", 0.853154)):
        cameras.append(
            {
                "camera_id": camera_id,
                "image_size_px": [1440, 1080],
                "camera_matrix": [
                    [1000.0, 0.0, 720.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_model": "brown_conrady",
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "transform_body_from_camera": _identity_with_x(x_m),
                "rms_reprojection_error_px": 0.25,
                "clock_domain": "camera-clock",
                "extrinsic_covariance": None,
            }
        )
    return {
        "schema_version": "2.1.0",
        "calibration_type": "vehicle_mounted_stereo",
        "calibration_id": "vehicle-stereo-main",
        "created_at_utc": "2026-10-22T00:00:00Z",
        "body_frame_id": "base_link",
        "camera_roles": {"left": "left-camera", "right": "right-camera"},
        "estimated_sync_error_s": 0.0001,
        "cameras": cameras,
        "baseline_m": 0.853154,
        "notes": "contract fixture",
    }


def _metadata(trial_id: str = "flight-000001") -> dict[str, object]:
    pipeline = with_pipeline_fingerprint(
        {
            "pipeline_id": "yolo26-test",
            "detector_family": "yolo26",
            "detector_version": "test",
            "code_revision": "0123456789abcdef",
            "config_sha256": "1" * 64,
            "weights_sha256": "2" * 64,
            "input_size_px": [640, 640],
            "resize_mode": "letterbox",
            "score_threshold": 0.3,
            "nms_iou_threshold": 0.4,
            "center_method": "bbox_center",
        }
    )
    return {
        "trial_id": trial_id,
        "session_id": "session-a",
        "ball_id": "ball-01",
        "calibration_id": "vehicle-stereo-main",
        "captured_at_utc": "2026-10-22T00:00:00Z",
        "trajectory_kind": "free_flight",
        "coordinate_frame": {
            "name": "stereo_rig",
            "origin": "left camera optical origin",
            "x_axis": "right",
            "y_axis": "down",
            "z_axis": "forward",
            "handedness": "right",
            "position_unit": "m",
            "time_unit": "s",
        },
        "capture": {
            "nominal_fps": 90.0,
            "sync_method": "hardware_trigger",
            "estimated_sync_error_s": 0.0001,
            "trajectory_timebase": {
                "clock_domain": "camera-clock",
                "origin_ns": 1_000_000_000,
                "timestamp_semantics": "stereo_midpoint_capture",
            },
        },
        "perception_pipelines": [pipeline],
        "source_files": {},
        "conditions": {"throw_label": "fixture"},
        "notes": "minimal valid capture",
    }


def _frame(camera_id: str) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "frame_index": 0,
        "hardware_timestamp_ns": 999_999_900,
        "clock_domain": "camera-clock",
        "exposure_start_ns": 999_999_950,
        "capture_timestamp_ns": 1_000_000_000,
        "receive_timestamp_ns": 1_000_000_100,
        "inference_done_timestamp_ns": 1_000_000_200,
        "exposure_us": 9000.0,
        "gain_db": 2.0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_width": 1440,
        "roi_height": 1080,
        "pixel_format": "BayerRG8",
        "dropped": False,
        "drop_reason": None,
    }


def _detection(camera_id: str, u_px: float) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "frame_index": 0,
        "candidate_id": 0,
        "u_px": u_px,
        "v_px": 500.0,
        "radius_px": 20.0,
        "bbox_x": u_px - 20.0,
        "bbox_y": 480.0,
        "bbox_w": 40.0,
        "bbox_h": 40.0,
        "du_dt_pxps": None,
        "dv_dt_pxps": None,
        "score": 0.95,
        "cov_uu_px2": 0.25,
        "cov_uv_px2": 0.0,
        "cov_vv_px2": 0.25,
        "selected": True,
        "source": "yolo26-test",
    }


def _write_valid_trial(root: Path, trial_id: str = "flight-000001") -> Path:
    calibrations = root / "calibrations"
    calibrations.mkdir(parents=True, exist_ok=True)
    (calibrations / "vehicle-stereo-main.json").write_text(
        json.dumps(_calibration(), indent=2),
        encoding="utf-8",
    )
    raw_source = root / "left.raw"
    raw_source.write_bytes(b"fixture-raw-frame")
    with TrialPackageWriter(root, _metadata(trial_id), project_root=ROOT) as writer:
        writer.append("camera_frames", _frame("left-camera"))
        writer.append("camera_frames", _frame("right-camera"))
        writer.append("detections_2d", _detection("left-camera", 700.0))
        writer.append("detections_2d", _detection("right-camera", 650.0))
        writer.append(
            "stereo_pairs",
            {
                "pair_id": 0,
                "left_frame_index": 0,
                "right_frame_index": 0,
                "left_candidate_id": 0,
                "right_candidate_id": 0,
                "left_capture_ns": 1_000_000_000,
                "right_capture_ns": 1_000_000_000,
                "delta_t_lr_ns": 0,
                "disparity_px": 50.0,
                "epipolar_error_px": 0.1,
                "x_rig_m": 0.1,
                "y_rig_m": 0.2,
                "z_rig_m": 5.0,
                "cov_xx_m2": 0.001,
                "cov_xy_m2": 0.0,
                "cov_xz_m2": 0.0,
                "cov_yy_m2": 0.001,
                "cov_yz_m2": 0.0,
                "cov_zz_m2": 0.01,
                "accepted": True,
                "reject_reason": None,
            },
        )
        writer.append(
            "trajectory",
            {
                "sample_index": 0,
                "timestamp_s": 0.0,
                "left_frame_index": 0,
                "right_frame_index": 0,
                "x_m": 0.1,
                "y_m": 0.2,
                "z_m": 5.0,
                "confidence": 0.95,
                "reprojection_error_px": 0.1,
                "cov_xx_m2": 0.001,
                "cov_xy_m2": 0.0,
                "cov_xz_m2": 0.0,
                "cov_yy_m2": 0.001,
                "cov_yz_m2": 0.0,
                "cov_zz_m2": 0.01,
            },
        )
        writer.add_raw_source("left_raw", raw_source, filename="left.raw")
        return writer.finalize()


class DatasetContractTests(unittest.TestCase):
    def test_writer_atomically_publishes_valid_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = _write_valid_trial(root)
            self.assertEqual(trial.name, "flight-000001")
            self.assertFalse(any(path.name.startswith(".") for path in (root / "trials").iterdir()))
            summary = validate_dataset(root, project_root=ROOT)
            self.assertEqual(summary.calibrations, 1)
            self.assertEqual(len(summary.trials), 1)
            self.assertEqual(summary.trials[0].table_rows["trajectory"], 1)

    def test_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_valid_trial(root)
            with self.assertRaises(FileExistsError):
                TrialPackageWriter(root, _metadata(), project_root=ROOT)

    def test_validator_detects_raw_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            (trial / "raw" / "left.raw").write_bytes(b"changed")
            with self.assertRaisesRegex(DatasetError, "大小不匹配|SHA-256"):
                validate_trial(trial, project_root=ROOT)

    def test_validator_detects_pipeline_fingerprint_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            metadata_path = trial / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["perception_pipelines"][0]["score_threshold"] = 0.99
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "指纹不匹配"):
                validate_trial(trial, project_root=ROOT)

    def test_writer_rejects_contract_extra_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibrations = root / "calibrations"
            calibrations.mkdir(parents=True)
            (calibrations / "vehicle-stereo-main.json").write_text(
                json.dumps(_calibration()), encoding="utf-8"
            )
            with TrialPackageWriter(root, _metadata(), project_root=ROOT) as writer:
                record = _frame("left-camera")
                record["filtered_x_m"] = 1.0
                with self.assertRaisesRegex(DatasetError, "契约外字段"):
                    writer.append("camera_frames", record)

    def test_invalid_finalize_removes_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibrations = root / "calibrations"
            calibrations.mkdir(parents=True)
            (calibrations / "vehicle-stereo-main.json").write_text(
                json.dumps(_calibration()), encoding="utf-8"
            )
            writer = TrialPackageWriter(root, _metadata(), project_root=ROOT)
            writer.append("camera_frames", _frame("left-camera"))
            writer.append(
                "stereo_pairs",
                {
                    "pair_id": 0,
                    "left_frame_index": 0,
                    "right_frame_index": 0,
                    "left_candidate_id": 0,
                    "right_candidate_id": 0,
                    "left_capture_ns": 1_000_000_000,
                    "right_capture_ns": 1_000_000_000,
                    "delta_t_lr_ns": 0,
                    "accepted": False,
                    "reject_reason": "missing-right",
                },
            )
            with self.assertRaises(DatasetError):
                writer.finalize()
            self.assertFalse(writer.stage.exists())


if __name__ == "__main__":
    unittest.main()

