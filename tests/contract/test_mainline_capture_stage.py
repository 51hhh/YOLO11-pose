from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.capture import finalize_capture_stage, prepare_capture_stage
from volleyball_vision.config import load_profile
from volleyball_vision.dataset import (
    DatasetError,
    validate_dataset,
    with_pipeline_fingerprint,
)
from volleyball_vision.mainline_capture import build_mainline_capture_context
from volleyball_vision.mainline_stage_contract import (
    MAINLINE_ASSOCIATION_COLUMNS,
    MAINLINE_DETECTION_COLUMNS,
    MAINLINE_FRAME_COLUMNS,
    MAINLINE_OBSERVATION_COLUMNS,
    MAINLINE_STAGE_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_pgm(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(width * height))


class MainlineCaptureStageTests(unittest.TestCase):
    def test_mainline_context_uses_v5_without_duplicating_capture_identity(self) -> None:
        profile = load_profile("free-flight", project_root=ROOT)
        context = build_mainline_capture_context(
            profile,
            trial_id="mainline-plan-0001",
            session_id="session-a",
            ball_id="ball-01",
            verify_model=False,
            project_root=ROOT,
        )

        self.assertEqual(
            context["stage_schema_version"], MAINLINE_STAGE_SCHEMA_VERSION
        )
        self.assertTrue(context["planning_only"])
        self.assertEqual(context["config_sha256"], profile.fingerprint_sha256)

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
            "stage_schema_version": MAINLINE_STAGE_SCHEMA_VERSION,
            "trial_id": "mainline-stage-0001",
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
            "notes": "mainline fixture",
            "profile": "free-flight",
            "config_sha256": profile.fingerprint_sha256,
            "exposure_timestamp": {
                "shutter_type": "global",
                "source": "nx_gpio_rising_edge",
                "delay_status": "assumed_zero_unmeasured",
                "delay_ns": 0,
                "delay_uncertainty_ns": None,
                "observation_timestamp": "exposure_midpoint",
            },
            "perception_pipeline": pipeline,
        }
        stage = root / "stage"
        prepare_capture_stage(stage, context)
        width, height = profile.data["cameras"]["left"]["resolution"]
        frame_rows: list[dict[str, object]] = []
        for role, camera_id, hardware, frame_number, trigger_index in (
            ("left", "hik-00D39342665", 1_000_000_000, 10, 100),
            ("right", "hik-00219471413", 900_000_000, 20, 102),
        ):
            relative = f"raw/mainline/{role}/42.pgm"
            _write_pgm(stage / relative, width, height)
            frame_rows.append(
                {
                    "camera_id": camera_id,
                    "camera_role": role,
                    "frame_index": 42,
                    "hardware_timestamp_ns": hardware,
                    "hardware_clock_domain": f"hik-{role}-device",
                    "exposure_start_ns": 1_995_066_500,
                    "exposure_start_clock_domain": "nx-realtime",
                    "capture_timestamp_ns": 2_000_000_000,
                    "capture_clock_domain": "nx-realtime",
                    "receive_timestamp_ns": 2_001_000_000,
                    "receive_clock_domain": "nx-realtime",
                    "inference_done_timestamp_ns": 2_002_000_000,
                    "inference_done_clock_domain": "nx-realtime",
                    "exposure_us": 9867.0,
                    "gain_db": 11.9906,
                    "roi_x": 0,
                    "roi_y": 0,
                    "roi_width": width,
                    "roi_height": height,
                    "pixel_format": "gray8",
                    "dropped": False,
                    "drop_reason": "",
                    "receive_monotonic_ns": 3_001_000_000,
                    "receive_monotonic_clock_domain": "nx-monotonic",
                    "sdk_host_timestamp_raw": 123,
                    "sensor_frame_number": frame_number,
                    "sensor_frame_counter": 42,
                    "sensor_trigger_index": trigger_index,
                    "stereo_hardware_timestamp_residual_ns": 0,
                    "trigger_sequence": 42,
                    "trigger_scheduled_monotonic_ns": 2_999_990_000,
                    "trigger_scheduled_monotonic_clock_domain": "nx-monotonic",
                    "trigger_edge_monotonic_ns": 3_000_000_000,
                    "trigger_edge_monotonic_clock_domain": "nx-monotonic",
                    "trigger_edge_realtime_ns": 1_995_066_500,
                    "trigger_edge_realtime_clock_domain": "nx-realtime",
                    "trigger_write_uncertainty_ns": 1000,
                    "capture_timestamp_uncertainty_ns": "",
                    "capture_timestamp_mapping_valid": False,
                    "image_width": width,
                    "image_height": height,
                    "image_stride_bytes": width,
                    "image_size_bytes": width * height,
                    "image_pixel_format": "gray8",
                    "image_memory_kind": "host",
                    "raw_image": relative,
                }
            )
        detection_rows = []
        for role, camera_id, candidate, u_px in (
            ("left", "hik-00D39342665", 1, 700.0),
            ("right", "hik-00219471413", 2, 650.0),
        ):
            detection_rows.append(
                {
                    "camera_id": camera_id,
                    "camera_role": role,
                    "frame_index": 42,
                    "candidate_id": candidate,
                    "u_px": u_px,
                    "v_px": 500.0,
                    "radius_px": 20.0,
                    "bbox_x": u_px - 20.0,
                    "bbox_y": 480.0,
                    "bbox_w": 40.0,
                    "bbox_h": 40.0,
                    "du_dt_pxps": "",
                    "dv_dt_pxps": "",
                    "score": 0.95,
                    "cov_uu_px2": 0.25,
                    "cov_uv_px2": 0.0,
                    "cov_vv_px2": 0.25,
                    "selected": True,
                    "source": pipeline["pipeline_id"],
                }
            )
        association_rows = [
            {
                "pair_id": 7,
                "left_frame_index": 42,
                "right_frame_index": 42,
                "left_candidate_id": 1,
                "right_candidate_id": 2,
                "left_capture_ns": 2_000_000_000,
                "left_capture_clock_domain": "nx-realtime",
                "right_capture_ns": 2_000_000_000,
                "right_capture_clock_domain": "nx-realtime",
                "delta_t_lr_ns": 0,
                "disparity_px": 50.0,
                "epipolar_error_px": 0.0,
                "decision": "accepted",
                "reject_reason": "",
            }
        ]
        observation_rows = [
            {
                "observation_id": 3,
                "source_pair_id": 7,
                "capture_reference_ns": 2_000_000_000,
                "capture_reference_clock_domain": "nx-realtime",
                "coordinate_frame": "stereo_rig",
                "x_m": 0.1,
                "y_m": 0.2,
                "z_m": 5.0,
                "cov_xx_m2": 0.001,
                "cov_xy_m2": 0.0,
                "cov_xz_m2": 0.0,
                "cov_yy_m2": 0.001,
                "cov_yz_m2": 0.0,
                "cov_zz_m2": 0.01,
                "confidence": 0.95,
                "reprojection_error_px": 0.0,
            }
        ]
        _write_csv(stage / "frames_v5.csv", MAINLINE_FRAME_COLUMNS, frame_rows)
        _write_csv(
            stage / "detections_v5.csv", MAINLINE_DETECTION_COLUMNS, detection_rows
        )
        _write_csv(
            stage / "associations_v5.csv",
            MAINLINE_ASSOCIATION_COLUMNS,
            association_rows,
        )
        _write_csv(
            stage / "observations_v5.csv",
            MAINLINE_OBSERVATION_COLUMNS,
            observation_rows,
        )
        (stage / "status_v5.json").write_text(
            json.dumps(
                {
                    "stage_schema_version": MAINLINE_STAGE_SCHEMA_VERSION,
                    "complete": True,
                    "fatal_error": None,
                    "stereo_frames_written": 1,
                    "camera_frames_written": 2,
                    "detections_written": 2,
                    "associations_written": 1,
                    "observations_written": 1,
                    "raw_images_written": 2,
                }
            ),
            encoding="utf-8",
        )
        return stage, profile

    def test_mainline_stage_is_normalized_and_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            dataset = root / "dataset"

            trial = finalize_capture_stage(
                stage, dataset, profile, project_root=ROOT
            )
            summary = validate_dataset(dataset, project_root=ROOT)

            self.assertEqual(trial.name, "mainline-stage-0001")
            self.assertEqual(summary.trials[0].table_rows["camera_frames"], 2)
            self.assertEqual(summary.trials[0].table_rows["trajectory"], 1)
            metadata = json.loads((trial / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["capture"]["trajectory_timebase"]["clock_domain"],
                "hik-shared-device-clock",
            )
            self.assertEqual(len(metadata["raw_manifest"]), 8)
            with (trial / "camera_frames.csv").open(encoding="utf-8") as handle:
                camera_frames = list(csv.DictReader(handle))
            self.assertEqual(
                {row["capture_timestamp_ns"] for row in camera_frames},
                {"1000000000"},
            )
            with (trial / "clock_mappings.csv").open(encoding="utf-8") as handle:
                mappings = list(csv.DictReader(handle))
            self.assertEqual(
                {row["source_clock_domain"] for row in mappings},
                {
                    "hik-left-device",
                    "hik-right-device",
                    "nx-realtime",
                    "nx-monotonic",
                },
            )

    def test_partial_mainline_stage_is_not_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            (stage / "frames_v5.csv.partial").write_text("unfinished", encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, "未完成文件"):
                finalize_capture_stage(
                    stage, root / "dataset", profile, project_root=ROOT
                )

    def test_observation_timestamp_must_be_stereo_midpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            path = stage / "observations_v5.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["capture_reference_ns"] = "2000000001"
            path.unlink()
            _write_csv(path, MAINLINE_OBSERVATION_COLUMNS, rows)

            with self.assertRaisesRegex(DatasetError, "整数中点"):
                finalize_capture_stage(
                    stage, root / "dataset", profile, project_root=ROOT
                )

    def test_selected_candidates_must_equal_accepted_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            path = stage / "detections_v5.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["selected"] = "false"
            path.unlink()
            _write_csv(path, MAINLINE_DETECTION_COLUMNS, rows)

            with self.assertRaisesRegex(DatasetError, "selected"):
                finalize_capture_stage(
                    stage, root / "dataset", profile, project_root=ROOT
                )


if __name__ == "__main__":
    unittest.main()
