from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.calibration import build_capture_calibration
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
from volleyball_vision.stereo_measurements import (
    STEREO_MEASUREMENT_COLUMNS,
    STEREO_MEASUREMENT_SCHEMA_VERSION,
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
    def test_mainline_context_uses_v6_without_duplicating_capture_identity(self) -> None:
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
        self.assertEqual(
            context["stereo_measurement_schema_version"],
            STEREO_MEASUREMENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            context["perception_pipeline"]["output_contract"],
            "yolo_pre_nms",
        )
        self.assertEqual(
            context["perception_pipeline"]["class_names"], ["volleyball"]
        )
        self.assertEqual(
            context["perception_pipeline"]["target_class_name"],
            "volleyball",
        )
        self.assertTrue(context["planning_only"])
        self.assertEqual(context["config_sha256"], profile.fingerprint_sha256)

    def _stage(
        self,
        root: Path,
        *,
        frame_count: int = 1,
        measurement_inputs: bool = False,
        consistent_geometry: bool = False,
    ) -> tuple[Path, object]:
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
        if measurement_inputs:
            context["stereo_measurement_schema_version"] = (
                STEREO_MEASUREMENT_SCHEMA_VERSION
            )
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
                "coordinate_frame": "nx_left_rectified_optical_frame",
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
                "depth_method": "circle_center",
                "measurement_disparity_px": 49.75,
                "disparity_stddev_px": 0.2,
                "support_points": 8,
            }
        ]
        # 同一生成器用于单仓合同测试及跨仓三帧正式格式集成测试。
        if frame_count > 1:
            originals = copy.deepcopy(
                (
                    frame_rows,
                    detection_rows,
                    association_rows,
                    observation_rows,
                )
            )
            for index in range(1, frame_count):
                dt = index * 10_000_000
                for original in originals[0]:
                    row = dict(original)
                    for key in (
                        "hardware_timestamp_ns",
                        "exposure_start_ns",
                        "capture_timestamp_ns",
                        "receive_timestamp_ns",
                        "inference_done_timestamp_ns",
                        "receive_monotonic_ns",
                        "trigger_scheduled_monotonic_ns",
                        "trigger_edge_monotonic_ns",
                        "trigger_edge_realtime_ns",
                    ):
                        row[key] += dt
                    for key in (
                        "frame_index",
                        "sensor_frame_number",
                        "sensor_frame_counter",
                        "sensor_trigger_index",
                        "trigger_sequence",
                    ):
                        row[key] += index
                    row["raw_image"] = (
                        f"raw/mainline/{row['camera_role']}/{42 + index}.pgm"
                    )
                    _write_pgm(stage / row["raw_image"], width, height)
                    frame_rows.append(row)
                for original in originals[1]:
                    detection_rows.append(
                        {**original, "frame_index": 42 + index}
                    )
                association_rows.append(
                    {
                        **originals[2][0],
                        "pair_id": 7 + index,
                        "left_frame_index": 42 + index,
                        "right_frame_index": 42 + index,
                        "left_capture_ns": 2_000_000_000 + dt,
                        "right_capture_ns": 2_000_000_000 + dt,
                    }
                )
                observation_rows.append(
                    {
                        **originals[3][0],
                        "observation_id": 3 + index,
                        "source_pair_id": 7 + index,
                        "capture_reference_ns": 2_000_000_000 + dt,
                    }
                )
        measurement_rows = []
        calibration = build_capture_calibration(profile, project_root=ROOT)
        matrix = calibration["cameras"][0]["camera_matrix"]
        for index, row in enumerate(observation_rows):
            u, v = 700.25, 500.0 + index * 0.2
            disparity = row["measurement_disparity_px"]
            if consistent_geometry:
                disparity = matrix[0][0] * calibration["baseline_m"] / 10.0
                row["measurement_disparity_px"] = disparity
                row["z_m"] = 10.0
                row["x_m"] = (u - matrix[0][2]) * 10.0 / matrix[0][0]
                row["y_m"] = (v - matrix[1][2]) * 10.0 / matrix[1][1]
            measurement_rows.append(
                {
                    "schema_version": STEREO_MEASUREMENT_SCHEMA_VERSION,
                    "pair_id": row["source_pair_id"],
                    "method": row["depth_method"],
                    "anchor_u_px": u,
                    "anchor_v_px": v,
                    "disparity_px": disparity,
                }
            )
        if measurement_inputs:
            _write_csv(
                stage / "stereo_measurements_v1.csv",
                STEREO_MEASUREMENT_COLUMNS,
                measurement_rows,
            )
        _write_csv(stage / "frames_v6.csv", MAINLINE_FRAME_COLUMNS, frame_rows)
        _write_csv(
            stage / "detections_v6.csv", MAINLINE_DETECTION_COLUMNS, detection_rows
        )
        _write_csv(
            stage / "associations_v6.csv",
            MAINLINE_ASSOCIATION_COLUMNS,
            association_rows,
        )
        _write_csv(
            stage / "observations_v6.csv",
            MAINLINE_OBSERVATION_COLUMNS,
            observation_rows,
        )
        (stage / "status_v6.json").write_text(
            json.dumps(
                {
                    "stage_schema_version": MAINLINE_STAGE_SCHEMA_VERSION,
                    "complete": True,
                    "fatal_error": None,
                    "stereo_frames_written": frame_count,
                    "camera_frames_written": 2 * frame_count,
                    "detections_written": 2 * frame_count,
                    "associations_written": frame_count,
                    "observations_written": frame_count,
                    "raw_images_written": 2 * frame_count,
                }
            ),
            encoding="utf-8",
        )
        return stage, profile

    def test_stereo_measurement_extension_is_hashed_and_declared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(
                root,
                frame_count=3,
                measurement_inputs=True,
                consistent_geometry=True,
            )
            trial = finalize_capture_stage(
                stage,
                root / "dataset",
                profile,
                project_root=ROOT,
            )
            metadata = json.loads(
                (trial / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertIn("direct_stereo_measurements", metadata["source_files"])
            self.assertNotIn("stereo_replay", metadata["conditions"])
            measurement_entries = [
                entry
                for entry in metadata["raw_manifest"]
                if entry["label"] == "direct_stereo_measurements"
            ]
            self.assertEqual(len(measurement_entries), 1)
            with (
                trial / metadata["source_files"]["direct_stereo_measurements"]
            ).open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            validate_dataset(root / "dataset", project_root=ROOT)

    def test_declared_measurement_extension_cannot_disappear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(
                root,
                measurement_inputs=True,
                consistent_geometry=True,
            )
            (stage / "stereo_measurements_v1.csv").unlink()
            with self.assertRaisesRegex(DatasetError, "文件缺失"):
                finalize_capture_stage(
                    stage,
                    root / "dataset",
                    profile,
                    project_root=ROOT,
                )

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
            self.assertNotIn(
                "direct_stereo_measurements",
                metadata["source_files"],
            )
            with (trial / "camera_frames.csv").open(encoding="utf-8") as handle:
                camera_frames = list(csv.DictReader(handle))
            self.assertEqual(
                {row["hardware_timestamp_ns"] for row in camera_frames},
                {"995066500"},
            )
            self.assertEqual(
                {row["exposure_start_ns"] for row in camera_frames},
                {"995066500"},
            )
            self.assertEqual(
                {row["capture_timestamp_ns"] for row in camera_frames},
                {"1000000000"},
            )
            with (trial / "stereo_pairs.csv").open(encoding="utf-8") as handle:
                stereo_pairs = list(csv.DictReader(handle))
            self.assertEqual(stereo_pairs[0]["disparity_px"], "49.75")
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
            monotonic_mapping = next(
                row
                for row in mappings
                if row["source_clock_domain"] == "nx-monotonic"
            )
            self.assertEqual(monotonic_mapping["source_anchor_ns"], "3000000000")
            self.assertEqual(monotonic_mapping["reference_anchor_ns"], "995066500")

    def test_frame_identity_remains_monotonic_after_uint32_wrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            logical_index = 1 << 32

            frames_path = stage / "frames_v6.csv"
            with frames_path.open(encoding="utf-8", newline="") as handle:
                frames = list(csv.DictReader(handle))
            for row in frames:
                row["frame_index"] = str(logical_index)
                row["sensor_frame_counter"] = "0"
                row["trigger_sequence"] = str(logical_index)
            frames_path.unlink()
            _write_csv(frames_path, MAINLINE_FRAME_COLUMNS, frames)

            detections_path = stage / "detections_v6.csv"
            with detections_path.open(encoding="utf-8", newline="") as handle:
                detections = list(csv.DictReader(handle))
            for row in detections:
                row["frame_index"] = str(logical_index)
            detections_path.unlink()
            _write_csv(
                detections_path, MAINLINE_DETECTION_COLUMNS, detections
            )

            associations_path = stage / "associations_v6.csv"
            with associations_path.open(encoding="utf-8", newline="") as handle:
                associations = list(csv.DictReader(handle))
            associations[0]["left_frame_index"] = str(logical_index)
            associations[0]["right_frame_index"] = str(logical_index)
            associations_path.unlink()
            _write_csv(
                associations_path,
                MAINLINE_ASSOCIATION_COLUMNS,
                associations,
            )

            trial = finalize_capture_stage(
                stage, root / "dataset", profile, project_root=ROOT
            )
            with (trial / "camera_frames.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                published = list(csv.DictReader(handle))
            self.assertEqual(
                {int(row["frame_index"]) for row in published},
                {logical_index},
            )
            with (trial / "raw" / "audit" / "frames_v6.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                audit_frames = list(csv.DictReader(handle))
            self.assertEqual(
                {int(row["sensor_frame_counter"]) for row in audit_frames},
                {0},
            )

    def test_partial_mainline_stage_is_not_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            (stage / "frames_v6.csv.partial").write_text("unfinished", encoding="utf-8")

            with self.assertRaisesRegex(DatasetError, "未完成文件"):
                finalize_capture_stage(
                    stage, root / "dataset", profile, project_root=ROOT
                )

    def test_observation_timestamp_must_be_stereo_midpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            path = stage / "observations_v6.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["capture_reference_ns"] = "2000000001"
            path.unlink()
            _write_csv(path, MAINLINE_OBSERVATION_COLUMNS, rows)

            with self.assertRaisesRegex(DatasetError, "整数中点"):
                finalize_capture_stage(
                    stage, root / "dataset", profile, project_root=ROOT
                )

    def test_observation_depth_method_must_be_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            path = stage / "observations_v6.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["depth_method"] = "unconfigured_method"
            path.unlink()
            _write_csv(path, MAINLINE_OBSERVATION_COLUMNS, rows)

            with self.assertRaisesRegex(DatasetError, "配置外"):
                finalize_capture_stage(
                    stage, root / "dataset", profile, project_root=ROOT
                )

    def test_selected_candidates_must_equal_accepted_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            path = stage / "detections_v6.csv"
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
