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
from volleyball_vision.dataset import (
    DatasetError,
    validate_dataset,
    with_pipeline_fingerprint,
)
from volleyball_vision.replay import run_replay


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
        (stage / "pipeline.yaml").write_text("capture_v21: {enabled: true}\n", encoding="utf-8")
        raw = stage / "raw"
        raw.mkdir()
        frame_rows = []
        detection_rows = []
        pair_rows = []
        for frame_id in range(2):
            left_device = 1_000_000_000 + frame_id * 11_111_111
            host = 2_000_000_000 + frame_id * 11_111_111
            trigger_monotonic = 3_000_010_000 + frame_id * 11_111_111
            # 9867 us 全局曝光完成后 USB payload 才能到达；让夹具保持真实的
            # 曝光中心 < 接收 < callback 顺序。
            trigger_realtime = host - 12_000_000
            # 刻意让左右 USB 到达先后在相邻帧间反转；配对仍必须由
            # FrameCounter/触发序号决定，不得由主机到达顺序决定。
            if frame_id == 0:
                left_receive_delay = 10_500_000
                right_receive_delay = 10_000_000
            else:
                left_receive_delay = 9_900_000
                right_receive_delay = 10_600_000
            left_rel = f"raw/left-{frame_id}.pgm"
            right_rel = f"raw/right-{frame_id}.pgm"
            (stage / left_rel).write_bytes(b"P5\n1 1\n255\n\x00")
            (stage / right_rel).write_bytes(b"P5\n1 1\n255\n\x00")
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "host_capture_timestamp_ns": host,
                    "callback_timestamp_ns": host + 1_000_000,
                    "left_receive_monotonic_ns": (
                        trigger_monotonic + left_receive_delay
                    ),
                    "right_receive_monotonic_ns": (
                        trigger_monotonic + right_receive_delay
                    ),
                    "left_receive_realtime_ns": (
                        trigger_realtime + left_receive_delay
                    ),
                    "right_receive_realtime_ns": (
                        trigger_realtime + right_receive_delay
                    ),
                    "left_sdk_host_timestamp_raw": host - 600_000,
                    "right_sdk_host_timestamp_raw": host - 500_000,
                    "left_device_timestamp_ns": left_device,
                    "right_device_timestamp_ns": left_device - 50_000,
                    "stereo_timestamp_residual_ns": 100,
                    "trigger_sequence": 100 + frame_id,
                    "trigger_scheduled_monotonic_ns": trigger_monotonic - 10_000,
                    "trigger_edge_monotonic_ns": trigger_monotonic,
                    "trigger_edge_realtime_ns": trigger_realtime,
                    "trigger_write_uncertainty_ns": 1_000,
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
            self.assertEqual(metadata["coordinate_frame"]["position_unit"], "m")
            self.assertEqual(metadata["coordinate_frame"]["time_unit"], "s")
            self.assertEqual(metadata["capture"]["trajectory_timebase"]["clock_domain"], "hik-shared-device-clock")
            self.assertEqual(
                metadata["capture"]["trajectory_timebase"]["timestamp_semantics"],
                "stereo_midpoint_capture",
            )
            self.assertEqual(
                metadata["capture"]["exposure_timing"],
                {
                    "shutter_type": "global",
                    "timestamp_source": "nx_gpio_rising_edge",
                    "delay_status": "assumed_zero_unmeasured",
                    "delay_ns": 0,
                    "delay_uncertainty_ns": None,
                    "observation_timestamp": "exposure_midpoint",
                },
            )
            with (trial / "camera_frames.csv").open(encoding="utf-8") as handle:
                camera_frames = list(csv.DictReader(handle))
            self.assertEqual(
                camera_frames[0]["capture_timestamp_ns"],
                camera_frames[1]["capture_timestamp_ns"],
            )
            self.assertEqual(
                int(camera_frames[0]["capture_timestamp_ns"])
                - int(camera_frames[0]["exposure_start_ns"]),
                4_933_500,
            )
            tampered = list(camera_frames)
            tampered[0] = dict(tampered[0])
            tampered[0]["capture_timestamp_ns"] = str(
                int(tampered[0]["capture_timestamp_ns"]) - 1
            )
            with (trial / "camera_frames.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(tampered[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(tampered)
            with self.assertRaisesRegex(
                DatasetError, "必须等于曝光开始加半个曝光时长"
            ):
                validate_dataset(dataset, project_root=ROOT)
            with (trial / "camera_frames.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(camera_frames[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(camera_frames)
            left_frames = [
                row
                for row in camera_frames
                if row["camera_id"] == "hik-00D39342665"
            ]
            right_frames = [
                row
                for row in camera_frames
                if row["camera_id"] == "hik-00219471413"
            ]
            self.assertGreater(
                int(left_frames[0]["receive_timestamp_ns"]),
                int(right_frames[0]["receive_timestamp_ns"]),
            )
            self.assertLess(
                int(left_frames[1]["receive_timestamp_ns"]),
                int(right_frames[1]["receive_timestamp_ns"]),
            )
            self.assertTrue(
                all(row["inference_done_timestamp_ns"] for row in camera_frames)
            )
            with (trial / "processing_timing.csv").open(encoding="utf-8") as handle:
                timings = list(csv.DictReader(handle))
            self.assertEqual(
                sum(row["stage"] == "gpio_rising_edge_write" for row in timings),
                2,
            )
            self.assertEqual(
                sum(row["stage"] == "camera_frame_receive" for row in timings),
                4,
            )
            with (trial / "clock_mappings.csv").open(encoding="utf-8") as handle:
                mappings = list(csv.DictReader(handle))
            self.assertEqual(
                {row["source_clock_domain"] for row in mappings},
                {"nx-monotonic", "nx-realtime"},
            )
            self.assertGreaterEqual(len(metadata["raw_manifest"]), 9)

            replay_profile = load_profile("replay", project_root=ROOT)
            replay = run_replay(
                trial,
                replay_profile,
                output_root=root / "derived-report",
                project_root=ROOT,
            )
            transport = replay.metrics["quality"]["transport_timing"]
            self.assertEqual(
                transport["arrival_order"],
                {
                    "left_first": 1,
                    "right_first": 1,
                    "tie": 0,
                    "unknown": 0,
                },
            )
            self.assertEqual(transport["pairing_key"], "frame_counter")
            self.assertFalse(transport["arrival_time_used_for_pairing"])

    def test_multiple_staging_observations_publish_one_single_ball_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)

            detections_path = stage / "detections.csv"
            with detections_path.open(encoding="utf-8", newline="") as handle:
                detections = list(csv.DictReader(handle))
            for role, cx in (("left", 710.0), ("right", 665.0)):
                detections.append(
                    {
                        "frame_id": 0,
                        "camera_role": role,
                        "candidate_id": 1,
                        "cx_px": cx,
                        "cy_px": 501.0,
                        "width_px": 38.0,
                        "height_px": 38.0,
                        "confidence": 0.99,
                        "class_id": 0,
                        "selected": True,
                    }
                )
            detections_path.unlink()
            _write_csv(
                detections_path,
                STAGE_DETECTION_COLUMNS,
                detections,
            )

            pairs_path = stage / "pairs.csv"
            with pairs_path.open(encoding="utf-8", newline="") as handle:
                pairs = list(csv.DictReader(handle))
            pairs.append(
                {
                    "pair_id": 99,
                    "frame_id": 0,
                    "left_candidate_id": 1,
                    "right_candidate_id": 1,
                    "disparity_px": 45.0,
                    "epipolar_error_px": 0.05,
                    "pair_score": 0.05,
                    "semantic_confidence": 0.99,
                    "x_rig_m": 0.9,
                    "y_rig_m": 0.8,
                    "z_rig_m": 4.0,
                    "accepted": True,
                    "reject_reason": "",
                }
            )
            pairs_path.unlink()
            _write_csv(pairs_path, STAGE_PAIR_COLUMNS, pairs)

            trial = finalize_capture_stage(
                stage,
                root / "dataset",
                profile,
                project_root=ROOT,
            )
            with (trial / "stereo_pairs.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                final_pairs = list(csv.DictReader(handle))
            frame_zero_pairs = [
                row for row in final_pairs
                if row["left_frame_index"] == "0"
            ]
            self.assertEqual(
                [row["pair_id"] for row in frame_zero_pairs if row["accepted"] == "true"],
                ["99"],
            )
            demoted = next(row for row in frame_zero_pairs if row["pair_id"] == "0")
            self.assertEqual(demoted["reject_reason"], "not_selected")
            self.assertEqual(demoted["x_rig_m"], "")

            with (trial / "detections_2d.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                final_detections = list(csv.DictReader(handle))
            selected_frame_zero = {
                (row["camera_id"], row["candidate_id"])
                for row in final_detections
                if row["frame_index"] == "0" and row["selected"] == "true"
            }
            self.assertEqual(
                selected_frame_zero,
                {
                    ("hik-00D39342665", "1"),
                    ("hik-00219471413", "1"),
                },
            )

            with (trial / "trajectory.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                trajectory = list(csv.DictReader(handle))
            self.assertEqual(trajectory[0]["x_m"], "0.9")
            self.assertEqual(trajectory[0]["z_m"], "4.0")

            with (trial / "raw" / "audit" / "stage_pairs.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                staging_pairs = list(csv.DictReader(handle))
            self.assertEqual(
                sum(
                    row["frame_id"] == "0" and row["accepted"] == "True"
                    for row in staging_pairs
                ),
                2,
            )

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
            rows[0]["trigger_sequence"] = str(0xFFFFFFFF)
            rows[1]["trigger_sequence"] = str(0x100000000)
            (stage / "frames.csv").unlink()
            _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, rows)
            trial = finalize_capture_stage(
                stage,
                root / "dataset",
                profile,
                project_root=ROOT,
            )
            self.assertTrue((trial / "metadata.json").is_file())

    def test_trigger_sequence_must_match_camera_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            with (stage / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["trigger_sequence"] = "102"
            (stage / "frames.csv").unlink()
            _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, rows)
            with self.assertRaisesRegex(Exception, "FrameCounter"):
                finalize_capture_stage(stage, root / "dataset", profile, project_root=ROOT)

    def test_usb_receive_time_must_follow_trigger_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            with (stage / "frames.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["left_receive_monotonic_ns"] = str(
                int(rows[0]["trigger_edge_monotonic_ns"]) - 1
            )
            (stage / "frames.csv").unlink()
            _write_csv(stage / "frames.csv", STAGE_FRAME_COLUMNS, rows)
            with self.assertRaisesRegex(Exception, "USB.*GPIO"):
                finalize_capture_stage(
                    stage,
                    root / "dataset",
                    profile,
                    project_root=ROOT,
                )

    def test_planning_stage_can_never_be_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, profile = self._stage(root)
            context_path = stage / "context.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["planning_only"] = True
            context_path.write_text(json.dumps(context), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "预演 staging"):
                finalize_capture_stage(
                    stage,
                    root / "dataset",
                    profile,
                    project_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()

