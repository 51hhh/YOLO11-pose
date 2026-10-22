from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.dataset import (
    DatasetError,
    TrialPackageWriter,
    validate_dataset,
    validate_trial,
)
from tests.support.trial_factory import (
    calibration_fixture as _calibration,
    detection_fixture as _detection,
    frame_fixture as _frame,
    metadata_fixture as _metadata,
    write_valid_trial as _write_valid_trial,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _set_declared_rows(trial: Path, table: str, count: int) -> None:
    metadata_path = trial / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if table == "trajectory":
        metadata["trajectory"]["sample_count"] = count
    else:
        metadata["sidecars"][table]["row_count"] = count
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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

    def test_validator_rejects_accepted_pair_without_selected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            detections_path = trial / "detections_2d.csv"
            fieldnames, rows = _read_csv(detections_path)
            rows[0]["selected"] = "false"
            _write_csv(detections_path, fieldnames, rows)

            with self.assertRaisesRegex(DatasetError, "selected.*accepted"):
                validate_trial(trial, project_root=ROOT)

    def test_validator_rejects_invalid_detection_values(self) -> None:
        cases = (
            ("u_px", "nan", "必须为有限数"),
            ("u_px", "1440", "来源图像内"),
            ("radius_px", "0", "radius_px 必须大于 0"),
            ("radius_px", "2000", "不超过图像尺度"),
            ("bbox_w", "0", "bbox 宽高必须大于 0"),
            ("bbox_w", "2000", "不超过图像尺寸"),
            ("bbox_x", "800", "检测中心必须位于 bbox 内"),
            ("cov_uv_px2", "2", "2D 协方差不是半正定"),
        )
        for field, replacement, message in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                trial = _write_valid_trial(Path(directory))
                path = trial / "detections_2d.csv"
                fieldnames, rows = _read_csv(path)
                rows[0][field] = replacement
                _write_csv(path, fieldnames, rows)
                with self.assertRaisesRegex(DatasetError, message):
                    validate_trial(trial, project_root=ROOT)

    def test_validator_requires_complete_pixel_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            path = trial / "detections_2d.csv"
            fieldnames, rows = _read_csv(path)
            rows[0]["du_dt_pxps"] = "1.0"
            _write_csv(path, fieldnames, rows)
            with self.assertRaisesRegex(DatasetError, "必须整组填写或留空"):
                validate_trial(trial, project_root=ROOT)

    def test_validator_normalizes_large_covariance_before_psd_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            path = trial / "detections_2d.csv"
            fieldnames, rows = _read_csv(path)
            rows[0]["cov_uu_px2"] = "9e307"
            rows[0]["cov_uv_px2"] = "1e308"
            rows[0]["cov_vv_px2"] = "9e307"
            _write_csv(path, fieldnames, rows)
            with self.assertRaisesRegex(DatasetError, "不是半正定"):
                validate_trial(trial, project_root=ROOT)

    def test_validator_rejects_detection_on_dropped_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            path = trial / "camera_frames.csv"
            fieldnames, rows = _read_csv(path)
            rows[0]["dropped"] = "true"
            rows[0]["drop_reason"] = "injected drop"
            _write_csv(path, fieldnames, rows)
            with self.assertRaisesRegex(DatasetError, "不得引用 dropped 帧"):
                validate_trial(trial, project_root=ROOT)

    def test_validator_binds_trajectory_values_to_accepted_pair(self) -> None:
        cases = (
            ("x_m", "0.2", "3D 位置"),
            ("cov_zz_m2", "0.02", "协方差"),
            ("reprojection_error_px", "0.2", "重投影误差"),
        )
        for field, replacement, message in cases:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                trial = _write_valid_trial(Path(directory))
                path = trial / "trajectory.csv"
                fieldnames, rows = _read_csv(path)
                rows[0][field] = replacement
                _write_csv(path, fieldnames, rows)
                with self.assertRaisesRegex(DatasetError, message):
                    validate_trial(trial, project_root=ROOT)

    def test_validator_rejects_multiple_accepted_pairs_for_one_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            path = trial / "stereo_pairs.csv"
            fieldnames, rows = _read_csv(path)
            duplicate = dict(rows[0])
            duplicate["pair_id"] = "1"
            rows.append(duplicate)
            _write_csv(path, fieldnames, rows)
            _set_declared_rows(trial, "stereo_pairs", len(rows))

            with self.assertRaisesRegex(DatasetError, "只能有一个 accepted"):
                validate_trial(trial, project_root=ROOT)

    def test_validator_rejects_accepted_pair_without_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = _write_valid_trial(Path(directory))
            path = trial / "trajectory.csv"
            fieldnames, _ = _read_csv(path)
            _write_csv(path, fieldnames, [])
            _set_declared_rows(trial, "trajectory", 0)

            with self.assertRaisesRegex(DatasetError, "必须各自生成一个"):
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

