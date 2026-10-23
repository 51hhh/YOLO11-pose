from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from volleyball_vision.dataset import DatasetError
from volleyball_vision.stereo_measurements import (
    STEREO_MEASUREMENT_COLUMNS,
    STEREO_MEASUREMENT_SCHEMA_VERSION,
    validate_stereo_measurements,
)


CALIBRATION = {
    "camera_roles": {"left": "left-camera", "right": "right-camera"},
    "cameras": [
        {
            "camera_id": "left-camera",
            "camera_matrix": [
                [800.0, 0.0, 640.0],
                [0.0, 800.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
        }
    ],
    "baseline_m": 0.8,
}
OBSERVATION = {
    "source_pair_id": "7",
    "depth_method": "circle_center",
    "measurement_disparity_px": "80.0",
    "x_m": "0.8",
    "y_m": "0.4",
    "z_m": "8.0",
}
MEASUREMENT = {
    "schema_version": STEREO_MEASUREMENT_SCHEMA_VERSION,
    "pair_id": "7",
    "method": "circle_center",
    "anchor_u_px": "720.0",
    "anchor_v_px": "400.0",
    "disparity_px": "80.0",
}


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=STEREO_MEASUREMENT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


class StereoMeasurementTests(unittest.TestCase):
    def test_missing_legacy_extension_is_explicitly_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo_measurements_v1.csv"
            self.assertFalse(
                validate_stereo_measurements(
                    path,
                    [OBSERVATION],
                    CALIBRATION,
                )
            )

    def test_versioned_uvd_reconstructs_recorded_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo_measurements_v1.csv"
            _write(path, [MEASUREMENT])
            self.assertTrue(
                validate_stereo_measurements(
                    path,
                    [OBSERVATION],
                    CALIBRATION,
                )
            )

    def test_uvd_must_match_recorded_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo_measurements_v1.csv"
            _write(path, [MEASUREMENT])
            inconsistent = {**OBSERVATION, "z_m": "8.1"}
            with self.assertRaisesRegex(DatasetError, "UVD.*XYZ"):
                validate_stereo_measurements(
                    path,
                    [inconsistent],
                    CALIBRATION,
                )

    def test_schema_version_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo_measurements_v1.csv"
            _write(path, [{**MEASUREMENT, "schema_version": "2.0.0"}])
            with self.assertRaisesRegex(DatasetError, "版本"):
                validate_stereo_measurements(
                    path,
                    [OBSERVATION],
                    CALIBRATION,
                )

    def test_every_observation_requires_one_unique_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo_measurements_v1.csv"
            _write(path, [MEASUREMENT, MEASUREMENT])
            with self.assertRaisesRegex(DatasetError, "唯一性"):
                validate_stereo_measurements(
                    path,
                    [OBSERVATION],
                    CALIBRATION,
                )


if __name__ == "__main__":
    unittest.main()
