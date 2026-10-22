from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from tests.support.trial_factory import write_valid_trial
from volleyball_vision.config import load_profile
from volleyball_vision.dataset import DatasetError, validate_trial
from volleyball_vision.replay import run_replay


ROOT = Path(__file__).resolve().parents[2]


class ReplayTests(unittest.TestCase):
    def test_replay_writes_only_fingerprinted_derived_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = write_valid_trial(root / "dataset")
            trajectory_before = (trial / "trajectory.csv").read_bytes()
            config = load_profile("replay", project_root=ROOT)

            result = run_replay(trial, config, project_root=ROOT)

            self.assertEqual(result.output_directory.parent, trial / "derived")
            self.assertEqual(result.output_directory.name, result.fingerprint_sha256)
            self.assertEqual(result.metrics["quality"]["direct_observations"], 1)
            self.assertEqual(result.metrics["quality"]["accepted_pairs"], 1)
            timing = result.metrics["quality"]["transport_timing"]
            self.assertEqual(timing["arrival_order"]["tie"], 1)
            self.assertEqual(timing["arrival_order"]["left_first"], 0)
            self.assertEqual(timing["arrival_order"]["right_first"], 0)
            self.assertEqual(timing["pairing_key"], "frame_counter")
            self.assertFalse(timing["arrival_time_used_for_pairing"])
            self.assertAlmostEqual(
                timing["exposure_to_receive_ms"]["left"]["p50"],
                0.00015,
            )
            self.assertIsNone(result.metrics["shadow_comparison"])
            self.assertEqual((trial / "trajectory.csv").read_bytes(), trajectory_before)
            validate_trial(trial, project_root=ROOT)

    def test_shadow_comparison_is_deterministic_and_reports_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = write_valid_trial(root / "dataset")
            shadow = root / "shadow.csv"
            columns = [
                "sample_index",
                "timestamp_s",
                "x_m",
                "y_m",
                "z_m",
                "cov_xx_m2",
                "cov_xy_m2",
                "cov_xz_m2",
                "cov_yy_m2",
                "cov_yz_m2",
                "cov_zz_m2",
            ]
            with shadow.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_index": 7,
                        "timestamp_s": 0.001,
                        "x_m": 0.2,
                        "y_m": 0.2,
                        "z_m": 5.0,
                        "cov_xx_m2": 0.01,
                        "cov_xy_m2": 0.0,
                        "cov_xz_m2": 0.0,
                        "cov_yy_m2": 0.01,
                        "cov_yz_m2": 0.0,
                        "cov_zz_m2": 0.01,
                    }
                )
            config = load_profile("replay", project_root=ROOT)

            first = run_replay(
                trial,
                config,
                shadow_trajectory=shadow,
                output_root=root / "reports",
                max_time_delta_ms=2.0,
                project_root=ROOT,
            )
            second = run_replay(
                trial,
                config,
                shadow_trajectory=shadow,
                output_root=root / "reports",
                max_time_delta_ms=2.0,
                project_root=ROOT,
            )

            comparison = first.metrics["shadow_comparison"]
            self.assertEqual(comparison["matched_samples"], 1)
            self.assertAlmostEqual(comparison["position_error_m"]["rmse"], 0.1)
            self.assertAlmostEqual(
                comparison["uncertainty"]["mean_mahalanobis_sq"],
                1.0,
            )
            self.assertEqual(first.fingerprint_sha256, second.fingerprint_sha256)
            self.assertTrue(second.reused)

    def test_replay_rejects_runtime_capture_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trial = write_valid_trial(Path(directory) / "dataset")
            config = load_profile("free-flight", project_root=ROOT)
            with self.assertRaisesRegex(DatasetError, "runtime.mode=replay"):
                run_replay(trial, config, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
