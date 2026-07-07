#!/usr/bin/env python3
"""Synthetic coverage for trajectory fusion CSV tooling."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from trajectory_fusion import evaluate_fusion  # noqa: E402
from trajectory_fusion import run_dataset_workflow as workflow_module  # noqa: E402
from trajectory_fusion import run_reliability_sweep as sweep_module  # noqa: E402
from trajectory_fusion.analyze_candidate_consistency import analyze_candidate_consistency  # noqa: E402
from trajectory_fusion.audit_reliability_methods import audit_reliability_methods  # noqa: E402
from trajectory_fusion.audit_training_inputs import (  # noqa: E402
    audit_training_inputs,
    write_feature_csv as write_training_feature_csv,
    write_json as write_training_audit_json,
    write_method_csv as write_training_method_csv,
)
from trajectory_fusion.build_dataset_manifest import (  # noqa: E402
    build_manifest,
    discover_csvs,
    write_manifest,
)
from trajectory_fusion.check_dataset import analyze_dataset  # noqa: E402
from trajectory_fusion.compare_workflows import compare_workflows, write_csv as write_workflow_compare_csv  # noqa: E402
from trajectory_fusion.dataset import (  # noqa: E402
    METHOD_COLUMNS,
    build_legacy_arrays,
    derive_frame_summary_path,
    find_metadata_for_csv,
    legacy_feature_names,
    load_legacy_sequences,
    read_metadata,
)
from trajectory_fusion.evaluate_calibrated_smoother import apply_calibrated_smoother  # noqa: E402
from trajectory_fusion.evaluate_reliability_smoother import (  # noqa: E402
    LearnedObservationConfig,
    _build_learned_observations,
)
from trajectory_fusion.fit_method_calibration import (  # noqa: E402
    CalibrationConfig,
    fit_method_calibration,
    write_calibration,
)
from trajectory_fusion.manifest import is_manifest_path, load_manifest  # noqa: E402
from trajectory_fusion.models import ReliabilityOutput  # noqa: E402
from trajectory_fusion.rank_sweep_metrics import rank_metrics  # noqa: E402
from trajectory_fusion.robust_smoother import group_correlated_z_measurements  # noqa: E402
from trajectory_fusion.run_evaluation_suite import run_suite  # noqa: E402
from trajectory_fusion.run_dataset_workflow import run_workflow  # noqa: E402
from trajectory_fusion.run_workflow_matrix import run_workflow_matrix  # noqa: E402
from trajectory_fusion.run_reliability_sweep import build_train_command, load_sweep_configs  # noqa: E402
from trajectory_fusion.select_reliability_model import select_reliability_models  # noqa: E402
from trajectory_fusion.summarize_evaluation_suite import summarize_reliability_methods, summarize_suite  # noqa: E402
from trajectory_fusion.summarize_workflow import build_workflow_report  # noqa: E402
from trajectory_fusion.train_reliability import (  # noqa: E402
    _training_label_summary,
    load_sequences_from_clips,
    resolve_input_clips,
)
from trajectory_fusion.validate_dataset_manifest import analyze_manifest  # noqa: E402


HEADER = [
    "frame_id",
    "timestamp",
    "track_id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "z_mono",
    "z_stereo",
    "depth_method",
    "confidence",
    "class_id",
    "z_bbox_center",
    "z_bbox_left_edge",
    "z_bbox_right_edge",
    "z_circle_center",
    "z_circle_left_edge",
    "z_circle_right_edge",
    "z_roi_edge_centroid",
    "z_roi_radial_center",
    "z_roi_edge_pair_center",
    "z_roi_center_patch",
    "z_roi_multi_point",
    "z_roi_neural_feature",
    "z_roi_cuda_stereo_bm",
    "z_roi_ring_edge_profile",
    "disparity_roi_cuda_stereo_bm",
    "disparity_roi_ring_edge_profile",
    "roi_cuda_stereo_bm_support",
    "roi_cuda_stereo_bm_std_px",
    "roi_cuda_stereo_bm_confidence",
    "roi_ring_edge_profile_support",
    "roi_ring_edge_profile_std_px",
    "roi_ring_edge_profile_confidence",
    "z_fallback",
    "z_fallback_epipolar",
    "z_fallback_template",
    "z_fallback_feature_points",
    "raw_observation_valid",
    "left_circle_source",
    "right_circle_source",
    "frame_counter_delta",
    "frame_number_delta",
    "stereo_match_source",
    "stereo_depth_source",
    "pair_positive_disparity",
]


def _write_synthetic_clip(root: Path) -> Path:
    csv_path = root / "traj_p0p1_001.csv"
    rows = []
    for index in range(4):
        z = 3.0 + index * 0.01
        is_fallback = index == 2
        is_r2l_fallback = index == 3
        rows.append(
            {
                "frame_id": str(index + 1),
                "timestamp": f"{100.0 + index * 0.01:.2f}",
                "track_id": "0",
                "x": "0.0",
                "y": "0.0",
                "z": f"{z:.3f}",
                "vx": "0.0",
                "vy": "0.0",
                "vz": "0.0",
                "ax": "0.0",
                "ay": "0.0",
                "az": "0.0",
                "z_mono": f"{z + 0.10:.3f}",
                "z_stereo": f"{z + 0.20:.3f}",
                "depth_method": "1",
                "confidence": "0.9",
                "class_id": "0",
                "z_bbox_center": f"{z - 0.01:.3f}",
                "z_bbox_left_edge": f"{z - 0.02:.3f}",
                "z_bbox_right_edge": f"{z:.3f}",
                "z_circle_center": "-1" if is_fallback or is_r2l_fallback else f"{z:.3f}",
                "z_circle_left_edge": "-1",
                "z_circle_right_edge": "-1",
                "z_roi_edge_centroid": f"{z + 0.001:.3f}",
                "z_roi_radial_center": f"{z + 0.002:.3f}",
                "z_roi_edge_pair_center": f"{z + 0.003:.3f}",
                "z_roi_center_patch": f"{z + 0.05:.3f}",
                "z_roi_multi_point": f"{z - 0.05:.3f}",
                "z_roi_neural_feature": "-1",
                "z_roi_cuda_stereo_bm": "3.200" if index == 0 else "-1",
                "z_roi_ring_edge_profile": "3.300" if index == 0 else "-1",
                "disparity_roi_cuda_stereo_bm": "456.0" if index == 0 else "-1",
                "disparity_roi_ring_edge_profile": "444.0" if index == 0 else "-1",
                "roi_cuda_stereo_bm_support": "7" if index == 0 else "0",
                "roi_cuda_stereo_bm_std_px": "0.4" if index == 0 else "-1",
                "roi_cuda_stereo_bm_confidence": "0.7" if index == 0 else "0",
                "roi_ring_edge_profile_support": "9" if index == 0 else "0",
                "roi_ring_edge_profile_std_px": "0.6" if index == 0 else "-1",
                "roi_ring_edge_profile_confidence": "0.8" if index == 0 else "0",
                "z_fallback": f"{z + 0.20:.3f}" if is_fallback or is_r2l_fallback else "-1",
                "z_fallback_epipolar": f"{z + 0.20:.3f}" if is_fallback or is_r2l_fallback else "-1",
                "z_fallback_template": "-1",
                "z_fallback_feature_points": "-1",
                "raw_observation_valid": "1",
                "left_circle_source": "3" if is_r2l_fallback else "2",
                "right_circle_source": "3" if is_fallback else "2",
                "frame_counter_delta": "0",
                "frame_number_delta": "0",
                "stereo_match_source": "3" if is_r2l_fallback else ("2" if is_fallback else "1"),
                "stereo_depth_source": "1",
                "pair_positive_disparity": "0" if is_fallback or is_r2l_fallback else "1",
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    frames_path = root / "traj_p0p1_001.frames.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame_id",
            "timestamp",
            "result_count",
            "tracked_count",
            "raw_observation_count",
            "stereo_observation_count",
            "direct_pair_count",
            "fallback_l2r_count",
            "fallback_r2l_count",
            "pair_positive_count",
            "pair_shifted_iou_min",
            "pair_shifted_iou_mean",
            "pair_score_mean",
            "pair_bbox_prior_penalty_mean",
            "pair_epipolar_dy_max",
            "roi_iou_region_color_patch_support_max",
            "roi_patch_iou_color_edge_support_max",
            "roi_neural_feature_support_max",
            "p2_candidate_observed_count",
            "p2_candidate_valid_count",
            "p2_feature_valid_count",
            "p2_cuda_valid_count",
            "p2_neural_valid_count",
            "best_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "frame_id": index + 1,
                    "timestamp": f"{100.0 + index * 0.01:.2f}",
                    "result_count": 1,
                    "tracked_count": 1,
                    "raw_observation_count": 1,
                    "stereo_observation_count": 1,
                    "direct_pair_count": 1 if index < 2 else 0,
                    "fallback_l2r_count": 1 if index == 2 else 0,
                    "fallback_r2l_count": 1 if index == 3 else 0,
                    "pair_positive_count": 1 if index < 2 else 0,
                    "pair_shifted_iou_min": "0.4" if index < 2 else "-1",
                    "pair_shifted_iou_mean": "0.4" if index < 2 else "-1",
                    "pair_score_mean": "0.1" if index < 2 else "0",
                    "pair_bbox_prior_penalty_mean": "0.0",
                    "pair_epipolar_dy_max": "0.5" if index < 2 else "-1",
                    "roi_iou_region_color_patch_support_max": 0,
                    "roi_patch_iou_color_edge_support_max": 0,
                    "roi_neural_feature_support_max": 0,
                    "p2_candidate_observed_count": 2 if index < 2 else 0,
                    "p2_candidate_valid_count": 1 if index < 2 else 0,
                    "p2_feature_valid_count": 1 if index < 2 else 0,
                    "p2_cuda_valid_count": 0,
                    "p2_neural_valid_count": 0,
                    "best_confidence": "0.9",
                }
            )
    (root / "traj_p0p1_001.metadata.yaml").write_text(
        "known_z: 3.0\nstatic: true\nscene: synthetic\n",
        encoding="utf-8",
    )
    return csv_path


def _write_known_distance_clip(root: Path, stem: str, known_z: float, *, rows: int = 24) -> Path:
    csv_path = root / f"{stem}.csv"
    jitter = (-0.002, -0.001, 0.0, 0.001, 0.002, 0.001, 0.0, -0.001)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for index in range(rows):
            noise = jitter[index % len(jitter)]
            z = known_z + noise
            writer.writerow(
                {
                    "frame_id": str(index + 1),
                    "timestamp": f"{100.0 + index * 0.01:.2f}",
                    "track_id": "0",
                    "x": "0.0",
                    "y": "0.0",
                    "z": f"{known_z + 0.20 + noise:.4f}",
                    "vx": "0.0",
                    "vy": "0.0",
                    "vz": "0.0",
                    "ax": "0.0",
                    "ay": "0.0",
                    "az": "0.0",
                    "z_mono": f"{known_z + 0.12 + noise:.4f}",
                    "z_stereo": f"{known_z + 0.03 + noise:.4f}",
                    "depth_method": "1",
                    "confidence": "0.9",
                    "class_id": "0",
                    "z_bbox_center": f"{known_z + 0.020 + noise:.4f}",
                    "z_bbox_left_edge": f"{known_z + 0.030 + noise:.4f}",
                    "z_bbox_right_edge": f"{known_z + 0.025 + noise:.4f}",
                    "z_circle_center": f"{known_z + 0.005 + noise:.4f}",
                    "z_circle_left_edge": "-1",
                    "z_circle_right_edge": "-1",
                    "z_roi_edge_centroid": f"{known_z - 0.010 + noise:.4f}",
                    "z_roi_radial_center": f"{known_z + 0.004 + noise:.4f}",
                    "z_roi_edge_pair_center": f"{known_z + 0.006 + noise:.4f}",
                    "z_roi_center_patch": f"{known_z + 0.060 + noise:.4f}",
                    "z_roi_multi_point": f"{known_z - 0.055 + noise:.4f}",
                    "z_roi_neural_feature": f"{known_z - 0.020 + noise:.4f}",
                    "z_roi_cuda_stereo_bm": f"{known_z - 0.015 + noise:.4f}",
                    "z_roi_ring_edge_profile": f"{known_z + 0.010 + noise:.4f}",
                    "disparity_roi_cuda_stereo_bm": "450.0",
                    "disparity_roi_ring_edge_profile": "445.0",
                    "roi_cuda_stereo_bm_support": "7",
                    "roi_cuda_stereo_bm_std_px": "0.4",
                    "roi_cuda_stereo_bm_confidence": "0.7",
                    "roi_ring_edge_profile_support": "9",
                    "roi_ring_edge_profile_std_px": "0.6",
                    "roi_ring_edge_profile_confidence": "0.8",
                    "z_fallback": "-1",
                    "z_fallback_epipolar": "-1",
                    "z_fallback_template": "-1",
                    "z_fallback_feature_points": "-1",
                    "raw_observation_valid": "1",
                    "left_circle_source": "2",
                    "right_circle_source": "2",
                    "frame_counter_delta": "0",
                    "frame_number_delta": "0",
                    "stereo_match_source": "1",
                    "stereo_depth_source": "1",
                    "pair_positive_disparity": "1",
                }
            )

    frames_path = root / f"{stem}.frames.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame_id",
            "timestamp",
            "result_count",
            "tracked_count",
            "raw_observation_count",
            "stereo_observation_count",
            "direct_pair_count",
            "fallback_l2r_count",
            "fallback_r2l_count",
            "pair_positive_count",
            "pair_shifted_iou_min",
            "pair_shifted_iou_mean",
            "pair_score_mean",
            "pair_bbox_prior_penalty_mean",
            "pair_epipolar_dy_max",
            "roi_iou_region_color_patch_support_max",
            "roi_patch_iou_color_edge_support_max",
            "roi_neural_feature_support_max",
            "p2_candidate_observed_count",
            "p2_candidate_valid_count",
            "p2_feature_valid_count",
            "p2_cuda_valid_count",
            "p2_neural_valid_count",
            "best_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "frame_id": index + 1,
                    "timestamp": f"{100.0 + index * 0.01:.2f}",
                    "result_count": 1,
                    "tracked_count": 1,
                    "raw_observation_count": 1,
                    "stereo_observation_count": 1,
                    "direct_pair_count": 1,
                    "fallback_l2r_count": 0,
                    "fallback_r2l_count": 0,
                    "pair_positive_count": 1,
                    "pair_shifted_iou_min": "0.5",
                    "pair_shifted_iou_mean": "0.5",
                    "pair_score_mean": "0.1",
                    "pair_bbox_prior_penalty_mean": "0.0",
                    "pair_epipolar_dy_max": "0.5",
                    "roi_iou_region_color_patch_support_max": 0,
                    "roi_patch_iou_color_edge_support_max": 0,
                    "roi_neural_feature_support_max": 0,
                    "p2_candidate_observed_count": 4,
                    "p2_candidate_valid_count": 4,
                    "p2_feature_valid_count": 1,
                    "p2_cuda_valid_count": 1,
                    "p2_neural_valid_count": 1,
                    "best_confidence": "0.9",
                }
            )
    (root / f"{stem}.metadata.yaml").write_text(
        f"known_z: {known_z:.4f}\nknown_z_tolerance_m: 0.05\nstatic: true\nscene: synthetic_known_z\n",
        encoding="utf-8",
    )
    return csv_path


class SyntheticDatasetTest(unittest.TestCase):
    def test_metadata_autodiscovery_and_frame_summary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_synthetic_clip(Path(tmp))
            metadata_path = find_metadata_for_csv(csv_path)
            self.assertEqual(metadata_path, Path(tmp) / "traj_p0p1_001.metadata.yaml")
            self.assertEqual(read_metadata(metadata_path)["known_z"], 3.0)
            self.assertEqual(
                derive_frame_summary_path(csv_path),
                Path(tmp) / "traj_p0p1_001.frames.csv",
            )
            self.assertEqual(
                derive_frame_summary_path(Path(tmp) / "clip"),
                Path(tmp) / "clip.frames.csv",
            )

    def test_training_candidates_do_not_include_legacy_outputs(self) -> None:
        method_keys = {key for _, key in METHOD_COLUMNS}
        feature_names = set(legacy_feature_names())
        self.assertNotIn("z_stereo", method_keys)
        self.assertNotIn("z", method_keys)
        self.assertNotIn("z_circle_left_edge", method_keys)
        self.assertNotIn("z_circle_right_edge", method_keys)
        self.assertNotIn("z_stereo", feature_names)
        self.assertNotIn("z", feature_names)
        self.assertNotIn("z_circle_left_edge", feature_names)
        self.assertNotIn("z_circle_right_edge", feature_names)
        self.assertNotIn("disparity_circle_left_edge", feature_names)
        self.assertNotIn("disparity_circle_right_edge", feature_names)
        for online_state_name in (
            "x",
            "y",
            "vx",
            "vy",
            "vz",
            "ax",
            "ay",
            "az",
            "depth_method",
            "stereo_depth_source",
        ):
            self.assertNotIn(online_state_name, feature_names)

    def test_build_legacy_arrays_covers_training_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_synthetic_clip(Path(tmp))
            sequence = load_legacy_sequences(csv_path)[0]
            arrays = build_legacy_arrays(sequence)

            self.assertEqual(len(arrays["features"]), sequence.length)
            self.assertEqual(len(arrays["dt"]), sequence.length)
            self.assertEqual(len(arrays["features"][0]), len(legacy_feature_names()))
            self.assertEqual(len(arrays["measurements"][0]), len(METHOD_COLUMNS))
            self.assertEqual(len(arrays["valid"][0]), len(METHOD_COLUMNS))
            self.assertAlmostEqual(arrays["dt"][1][0], 0.01, places=6)

            method_keys = {key for _, key in METHOD_COLUMNS}
            self.assertNotIn("z_fallback", method_keys)
            fallback_feature_idx = legacy_feature_names().index("z_fallback")
            self.assertGreater(arrays["features"][2][fallback_feature_idx], 0.1)

            feature_index = {name: idx for idx, name in enumerate(legacy_feature_names())}
            self.assertEqual(
                arrays["features"][0][feature_index["roi_cuda_stereo_bm_support"]],
                7.0,
            )
            self.assertEqual(
                arrays["features"][0][feature_index["roi_ring_edge_profile_support"]],
                9.0,
            )

    def test_audit_training_inputs_reports_exact_model_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_synthetic_clip(root)
            report = audit_training_inputs([csv_path])

            methods = {
                (row["split"], row["method"]): row
                for row in report["method_coverage"]
            }
            feature_rows = {
                (row["split"], row["feature"]): row
                for row in report["feature_coverage"]
            }

            self.assertEqual(report["clip_count"], 1)
            self.assertEqual(report["sequence_count"], 1)
            self.assertEqual(report["frame_count"], 4)
            self.assertEqual(methods[("eval", "bbox_center")]["valid"], 4)
            self.assertEqual(methods[("eval", "circle_center")]["valid"], 2)
            self.assertEqual(feature_rows[("eval", "candidate_valid_count")]["nonzero"], 4)
            self.assertFalse(any(str(warning).startswith("legacy_") for warning in report["warnings"]))

            json_path = root / "training_input_audit.json"
            method_csv = root / "training_method_coverage.csv"
            feature_csv = root / "training_feature_coverage.csv"
            write_training_audit_json(json_path, report)
            write_training_method_csv(method_csv, report)
            write_training_feature_csv(feature_csv, report)
            self.assertTrue(json_path.exists())
            self.assertIn("bbox_center", method_csv.read_text(encoding="utf-8"))
            self.assertIn("candidate_valid_count", feature_csv.read_text(encoding="utf-8"))

    def test_p2_sidecar_merges_ncc_xfeat_and_superpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_synthetic_clip(root)
            sidecar_path = root / "traj_p0p1_001.p2_diagnostic.csv"
            fieldnames = [
                "frame_id",
                "mode",
                "valid",
                "z_m",
                "disparity",
                "support",
                "stddev",
                "confidence",
            ]
            rows = [
                {
                    "frame_id": "1",
                    "mode": "cuda_template",
                    "valid": "1",
                    "z_m": "3.50",
                    "disparity": "420.0",
                    "support": "1",
                    "stddev": "0.2",
                    "confidence": "0.8",
                },
                {
                    "frame_id": "1",
                    "mode": "neural_xfeat",
                    "valid": "1",
                    "z_m": "3.46",
                    "disparity": "425.0",
                    "support": "5",
                    "stddev": "0.9",
                    "confidence": "0.6",
                },
                {
                    "frame_id": "1",
                    "mode": "neural_superpoint",
                    "valid": "1",
                    "z_m": "3.41",
                    "disparity": "431.0",
                    "support": "21",
                    "stddev": "0.5",
                    "confidence": "0.9",
                },
                {
                    "frame_id": "2",
                    "mode": "cuda_stereo_bm",
                    "valid": "1",
                    "z_m": "3.25",
                    "disparity": "452.0",
                    "support": "8",
                    "stddev": "0.7",
                    "confidence": "0.5",
                },
                {
                    "frame_id": "2",
                    "mode": "cuda_ring_edge_profile",
                    "valid": "1",
                    "z_m": "3.35",
                    "disparity": "439.0",
                    "support": "11",
                    "stddev": "0.6",
                    "confidence": "0.7",
                },
            ]
            with sidecar_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            sequence = load_legacy_sequences(csv_path)[0]
            first = sequence.rows[0]
            self.assertEqual(first["z_roi_cuda_template_match"], 3.50)
            self.assertEqual(first["roi_cuda_template_match_support"], 1.0)
            self.assertEqual(first["z_roi_neural_xfeat"], 3.46)
            self.assertEqual(first["roi_neural_xfeat_support"], 5.0)
            self.assertEqual(first["z_roi_neural_superpoint"], 3.41)
            self.assertEqual(first["roi_neural_superpoint_support"], 21.0)
            second = sequence.rows[1]
            self.assertEqual(second["z_roi_cuda_stereo_bm"], 3.25)
            self.assertEqual(second["roi_cuda_stereo_bm_support"], 8.0)
            self.assertEqual(second["z_roi_ring_edge_profile"], 3.35)
            self.assertEqual(second["roi_ring_edge_profile_support"], 11.0)

    def test_correlated_depth_measurements_are_grouped_by_family(self) -> None:
        grouped = group_correlated_z_measurements(
            [
                (3.00, 0.01, "bbox_center"),
                (3.02, 0.0025, "bbox_left_edge"),
                (3.20, 0.01, "circle_center"),
                (3.30, 0.01, "roi_edge_centroid"),
                (3.34, 0.0025, "roi_radial_center"),
            ]
        )
        by_group = {name: (z_value, variance) for z_value, variance, name in grouped}
        self.assertIn("bbox", by_group)
        self.assertIn("circle", by_group)
        self.assertIn("roi_geometry", by_group)
        self.assertAlmostEqual(by_group["bbox"][0], 3.01, places=6)
        self.assertGreaterEqual(by_group["bbox"][1], 0.0001)
        self.assertAlmostEqual(by_group["roi_geometry"][0], 3.32, places=6)

    def test_check_dataset_and_evaluate_run_on_known_z_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_synthetic_clip(Path(tmp))
            report = analyze_dataset(csv_path)
            self.assertEqual(report["metadata"], str(Path(tmp) / "traj_p0p1_001.metadata.yaml"))
            self.assertEqual(report["rows"], 4)
            self.assertEqual(report["timing_source"], "frame_summary")
            self.assertEqual(report["frame_gaps"]["count"], 0)
            self.assertEqual(report["missing_fields"], [])
            self.assertEqual(report["watermarks"]["frame_counter_delta"]["nonzero"], 0)
            self.assertAlmostEqual(report["depth"]["z_circle_center"]["known_z_bias"], 0.005, places=6)
            self.assertEqual(report["source_breakdown"]["match_source"]["direct_pair"], 2)
            self.assertEqual(report["source_breakdown"]["match_source"]["fallback_l2r"], 1)
            self.assertEqual(report["source_breakdown"]["match_source"]["fallback_r2l"], 1)
            self.assertEqual(report["source_breakdown"]["epipolar_fallback"]["valid"], 2)
            self.assertTrue(report["frame_summary"]["present"])
            self.assertEqual(report["frame_summary"]["totals"]["direct_pair_count"], 2)
            self.assertEqual(report["frame_summary"]["totals"]["fallback_l2r_count"], 1)
            self.assertEqual(report["frame_summary"]["totals"]["fallback_r2l_count"], 1)
            self.assertEqual(report["frame_summary"]["totals"]["p2_candidate_observed_count"], 4)
            self.assertEqual(report["frame_summary"]["totals"]["p2_candidate_valid_count"], 2)
            self.assertEqual(report["frame_summary"]["max_per_frame"]["p2_feature_valid_count"], 1)
            self.assertEqual(report["depth_jump"]["z_bbox_center"]["pairs"], 3)
            self.assertAlmostEqual(
                report["depth_jump"]["z_bbox_center"]["max_abs_delta"],
                0.01,
                places=6,
            )

            old_argv = sys.argv[:]
            stdout = io.StringIO()
            json_out = Path(tmp) / "report.json"
            csv_out = Path(tmp) / "report.csv"
            try:
                sys.argv = [
                    "evaluate_fusion.py",
                    str(csv_path),
                    "--json-out",
                    str(json_out),
                    "--csv-out",
                    str(csv_out),
                ]
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(evaluate_fusion.main(), 0)
            finally:
                sys.argv = old_argv
            self.assertIn("known_z=3.0000m", stdout.getvalue())
            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertIn("0", report["tracks"])
            self.assertIn("z_circle_center", report["tracks"]["0"]["candidate_depths"])
            self.assertAlmostEqual(report["tracks"]["0"]["raw"]["speed_rms_mps"], 1.0, places=6)
            self.assertAlmostEqual(report["tracks"]["0"]["raw"]["ballistic_residual_rms_mps2"], 0.0, places=6)
            csv_text = csv_out.read_text(encoding="utf-8")
            self.assertIn("candidate_depths.z_circle_center.known_z_bias", csv_text)
            self.assertIn("raw.ballistic_residual_rms_mps2", csv_text)

    def test_known_z_loss_if_torch_available(self) -> None:
        try:
            import torch
            from trajectory_fusion.losses import bias_regularizer, known_z_loss, leave_one_method_loss
        except ImportError:
            self.skipTest("PyTorch is not installed")

        depth = torch.tensor([[[3.00], [3.02]]])
        known_z = torch.tensor([[3.0, 3.0]])
        valid = torch.tensor([[1.0, 1.0]])
        loss = known_z_loss(depth, known_z, valid)
        self.assertTrue(torch.isfinite(loss).item())

        bias = torch.tensor([[[[0.00], [0.05]], [[0.10], [0.20]]]])
        method_valid = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
        bias_loss = bias_regularizer(bias, method_valid)
        self.assertTrue(torch.isfinite(bias_loss).item())
        self.assertGreater(float(bias_loss), 0.0)

        measurements = torch.tensor([[[3.0, 3.1], [3.0, 3.2]]])
        method_valid = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
        predicted_depth = torch.tensor([[[3.02], [3.03]]])
        log_sigma = torch.full((1, 2, 2, 1), -2.3)
        method_bias = torch.zeros((1, 2, 2, 1))
        leave_loss = leave_one_method_loss(
            measurements,
            method_valid,
            predicted_depth,
            log_sigma,
            method_bias,
            method_index=1,
        )
        self.assertTrue(torch.isfinite(leave_loss).item())

    def test_reliability_smoother_reports_method_quality_stats_if_torch_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        method_names = tuple(name for name, _ in METHOD_COLUMNS)
        method_count = len(method_names)
        bbox_index = method_names.index("bbox_center")
        circle_index = method_names.index("circle_center")
        measurements = [[0.0] * method_count for _ in range(2)]
        valid = [[0.0] * method_count for _ in range(2)]
        for frame_index, bbox_z in enumerate((3.20, 3.22)):
            measurements[frame_index][bbox_index] = bbox_z
            measurements[frame_index][circle_index] = bbox_z + 0.20
            valid[frame_index][bbox_index] = 1.0
            valid[frame_index][circle_index] = 1.0

        log_sigma = torch.full((1, 2, method_count, 1), -1.0)
        bias = torch.zeros((1, 2, method_count, 1))
        outlier_logit = torch.full((1, 2, method_count, 1), -3.0)
        log_sigma[:, :, bbox_index, :] = torch.log(torch.tensor(0.05))
        log_sigma[:, :, circle_index, :] = torch.log(torch.tensor(0.20))
        bias[:, :, bbox_index, :] = 0.10
        output = ReliabilityOutput(
            log_sigma=log_sigma,
            bias=bias,
            outlier_logit=outlier_logit,
            common_log_sigma=torch.zeros((1, 2, 1)),
        )

        observations, diagnostics, summary = _build_learned_observations(
            measurements,
            valid,
            output,
            method_names,
            LearnedObservationConfig(min_sigma=0.01, max_sigma=1.0),
        )

        self.assertEqual(len(observations), 2)
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(summary["bbox_center"]["valid"], 2.0)
        self.assertEqual(summary["bbox_center"]["top_count"], 2.0)
        self.assertEqual(summary["bbox_center"]["top_rate"], 1.0)
        self.assertAlmostEqual(summary["bbox_center"]["mean_sigma"], 0.05, places=6)
        self.assertAlmostEqual(summary["bbox_center"]["mean_bias"], 0.10, places=6)
        self.assertAlmostEqual(summary["bbox_center"]["mean_corrected_minus_raw_z"], -0.10, places=6)
        self.assertGreater(summary["circle_center"]["mean_sigma"], summary["bbox_center"]["mean_sigma"])

    def test_physics_depth_loss_uses_sequence_dt_if_torch_available(self) -> None:
        try:
            import torch
            from trajectory_fusion.losses import physics_depth_loss
        except ImportError:
            self.skipTest("PyTorch is not installed")

        timestamps = torch.tensor([[0.00, 0.01, 0.03, 0.06, 0.10]], dtype=torch.float32)
        dt = torch.tensor([[0.01, 0.01, 0.02, 0.03, 0.04]], dtype=torch.float32).unsqueeze(-1)
        depth = (3.0 + 2.0 * timestamps).unsqueeze(-1)
        loss = physics_depth_loss(depth, dt)
        self.assertLess(float(loss), 1e-8)

    def test_dataset_manifest_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_clip(root)
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: traj_p0p1_001.csv",
                        "    metadata: traj_p0p1_001.metadata.yaml",
                        "    split: train",
                        "    name: static_3m",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            clips = load_manifest(manifest_path)
            self.assertTrue(is_manifest_path(manifest_path))
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0].csv, root / "traj_p0p1_001.csv")
            self.assertEqual(clips[0].metadata, root / "traj_p0p1_001.metadata.yaml")
            self.assertEqual(clips[0].split, "train")
            self.assertEqual(clips[0].name, "static_3m")

    def test_build_dataset_manifest_discovers_main_csvs_and_splits_known_z(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_synthetic_clip(root)
            second = root / "traj_p0p1_002.csv"
            second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "traj_p0p1_002.metadata.yaml").write_text(
                "known_z: 3.1\nstatic: true\n",
                encoding="utf-8",
            )
            (root / "traj_p0p1_002.frames.csv").write_text("frame_id\n1\n", encoding="utf-8")
            (root / "traj_p0p1_002.p2_diagnostic.csv").write_text("frame_id,mode\n1,x\n", encoding="utf-8")
            (root / "suite_metrics.csv").write_text("clip,variant\nx,raw\n", encoding="utf-8")

            discovered = discover_csvs([root])
            self.assertEqual(discovered, [first.resolve(), second.resolve()])

            manifest_path = root / "dataset_manifest.yaml"
            entries = build_manifest(
                [root],
                output_path=manifest_path,
                val_ratio=0.5,
                seed=3,
            )
            write_manifest(manifest_path, entries)
            clips = load_manifest(manifest_path)

            self.assertEqual(len(entries), 2)
            self.assertEqual(len(clips), 2)
            self.assertEqual({clip.split for clip in clips}, {"train", "val"})
            self.assertEqual({clip.metadata for clip in clips}, {
                root / "traj_p0p1_001.metadata.yaml",
                root / "traj_p0p1_002.metadata.yaml",
            })
            text = manifest_path.read_text(encoding="utf-8")
            self.assertIn("csv: traj_p0p1_001.csv", text)
            self.assertNotIn(".frames.csv", text)
            self.assertNotIn(".p2_diagnostic.csv", text)

    def test_build_dataset_manifest_can_stratify_known_z_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_3m_a", 3.0, rows=4)
            _write_known_distance_clip(root, "static_3m_b", 3.0, rows=4)
            _write_known_distance_clip(root, "static_4m_a", 4.0, rows=4)
            _write_known_distance_clip(root, "static_4m_b", 4.0, rows=4)
            _write_known_distance_clip(root, "static_5m_single", 5.0, rows=4)

            entries = build_manifest(
                [root],
                output_path=root / "dataset_manifest.yaml",
                val_ratio=0.5,
                seed=11,
                stratify_known_z=True,
            )

            split_by_z: dict[float, set[str]] = {}
            for entry in entries:
                self.assertIsNotNone(entry.known_z)
                split_by_z.setdefault(round(float(entry.known_z), 1), set()).add(entry.split)

            self.assertEqual(split_by_z[3.0], {"train", "val"})
            self.assertEqual(split_by_z[4.0], {"train", "val"})
            self.assertEqual(split_by_z[5.0], {"train"})

    def test_build_dataset_manifest_can_mark_unlabeled_eval_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_synthetic_clip(root)
            (root / "traj_p0p1_001.metadata.yaml").unlink()
            manifest_path = root / "dataset_manifest.yaml"

            entries = build_manifest(
                [first],
                output_path=manifest_path,
                split_mode="auto",
                unlabeled_split="eval",
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].split, "eval")
            self.assertIsNone(entries[0].metadata)

    def test_train_reliability_resolves_manifest_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_synthetic_clip(root)
            second = root / "traj_p0p1_002.csv"
            second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "traj_p0p1_002.metadata.yaml").write_text(
                "known_z: 3.1\nstatic: true\n",
                encoding="utf-8",
            )
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: traj_p0p1_001.csv",
                        "    metadata: traj_p0p1_001.metadata.yaml",
                        "    split: train",
                        "  - csv: traj_p0p1_002.csv",
                        "    metadata: traj_p0p1_002.metadata.yaml",
                        "    split: val",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            clips = resolve_input_clips([str(manifest_path)], metadata=None)
            train_items, heldout_items = load_sequences_from_clips(clips, train_split="train")
            self.assertEqual(len(clips), 2)
            self.assertEqual(len(train_items), 1)
            self.assertEqual(len(heldout_items), 1)
            self.assertEqual(train_items[0][0].split, "train")
            self.assertEqual(heldout_items[0][0].split, "val")

            train_arrays = [
                {"clip": clip.name, "track_id": sequence.track_id, **build_legacy_arrays(sequence)}
                for clip, sequence in train_items
            ]
            summary = _training_label_summary(train_arrays)
            self.assertEqual(summary["sequence_count"], 1)
            self.assertGreater(summary["frame_count"], 0)
            self.assertEqual(summary["known_z_frames"], summary["frame_count"])
            self.assertEqual(summary["static_frames"], summary["frame_count"])

    def test_validate_dataset_manifest_summarizes_splits_and_known_z(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_synthetic_clip(root)
            second = root / "traj_p0p1_002.csv"
            second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "traj_p0p1_002.metadata.yaml").write_text(
                "known_z: 3.1\nstatic: true\n",
                encoding="utf-8",
            )
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: traj_p0p1_001.csv",
                        "    metadata: traj_p0p1_001.metadata.yaml",
                        "    split: train",
                        "  - csv: traj_p0p1_002.csv",
                        "    metadata: traj_p0p1_002.metadata.yaml",
                        "    split: val",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = analyze_manifest(
                manifest_path,
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
            )
            self.assertEqual(report["clip_count"], 2)
            self.assertEqual(report["split_counts"], {"train": 1, "val": 1})
            self.assertEqual(report["known_z_counts"], {"train": 1, "val": 1})
            self.assertEqual(report["known_z_bucket_counts"], {"3.000": {"train": 1}, "3.100": {"val": 1}})
            self.assertNotIn("missing_val_split", report["warnings"])
            self.assertNotIn("missing_known_z_val_split", report["warnings"])
            self.assertNotIn("known_z_bucket_missing_val_split", report["warnings"])

            train_only_manifest = root / "train_only_manifest.yaml"
            train_only_manifest.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: traj_p0p1_001.csv",
                        "    metadata: traj_p0p1_001.metadata.yaml",
                        "    split: train",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            train_only = analyze_manifest(train_only_manifest, min_rows=1, min_fps=0.0, min_p0_hit=0.0)
            self.assertIn("missing_val_split", train_only["warnings"])
            self.assertIn("missing_known_z_val_split", train_only["warnings"])

    def test_validate_dataset_manifest_can_require_known_z_bucket_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_3m_a", 3.0, rows=4)
            _write_known_distance_clip(root, "static_3m_b", 3.0, rows=4)
            _write_known_distance_clip(root, "static_5m_single", 5.0, rows=4)
            manifest_path = root / "dataset_manifest.yaml"
            entries = build_manifest(
                [root],
                output_path=manifest_path,
                val_ratio=0.5,
                stratify_known_z=True,
                seed=11,
            )
            write_manifest(manifest_path, entries)

            report = analyze_manifest(
                manifest_path,
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
                require_stratified_known_z=True,
            )

            self.assertEqual(report["known_z_bucket_counts"]["3.000"], {"train": 1, "val": 1})
            self.assertEqual(report["known_z_bucket_counts"]["5.000"], {"train": 1})
            self.assertEqual(report["warning_counts"]["known_z_bucket_missing_val_split"], 1)
            self.assertEqual(
                report["known_z_bucket_warnings"],
                [
                    {
                        "known_z_bucket": "5.000",
                        "missing_split": "val",
                        "counts": {"train": 1},
                    }
                ],
            )

    def test_candidate_consistency_uses_known_z_and_pairwise_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_synthetic_clip(Path(tmp))
            report = analyze_candidate_consistency(
                [str(csv_path)],
                reference="auto",
                min_pair_count=1,
            )

            aggregate = report["aggregate"]
            self.assertEqual(aggregate["frames"], 4)
            self.assertEqual(aggregate["reference_counts"], {"known_z": 4})
            bbox = aggregate["methods"]["bbox_center"]
            self.assertEqual(bbox["valid"], 4)
            self.assertAlmostEqual(bbox["residual"]["median"], 0.005, places=6)
            self.assertIn("circle_center", aggregate["methods"])
            pairs = {
                (item["left"], item["right"]): item
                for item in aggregate["pairwise"]
            }
            self.assertIn(("bbox_center", "circle_center"), pairs)
            self.assertEqual(pairs[("bbox_center", "circle_center")]["count"], 2)

    def test_candidate_consistency_groups_known_z_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_3m", 3.0, rows=8)
            _write_known_distance_clip(root, "static_4m", 4.0, rows=8)
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: static_3m.csv",
                        "    metadata: static_3m.metadata.yaml",
                        "    split: train",
                        "  - csv: static_4m.csv",
                        "    metadata: static_4m.metadata.yaml",
                        "    split: val",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = analyze_candidate_consistency([str(manifest_path)], reference="auto")
            buckets = {
                (item["split"], item["known_z_bucket"]): item
                for item in report["known_z_buckets"]
            }
            self.assertEqual(set(buckets), {("train", "3.000"), ("val", "4.000")})
            self.assertEqual(buckets[("train", "3.000")]["frames"], 8)
            self.assertIn("bbox_center", buckets[("val", "4.000")]["methods"])

            csv_path = root / "candidate_consistency.csv"
            from trajectory_fusion.analyze_candidate_consistency import write_method_csv  # noqa: E402

            write_method_csv(csv_path, report)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            bucket_rows = [row for row in rows if row["scope"] == "known_z_bucket"]
            self.assertTrue(any(row["known_z_bucket"] == "3.000" for row in bucket_rows))
            self.assertTrue(any(row["known_z_bucket"] == "4.000" for row in bucket_rows))

    def test_method_calibration_fits_and_applies_smoother(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_synthetic_clip(root)
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: traj_p0p1_001.csv",
                        "    metadata: traj_p0p1_001.metadata.yaml",
                        "    split: train",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            calibration = fit_method_calibration(
                load_manifest(manifest_path),
                cfg=CalibrationConfig(min_count=1, min_sigma=0.01),
            )
            self.assertIn("bbox_center", calibration["methods"])
            self.assertAlmostEqual(
                calibration["methods"]["bbox_center"]["bias_median"],
                0.005,
                places=6,
            )
            calibration_path = root / "method_calibration.json"
            write_calibration(calibration_path, calibration)
            output_csv = root / "calibrated_smooth.csv"
            report = apply_calibrated_smoother(
                csv_path,
                calibration_path,
                output_csv,
                metadata_path=root / "traj_p0p1_001.metadata.yaml",
            )
            self.assertEqual(report["rows"], 4)
            self.assertTrue(output_csv.exists())
            text = output_csv.read_text(encoding="utf-8")
            self.assertIn("calibrated_smoother_valid_count", text)
            self.assertGreater(len(report["calibrated_methods"]), 0)

            suite_dir = root / "calibrated_suite"
            suite_report = run_suite(
                [str(csv_path)],
                suite_dir,
                metadata=str(root / "traj_p0p1_001.metadata.yaml"),
                calibration=calibration_path,
                gravity_y=0.0,
            )
            self.assertIn("calibrated_smoother_eval_json", suite_report["clips"][0])
            self.assertIn("calibrated_rts_smoother_eval_json", suite_report["clips"][0])
            rows = summarize_suite(suite_dir, suite_dir / "suite_metrics.csv")
            self.assertIn("calibrated_smoother", {row["variant"] for row in rows})
            self.assertIn("calibrated_rts_smoother", {row["variant"] for row in rows})

    def test_run_evaluation_suite_writes_baseline_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_synthetic_clip(root)
            output_dir = root / "suite"

            report = run_suite([str(csv_path)], output_dir, gravity_y=0.0)

            self.assertEqual(len(report["clips"]), 1)
            self.assertFalse(report["config"]["use_static_known_z"])
            clip = report["clips"][0]
            self.assertEqual(clip["check_rows"], 4)
            self.assertEqual(clip["robust_rows"], 4)
            self.assertEqual(clip["robust_rts_rows"], 4)
            self.assertEqual(clip["depth_polyfit_rows"], 4)
            self.assertTrue((output_dir / "suite_summary.json").exists())
            self.assertTrue(Path(clip["check_dataset_json"]).exists())
            self.assertTrue(Path(clip["raw_eval_json"]).exists())
            self.assertTrue(Path(clip["candidate_consistency_json"]).exists())
            self.assertTrue(Path(clip["candidate_consistency_csv"]).exists())
            self.assertTrue(Path(clip["candidate_pairwise_csv"]).exists())
            self.assertTrue(Path(clip["robust_smooth_csv"]).exists())
            self.assertTrue(Path(clip["robust_smooth_eval_json"]).exists())
            self.assertTrue(Path(clip["robust_rts_smooth_csv"]).exists())
            self.assertTrue(Path(clip["robust_rts_smooth_eval_json"]).exists())
            self.assertTrue(Path(clip["depth_polyfit_smooth_csv"]).exists())
            self.assertTrue(Path(clip["depth_polyfit_smooth_eval_json"]).exists())
            candidate_report = json.loads(
                Path(clip["candidate_consistency_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(candidate_report["aggregate"]["reference_counts"], {"known_z": 4})

            summary_csv = output_dir / "suite_metrics.csv"
            rows = summarize_suite(output_dir, summary_csv)
            self.assertEqual(len(rows), 4)
            self.assertTrue(summary_csv.exists())
            self.assertEqual(
                {row["variant"] for row in rows},
                {"raw", "robust_smooth", "robust_rts_smooth", "depth_polyfit_smooth"},
            )
            self.assertEqual({row["split"] for row in rows}, {"eval"})
            smooth_row = next(row for row in rows if row["variant"] == "robust_smooth")
            rts_row = next(row for row in rows if row["variant"] == "robust_rts_smooth")
            polyfit_row = next(row for row in rows if row["variant"] == "depth_polyfit_smooth")
            self.assertIn("known_z_bias", smooth_row)
            self.assertIn("known_z_bias", rts_row)
            self.assertIn("known_z_bias", polyfit_row)
            self.assertIn("ballistic_residual_rms_mps2", smooth_row)

    def test_summarize_reliability_methods_reads_apply_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_dir = root / "suite" / "clip"
            clip_dir.mkdir(parents=True)
            apply_json = clip_dir / "reliability_smoother_apply.json"
            apply_json.write_text(
                json.dumps(
                    {
                        "sequences": [
                            {
                                "track_id": 0.0,
                                "method_summary": {
                                    "bbox_center": {
                                        "valid": 10.0,
                                        "top_count": 6.0,
                                        "top_rate": 0.6,
                                        "mean_weight": 100.0,
                                        "mean_sigma": 0.02,
                                        "mean_bias": 0.03,
                                        "mean_abs_bias": 0.03,
                                        "mean_inlier_prob": 0.9,
                                        "mean_raw_z": 3.50,
                                        "mean_corrected_z": 3.47,
                                        "mean_corrected_minus_raw_z": -0.03,
                                    }
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            suite_summary = root / "suite" / "suite_summary.json"
            suite_summary.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "name": "clip",
                                "split": "val",
                                "reliability_smoother_apply_json": str(apply_json),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_csv = root / "suite" / "suite_reliability_methods.csv"
            rows = summarize_reliability_methods(root / "suite", output_csv)
            self.assertEqual(len(rows), 1)
            self.assertTrue(output_csv.exists())
            row = rows[0]
            self.assertEqual(row["clip"], "clip")
            self.assertEqual(row["split"], "val")
            self.assertEqual(row["variant"], "reliability_smoother")
            self.assertEqual(row["method"], "bbox_center")
            self.assertEqual(row["valid"], 10.0)
            self.assertEqual(row["top_count"], 6.0)
            self.assertEqual(row["mean_corrected_minus_raw_z"], -0.03)

    def test_audit_reliability_methods_flags_low_coverage_top_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep_reliability_methods.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "variant",
                    "split",
                    "clip",
                    "track_id",
                    "method",
                    "rows",
                    "valid",
                    "top_count",
                    "mean_sigma",
                    "mean_abs_bias",
                    "mean_inlier_prob",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "net_a",
                        "variant": "reliability_smoother",
                        "split": "val",
                        "clip": "clip",
                        "track_id": "0",
                        "method": "bbox_center",
                        "rows": "100",
                        "valid": "100",
                        "top_count": "10",
                        "mean_sigma": "0.05",
                        "mean_abs_bias": "0.01",
                        "mean_inlier_prob": "0.9",
                    }
                )
                writer.writerow(
                    {
                        "config": "net_a",
                        "variant": "reliability_smoother",
                        "split": "val",
                        "clip": "clip",
                        "track_id": "0",
                        "method": "roi_neural_xfeat",
                        "rows": "100",
                        "valid": "5",
                        "top_count": "90",
                        "mean_sigma": "0.01",
                        "mean_abs_bias": "0.12",
                        "mean_inlier_prob": "0.2",
                    }
                )

            rows = audit_reliability_methods(path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["dominant_top_method"], "roi_neural_xfeat")
            self.assertGreater(row["low_coverage_top_share"], 0.8)
            self.assertIn("low_coverage_methods_receive_top_weight", row["warnings"])
            self.assertIn("low_coverage_methods_have_tiny_sigma", row["warnings"])
            self.assertIn("large_method_bias", row["warnings"])
            self.assertIn("low_inlier_method_receives_top_weight", row["warnings"])

    def test_select_reliability_model_combines_rank_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "sweep_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "split",
                    "variant",
                    "z_std",
                    "z_peak_to_peak",
                    "known_z_bias",
                    "known_z_mad",
                    "checkpoint",
                    "suite_dir",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "smooth_but_risky",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "z_std": "0.001",
                        "z_peak_to_peak": "0.004",
                        "known_z_bias": "0.01",
                        "known_z_mad": "0.001",
                        "checkpoint": "risky.pt",
                        "suite_dir": "risky_suite",
                    }
                )
                writer.writerow(
                    {
                        "config": "stable",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "z_std": "0.004",
                        "z_peak_to_peak": "0.02",
                        "known_z_bias": "0.02",
                        "known_z_mad": "0.002",
                        "checkpoint": "stable.pt",
                        "suite_dir": "stable_suite",
                    }
                )
            audit_path = root / "sweep_reliability_method_audit.csv"
            with audit_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "variant",
                    "split",
                    "warnings",
                    "dominant_top_method",
                    "dominant_top_share",
                    "low_coverage_top_share",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "smooth_but_risky",
                        "variant": "reliability_smoother",
                        "split": "val",
                        "warnings": "dominant_method_top_share",
                        "dominant_top_method": "roi_neural_xfeat",
                        "dominant_top_share": "0.99",
                        "low_coverage_top_share": "0.00",
                    }
                )
                writer.writerow(
                    {
                        "config": "stable",
                        "variant": "reliability_smoother",
                        "split": "val",
                        "warnings": "",
                        "dominant_top_method": "bbox_center",
                        "dominant_top_share": "0.40",
                        "low_coverage_top_share": "0.00",
                    }
                )

            selected = select_reliability_models(metrics_path, audit_csv=audit_path)
            self.assertEqual(selected[0]["config"], "stable")
            self.assertEqual(selected[0]["decision"], "recommended")
            risky = next(row for row in selected if row["config"] == "smooth_but_risky")
            self.assertEqual(risky["decision"], "reject")
            self.assertEqual(risky["decision_reason"], "dominant_method_top_share")

    def test_select_reliability_model_uses_split_audit_for_all_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_path = root / "sweep_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "split",
                    "variant",
                    "z_std",
                    "z_peak_to_peak",
                    "ballistic_residual_rms_mps2",
                    "accel_z_rms_mps2",
                    "checkpoint",
                    "suite_dir",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "train_only",
                        "split": "train",
                        "variant": "reliability_smoother",
                        "z_std": "0.02",
                        "z_peak_to_peak": "0.08",
                        "ballistic_residual_rms_mps2": "5.0",
                        "accel_z_rms_mps2": "2.0",
                        "checkpoint": "train_only.pt",
                        "suite_dir": "train_only_suite",
                    }
                )
            audit_path = root / "sweep_reliability_method_audit.csv"
            with audit_path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "variant",
                    "split",
                    "warnings",
                    "dominant_top_method",
                    "dominant_top_count",
                    "top_total",
                    "low_coverage_top_count",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "train_only",
                        "variant": "reliability_smoother",
                        "split": "train",
                        "warnings": "large_method_bias",
                        "dominant_top_method": "roi_multi_point",
                        "dominant_top_count": "42",
                        "top_total": "100",
                        "low_coverage_top_count": "5",
                    }
                )

            selected = select_reliability_models(metrics_path, audit_csv=audit_path)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["split"], "all")
            self.assertEqual(selected[0]["decision"], "reject")
            self.assertEqual(selected[0]["decision_reason"], "no_known_z;large_method_bias")
            self.assertEqual(selected[0]["audit_warnings"], "large_method_bias")
            self.assertEqual(selected[0]["dominant_top_method"], "roi_multi_point")

    def test_reliability_sweep_config_and_command_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "sweep.json"
            config_path.write_text(
                json.dumps(
                    {
                        "configs": [
                            {
                                "name": "quick test",
                                "epochs": 2,
                                "hidden": 16,
                                "bias_reg_weight": 0.5,
                                "leave_one_weight": 0.02,
                                "seed": 123,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            configs = load_sweep_configs(config_path)
            self.assertEqual(configs[0]["name"], "quick_test")
            command = build_train_command(
                ["clip.csv"],
                root / "model.pt",
                configs[0],
                metadata="clip.metadata.yaml",
                train_split="train",
                device="cpu",
            )
            self.assertIn("--leave-one-weight", command)
            self.assertIn("0.02", command)
            self.assertIn("--bias-reg-weight", command)
            self.assertIn("--metadata", command)
            self.assertIn("--seed", command)
            self.assertIn("123", command)

    def test_repository_sweep_configs_load(self) -> None:
        config_dir = PROJECT / "trajectory_fusion" / "configs"
        expected = {
            "sweep_smoke.json": 1,
            "sweep_known_distance_selection.json": 5,
            "sweep_dynamic_regularization.json": 5,
        }
        for filename, count in expected.items():
            configs = load_sweep_configs(config_dir / filename)
            self.assertEqual(len(configs), count, filename)
            self.assertTrue(all(config["epochs"] > 0 for config in configs))
            self.assertTrue(all(config["hidden"] > 0 for config in configs))
            self.assertTrue(all(config["bias_reg_weight"] >= 1.0 for config in configs))
            self.assertTrue(all(config["seed"] > 0 for config in configs))

    def test_reliability_sweep_passes_calibration_to_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "sweep.json"
            calibration_path = root / "method_calibration.json"
            calibration_path.write_text("{}", encoding="utf-8")
            config_path.write_text(
                json.dumps({"configs": [{"name": "quick", "epochs": 1, "hidden": 8}]}),
                encoding="utf-8",
            )
            metric_row = {
                "clip": "clip",
                "split": "val",
                "variant": "reliability_smoother",
                "track_id": "0",
                "known_z": "3.0",
                "z_std": "0.01",
                "z_peak_to_peak": "0.02",
                "known_z_bias": "0.01",
                "known_z_mad": "0.002",
            }
            method_row = {
                "clip": "clip",
                "split": "val",
                "variant": "reliability_smoother",
                "track_id": "0",
                "method": "bbox_center",
                "valid": "10",
                "top_count": "4",
                "mean_sigma": "0.02",
            }

            with mock.patch.object(sweep_module.subprocess, "run") as run_mock, mock.patch.object(
                sweep_module,
                "run_suite",
                return_value={"clips": [{"name": "clip"}]},
            ) as suite_mock, mock.patch.object(
                sweep_module,
                "summarize_suite",
                return_value=[metric_row],
            ), mock.patch.object(
                sweep_module,
                "summarize_reliability_methods",
                return_value=[method_row],
            ), mock.patch.object(
                sweep_module,
                "select_reliability_models",
                return_value=[{"decision": "recommended"}],
            ) as select_mock:
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = sweep_module.run_sweep(
                        ["dataset_manifest.yaml"],
                        root / "sweep_out",
                        configs_path=config_path,
                        calibration=calibration_path,
                        gravity_y=0.0,
                        rank_split="val",
                    )

            run_mock.assert_called_once()
            suite_mock.assert_called_once()
            self.assertEqual(summary["calibration"], str(calibration_path))
            self.assertEqual(suite_mock.call_args.kwargs["calibration"], calibration_path)
            self.assertEqual(summary["runs"][0]["calibration"], str(calibration_path))
            self.assertEqual(summary["sweep_variant_ranking"], str(root / "sweep_out" / "sweep_variant_ranking.csv"))
            self.assertTrue((root / "sweep_out" / "sweep_variant_ranking.csv").exists())
            self.assertEqual(
                summary["sweep_reliability_methods"],
                str(root / "sweep_out" / "sweep_reliability_methods.csv"),
            )
            self.assertTrue((root / "sweep_out" / "sweep_reliability_methods.csv").exists())
            self.assertEqual(
                summary["sweep_reliability_method_audit"],
                str(root / "sweep_out" / "sweep_reliability_method_audit.csv"),
            )
            self.assertTrue((root / "sweep_out" / "sweep_reliability_method_audit.csv").exists())
            self.assertEqual(
                summary["sweep_model_selection"],
                str(root / "sweep_out" / "sweep_model_selection.csv"),
            )
            self.assertTrue((root / "sweep_out" / "sweep_model_selection.csv").exists())
            self.assertEqual(select_mock.call_args.kwargs["split"], "val")

    def test_dataset_workflow_builds_manifest_calibration_and_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _write_synthetic_clip(root)
            second = root / "traj_p0p1_002.csv"
            second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "traj_p0p1_002.metadata.yaml").write_text(
                "known_z: 3.1\nstatic: true\n",
                encoding="utf-8",
            )
            output_dir = root / "workflow"

            summary = run_workflow(
                [root],
                output_dir,
                val_ratio=0.5,
                seed=3,
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
                calibration_min_count=1,
                skip_sweep=True,
                include_depth_polyfit=False,
                include_rts_smoother=False,
            )

            self.assertTrue((output_dir / "dataset_manifest.yaml").exists())
            self.assertTrue((output_dir / "manifest_validation.json").exists())
            self.assertTrue((output_dir / "workflow_summary.json").exists())
            self.assertTrue((output_dir / "workflow_report.json").exists())
            self.assertTrue((output_dir / "workflow_report.md").exists())
            self.assertEqual(summary["manifest"]["clip_count"], 2)
            self.assertEqual(summary["validation"]["split_counts"], {"train": 1, "val": 1})
            self.assertTrue(Path(summary["training_input_audit"]["json"]).exists())
            self.assertTrue(Path(summary["training_input_audit"]["method_csv"]).exists())
            self.assertTrue(Path(summary["candidate_consistency"]["json"]).exists())
            self.assertTrue(Path(summary["candidate_consistency"]["method_csv"]).exists())
            self.assertGreater(summary["candidate_consistency"]["frames"], 0)
            self.assertGreater(summary["training_input_audit"]["frame_count"], 0)
            self.assertFalse(summary["config"]["use_static_known_z"])
            self.assertGreater(summary["calibration"]["method_count"], 0)
            self.assertTrue(summary["calibration"]["used_for_suite"])
            self.assertEqual(summary["sweep"]["reason"], "skip_sweep")
            self.assertTrue(Path(summary["baseline_suite"]["metrics_csv"]).exists())
            self.assertIn("raw", summary["baseline_suite"]["variants"])
            self.assertIn("robust_smooth", summary["baseline_suite"]["variants"])
            self.assertIn("calibrated_smoother", summary["baseline_suite"]["variants"])
            self.assertNotIn("robust_rts_smooth", summary["baseline_suite"]["variants"])
            report = build_workflow_report(output_dir)
            self.assertIn("sweep:skipped", report["warnings"])
            self.assertIn("calibrated_smoother", report["baseline_suite"]["variants"])
            self.assertEqual(report["readiness"]["status"], "ready_for_sweep")
            self.assertTrue(report["readiness"]["ready_for_sweep"])
            self.assertFalse(report["readiness"]["ready_for_model_selection"])
            self.assertTrue(report["candidate_consistency"]["top_aggregate"])
            self.assertTrue(any("ReliabilityNet sweep" in action for action in report["recommended_actions"]))

    def test_dataset_workflow_can_generate_stratified_known_z_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_3m_a", 3.0, rows=4)
            _write_known_distance_clip(root, "static_3m_b", 3.0, rows=4)
            _write_known_distance_clip(root, "static_4m_a", 4.0, rows=4)
            _write_known_distance_clip(root, "static_4m_b", 4.0, rows=4)
            output_dir = root / "workflow_stratified"

            summary = run_workflow(
                [root],
                output_dir,
                val_ratio=0.5,
                seed=11,
                stratify_known_z=True,
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
                skip_calibration=True,
                skip_sweep=True,
                include_depth_polyfit=False,
                include_rts_smoother=False,
                include_candidate_consistency=False,
            )

            clips = load_manifest(output_dir / "dataset_manifest.yaml")
            split_by_z: dict[float, set[str]] = {}
            for clip in clips:
                known_z = read_metadata(clip.metadata)["known_z"] if clip.metadata else None
                self.assertIsNotNone(known_z)
                split_by_z.setdefault(round(float(known_z), 1), set()).add(clip.split)

            self.assertTrue(summary["manifest"]["stratify_known_z"])
            self.assertTrue(summary["config"]["stratify_known_z"])
            self.assertEqual(summary["validation"]["known_z_counts"], {"train": 2, "val": 2})
            self.assertEqual(summary["validation"]["known_z_bucket_warnings"], [])
            self.assertEqual(summary["training_input_audit"]["clip_count"], 4)
            self.assertEqual(split_by_z[3.0], {"train", "val"})
            self.assertEqual(split_by_z[4.0], {"train", "val"})

    def test_build_manifest_can_hold_out_known_z_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_3m_a", 3.0, rows=4)
            _write_known_distance_clip(root, "static_3m_b", 3.0, rows=4)
            _write_known_distance_clip(root, "static_4m_a", 4.0, rows=4)
            _write_known_distance_clip(root, "static_4m_b", 4.0, rows=4)

            entries = build_manifest(
                [root],
                output_path=root / "dataset_manifest.yaml",
                holdout_known_z="4.0",
                holdout_split="val",
            )
            splits_by_z: dict[float, set[str]] = {}
            for entry in entries:
                self.assertIsNotNone(entry.known_z)
                splits_by_z.setdefault(round(float(entry.known_z or 0.0), 1), set()).add(entry.split)

            self.assertEqual(splits_by_z[3.0], {"train"})
            self.assertEqual(splits_by_z[4.0], {"val"})

            summary = run_workflow(
                [root],
                root / "workflow_holdout",
                holdout_known_z="4.0",
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
                calibration_min_count=1,
                skip_sweep=True,
                include_depth_polyfit=False,
                include_rts_smoother=False,
                include_candidate_consistency=False,
            )
            self.assertEqual(summary["manifest"]["holdout_known_z"], "4.0")
            self.assertEqual(summary["manifest"]["holdout_split"], "val")
            self.assertEqual(summary["manifest"]["split_counts"], {"train": 2, "val": 2})
            self.assertEqual(summary["validation"]["known_z_counts"], {"train": 2, "val": 2})
            self.assertEqual(summary["config"]["holdout_known_z"], "4.0")

    def test_build_manifest_reads_known_z_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_known_z_m", 5.0, rows=4)
            (root / "static_known_z_m.metadata.yaml").write_text(
                "known_z_m: 5.0\nstatic: true\n",
                encoding="utf-8",
            )
            _write_known_distance_clip(root, "static_known_distance_m", 6.0, rows=4)
            (root / "static_known_distance_m.metadata.yaml").write_text(
                "known_distance_m: 6.0\nstatic: true\n",
                encoding="utf-8",
            )

            entries = build_manifest(
                [root],
                output_path=root / "dataset_manifest.yaml",
                split_mode="auto",
                val_ratio=0.5,
                seed=3,
            )
            known_by_name = {entry.name: entry.known_z for entry in entries}

            self.assertEqual(known_by_name["static_known_z_m"], 5.0)
            self.assertEqual(known_by_name["static_known_distance_m"], 6.0)

    def test_dataset_workflow_known_distance_report_is_ready_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_known_distance_clip(root, "static_train_3m", 3.0)
            _write_known_distance_clip(root, "static_val_4m", 4.0)
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: static_train_3m.csv",
                        "    metadata: static_train_3m.metadata.yaml",
                        "    split: train",
                        "    name: static_train_3m",
                        "  - csv: static_val_4m.csv",
                        "    metadata: static_val_4m.metadata.yaml",
                        "    split: val",
                        "    name: static_val_4m",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "workflow_known_z"

            summary = run_workflow(
                [manifest_path],
                output_dir,
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
                calibration_min_count=4,
                skip_sweep=True,
                include_depth_polyfit=False,
                include_rts_smoother=False,
            )
            report = build_workflow_report(output_dir)

            self.assertEqual(summary["validation"]["known_z_counts"], {"train": 1, "val": 1})
            self.assertNotIn("missing_known_z_clips", summary["validation"]["warning_counts"])
            self.assertNotIn("missing_val_split", summary["validation"]["warning_counts"])
            self.assertTrue(summary["calibration"]["used_for_suite"])
            self.assertGreater(summary["calibration"]["method_count"], 4)
            self.assertIn("calibrated_smoother", summary["baseline_suite"]["variants"])
            self.assertNotIn("validation:missing_known_z_clips", report["warnings"])
            self.assertNotIn("validation:missing_val_split", report["warnings"])
            self.assertNotIn("calibration:not_used", report["warnings"])
            self.assertIn("sweep:skipped", report["warnings"])
            self.assertEqual(report["readiness"]["status"], "ready_for_sweep")
            self.assertTrue(report["readiness"]["ready_for_sweep"])
            self.assertFalse(report["readiness"]["ready_for_model_selection"])
            self.assertTrue(report["candidate_consistency"]["top_aggregate"])
            self.assertTrue(report["candidate_consistency"]["top_known_z_buckets"])
            markdown = (output_dir / "workflow_report.md").read_text(encoding="utf-8")
            self.assertIn("readiness: `ready_for_sweep`", markdown)
            self.assertIn("## Candidate Consistency", markdown)
            self.assertIn("## Known-Z Bucket Candidates", markdown)
            self.assertIn("static_val_4m", markdown)
            self.assertIn("calibrated_smoother", markdown)

    def test_compare_workflows_sorts_ready_model_selection_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_workflow(
                name: str,
                *,
                known: bool,
                decision: str,
                reason: str,
                config: str,
            ) -> Path:
                workflow = root / name
                workflow.mkdir()
                selection_path = workflow / "sweep_model_selection.csv"
                with selection_path.open("w", newline="", encoding="utf-8") as handle:
                    fieldnames = [
                        "selection_rank",
                        "decision",
                        "decision_reason",
                        "metric_rank",
                        "config",
                        "variant",
                        "split",
                        "score",
                        "known_clip_count",
                        "mean_abs_known_z_bias",
                        "mean_known_z_mad",
                        "mean_z_std",
                        "audit_warnings",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "selection_rank": "1",
                            "decision": decision,
                            "decision_reason": reason,
                            "metric_rank": "1",
                            "config": config,
                            "variant": "reliability_smoother",
                            "split": "val" if known else "all",
                            "score": "0.01" if known else "1.0",
                            "known_clip_count": "2" if known else "0",
                            "mean_abs_known_z_bias": "0.01" if known else "",
                            "mean_known_z_mad": "0.002" if known else "",
                            "mean_z_std": "0.003" if known else "0.04",
                            "audit_warnings": "",
                        }
                    )
                variant_path = workflow / "sweep_variant_ranking.csv"
                with variant_path.open("w", newline="", encoding="utf-8") as handle:
                    fieldnames = [
                        "rank",
                        "config",
                        "variant",
                        "split",
                        "score",
                        "known_clip_count",
                        "mean_abs_known_z_bias",
                        "mean_z_std",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "rank": "1",
                            "config": "baseline",
                            "variant": "calibrated_smoother" if known else "robust_smooth",
                            "split": "val" if known else "all",
                            "score": "0.02",
                            "known_clip_count": "2" if known else "0",
                            "mean_abs_known_z_bias": "0.012" if known else "",
                            "mean_z_std": "0.004",
                        }
                    )
                candidate_path = workflow / "candidate_consistency.csv"
                with candidate_path.open("w", newline="", encoding="utf-8") as handle:
                    fieldnames = [
                        "scope",
                        "clip",
                        "split",
                        "known_z_bucket",
                        "track_id",
                        "method",
                        "key",
                        "valid",
                        "total",
                        "hit_rate",
                        "z_median",
                        "z_mad",
                        "residual_median",
                        "residual_mad",
                        "residual_abs_p95",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "scope": "aggregate",
                            "method": "circle_center",
                            "key": "z_circle_center",
                            "valid": "10",
                            "total": "10",
                            "hit_rate": "1.0",
                            "z_median": "3.0",
                            "z_mad": "0.001",
                            "residual_median": "0.002",
                            "residual_mad": "0.001",
                            "residual_abs_p95": "0.005",
                        }
                    )
                summary = {
                    "output_dir": str(workflow),
                    "validation": {
                        "split_counts": {"train": 1, "val": 1} if known else {"train": 1},
                        "known_z_counts": {"train": 1, "val": 1} if known else {},
                        "known_z_bucket_counts": {"3.000": {"train": 1, "val": 1}} if known else {},
                        "warning_counts": {} if known else {"missing_known_z_clips": 1, "missing_val_split": 1},
                    },
                    "training_input_audit": {
                        "frame_count": 10,
                        "method_count": 3,
                        "feature_count": 5,
                        "warnings": [],
                    },
                    "candidate_consistency": {
                        "frames": 10,
                        "method_count": 1,
                        "known_z_bucket_count": 1 if known else 0,
                        "method_csv": str(candidate_path),
                    },
                    "calibration": {
                        "used_for_suite": known,
                        "method_count": 3 if known else 0,
                    },
                    "baseline_suite": {
                        "variants": ["raw", "calibrated_smoother"] if known else ["raw", "robust_smooth"],
                    },
                    "sweep": {
                        "skipped": False,
                        "sweep_model_selection": str(selection_path),
                        "sweep_variant_ranking": str(variant_path),
                    },
                }
                (workflow / "workflow_summary.json").write_text(
                    json.dumps(summary, indent=2),
                    encoding="utf-8",
                )
                return workflow

            ready = write_workflow(
                "ready_workflow",
                known=True,
                decision="recommended",
                reason="",
                config="net_ok",
            )
            smoke = write_workflow(
                "smoke_workflow",
                known=False,
                decision="reject",
                reason="no_known_z",
                config="net_smoke",
            )

            rows = compare_workflows([smoke, ready])
            self.assertEqual(rows[0]["workflow"], "ready_workflow")
            self.assertEqual(rows[0]["readiness"], "ready_for_model_selection")
            self.assertEqual(rows[0]["top_config"], "net_ok")
            self.assertEqual(rows[1]["workflow"], "smoke_workflow")
            self.assertIn("selection:no_known_z", rows[1]["warnings"])

            csv_path = root / "workflow_compare.csv"
            write_workflow_compare_csv(csv_path, rows)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[0]["workflow"], "ready_workflow")

    def test_workflow_matrix_runs_stratified_holdout_and_dynamic_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            _write_known_distance_clip(data_dir, "static_3m_a", 3.0, rows=8)
            _write_known_distance_clip(data_dir, "static_3m_b", 3.0, rows=8)
            _write_known_distance_clip(data_dir, "static_4m_a", 4.0, rows=8)
            _write_known_distance_clip(data_dir, "static_4m_b", 4.0, rows=8)
            output_dir = root / "matrix"

            summary = run_workflow_matrix(
                [data_dir],
                output_dir,
                include_dynamic=True,
                skip_sweep=True,
                include_depth_polyfit=False,
                include_rts_smoother=False,
                calibration_min_count=1,
                min_rows=1,
                min_fps=0.0,
                min_p0_hit=0.0,
            )

            names = {item["name"] for item in summary["workflows"]}
            self.assertEqual(
                names,
                {"known_stratified", "holdout_3m000", "holdout_4m000", "dynamic_regularization"},
            )
            self.assertEqual(summary["known_z_buckets"], ["3.000", "4.000"])
            self.assertEqual(summary["workflow_count"], 4)
            self.assertTrue((output_dir / "workflow_compare.csv").exists())
            self.assertTrue((output_dir / "workflow_compare.json").exists())
            self.assertTrue((output_dir / "workflow_matrix_summary.json").exists())
            rows = compare_workflows([Path(item["dir"]) for item in summary["workflows"]])
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["readiness"] == "ready_for_sweep" for row in rows))

    def test_dataset_workflow_passes_sweep_options_without_known_z_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_clip(root)
            output_dir = root / "workflow"
            sweep_stub = {
                "sweep_ranking": str(output_dir / "reliability_sweep" / "sweep_ranking.csv"),
                "sweep_variant_ranking": str(output_dir / "reliability_sweep" / "sweep_variant_ranking.csv"),
                "sweep_reliability_methods": str(output_dir / "reliability_sweep" / "sweep_reliability_methods.csv"),
                "sweep_reliability_method_audit": str(
                    output_dir / "reliability_sweep" / "sweep_reliability_method_audit.csv"
                ),
                "sweep_model_selection": str(output_dir / "reliability_sweep" / "sweep_model_selection.csv"),
                "runs": [{"name": "quick"}],
            }

            with mock.patch.object(workflow_module, "run_sweep", return_value=sweep_stub) as sweep_mock:
                summary = run_workflow(
                    [root],
                    output_dir,
                    min_rows=1,
                    min_fps=0.0,
                    min_p0_hit=0.0,
                    calibration_min_count=1,
                    skip_sweep=False,
                    include_depth_polyfit=False,
                    include_rts_smoother=False,
                    include_candidate_consistency=False,
                    rank_split="val",
                )

            sweep_mock.assert_called_once()
            kwargs = sweep_mock.call_args.kwargs
            self.assertFalse(kwargs["use_static_known_z"])
            self.assertFalse(kwargs["include_depth_polyfit"])
            self.assertFalse(kwargs["include_rts_smoother"])
            self.assertFalse(kwargs["include_candidate_consistency"])
            self.assertEqual(kwargs["rank_split"], "val")
            self.assertIsNotNone(kwargs["calibration"])
            self.assertFalse(summary["sweep"]["skipped"])
            self.assertEqual(summary["sweep"]["run_count"], 1)

    def test_dataset_workflow_rejects_manifest_input_with_manifest_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_clip(root)
            manifest_path = root / "dataset_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "clips:",
                        "  - csv: traj_p0p1_001.csv",
                        "    metadata: traj_p0p1_001.metadata.yaml",
                        "    split: train",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                run_workflow(
                    [manifest_path],
                    root / "workflow",
                    manifest=root / "other_manifest.yaml",
                    skip_sweep=True,
                )

    def test_rank_sweep_metrics_prefers_known_z_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep_metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "split",
                    "variant",
                    "z_std",
                    "z_peak_to_peak",
                    "known_z_bias",
                    "known_z_mad",
                    "checkpoint",
                    "suite_dir",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "smooth_but_wrong",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "z_std": "0.001",
                        "z_peak_to_peak": "0.004",
                        "known_z_bias": "0.25",
                        "known_z_mad": "0.01",
                        "checkpoint": "wrong.pt",
                        "suite_dir": "wrong_suite",
                    }
                )
                writer.writerow(
                    {
                        "config": "accurate",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "z_std": "0.004",
                        "z_peak_to_peak": "0.02",
                        "known_z_bias": "0.01",
                        "known_z_mad": "0.01",
                        "checkpoint": "accurate.pt",
                        "suite_dir": "accurate_suite",
                    }
                )

                writer.writerow(
                    {
                        "config": "smooth_but_wrong",
                        "split": "train",
                        "variant": "reliability_smoother",
                        "z_std": "0.001",
                        "z_peak_to_peak": "0.004",
                        "known_z_bias": "0.001",
                        "known_z_mad": "0.001",
                        "checkpoint": "wrong.pt",
                        "suite_dir": "wrong_suite",
                    }
                )

            ranked = rank_metrics(path)
            self.assertEqual(ranked[0]["config"], "accurate")
            self.assertEqual(ranked[0]["split"], "val")
            self.assertEqual(ranked[0]["checkpoint"], "accurate.pt")
            self.assertEqual(ranked[0]["rank"], 1)

    def test_rank_sweep_metrics_can_compare_all_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep_metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "split",
                    "variant",
                    "clip",
                    "track_id",
                    "z_std",
                    "z_peak_to_peak",
                    "known_z_bias",
                    "known_z_mad",
                    "checkpoint",
                    "suite_dir",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for config in ("net_a", "net_b"):
                    writer.writerow(
                        {
                            "config": config,
                            "split": "val",
                            "variant": "calibrated_smoother",
                            "clip": "static_3m",
                            "track_id": "0",
                            "z_std": "0.003",
                            "z_peak_to_peak": "0.01",
                            "known_z_bias": "0.005",
                            "known_z_mad": "0.002",
                            "checkpoint": f"{config}.pt",
                            "suite_dir": f"{config}_suite",
                        }
                    )
                    writer.writerow(
                        {
                            "config": config,
                            "split": "val",
                            "variant": "calibrated_rts_smoother",
                            "clip": "static_3m",
                            "track_id": "0",
                            "z_std": "0.002",
                            "z_peak_to_peak": "0.006",
                            "known_z_bias": "0.004",
                            "known_z_mad": "0.001",
                            "checkpoint": f"{config}.pt",
                            "suite_dir": f"{config}_suite",
                        }
                    )
                writer.writerow(
                    {
                        "config": "net_a",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "clip": "static_3m",
                        "track_id": "0",
                        "z_std": "0.002",
                        "z_peak_to_peak": "0.008",
                        "known_z_bias": "0.02",
                        "known_z_mad": "0.003",
                        "checkpoint": "net_a.pt",
                        "suite_dir": "net_a_suite",
                    }
                )

            ranked = rank_metrics(path, variant="all", split="val")
            variants = {(row["config"], row["variant"]): row for row in ranked}
            self.assertIn(("baseline", "calibrated_smoother"), variants)
            self.assertIn(("baseline", "calibrated_rts_smoother"), variants)
            self.assertIn(("net_a", "reliability_smoother"), variants)
            self.assertEqual(variants[("baseline", "calibrated_smoother")]["clip_count"], 1)
            self.assertEqual(variants[("baseline", "calibrated_rts_smoother")]["clip_count"], 1)

    def test_rank_sweep_metrics_uses_motion_when_known_z_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep_metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                fieldnames = [
                    "config",
                    "split",
                    "variant",
                    "z_std",
                    "z_peak_to_peak",
                    "ballistic_residual_rms_mps2",
                    "accel_z_rms_mps2",
                    "known_z_bias",
                    "known_z_mad",
                    "checkpoint",
                    "suite_dir",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "config": "smooth_bad_motion",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "z_std": "0.001",
                        "z_peak_to_peak": "0.002",
                        "ballistic_residual_rms_mps2": "200.0",
                        "accel_z_rms_mps2": "150.0",
                        "known_z_bias": "",
                        "known_z_mad": "",
                        "checkpoint": "bad.pt",
                        "suite_dir": "bad_suite",
                    }
                )
                writer.writerow(
                    {
                        "config": "slightly_noisier_good_motion",
                        "split": "val",
                        "variant": "reliability_smoother",
                        "z_std": "0.003",
                        "z_peak_to_peak": "0.006",
                        "ballistic_residual_rms_mps2": "2.0",
                        "accel_z_rms_mps2": "1.0",
                        "known_z_bias": "",
                        "known_z_mad": "",
                        "checkpoint": "good.pt",
                        "suite_dir": "good_suite",
                    }
                )

            ranked = rank_metrics(path, split="val")
            self.assertEqual(ranked[0]["config"], "slightly_noisier_good_motion")
            self.assertEqual(ranked[0]["known_clip_count"], 0)


if __name__ == "__main__":
    unittest.main()
