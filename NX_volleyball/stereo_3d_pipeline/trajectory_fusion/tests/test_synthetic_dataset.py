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
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from trajectory_fusion import evaluate_fusion  # noqa: E402
from trajectory_fusion.check_dataset import analyze_dataset  # noqa: E402
from trajectory_fusion.dataset import (  # noqa: E402
    METHOD_COLUMNS,
    build_legacy_arrays,
    derive_frame_summary_path,
    find_metadata_for_csv,
    legacy_feature_names,
    load_legacy_sequences,
    read_metadata,
)
from trajectory_fusion.manifest import is_manifest_path, load_manifest  # noqa: E402
from trajectory_fusion.robust_smoother import group_correlated_z_measurements  # noqa: E402
from trajectory_fusion.run_evaluation_suite import run_suite  # noqa: E402
from trajectory_fusion.train_reliability import load_sequences_from_clips, resolve_input_clips  # noqa: E402


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
            csv_text = csv_out.read_text(encoding="utf-8")
            self.assertIn("candidate_depths.z_circle_center.known_z_bias", csv_text)

    def test_known_z_loss_if_torch_available(self) -> None:
        try:
            import torch
            from trajectory_fusion.losses import bias_regularizer, known_z_loss
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
            self.assertTrue((output_dir / "suite_summary.json").exists())
            self.assertTrue(Path(clip["check_dataset_json"]).exists())
            self.assertTrue(Path(clip["raw_eval_json"]).exists())
            self.assertTrue(Path(clip["robust_smooth_csv"]).exists())
            self.assertTrue(Path(clip["robust_smooth_eval_json"]).exists())


if __name__ == "__main__":
    unittest.main()
