#!/usr/bin/env python3
"""Synthetic coverage for trajectory fusion CSV tooling."""

from __future__ import annotations

import contextlib
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
    derive_frame_summary_path,
    find_metadata_for_csv,
    legacy_feature_names,
    read_metadata,
)
from trajectory_fusion.manifest import is_manifest_path, load_manifest  # noqa: E402
from trajectory_fusion.train_reliability import load_sequences_from_clips, resolve_input_clips  # noqa: E402
from trajectory_fusion.tests.synthetic_dataset_fixture import write_synthetic_clip  # noqa: E402


class SyntheticDatasetTest(unittest.TestCase):
    def test_metadata_autodiscovery_and_frame_summary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_synthetic_clip(Path(tmp))
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
        self.assertNotIn("z_stereo", feature_names)
        self.assertNotIn("z", feature_names)

    def test_check_dataset_and_evaluate_run_on_known_z_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_synthetic_clip(Path(tmp))
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
            from trajectory_fusion.losses import known_z_loss
        except ImportError:
            self.skipTest("PyTorch is not installed")

        depth = torch.tensor([[[3.00], [3.02]]])
        known_z = torch.tensor([[3.0, 3.0]])
        valid = torch.tensor([[1.0, 1.0]])
        loss = known_z_loss(depth, known_z, valid)
        self.assertTrue(torch.isfinite(loss).item())

    def test_dataset_manifest_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_clip(root)
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
            first = write_synthetic_clip(root)
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


if __name__ == "__main__":
    unittest.main()
