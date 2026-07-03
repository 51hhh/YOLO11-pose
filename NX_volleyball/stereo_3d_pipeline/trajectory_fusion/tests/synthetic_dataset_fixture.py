"""Synthetic trajectory CSV fixtures used by trajectory fusion tests."""

from __future__ import annotations

import csv
from pathlib import Path


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


def write_synthetic_clip(root: Path) -> Path:
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
                "z_circle_left_edge": f"{z - 0.01:.3f}",
                "z_circle_right_edge": f"{z + 0.01:.3f}",
                "z_roi_edge_centroid": f"{z + 0.001:.3f}",
                "z_roi_radial_center": f"{z + 0.002:.3f}",
                "z_roi_edge_pair_center": f"{z + 0.003:.3f}",
                "z_roi_center_patch": f"{z + 0.05:.3f}",
                "z_roi_multi_point": f"{z - 0.05:.3f}",
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

    _write_frame_summary(root / "traj_p0p1_001.frames.csv")
    (root / "traj_p0p1_001.metadata.yaml").write_text(
        "known_z: 3.0\nstatic: true\nscene: synthetic\n",
        encoding="utf-8",
    )
    return csv_path


def _write_frame_summary(frames_path: Path) -> None:
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
    with frames_path.open("w", newline="", encoding="utf-8") as handle:
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
