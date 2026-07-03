"""Shared algorithm matrix case types."""

from __future__ import annotations

from dataclasses import dataclass, field


MODE_KEYS = {
    "bbox_pair",
    "bbox_edges",
    "circle_center",
    "circle_edges",
    "roi_edge_centroid",
    "roi_radial_center",
    "roi_edge_pair_center",
    "roi_corner_points",
    "roi_texture_points",
    "roi_binary_points",
    "roi_orb_points",
    "roi_brisk_points",
    "roi_akaze_points",
    "roi_sift_points",
    "roi_iou_region_color_patch",
    "roi_patch_iou_color_edge",
    "roi_cuda_template_match",
    "roi_cuda_stereo_bm",
    "roi_cuda_stereo_sgm",
    "roi_center_patch",
    "roi_subpixel",
    "epipolar_fallback",
    "fallback_template",
    "fallback_feature_points",
}


@dataclass
class Case:
    name: str
    modes: dict[str, bool] = field(default_factory=dict)
    candidate_fields: tuple[str, ...] = ()
    support_field: str | None = None
    subpixel_enabled: bool | None = None
    yaml_scalars: dict[str, str] = field(default_factory=dict)
    neural_backend: str | None = None
    neural_engine: str | None = None
    roi_size: int = 224
    top_k: int = 128
    descriptor_dim: int = 64
    neural_min_matches: int = 8
    neural_max_y_error_px: float = 2.0
    neural_max_disp_delta_px: float = 32.0
    neural_final_disp_gate_px: float = 2.0
    neural_min_score: float = 0.0
    note: str = ""
