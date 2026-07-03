"""Shared scalar overrides for NX algorithm sweep cases."""

from __future__ import annotations


ORB_FAST_SWEEP = {
    "subpixel_patch_radius": "3",
    "subpixel_search_radius_px": "4",
    "subpixel_max_points": "8",
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.08",
    "subpixel_max_disp_delta_px": "6.0",
    "subpixel_max_stddev_px": "2.5",
    "feature_y_tolerance_px": "3.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.70",
    "feature_mad_scale": "3.5",
    "feature_ransac_gate_px": "2.0",
}

ORB_WIDE_Y_SWEEP = {
    "subpixel_patch_radius": "5",
    "subpixel_search_radius_px": "8",
    "subpixel_max_points": "12",
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.08",
    "subpixel_max_disp_delta_px": "10.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "8.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.90",
    "feature_mad_scale": "4.0",
    "feature_ransac_gate_px": "3.0",
    "feature_sphere_radius_scale": "2.8",
}

COLOR_OFFLINE_TUNED = {
    "subpixel_patch_radius": "9",
    "subpixel_search_radius_px": "12",
    "subpixel_min_points": "4",
    "subpixel_min_confidence": "0.54",
    "subpixel_max_disp_delta_px": "6.0",
    "subpixel_max_stddev_px": "1.5",
    "feature_y_tolerance_px": "1.0",
    "feature_reverse_check_px": "1.0",
    "feature_overlap_scale": "0.55",
    "feature_mad_scale": "2.5",
    "feature_ransac_gate_px": "0.75",
}

COLOR_WIDE_SEARCH = {
    "subpixel_patch_radius": "9",
    "subpixel_search_radius_px": "24",
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.25",
    "subpixel_max_disp_delta_px": "18.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "4.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.85",
    "feature_mad_scale": "4.0",
    "feature_ransac_gate_px": "3.0",
    "feature_sphere_radius_scale": "2.5",
}

RELAXED_FEATURE_GATES = {
    "subpixel_min_points": "3",
    "subpixel_min_confidence": "0.10",
    "subpixel_max_disp_delta_px": "8.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "6.0",
    "feature_reverse_check_px": "-1.0",
    "feature_overlap_scale": "0.90",
    "feature_mad_scale": "4.0",
    "feature_ransac_gate_px": "3.0",
}

TEMPLATE_PATCH9_SWEEP = {
    "subpixel_patch_radius": "9",
    "subpixel_search_radius_px": "16",
    "subpixel_min_confidence": "0.35",
    "subpixel_max_disp_delta_px": "10.0",
    "feature_y_tolerance_px": "2.0",
    "feature_overlap_scale": "0.70",
    "feature_sphere_radius_scale": "2.2",
}

DENSE_PATCH9_SWEEP = {
    "subpixel_patch_radius": "9",
    "subpixel_min_points": "4",
    "subpixel_min_confidence": "0.12",
    "subpixel_max_disp_delta_px": "18.0",
    "subpixel_max_stddev_px": "3.0",
    "feature_y_tolerance_px": "3.0",
    "feature_overlap_scale": "0.80",
    "feature_sphere_radius_scale": "2.5",
}
