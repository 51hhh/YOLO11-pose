"""Sparse feature geometry validation for the realtime contract mirror."""

from __future__ import annotations

import math

from .realtime_contract_types import (
    Detection,
    FeatureValidationConfig,
    SparseFeatureObservation,
)


def point_inside_detection_ellipse(
    det: Detection,
    x: float,
    y: float,
    scale: float,
) -> bool:
    if det.width <= 1.0 or det.height <= 1.0:
        return False
    rx = max(1.0, det.width * scale)
    ry = max(1.0, det.height * scale)
    nx = (x - det.cx) / rx
    ny = (y - det.cy) / ry
    return nx * nx + ny * ny <= 1.0


def validate_sparse_feature_geometry(
    observation: SparseFeatureObservation,
    left_det: Detection,
    right_det: Detection,
    initial_disparity: float,
    config: FeatureValidationConfig,
    focal_px: float,
    baseline_m: float,
) -> bool:
    if (
        not observation.valid
        or observation.disparity_px <= 0.5
        or initial_disparity <= 0.5
        or focal_px <= 1e-3
        or baseline_m <= 1e-6
    ):
        return False
    if observation.support < max(1, config.min_support):
        return False
    if observation.stddev_px > max(0.05, config.max_stddev_px):
        return False

    left_x = observation.anchor_left_x
    left_y = observation.anchor_left_y
    expected_y = config.feature_y_offset_px + config.feature_y_slope * (left_x - left_det.cx)
    has_right_anchor = (
        observation.anchor_right_x is not None
        and observation.anchor_right_y is not None
        and math.isfinite(observation.anchor_right_x)
        and math.isfinite(observation.anchor_right_y)
    )
    right_x = observation.anchor_right_x if has_right_anchor else left_x - observation.disparity_px
    right_y = observation.anchor_right_y if has_right_anchor else left_y - expected_y
    if abs((left_y - right_y) - expected_y) > float(
        min(8.0, max(0.5, config.feature_y_tolerance_px))
    ):
        return False

    scale = float(min(0.90, max(0.35, config.feature_overlap_scale)))
    projection_scale = min(0.98, scale + 0.12)
    if not point_inside_detection_ellipse(left_det, left_x, left_y, scale):
        return False
    if not point_inside_detection_ellipse(right_det, right_x, right_y, scale):
        return False
    if not point_inside_detection_ellipse(
        right_det, left_x - initial_disparity, left_y, projection_scale
    ):
        return False
    if not point_inside_detection_ellipse(
        left_det, right_x + initial_disparity, right_y, projection_scale
    ):
        return False

    if config.feature_sphere_radius_m > 0.0:
        fb = focal_px * baseline_m
        center_z = fb / initial_disparity
        z = fb / observation.disparity_px
        if not math.isfinite(center_z) or not math.isfinite(z):
            return False
        dx = (left_x - left_det.cx) * z / focal_px
        dy = (left_y - left_det.cy) * z / focal_px
        dz = z - center_z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        max_distance = (
            config.feature_sphere_radius_m * max(1.0, config.feature_sphere_radius_scale)
            + max(0.0, config.feature_sphere_margin_m)
        )
        if distance > max_distance:
            return False
    return True
