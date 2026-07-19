"""Depth math helpers shared by offline volleyball probes."""

from __future__ import annotations


def depth_from_disparity(disparity: float, focal_px: float, baseline_m: float) -> float:
    if disparity <= 0.0:
        return -1.0
    return focal_px * baseline_m / disparity
