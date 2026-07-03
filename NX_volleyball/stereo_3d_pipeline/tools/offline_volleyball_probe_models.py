"""Shared data models for offline volleyball probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class BallROI:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    radius: float
    mask: np.ndarray
    source: str = "auto"


@dataclass
class BallCandidate:
    index: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: int
    score: float
    mask: np.ndarray


@dataclass
class RoughBallROI:
    roi: BallROI
    seed: BallCandidate
    rank: int


@dataclass
class MatchResult:
    name: str
    left_keypoints: List[cv2.KeyPoint]
    right_keypoints: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    candidates: int
    disparity: float
    std_px: float
    depth_m: float
    confidence: float
    notes: str = ""
    extras: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ValidationThresholds:
    min_valid_matches: int = 8
    max_y_error_px: float = 2.0
    max_disparity_mad_px: float = 1.0
    max_disparity_range_px: float = 4.0
    max_z_mad_m: float = 0.020
    max_z_range_m: float = 0.060
    max_sphere_residual_m: float = 0.030
    max_depth_vs_center_m: float = 0.140
