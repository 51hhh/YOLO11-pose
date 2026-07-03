"""Calibration loading and rectification helpers for offline volleyball probes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _node_mat(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise KeyError(f"missing calibration key: {key}")
    value = node.mat()
    if value is None:
        raise ValueError(f"calibration key is not a matrix: {key}")
    return value


def load_calibration(path: Path) -> Dict[str, np.ndarray | float | int]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    try:
        width = int(fs.getNode("image_width").real())
        height = int(fs.getNode("image_height").real())
        calib: Dict[str, np.ndarray | float | int] = {
            "width": width,
            "height": height,
            "K1": _node_mat(fs, "camera_matrix_left"),
            "D1": _node_mat(fs, "distortion_coefficients_left"),
            "R1": _node_mat(fs, "rectification_left"),
            "P1": _node_mat(fs, "projection_left"),
            "K2": _node_mat(fs, "camera_matrix_right"),
            "D2": _node_mat(fs, "distortion_coefficients_right"),
            "R2": _node_mat(fs, "rectification_right"),
            "P2": _node_mat(fs, "projection_right"),
            "baseline_m": float(fs.getNode("baseline").real()) / 1000.0,
        }
        return calib
    finally:
        fs.release()


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    calib: Dict[str, np.ndarray | float | int],
) -> Tuple[np.ndarray, np.ndarray]:
    size = (int(calib["width"]), int(calib["height"]))
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size, cv2.CV_32FC1
    )
    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect
