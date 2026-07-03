"""Input helpers for offline YOLO IoU fallback regression."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from stereo_feature_matching.realtime_contract import Detection


def load_baseline_from_calib(path: Path) -> float:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    try:
        node = fs.getNode("baseline")
        if node.empty():
            raise KeyError("missing calibration key: baseline")
        return float(node.real()) / 1000.0
    finally:
        fs.release()


def read_rows(path: Path, max_frames: int) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_frames > 0:
        rows = rows[:max_frames]
    return rows


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def row_detection(row: Dict[str, str], side: str) -> Detection | None:
    try:
        count = int(float(row[f"{side}_count"]))
    except (KeyError, ValueError):
        count = 1
    if count <= 0:
        return None
    try:
        return Detection(
            cx=float(row[f"{side}_cx"]),
            cy=float(row[f"{side}_cy"]),
            width=float(row[f"{side}_w"]),
            height=float(row[f"{side}_h"]),
            confidence=float(row.get(f"{side}_conf", 1.0)),
            class_id=int(float(row.get(f"{side}_class_id", 0))),
        )
    except (KeyError, ValueError):
        return None
