"""Patch scoring helpers shared by offline volleyball match probes."""

from __future__ import annotations

import numpy as np


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aa -= float(aa.mean())
    bb -= float(bb.mean())
    denom = float(np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    if denom < 1e-6:
        return -1.0
    return float((aa * bb).sum() / denom)


def _patch_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)
