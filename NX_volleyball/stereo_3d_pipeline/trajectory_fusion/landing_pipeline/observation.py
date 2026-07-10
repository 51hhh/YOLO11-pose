"""BBox-primary stereo observation formation with d0 correction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def _finite(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


@dataclass(frozen=True)
class StereoGeometry:
    fx: float
    fy: float
    cx: float
    cy: float
    fB: float
    d0: float
    baseline_m: float = 0.0

    def depth_from_disparity(self, disparity: float) -> Optional[float]:
        denom = float(disparity) - self.d0
        if denom <= 1.0:
            return None
        z = self.fB / denom
        if not math.isfinite(z) or z <= 0.0:
            return None
        return z

    def reproject(self, u: float, v: float, disparity: float) -> Optional[np.ndarray]:
        z = self.depth_from_disparity(disparity)
        if z is None:
            return None
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=float)


@dataclass
class Observation:
    t: float
    p: np.ndarray
    source: str
    disparity: float
    u: float
    v: float
    quality: Dict[str, float] = field(default_factory=dict)
    circle_p: Optional[np.ndarray] = None
    circle_delta_z: Optional[float] = None


class BBoxObservationBuilder:
    """Build one 3D observation per frame from bbox_center (+ optional circle).

    Design rules:
    - Primary source is always bbox_center when valid.
    - Circle is only used as fallback when bbox is invalid, or as a consistency
      sentinel that can inflate R (not as a second independent measurement).
    - Depth always comes from disparity - d0, never from raw z_* columns.
    """

    def __init__(
        self,
        geometry: StereoGeometry,
        prefer_bbox: bool = True,
        enable_circle_fallback: bool = True,
        circle_consistency_m: float = 0.35,
        min_trust: float = -1.0,
    ) -> None:
        self.geometry = geometry
        self.prefer_bbox = prefer_bbox
        self.enable_circle_fallback = enable_circle_fallback
        self.circle_consistency_m = circle_consistency_m
        self.min_trust = min_trust

    def _candidate(
        self,
        row: Mapping[str, Any],
        disp_key: str,
        u_key: str,
        v_key: str,
        trust_key: str,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        disp = _finite(row.get(disp_key))
        u = _finite(row.get(u_key))
        v = _finite(row.get(v_key))
        if not (math.isfinite(disp) and disp > 0.0 and math.isfinite(u) and math.isfinite(v)):
            return None
        if u < 0.0 or v < 0.0:
            return None
        p = self.geometry.reproject(u, v, disp)
        if p is None:
            return None
        trust = _finite(row.get(trust_key), 0.0)
        if self.min_trust >= 0.0 and trust < self.min_trust:
            return None
        return {
            "source": source,
            "p": p,
            "disparity": disp,
            "u": u,
            "v": v,
            "trust": trust,
        }

    def from_row(self, row: Mapping[str, Any], t_key: str = "timestamp") -> Optional[Observation]:
        t = _finite(row.get(t_key))
        if not math.isfinite(t) or t <= 0.0:
            return None

        bbox = self._candidate(
            row,
            "disparity_bbox_center",
            "left_bbox_cx",
            "left_bbox_cy",
            "p0p1_bbox_center_trust",
            "bbox_center",
        )
        circle = self._candidate(
            row,
            "disparity_circle_center",
            "left_circle_cx",
            "left_circle_cy",
            "p0p1_circle_center_trust",
            "circle_center",
        )

        chosen = None
        if self.prefer_bbox and bbox is not None:
            chosen = bbox
        elif self.enable_circle_fallback and circle is not None:
            chosen = circle
        elif bbox is not None:
            chosen = bbox
        else:
            return None

        quality = {
            "trust": float(chosen["trust"]),
            "dy_mad": _finite(row.get("p0p1_dy_mad"), 0.0),
            "pair_score": _finite(row.get("pair_score"), 0.0),
            "bbox_valid": 1.0 if bbox is not None else 0.0,
            "circle_valid": 1.0 if circle is not None else 0.0,
        }

        circle_p = None
        circle_delta_z = None
        if circle is not None:
            circle_p = circle["p"]
            if bbox is not None:
                circle_delta_z = float(circle["p"][2] - bbox["p"][2])
                quality["circle_delta_z"] = circle_delta_z
                # If bbox/circle disagree strongly, keep bbox but mark low consistency.
                quality["consistency"] = float(
                    math.exp(-abs(circle_delta_z) / max(self.circle_consistency_m, 1e-3))
                )
            else:
                quality["consistency"] = 1.0
        else:
            quality["consistency"] = 1.0 if chosen["source"] == "bbox_center" else 0.5

        return Observation(
            t=float(t),
            p=np.asarray(chosen["p"], dtype=float),
            source=str(chosen["source"]),
            disparity=float(chosen["disparity"]),
            u=float(chosen["u"]),
            v=float(chosen["v"]),
            quality=quality,
            circle_p=circle_p,
            circle_delta_z=circle_delta_z,
        )

    def from_values(
        self,
        t: float,
        disparity: float,
        u: float,
        v: float,
        source: str = "bbox_center",
        quality: Optional[Dict[str, float]] = None,
    ) -> Optional[Observation]:
        p = self.geometry.reproject(u, v, disparity)
        if p is None:
            return None
        return Observation(
            t=float(t),
            p=p,
            source=source,
            disparity=float(disparity),
            u=float(u),
            v=float(v),
            quality=dict(quality or {}),
        )
