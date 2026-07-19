"""Streaming landing pipeline: bbox+d0 -> EKF -> RK4 -> optional residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .config import LandingPipelineConfig, load_pipeline_config
from .ekf import StudentTDragEKF
from .observation import BBoxObservationBuilder, Observation
from .residual_gru import TinyGRUResidual


@dataclass
class LandingResult:
    t: float
    position: np.ndarray
    velocity: np.ndarray
    landing: np.ndarray
    landing_physics: np.ndarray
    t_impact: float
    time_to_land: float
    source: str
    residual_applied: bool
    residual_corr: np.ndarray
    residual_weight: float
    student_w: float
    quality: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t": float(self.t),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "landing": self.landing.tolist(),
            "landing_physics": self.landing_physics.tolist(),
            "t_impact": float(self.t_impact),
            "time_to_land": float(self.time_to_land),
            "source": self.source,
            "residual_applied": bool(self.residual_applied),
            "residual_corr": self.residual_corr.tolist(),
            "residual_weight": float(self.residual_weight),
            "student_w": float(self.student_w),
            "quality": dict(self.quality),
        }


class LandingPipeline:
    """Causal per-frame API.

    Usage:
        pipe = LandingPipeline.from_default()
        for row in csv_rows:
            out = pipe.update_row(row)
            if out is not None:
                use(out.landing, out.time_to_land)
    """

    def __init__(self, cfg: LandingPipelineConfig) -> None:
        self.cfg = cfg
        self.obs_builder = BBoxObservationBuilder(
            geometry=cfg.geometry(),
            prefer_bbox=cfg.prefer_bbox,
            enable_circle_fallback=cfg.enable_circle_fallback,
            circle_consistency_m=cfg.circle_consistency_m,
        )
        # Align EKF fB with geometry for sigma_z scaling.
        self.cfg.ekf.fB = float(cfg.fB if cfg.ekf.fB <= 0 else cfg.ekf.fB)
        self.ekf = StudentTDragEKF(
            cfg=cfg.ekf,
            g_hat=cfg.g_hat,
            ground_h=cfg.ground_h,
            rk4_dt=cfg.rk4_dt,
            max_predict_time=cfg.max_predict_time,
        )
        self.residual = TinyGRUResidual(
            g_hat=cfg.g_hat,
            ground_h=cfg.ground_h,
            checkpoint=cfg.residual_checkpoint,
            cfg=cfg.residual,
        )
        self.last_source = "none"

    @classmethod
    def from_default(
        cls,
        config_path: Optional[str] = None,
        enable_residual: bool = True,
        use_runtime_d0: bool = True,
        residual_checkpoint: Optional[str] = None,
    ) -> "LandingPipeline":
        cfg = load_pipeline_config(
            config_path,
            use_runtime_d0=use_runtime_d0,
            enable_residual=enable_residual,
            residual_checkpoint=residual_checkpoint,
        )
        return cls(cfg)

    def reset(self) -> None:
        self.ekf.reset()
        self.residual.reset()
        self.last_source = "none"

    def update_observation(self, obs: Observation) -> Optional[LandingResult]:
        self.last_source = obs.source
        out = self.ekf.update(obs.t, obs.p, obs.quality)
        if out is None:
            return None
        land_phys = np.asarray(out["landing_physics"], dtype=float)
        land, corr, w = self.residual.correct(
            t=obs.t,
            p=out["position"],
            v=out["velocity"],
            landing_physics=land_phys,
            t_impact=float(out["t_impact"]),
        )
        residual_applied = bool(self.residual.available and self.residual.cfg.enabled and w > 0.0)
        return LandingResult(
            t=float(obs.t),
            position=np.asarray(out["position"], dtype=float),
            velocity=np.asarray(out["velocity"], dtype=float),
            landing=np.asarray(land, dtype=float),
            landing_physics=land_phys,
            t_impact=float(out["t_impact"]),
            time_to_land=float(out["time_to_land"]),
            source=obs.source,
            residual_applied=residual_applied,
            residual_corr=np.asarray(corr, dtype=float),
            residual_weight=float(w),
            student_w=float(out.get("student_w", 1.0)),
            quality=dict(obs.quality),
        )

    def update_row(self, row: Mapping[str, Any]) -> Optional[LandingResult]:
        obs = self.obs_builder.from_row(row)
        if obs is None:
            return None
        return self.update_observation(obs)

    def update_bbox(
        self,
        t: float,
        disparity: float,
        u: float,
        v: float,
        quality: Optional[Dict[str, float]] = None,
    ) -> Optional[LandingResult]:
        obs = self.obs_builder.from_values(t, disparity, u, v, source="bbox_center", quality=quality)
        if obs is None:
            return None
        return self.update_observation(obs)
