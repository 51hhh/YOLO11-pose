"""Configuration loading for the deployable landing pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .ekf import EkfConfig
from .observation import StereoGeometry
from .residual_gru import ResidualConfig


DEFAULT_G_HAT = (0.0018133971960385172, 0.9701361127202714, 0.24255439469655052)
DEFAULT_GROUND_H = -1.2592940759920357


@dataclass
class LandingPipelineConfig:
    # Geometry / d0
    fx: float = 1613.873447565926
    fy: float = 1613.873447565926
    cx: float = 681.58988952636719
    cy: float = 620.79562759399414
    fB: float = 1497.079
    baseline_m: float = 0.92763108240240683
    d0: float = -5.324
    prefer_bbox: bool = True
    enable_circle_fallback: bool = True
    circle_consistency_m: float = 0.35

    # World / ground
    g_hat: Sequence[float] = field(default_factory=lambda: list(DEFAULT_G_HAT))
    ground_h: float = DEFAULT_GROUND_H

    # Filter / rollout
    ekf: EkfConfig = field(default_factory=EkfConfig)
    rk4_dt: float = 0.008
    max_predict_time: float = 3.0

    # Optional residual
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    residual_checkpoint: Optional[str] = None

    def geometry(self) -> StereoGeometry:
        return StereoGeometry(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            fB=self.fB,
            d0=self.d0,
            baseline_m=self.baseline_m,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pipeline_config(
    config_path: Optional[str | Path] = None,
    *,
    use_runtime_d0: bool = True,
    enable_residual: bool = True,
    residual_checkpoint: Optional[str | Path] = None,
) -> LandingPipelineConfig:
    """Load defaults, optionally overlaying JSON config and known project files."""
    here = Path(__file__).resolve().parent
    root = here.parents[1]  # stereo_3d_pipeline
    cfg = LandingPipelineConfig()

    # Prefer runtime d0 used by C++ pipeline if present.
    runtime_d0 = root / "config" / "disparity_offset_fit_20260709.json"
    eval_gt = here.parent / "landing_eval" / "throws_gt.json"
    eval_d0 = here.parent / "landing_eval" / "d0_fit.json"
    default_ckpt = here.parent / "landing_eval" / "gru_final.pt"

    if use_runtime_d0 and runtime_d0.exists():
        blob = _read_json(runtime_d0)
        fit = blob.get("fit", blob)
        if "d0" in fit:
            cfg.d0 = float(fit["d0"])
        if "fB" in fit:
            cfg.fB = float(fit["fB"])
            cfg.ekf.fB = float(fit["fB"])
    elif eval_d0.exists():
        blob = _read_json(eval_d0)
        cfg.d0 = float(blob.get("d0", cfg.d0))
        if "fB_fit" in blob:
            # Keep runtime scale from calibration fB by default; only copy if asked
            # via explicit config. Still store fitted value for reference in ekf R.
            cfg.ekf.fB = float(blob["fB_fit"])

    if eval_gt.exists():
        blob = _read_json(eval_gt)
        if "g_hat" in blob:
            cfg.g_hat = list(blob["g_hat"])
        if "ground_h" in blob:
            cfg.ground_h = float(blob["ground_h"])
        # landing_eval d0 is kept as optional overlay only when runtime d0 missing.
        if not (use_runtime_d0 and runtime_d0.exists()) and "d0" in blob:
            cfg.d0 = float(blob["d0"])

    if residual_checkpoint is not None:
        cfg.residual_checkpoint = str(residual_checkpoint)
    elif default_ckpt.exists():
        cfg.residual_checkpoint = str(default_ckpt)

    if config_path is not None:
        path = Path(config_path)
        blob = _read_json(path)
        cfg = _apply_dict(cfg, blob)

    # Explicit function args win over JSON defaults.
    cfg.residual.enabled = bool(enable_residual)
    if residual_checkpoint is not None:
        cfg.residual_checkpoint = str(residual_checkpoint)

    # Resolve relative residual checkpoint against stereo_3d_pipeline root.
    if cfg.residual_checkpoint:
        ckpt = Path(cfg.residual_checkpoint)
        if not ckpt.is_absolute():
            cand = (root / ckpt).resolve()
            if cand.exists():
                cfg.residual_checkpoint = str(cand)
            else:
                cand2 = (Path.cwd() / ckpt).resolve()
                cfg.residual_checkpoint = str(cand2)
    return cfg


def _apply_dict(cfg: LandingPipelineConfig, blob: Dict[str, Any]) -> LandingPipelineConfig:
    for key in (
        "fx",
        "fy",
        "cx",
        "cy",
        "fB",
        "baseline_m",
        "d0",
        "prefer_bbox",
        "enable_circle_fallback",
        "circle_consistency_m",
        "ground_h",
        "rk4_dt",
        "max_predict_time",
        "residual_checkpoint",
    ):
        if key in blob:
            setattr(cfg, key, blob[key])
    if "g_hat" in blob:
        cfg.g_hat = list(blob["g_hat"])
    if "ekf" in blob and isinstance(blob["ekf"], dict):
        for k, v in blob["ekf"].items():
            if hasattr(cfg.ekf, k):
                setattr(cfg.ekf, k, v)
    if "residual" in blob and isinstance(blob["residual"], dict):
        for k, v in blob["residual"].items():
            if hasattr(cfg.residual, k):
                setattr(cfg.residual, k, v)
    # Keep filter depth-noise scale aligned with geometry fB unless overridden.
    if "fB" in blob and "ekf" in blob and "fB" not in blob["ekf"]:
        cfg.ekf.fB = float(blob["fB"])
    elif "fB" in blob and "ekf" not in blob:
        cfg.ekf.fB = float(blob["fB"])
    return cfg


def write_default_config(path: str | Path) -> Path:
    cfg = load_pipeline_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return path
