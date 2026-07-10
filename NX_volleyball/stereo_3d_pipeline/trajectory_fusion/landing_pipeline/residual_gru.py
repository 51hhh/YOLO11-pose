"""Optional TinyGRU residual corrector for landing plane residuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .physics import as_unit, height_above_ground, inplane_coords, plane_basis


@dataclass
class ResidualConfig:
    enabled: bool = True
    hidden: int = 16
    max_abs_m: float = 1.0
    fade_start_s: float = 0.15
    fade_full_s: float = 0.50
    gap_reset_s: float = 0.5
    bounce_down_mps: float = 1.0
    bounce_up_mps: float = -1.0


class TinyGRUResidual:
    """Apply a 2D in-plane residual on top of physics landing.

    Torch is optional. If torch/checkpoint is unavailable, correction is zero and
    the pipeline degrades cleanly to pure physics EKF.
    """

    def __init__(
        self,
        g_hat: Sequence[float],
        ground_h: float,
        checkpoint: Optional[str | Path] = None,
        cfg: Optional[ResidualConfig] = None,
    ) -> None:
        self.cfg = cfg or ResidualConfig()
        self.g_hat, self.e1, self.e2 = plane_basis(g_hat)
        self.ground_h = float(ground_h)
        self.net = None
        self.hidden = None
        self.prev_t: Optional[float] = None
        self.prev_vdown: Optional[float] = None
        self.available = False
        self.reason = "disabled"
        if self.cfg.enabled and checkpoint is not None:
            self._load(checkpoint)

    def _load(self, checkpoint: str | Path) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - environment dependent
            self.reason = f"torch_unavailable: {exc}"
            return

        class TinyGRU(nn.Module):
            def __init__(self, nf: int, hidden: int):
                super().__init__()
                self.gru = nn.GRU(nf, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 2)

            def forward(self, x):
                y, h = self.gru(x)
                return self.head(y), h

        path = Path(checkpoint)
        if not path.exists():
            self.reason = f"checkpoint_missing: {path}"
            return
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            nf = int(blob.get("nf", 7))
            hidden = int(blob.get("hidden", self.cfg.hidden))
            net = TinyGRU(nf, hidden)
            state = blob.get("state", blob)
            net.load_state_dict(state)
            net.eval()
            self.net = net
            self.torch = torch
            self.available = True
            self.reason = "ok"
            self.nf = nf
        except Exception as exc:  # pragma: no cover
            self.reason = f"checkpoint_load_failed: {exc}"
            self.net = None
            self.available = False

    def reset(self) -> None:
        self.hidden = None
        self.prev_t = None
        self.prev_vdown = None

    def _fade(self, tti: float) -> float:
        # Near impact, trust pure physics more.
        return float(min(1.0, max(0.0, (tti - self.cfg.fade_start_s) / max(1e-6, self.cfg.fade_full_s - self.cfg.fade_start_s))))

    def correct(
        self,
        t: float,
        p: Sequence[float],
        v: Sequence[float],
        landing_physics: Sequence[float],
        t_impact: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        land = np.asarray(landing_physics, dtype=float)
        if not self.available or self.net is None:
            return land, np.zeros(2, dtype=float), 0.0

        if self.prev_t is not None and (t - self.prev_t) > self.cfg.gap_reset_s:
            self.hidden = None
        self.prev_t = float(t)

        v = np.asarray(v, dtype=float)
        p = np.asarray(p, dtype=float)
        vdown = float(np.dot(v, self.g_hat))
        if (
            self.prev_vdown is not None
            and self.prev_vdown > self.cfg.bounce_down_mps
            and vdown < self.cfg.bounce_up_mps
        ):
            self.hidden = None
        self.prev_vdown = vdown

        tti = float(t_impact - t)
        hgt = height_above_ground(p, self.g_hat, self.ground_h)
        feat = np.array(
            [[
                hgt / 3.0,
                p[2] / 10.0,
                vdown / 10.0,
                float(np.dot(v, self.e1)) / 10.0,
                float(np.dot(v, self.e2)) / 10.0,
                tti / 1.5,
                float(np.linalg.norm(inplane_coords(land - p, self.e1, self.e2))) / 10.0,
            ]],
            dtype=np.float32,
        )
        if feat.shape[1] != getattr(self, "nf", 7):
            # Feature contract mismatch: fail closed.
            return land, np.zeros(2, dtype=float), 0.0

        x = self.torch.from_numpy(feat).unsqueeze(0)  # [1,1,F]
        with self.torch.no_grad():
            y, self.hidden = self.net.gru(x, self.hidden)
            corr = self.net.head(y)[0, 0].cpu().numpy()
        w = self._fade(tti)
        corr = np.clip(corr, -self.cfg.max_abs_m, self.cfg.max_abs_m) * w
        land_corr = land + corr[0] * self.e1 + corr[1] * self.e2
        return land_corr, corr.astype(float), w
