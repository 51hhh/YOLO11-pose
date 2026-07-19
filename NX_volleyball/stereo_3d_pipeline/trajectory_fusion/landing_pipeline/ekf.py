"""Student-t robust EKF with gravity + quadratic drag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .physics import DRAG_K, G, RolloutResult, as_unit, height_above_ground, rollout_landing


@dataclass
class EkfConfig:
    cd: float = 0.10
    nu: float = 12.0
    sigma_d_px: float = 0.4
    fB: float = 1497.079
    q_pos: float = 1e-4
    q_vel: float = 1.5
    max_dt: float = 0.5
    xy_sigma_scale: float = 0.0012
    xy_sigma_floor: float = 0.004
    consistency_inflate: float = 3.0


class StudentTDragEKF:
    """Causal 6D filter: state = [p(3), v(3)]."""

    def __init__(
        self,
        cfg: EkfConfig,
        g_hat: Sequence[float],
        ground_h: float,
        rk4_dt: float = 0.008,
        max_predict_time: float = 3.0,
    ) -> None:
        self.cfg = cfg
        self.g_hat = as_unit(g_hat)
        self.ground_h = float(ground_h)
        self.rk4_dt = float(rk4_dt)
        self.max_predict_time = float(max_predict_time)
        self.reset()

    def reset(self) -> None:
        self.t: Optional[float] = None
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.last_innovation = None
        self.last_student_w = 1.0

    def _accel(self, v: np.ndarray, cd: float) -> np.ndarray:
        return G * self.g_hat - DRAG_K * cd * np.linalg.norm(v) * v

    def _f(self, x: np.ndarray, dt: float) -> np.ndarray:
        p, v = x[:3], x[3:6]
        a = self._accel(v, self.cfg.cd)
        xn = x.copy()
        xn[:3] = p + v * dt + 0.5 * a * dt * dt
        xn[3:6] = v + a * dt
        return xn

    def _measurement_R(self, zc: float, quality: Optional[Dict[str, float]]) -> np.ndarray:
        zc = max(float(zc), 0.5)
        sz = (zc * zc / max(self.cfg.fB, 1e-6)) * self.cfg.sigma_d_px
        sxy = max(self.cfg.xy_sigma_floor, self.cfg.xy_sigma_scale * zc)
        # Inflate when bbox/circle disagree or trust is low.
        inflate = 1.0
        if quality:
            consistency = float(quality.get("consistency", 1.0))
            consistency = min(1.0, max(0.05, consistency))
            inflate *= 1.0 + (self.cfg.consistency_inflate - 1.0) * (1.0 - consistency)
            trust = float(quality.get("trust", 1.0))
            if trust >= 0.0:
                inflate *= 1.0 + max(0.0, 1.0 - trust)
        return np.diag([(sxy * inflate) ** 2, (sxy * inflate) ** 2, (sz * inflate) ** 2])

    def _predict_matrices(self, dt: float) -> np.ndarray:
        n = 6
        Fm = np.eye(n)
        Fm[:3, 3:6] = np.eye(3) * dt
        v = self.x[3:6]
        vn = float(np.linalg.norm(v))
        if vn > 1e-3:
            Jd = -DRAG_K * self.cfg.cd * (vn * np.eye(3) + np.outer(v, v) / vn)
            Fm[3:6, 3:6] += Jd * dt
        return Fm

    def update(
        self,
        t: float,
        p_obs: Sequence[float],
        quality: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, object]]:
        z = np.asarray(p_obs, dtype=float)
        if self.x is None:
            self.x = np.zeros(6, dtype=float)
            self.x[:3] = z
            self.P = np.diag([0.05, 0.05, 0.05, 25.0, 25.0, 25.0])
            self.t = float(t)
            self.last_innovation = np.zeros(3)
            self.last_student_w = 1.0
            return None

        dt = float(t) - float(self.t)
        if dt <= 0.0:
            return self.predict_output(float(t))
        if dt > self.cfg.max_dt:
            self.x[:3] = z
            self.P[:3, :3] = np.eye(3) * 0.05
            self.P[3:6, 3:6] = np.eye(3) * 25.0
            self.t = float(t)
            self.last_innovation = np.zeros(3)
            self.last_student_w = 1.0
            return self.predict_output(float(t))

        Fm = self._predict_matrices(dt)
        self.x = self._f(self.x, dt)
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * self.cfg.q_pos * dt
        Q[3:6, 3:6] = np.eye(3) * self.cfg.q_vel * dt
        self.P = Fm @ self.P @ Fm.T + Q
        self.t = float(t)

        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = self._measurement_R(self.x[2], quality)
        innov = z - self.x[:3]
        S = self.P[:3, :3] + R
        try:
            Sin = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.predict_output(float(t))
        delta = float(innov @ Sin @ innov)
        w = (self.cfg.nu + 3.0) / (self.cfg.nu + delta)
        self.last_innovation = innov
        self.last_student_w = float(w)
        R_eff = R / max(w, 0.05)
        S = self.P[:3, :3] + R_eff
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.predict_output(float(t))
        self.x = self.x + K @ innov
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.predict_output(float(t))

    def predict_output(self, t: float) -> Optional[Dict[str, object]]:
        if self.x is None:
            return None
        p = self.x[:3]
        v = self.x[3:6]
        # Avoid pre-throw upward/hover predictions with tiny speed.
        speed = float(np.linalg.norm(v))
        h = height_above_ground(p, self.g_hat, self.ground_h)
        if h <= 0.05 or speed < 0.2:
            return None
        roll = rollout_landing(
            p,
            v,
            self.cfg.cd,
            t,
            self.g_hat,
            self.ground_h,
            dt=self.rk4_dt,
            t_max=self.max_predict_time,
        )
        if roll is None:
            return None
        return {
            "position": p.copy(),
            "velocity": v.copy(),
            "landing_physics": roll.landing.copy(),
            "t_impact": roll.t_impact,
            "time_to_land": roll.time_to_land,
            "student_w": self.last_student_w,
            "innovation": None if self.last_innovation is None else self.last_innovation.copy(),
            "height": h,
        }
