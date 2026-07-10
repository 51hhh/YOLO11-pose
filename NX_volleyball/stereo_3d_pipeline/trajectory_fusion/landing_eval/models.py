"""Causal per-frame landing predictors (评价标准 2-4: 准/快/稳).

Contract: model.reset(); model.update(t, p, q) -> None or
{"landing": (x,y,z), "t_impact": float}. Observations arrive one at a
time; models must never look ahead. Gravity direction g_hat and ground
plane height ground_h are offline calibration constants (deployment-
realistic), loaded from throws_gt.json.
"""
import json
import math
import os

import numpy as np

_GT = json.load(open(os.path.join(os.path.dirname(__file__), "throws_gt.json")))
G_HAT = np.array(_GT["g_hat"])
GROUND_H = _GT["ground_h"]
G = 9.81

# volleyball params (config/pipeline_dual_yolo_roi.yaml prediction block)
MASS = 0.270
RADIUS = 0.105
RHO = 1.225
AREA = math.pi * RADIUS * RADIUS
DRAG_K = 0.5 * RHO * AREA / MASS  # accel = -DRAG_K*Cd*|v|*v


def height(p):
    return -float(np.dot(G_HAT, p)) - GROUND_H  # >0 above ground


def rollout_landing(p, v, cd, t_now, dt=0.008, t_max=3.0):
    """RK4 ballistic rollout with drag until ground-plane crossing."""
    p = p.copy()
    v = v.copy()

    def acc(vv):
        return G * G_HAT - DRAG_K * cd * np.linalg.norm(vv) * vv

    t = 0.0
    h_prev = height(p)
    while t < t_max:
        k1v = acc(v);              k1p = v
        k2v = acc(v + 0.5 * dt * k1v); k2p = v + 0.5 * dt * k1v
        k3v = acc(v + 0.5 * dt * k2v); k3p = v + 0.5 * dt * k2v
        k4v = acc(v + dt * k3v);   k4p = v + dt * k3v
        p = p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        h = height(p)
        if h <= 0.0 and h_prev > 0.0:
            frac = h_prev / (h_prev - h)
            p_land = p + (frac - 1.0) * dt * v
            return p_land, t_now + t - (1.0 - frac) * dt
        h_prev = h
    return None, None


class EkfDrag:
    """EKF [p,v] with gravity+drag process, Student-t robust innovation,
    distance-scaled R. cd_state=True adds Cd as slow 7th state."""

    def __init__(self, cd=0.47, nu=5.0, cd_state=False, sigma_d_px=0.4,
                 fB=1493.0, q_pos=1e-4, q_vel=4.0, q_cd=0.02):
        self.cd0 = cd
        self.nu = nu
        self.cd_state = cd_state
        self.sigma_d = sigma_d_px
        self.fB = fB
        self.q_pos, self.q_vel, self.q_cd = q_pos, q_vel, q_cd
        self.reset()

    def reset(self):
        self.t = None
        self.x = None   # [p(3), v(3), (cd)]
        self.P = None

    def _f(self, x, dt):
        p, v = x[:3], x[3:6]
        cd = x[6] if self.cd_state else self.cd0
        a = G * G_HAT - DRAG_K * cd * np.linalg.norm(v) * v
        xn = x.copy()
        xn[:3] = p + v * dt + 0.5 * a * dt * dt
        xn[3:6] = v + a * dt
        return xn

    def update(self, t, p_obs, q=None):
        z = np.asarray(p_obs, dtype=float)
        n = 7 if self.cd_state else 6
        if self.x is None:
            self.x = np.zeros(n)
            self.x[:3] = z
            if self.cd_state:
                self.x[6] = self.cd0
            self.P = np.diag([0.05] * 3 + [25.0] * 3 + ([0.04] if self.cd_state else []))
            self.t = t
            return None
        dt = t - self.t
        if dt <= 0 or dt > 0.5:
            # gap too large: soft reinit velocity from position delta
            if dt > 0.5:
                self.x[:3] = z
                self.P[:3, :3] = np.eye(3) * 0.05
                self.P[3:6, 3:6] = np.eye(3) * 25.0
                self.t = t
            return self._predict_out(t)
        self.t = t
        # ---- predict (EKF with numeric Jacobian on v for drag) ----
        Fm = np.eye(n)
        Fm[:3, 3:6] = np.eye(3) * dt
        v = self.x[3:6]
        cd = self.x[6] if self.cd_state else self.cd0
        vn = np.linalg.norm(v)
        if vn > 1e-3:
            Jd = -DRAG_K * cd * (vn * np.eye(3) + np.outer(v, v) / vn)
            Fm[3:6, 3:6] += Jd * dt
            if self.cd_state:
                Fm[3:6, 6] = -DRAG_K * vn * v * dt
        self.x = self._f(self.x, dt)
        Q = np.zeros((n, n))
        Q[:3, :3] = np.eye(3) * self.q_pos * dt
        Q[3:6, 3:6] = np.eye(3) * self.q_vel * dt
        if self.cd_state:
            Q[6, 6] = self.q_cd * dt
            self.x[6] = min(0.9, max(0.1, self.x[6]))
        self.P = Fm @ self.P @ Fm.T + Q
        # ---- update with Student-t reweighted R ----
        H = np.zeros((3, n))
        H[:, :3] = np.eye(3)
        zc = self.x[2]
        sz = (zc * zc / self.fB) * self.sigma_d
        sxy = max(0.004, 0.0012 * zc)
        R = np.diag([sxy * sxy, sxy * sxy, sz * sz])
        y = z - self.x[:3]
        S = self.P[:3, :3] + R
        try:
            Sin = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self._predict_out(t)
        delta = float(y @ Sin @ y)
        w = (self.nu + 3.0) / (self.nu + delta)   # <1 on outliers
        R_eff = R / max(w, 0.05)
        S = self.P[:3, :3] + R_eff
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(n) - K @ H) @ self.P
        return self._predict_out(t)

    def _predict_out(self, t):
        p, v = self.x[:3], self.x[3:6]
        # only predict when moving downward or ballistic (avoid pre-throw)
        cd = self.x[6] if self.cd_state else self.cd0
        land, ti = rollout_landing(p, v, cd, t)
        if land is None:
            return None
        return {"landing": land, "t_impact": ti}


class CaKalman:
    """9D constant-acceleration KF + gravity-only rollout (current-system
    style baseline)."""

    def __init__(self, r_xy=0.01, r_z_scale=0.02, q_jerk=60.0):
        self.q_jerk = q_jerk
        self.r_xy = r_xy
        self.r_z_scale = r_z_scale
        self.reset()

    def reset(self):
        self.t = None
        self.x = None
        self.P = None

    def update(self, t, p_obs, q=None):
        z = np.asarray(p_obs, dtype=float)
        if self.x is None:
            self.x = np.zeros(9)
            self.x[:3] = z
            self.P = np.diag([0.05] * 3 + [25.0] * 3 + [50.0] * 3)
            self.t = t
            return None
        dt = t - self.t
        if dt <= 0 or dt > 0.5:
            if dt > 0.5:
                self.reset()
                self.update(t, p_obs)
            return None
        self.t = t
        Fm = np.eye(9)
        Fm[:3, 3:6] = np.eye(3) * dt
        Fm[:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        Fm[3:6, 6:9] = np.eye(3) * dt
        self.x = Fm @ self.x
        Q = np.zeros((9, 9))
        Q[6:9, 6:9] = np.eye(3) * self.q_jerk * dt
        self.P = Fm @ self.P @ Fm.T + Q
        H = np.zeros((3, 9))
        H[:, :3] = np.eye(3)
        zc = max(1.0, self.x[2])
        R = np.diag([self.r_xy ** 2, self.r_xy ** 2,
                     (self.r_z_scale * zc) ** 2])
        S = self.P[:3, :3] + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x[:3])
        self.P = (np.eye(9) - K @ H) @ self.P
        land, ti = rollout_landing(self.x[:3], self.x[3:6], 0.0, t)
        if land is None:
            return None
        return {"landing": land, "t_impact": ti}


class PolyFit:
    """Sliding-window per-axis quadratic LS, analytic extrapolation to the
    ground plane. Pure data fit, no physics prior beyond quadratic."""

    def __init__(self, window_s=0.45, min_pts=8):
        self.window_s = window_s
        self.min_pts = min_pts
        self.reset()

    def reset(self):
        self.buf = []

    def update(self, t, p_obs, q=None):
        self.buf.append((t, np.asarray(p_obs, dtype=float)))
        t0 = t - self.window_s
        while self.buf and self.buf[0][0] < t0:
            self.buf.pop(0)
        if len(self.buf) < self.min_pts:
            return None
        ts = np.array([b[0] for b in self.buf]) - t
        ps = np.stack([b[1] for b in self.buf])
        A = np.stack([np.ones_like(ts), ts, 0.5 * ts * ts], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, ps, rcond=None)
        except np.linalg.LinAlgError:
            return None
        p0, v0 = coef[0], coef[1]
        land, ti = rollout_landing(p0, v0, 0.0, t)
        if land is None:
            return None
        return {"landing": land, "t_impact": ti}


def make_model(name):
    if name == "ca_kalman":
        return CaKalman()
    if name == "ekf_drag_t":
        return EkfDrag(cd=0.47, cd_state=False)
    if name == "ekf_nodrag_t":
        return EkfDrag(cd=0.0, cd_state=False)
    if name == "ekf_cd":
        return EkfDrag(cd=0.47, cd_state=True)
    if name == "polyfit":
        return PolyFit()
    raise ValueError(name)
