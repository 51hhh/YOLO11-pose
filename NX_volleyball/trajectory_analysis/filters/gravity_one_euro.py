"""Gravity EKF + One-Euro hybrid filter.

Strategy: Use gravity EKF for physics-correct prediction, then apply
One-Euro style adaptive smoothing on the EKF output.

This combines:
- EKF's gravity model for accurate trajectory shape
- One-Euro's speed-adaptive smoothing for low-latency response
"""

import numpy as np
from .base import FilterBase, FilterState


class GravityOneEuroHybrid(FilterBase):
    """Gravity EKF with One-Euro post-smoothing."""

    def __init__(self, sigma_a=5.0, R_base=0.015, noise_exponent=2.85,
                 innovation_gate=25.0, min_cutoff=3.0, beta=0.01,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.focal = focal
        self.gravity = gravity
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.P[3:, 3:] *= 10.0
        # One-Euro state
        self.smooth_pos = np.zeros(3)
        self.smooth_vel = np.zeros(3)
        self._initialized = False

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def predict(self, dt: float):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x[3] += self.g_vec[0] * dt
        self.x[4] += self.g_vec[1] * dt
        self.x[5] += self.g_vec[2] * dt
        self.x[:3] += self.x[3:6] * dt

        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        Q[0, 0] = q * dt**4 / 4
        Q[1, 1] = q * dt**4 / 4
        Q[2, 2] = q * dt**4 / 4
        Q[0, 3] = q * dt**3 / 2
        Q[1, 4] = q * dt**3 / 2
        Q[2, 5] = q * dt**3 / 2
        Q[3, 0] = q * dt**3 / 2
        Q[4, 1] = q * dt**3 / 2
        Q[5, 2] = q * dt**3 / 2
        Q[3, 3] = q * dt**2
        Q[4, 4] = q * dt**2
        Q[5, 5] = q * dt**2

        self.P = F @ self.P @ F.T + Q
        self._last_dt = dt

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self.smooth_pos = self.x[:3].copy()
            self._initialized = True
            self._last_dt = 1.0 / 60.0
            return self.get_state()

        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        z_c = max(1.0, abs(obs_z))
        Rz = self.R_base * (z_c ** self.noise_exponent)
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        R = np.diag([Rxy, Rxy, Rz])

        y = np.array([obs_x, obs_y, obs_z]) - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # One-Euro post-smoothing on EKF output
        dt = getattr(self, '_last_dt', 1.0/60.0)
        ekf_pos = self.x[:3].copy()

        # Derivative (speed) of EKF output
        d_cutoff = 1.0
        a_d = self._alpha(d_cutoff, dt)
        vel = (ekf_pos - self.smooth_pos) / dt
        self.smooth_vel = a_d * vel + (1 - a_d) * self.smooth_vel

        # Adaptive cutoff
        speed = np.abs(self.smooth_vel)
        cutoff = self.min_cutoff + self.beta * speed

        # Smooth
        a = np.array([self._alpha(c, dt) for c in cutoff])
        self.smooth_pos = a * ekf_pos + (1 - a) * self.smooth_pos

        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.smooth_pos[0], y=self.smooth_pos[1], z=self.smooth_pos[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )
