"""Adaptive EKF (AEKF) - online estimation of Q and R from innovation.

Reference: Mohamed & Schwarz (1999) "Adaptive Kalman Filtering for INS/GPS"
Also used in SMASH (HKU badminton tracking).

Key idea: estimate R from innovation covariance, and Q from state residuals,
using a sliding window of recent innovations.
"""

import numpy as np
from .base import FilterBase, FilterState


class AdaptiveEKF(FilterBase):
    """6D Gravity EKF with adaptive Q and R estimation."""

    def __init__(self, sigma_a_init=5.0, R_base=0.015, noise_exponent=2.85,
                 innovation_gate=25.0, window_size=20,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a_init = sigma_a_init
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.window_size = window_size
        self.focal = focal
        self.gravity = gravity
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1.0
        self.P[3:, 3:] *= 10.0
        self.innovations = []  # sliding window of innovations
        self.S_history = []  # sliding window of S matrices
        self.sigma_a = self.sigma_a_init
        self.R_scale = 1.0

    def predict(self, dt: float):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Apply gravity
        self.x[3] += self.g_vec[0] * dt
        self.x[4] += self.g_vec[1] * dt
        self.x[5] += self.g_vec[2] * dt
        self.x[:3] += self.x[3:6] * dt

        # Process noise
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

    def _compute_R(self, oz):
        z_c = max(1.0, abs(oz))
        Rz = self.R_base * (z_c ** self.noise_exponent) * self.R_scale
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def _adapt(self):
        """Adapt R_scale from innovation statistics."""
        if len(self.innovations) < self.window_size:
            return

        # Empirical innovation covariance
        innov = np.array(self.innovations[-self.window_size:])
        C_innov = np.mean([np.outer(v, v) for v in innov], axis=0)

        # Expected innovation covariance: S = H P H' + R
        # If C_innov > mean(S), R is too small → increase R_scale
        S_mean = np.mean(self.S_history[-self.window_size:], axis=0)

        # Ratio of traces
        ratio = np.trace(C_innov) / (np.trace(S_mean) + 1e-10)
        # Smooth adaptation
        self.R_scale = 0.9 * self.R_scale + 0.1 * max(0.1, min(10.0, ratio))

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not hasattr(self, '_initialized') or not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            return self.get_state()

        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = self._compute_R(obs_z)
        y = np.array([obs_x, obs_y, obs_z]) - H @ self.x
        S = H @ self.P @ H.T + R

        # Store for adaptation
        self.innovations.append(y.copy())
        self.S_history.append(S.copy())
        self._adapt()

        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S

        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )
