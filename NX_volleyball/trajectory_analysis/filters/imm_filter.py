"""Interacting Multiple Model (IMM) filter.

Uses 3 models:
1. Static (near-zero process noise)
2. Ballistic flight (gravity + moderate process noise)
3. High-maneuver (very high process noise for impacts/bounces)

Mode probabilities adapt automatically based on which model
best explains the observations.
"""

import numpy as np
from .base import FilterBase, FilterState


class IMMFilter(FilterBase):
    """IMM with 3 motion modes: static, ballistic, high-maneuver."""

    def __init__(self, R_base=0.015, noise_exponent=2.85,
                 innovation_gate=25.0,
                 sigma_static=0.5, sigma_flight=5.0, sigma_maneuver=50.0,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.sigma = [sigma_static, sigma_flight, sigma_maneuver]
        self.focal = focal
        self.gravity = gravity
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.n_models = 3

        # Markov transition matrix (rows sum to 1)
        self.TPM = np.array([
            [0.95, 0.04, 0.01],  # static → mostly stays static
            [0.02, 0.95, 0.03],  # flight → mostly stays flight
            [0.05, 0.20, 0.75],  # maneuver → quickly transitions out
        ])
        self.reset()

    def reset(self):
        self.x = [np.zeros(6) for _ in range(self.n_models)]
        self.P = [np.eye(6) * 1.0 for _ in range(self.n_models)]
        for p in self.P:
            p[3:, 3:] *= 10.0
        self.mu = np.array([0.33, 0.34, 0.33])  # mode probabilities
        self._initialized = False

    def _make_Q(self, sigma_a, dt):
        q = sigma_a ** 2
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
        return Q

    def _compute_R(self, oz):
        z_c = max(1.0, abs(oz))
        Rz = self.R_base * (z_c ** self.noise_exponent)
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def predict(self, dt: float):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        for j in range(self.n_models):
            # Gravity in all models (static model has near-zero Q to compensate)
            xj = self.x[j].copy()
            xj[3] += self.g_vec[0] * dt
            xj[4] += self.g_vec[1] * dt
            xj[5] += self.g_vec[2] * dt
            xj[:3] += xj[3:6] * dt
            self.x[j] = xj

            Q = self._make_Q(self.sigma[j], dt)
            self.P[j] = F @ self.P[j] @ F.T + Q

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            for j in range(self.n_models):
                self.x[j][:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            return self.get_state()

        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = self._compute_R(obs_z)
        z = np.array([obs_x, obs_y, obs_z])

        # --- IMM Mixing ---
        # Predicted mode probability
        c_bar = self.TPM.T @ self.mu
        # Mixing weights
        mu_mix = np.zeros((self.n_models, self.n_models))
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_mix[i, j] = self.TPM[i, j] * self.mu[i] / (c_bar[j] + 1e-30)

        # Mixed state/covariance for each model
        x_mixed = []
        P_mixed = []
        for j in range(self.n_models):
            xm = np.zeros(6)
            for i in range(self.n_models):
                xm += mu_mix[i, j] * self.x[i]
            x_mixed.append(xm)

            Pm = np.zeros((6, 6))
            for i in range(self.n_models):
                diff = self.x[i] - xm
                Pm += mu_mix[i, j] * (self.P[i] + np.outer(diff, diff))
            P_mixed.append(Pm)

        # --- Model-specific update ---
        likelihoods = np.zeros(self.n_models)
        for j in range(self.n_models):
            self.x[j] = x_mixed[j]
            self.P[j] = P_mixed[j]

            y = z - H @ self.x[j]
            S = H @ self.P[j] @ H.T + R
            S_inv = np.linalg.inv(S)
            maha2 = y @ S_inv @ y

            # Likelihood (Gaussian)
            det_S = np.linalg.det(S)
            likelihoods[j] = np.exp(-0.5 * maha2) / (np.sqrt((2*np.pi)**3 * det_S) + 1e-30)

            # Kalman update (always update, gate handled by mode switching)
            if maha2 <= self.innovation_gate:
                K = self.P[j] @ H.T @ S_inv
                self.x[j] += K @ y
                IKH = np.eye(6) - K @ H
                self.P[j] = IKH @ self.P[j] @ IKH.T + K @ R @ K.T

        # --- Mode probability update ---
        self.mu = c_bar * likelihoods
        mu_sum = np.sum(self.mu)
        if mu_sum > 1e-30:
            self.mu /= mu_sum
        else:
            self.mu = np.ones(self.n_models) / self.n_models

        return self.get_state()

    def get_state(self) -> FilterState:
        # Combined estimate (weighted by mode probability)
        x_comb = np.zeros(6)
        for j in range(self.n_models):
            x_comb += self.mu[j] * self.x[j]
        return FilterState(
            x=x_comb[0], y=x_comb[1], z=x_comb[2],
            vx=x_comb[3], vy=x_comb[4], vz=x_comb[5],
        )
