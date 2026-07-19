"""Gravity EKF with two-stage bounce detection."""

import numpy as np
from collections import deque
from .gravity_ekf_6d import GravityEKF6D
from .base import FilterState


class GravityBounce(GravityEKF6D):
    """6-state gravity EKF with two-stage bounce detection.

    Inherits from GravityEKF6D and adds bounce handling:
    - Stage 1 (candidate): vy_pred > threshold AND obs_y < pred_y - obs_threshold
                           AND |obs_z - pred_z| < 0.5
    - Stage 2 (confirm): next 2 frames obs_y continues below prediction
    - Trigger: vy = -vy * restitution, P[vy,vy] *= P_boost
    """

    def __init__(self, bounce_vy_threshold: float = 1.0,
                 bounce_obs_threshold: float = 0.15,
                 restitution: float = 0.75,
                 P_boost_vy: float = 100.0, **kwargs):
        self.bounce_vy_threshold = bounce_vy_threshold
        self.bounce_obs_threshold = bounce_obs_threshold
        self.restitution = restitution
        self.P_boost_vy = P_boost_vy
        self._bounce_state = 'idle'  # 'idle', 'candidate', 'confirming'
        self._confirm_count = 0
        self._history = deque(maxlen=3)  # Store recent (obs_y, pred_y) pairs
        super().__init__(**kwargs)
        self.name = "GravityBounce"

    def reset(self):
        super().reset()
        self._bounce_state = 'idle'
        self._confirm_count = 0
        self._history.clear()

    def _check_bounce_candidate(self, obs_y: float, pred_y: float,
                                obs_z: float, pred_z: float) -> bool:
        """Stage 1: Check if current frame is a bounce candidate."""
        vy_pred = self.x[4]  # predicted vy (positive = downward)

        cond_vy = vy_pred > self.bounce_vy_threshold
        cond_y = obs_y < (pred_y - self.bounce_obs_threshold)
        cond_z = abs(obs_z - pred_z) < 0.5

        return cond_vy and cond_y and cond_z

    def _apply_bounce(self):
        """Apply bounce: reverse vy with restitution, boost P."""
        self.x[4] = -abs(self.x[4]) * self.restitution
        self.P[4, 4] *= self.P_boost_vy

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        if not self._initialized:
            self.x[0:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._diagnostics['innovation'] = np.zeros(3)
            self._diagnostics['S'] = np.eye(3)
            self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])
            return self.get_state()

        # Save predicted state before update
        pred_y = self.x[1]
        pred_z = self.x[2]

        # Bounce detection state machine
        if self._bounce_state == 'idle':
            if self._check_bounce_candidate(obs_y, pred_y, obs_z, pred_z):
                self._bounce_state = 'confirming'
                self._confirm_count = 0
        elif self._bounce_state == 'confirming':
            # Check if obs_y continues below predicted
            if obs_y < pred_y - self.bounce_obs_threshold * 0.5:
                self._confirm_count += 1
                if self._confirm_count >= 2:
                    # Confirmed bounce
                    self._apply_bounce()
                    self._bounce_state = 'idle'
                    self._confirm_count = 0
            else:
                # Not confirmed, reset
                self._bounce_state = 'idle'
                self._confirm_count = 0

        # Store history
        self._history.append((obs_y, pred_y))

        # Standard EKF update (from parent class)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])

        return self.get_state()
