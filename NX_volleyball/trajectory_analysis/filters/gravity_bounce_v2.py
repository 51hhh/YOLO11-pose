"""Improved bounce detection with adaptive thresholds."""

import numpy as np
from .gravity_ekf_6d import GravityEKF6D
from .base import FilterState


class GravityBounceV2(GravityEKF6D):
    """6-state gravity EKF with improved adaptive bounce detection.

    Improvements over GravityBounce:
    - Adaptive thresholds: bounce_obs = 3*sqrt(R_y), vy_threshold = 1.5*sqrt(P[4,4])
    - Requires vy to actually become negative after update for confirmation
    - Post-bounce: pauses y-direction update for pause_frames to stabilize

    Uses calibrated R(z).
    """

    def __init__(self, restitution: float = 0.75, P_boost_vy: float = 50.0,
                 pause_frames: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.name = "GravityBounceV2"
        self.restitution = restitution
        self.P_boost_vy = P_boost_vy
        self.pause_frames = pause_frames
        self._bounce_state = 'idle'
        self._confirm_count = 0
        self._pause_count = 0

    def reset(self):
        super().reset()
        self._bounce_state = 'idle'
        self._confirm_count = 0
        self._pause_count = 0

    def _build_R(self, z_depth: float) -> np.ndarray:
        z = max(0.5, abs(z_depth))
        Rx = 0.000005 * (z ** 0.96)
        Ry = 0.000006 * (z ** 2.11)
        Rz = 0.000046 * (z ** 2.85)
        return np.diag([Rx, Ry, Rz])

    def _apply_bounce(self):
        """Reverse vy with restitution and inflate P[4,4]."""
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

        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        z_obs = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)

        y = z_obs - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        # Adaptive thresholds from current state
        vy_threshold = 1.5 * np.sqrt(max(self.P[4, 4], 0.01))
        bounce_obs_threshold = 3.0 * np.sqrt(max(R[1, 1], 1e-6))

        # Bounce detection state machine
        pred_y = self.x[1]
        vy_pred = self.x[4]

        if self._pause_count > 0:
            # During pause: update x/z normally, skip y update
            self._pause_count -= 1
            # Partial update: only x and z
            H_xz = np.zeros((2, 6))
            H_xz[0, 0] = 1.0  # x
            H_xz[1, 2] = 1.0  # z
            z_xz = np.array([obs_x, obs_z])
            R_xz = np.diag([R[0, 0], R[2, 2]])
            y_xz = z_xz - H_xz @ self.x
            S_xz = H_xz @ self.P @ H_xz.T + R_xz
            K_xz = self.P @ H_xz.T @ np.linalg.inv(S_xz)
            self.x = self.x + K_xz @ y_xz
            I_KH = np.eye(6) - K_xz @ H_xz
            self.P = I_KH @ self.P @ I_KH.T + K_xz @ R_xz @ K_xz.T

            self._diagnostics['innovation'] = np.array([y_xz[0], 0.0, y_xz[1]])
            self._diagnostics['S'] = S.copy()
            self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])
            return self.get_state()

        if self._bounce_state == 'idle':
            # Check candidate: vy positive (falling) and obs below prediction
            if vy_pred > vy_threshold and (obs_y < pred_y - bounce_obs_threshold):
                self._bounce_state = 'confirming'
                self._confirm_count = 1
        elif self._bounce_state == 'confirming':
            if obs_y < pred_y - bounce_obs_threshold * 0.5:
                self._confirm_count += 1
                if self._confirm_count >= 2:
                    # Confirmed: apply bounce
                    self._apply_bounce()
                    self._bounce_state = 'idle'
                    self._confirm_count = 0
                    self._pause_count = self.pause_frames
            else:
                self._bounce_state = 'idle'
                self._confirm_count = 0

        # Standard EKF update (with innovation gate)
        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])

        return self.get_state()
