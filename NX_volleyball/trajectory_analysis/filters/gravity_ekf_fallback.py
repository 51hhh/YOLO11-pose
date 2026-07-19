"""Gravity EKF with physics fallback on outliers."""

import numpy as np
from .gravity_ekf_6d import GravityEKF6D
from .base import FilterState


class GravityEKFFallback(GravityEKF6D):
    """6-state gravity EKF with physics-only fallback.

    When innovation gate rejects a measurement:
    - Pure physics prediction continues (no update)
    - After fallback_max consecutive outliers, P is inflated
      and a forced update re-acquires the target.

    Uses calibrated R(z).
    """

    def __init__(self, fallback_max: int = 3, P_inflate: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.name = "GravityEKF_Fallback"
        self.fallback_max = fallback_max
        self.P_inflate = P_inflate
        self._consecutive_outliers = 0

    def reset(self):
        super().reset()
        self._consecutive_outliers = 0

    def _build_R(self, z_depth: float) -> np.ndarray:
        z = max(0.5, abs(z_depth))
        Rx = 0.000005 * (z ** 0.96)
        Ry = 0.000006 * (z ** 2.11)
        Rz = 0.000046 * (z ** 2.85)
        return np.diag([Rx, Ry, Rz])

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        if not self._initialized:
            self.x[0:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._consecutive_outliers = 0
            self._diagnostics['innovation'] = np.zeros(3)
            self._diagnostics['S'] = np.eye(3)
            self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])
            return self.get_state()

        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            # Normal update
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self._consecutive_outliers = 0
        else:
            # Outlier: no update (pure physics fallback)
            self._consecutive_outliers += 1
            if self._consecutive_outliers >= self.fallback_max:
                # Force re-acquisition: inflate P and update
                self.P *= self.P_inflate
                S_new = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S_new)
                self.x = self.x + K @ y
                I_KH = np.eye(6) - K @ H
                self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
                self._consecutive_outliers = 0

        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])

        return self.get_state()
