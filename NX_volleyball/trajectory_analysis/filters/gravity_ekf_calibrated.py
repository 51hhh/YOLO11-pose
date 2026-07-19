"""Gravity EKF with calibrated R(z) from static data."""

import numpy as np
from .gravity_ekf_6d import GravityEKF6D


class GravityEKFCalibrated(GravityEKF6D):
    """6-state gravity EKF with calibrated measurement noise model.

    R(z) calibrated from static ball data:
        R_x = 0.000005 * z^0.96
        R_y = 0.000006 * z^2.11
        R_z = 0.000046 * z^2.85
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GravityEKF_Calibrated"

    def _build_R(self, z_depth: float) -> np.ndarray:
        z = max(0.5, abs(z_depth))
        Rx = 0.000005 * (z ** 0.96)
        Ry = 0.000006 * (z ** 2.11)
        Rz = 0.000046 * (z ** 2.85)
        return np.diag([Rx, Ry, Rz])
