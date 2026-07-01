"""6-dimensional gravity-prior EKF filter."""

import numpy as np
from .base import FilterBase, FilterState


class GravityEKF6D(FilterBase):
    """6-state EKF with gravity prior.

    State: [x, y, z, vx, vy, vz]
    Model: px' = px + vx*dt
           py' = py + vy*dt + 0.5*g*dt²
           pz' = pz + vz*dt
           vx' = vx
           vy' = vy + g*dt
           vz' = vz

    Gravity is along +y (downward in camera frame).
    """

    def __init__(self, sigma_a: float = 2.0, R_base: float = 0.01,
                 noise_exponent: float = 2.0, innovation_gate: float = 9.0,
                 focal: float = 727.0, gravity: float = 9.81,
                 gravity_vec: list = None, **kwargs):
        super().__init__(**kwargs)
        self.name = "GravityEKF6D"
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.focal = focal
        self.gravity = gravity
        # Default gravity vector: [0, g, 0] (y-down in OpenCV/ZED frame)
        if gravity_vec is not None:
            self.g_vec = np.array(gravity_vec, dtype=float)
        else:
            self.g_vec = np.array([0.0, gravity, 0.0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 100.0
        self.P[0:3, 0:3] = np.eye(3) * 1.0
        self.P[3:6, 3:6] = np.eye(3) * 10.0
        self._initialized = False

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix (linear part)."""
        F = np.eye(6)
        F[0, 3] = dt  # x += vx*dt
        F[1, 4] = dt  # y += vy*dt
        F[2, 5] = dt  # z += vz*dt
        return F

    def _predict_state(self, dt: float) -> np.ndarray:
        """Nonlinear state prediction with gravity."""
        x_pred = np.zeros(6)
        # Position prediction with gravity
        x_pred[0] = self.x[0] + self.x[3] * dt + 0.5 * self.g_vec[0] * dt * dt
        x_pred[1] = self.x[1] + self.x[4] * dt + 0.5 * self.g_vec[1] * dt * dt
        x_pred[2] = self.x[2] + self.x[5] * dt + 0.5 * self.g_vec[2] * dt * dt
        # Velocity prediction with gravity
        x_pred[3] = self.x[3] + self.g_vec[0] * dt
        x_pred[4] = self.x[4] + self.g_vec[1] * dt
        x_pred[5] = self.x[5] + self.g_vec[2] * dt
        return x_pred

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance.

        Q = sigma_a² * [[dt⁴/4*I₃, dt³/2*I₃], [dt³/2*I₃, dt²*I₃]]
        """
        I3 = np.eye(3)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (dt4 / 4.0) * I3
        Q[0:3, 3:6] = (dt3 / 2.0) * I3
        Q[3:6, 0:3] = (dt3 / 2.0) * I3
        Q[3:6, 3:6] = dt2 * I3

        Q *= self.sigma_a ** 2
        return Q

    def _build_R(self, z_depth: float) -> np.ndarray:
        """Measurement noise covariance (depth-dependent)."""
        z_clamped = max(1.0, abs(z_depth))
        Rz = self.R_base * (z_clamped ** self.noise_exponent)
        Rxy = Rz * (z_clamped ** 2) / (self.focal ** 2) + 0.001
        R = np.diag([Rxy, Rxy, Rz])
        return R

    def predict(self, dt: float):
        if dt <= 0:
            return
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = self._predict_state(dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        if not self._initialized:
            self.x[0:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._diagnostics['innovation'] = np.zeros(3)
            self._diagnostics['S'] = np.eye(3)
            self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])
            return self.get_state()

        # Measurement model: H = [I₃, 0₃]
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)

        # Innovation
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        # Innovation gate
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Store diagnostics
        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])

        return self.get_state()

    def get_state(self) -> FilterState:
        # 6D filter estimates acceleration as gravity (since it's the known model)
        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
            ax=self.g_vec[0], ay=self.g_vec[1], az=self.g_vec[2],
        )
