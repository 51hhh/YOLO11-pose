"""9-dimensional constant acceleration Kalman filter."""

import numpy as np
from .base import FilterBase, FilterState


class ConstAccel9D(FilterBase):
    """9-state constant acceleration Kalman filter.

    State: [x, y, z, vx, vy, vz, ax, ay, az]
    Model: position += velocity*dt + 0.5*accel*dt²
           velocity += accel*dt
           accel = const (process noise drives changes)
    """

    def __init__(self, sigma_a: float = 5.0, R_base: float = 0.01,
                 noise_exponent: float = 2.0, innovation_gate: float = 9.0,
                 focal: float = 727.0, **kwargs):
        super().__init__(**kwargs)
        self.name = "ConstAccel9D"
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.focal = focal
        self.reset()

    def reset(self):
        self.x = np.zeros(9)  # [x,y,z, vx,vy,vz, ax,ay,az]
        self.P = np.eye(9) * 100.0
        # Position uncertainty smaller, velocity/accel larger
        self.P[0:3, 0:3] = np.eye(3) * 1.0
        self.P[3:6, 3:6] = np.eye(3) * 10.0
        self.P[6:9, 6:9] = np.eye(3) * 100.0
        self._initialized = False

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix."""
        F = np.eye(9)
        # Position += velocity*dt + 0.5*accel*dt²
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        # Velocity += accel*dt
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance.

        Q = sigma_a² * G @ G.T
        G = [0.5*dt²*I₃; dt*I₃; I₃]
        """
        I3 = np.eye(3)
        G = np.zeros((9, 3))
        G[0:3, :] = 0.5 * dt * dt * I3
        G[3:6, :] = dt * I3
        G[6:9, :] = I3
        Q = self.sigma_a ** 2 * (G @ G.T)
        return Q

    def _build_R(self, z_depth: float) -> np.ndarray:
        """Measurement noise covariance (depth-dependent).

        Rz = R_base * max(1, z)^exponent
        Rxy = Rz * z²/f² + 0.001
        """
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
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        if not self._initialized:
            self.x[0:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._diagnostics['innovation'] = np.zeros(3)
            self._diagnostics['S'] = np.eye(3)
            self._diagnostics['P_diag'] = np.diag(self.P)
            return self.get_state()

        # Measurement model: H = [I₃, 0₃, 0₃]
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)

        z = np.array([obs_x, obs_y, obs_z])
        z_depth = obs_z

        R = self._build_R(z_depth)

        # Innovation
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        # Innovation gate (Mahalanobis distance squared)
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            # Kalman gain
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I_KH = np.eye(9) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form

        # Store diagnostics
        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.diag(self.P).copy()

        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
            ax=self.x[6], ay=self.x[7], az=self.x[8],
        )
