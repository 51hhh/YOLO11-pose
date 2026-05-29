"""7-dimensional EKF with gravity and aerodynamic drag estimation."""

import numpy as np
from .base import FilterBase, FilterState


class GravityDrag7D(FilterBase):
    """7-state EKF estimating position, velocity, and drag coefficient.

    State: [x, y, z, vx, vy, vz, k_drag]
    Model:
        a_drag = -k_drag * |v| * v
        p' = p + v*dt + 0.5*(g + a_drag)*dt²
        v' = v + (g + a_drag)*dt
        k' = k (random walk)

    Observation: [x, y, z] (position only)
    """

    def __init__(self, sigma_a: float = 2.0, sigma_k: float = 0.01,
                 innovation_gate: float = 9.0, k_drag_init: float = 0.023,
                 gravity: float = 9.81, gravity_vec: list = None,
                 focal: float = 727.0, **kwargs):
        super().__init__(**kwargs)
        self.name = "GravityDrag7D"
        self.sigma_a = sigma_a
        self.sigma_k = sigma_k
        self.innovation_gate = innovation_gate
        self.k_drag_init = k_drag_init
        self.gravity = gravity
        self.focal = focal
        if gravity_vec is not None:
            self.g_vec = np.array(gravity_vec, dtype=float)
        else:
            self.g_vec = np.array([0.0, gravity, 0.0])
        self.reset()

    def reset(self):
        self.x = np.zeros(7)
        self.x[6] = self.k_drag_init
        self.P = np.eye(7) * 100.0
        self.P[0:3, 0:3] = np.eye(3) * 1.0
        self.P[3:6, 3:6] = np.eye(3) * 10.0
        self.P[6, 6] = 1.0  # uncertain drag
        self._initialized = False

    def _predict_state(self, dt: float) -> np.ndarray:
        pos = self.x[0:3]
        vel = self.x[3:6]
        k = max(0.0, self.x[6])

        v_mag = np.linalg.norm(vel)
        if v_mag > 0.01:
            drag_accel = -k * v_mag * vel
        else:
            drag_accel = np.zeros(3)

        total_accel = self.g_vec + drag_accel

        x_pred = np.zeros(7)
        x_pred[0:3] = pos + vel * dt + 0.5 * total_accel * dt * dt
        x_pred[3:6] = vel + total_accel * dt
        x_pred[6] = k
        return x_pred

    def _build_F(self, dt: float) -> np.ndarray:
        """Jacobian of state transition (7x7).

        Analytical Jacobian for the drag model:
        dp/dp = I, dp/dv = dt*I + 0.5*dt²*d(a_drag)/dv
        dv/dp = 0, dv/dv = I + dt*d(a_drag)/dv
        dp/dk = 0.5*dt²*d(a_drag)/dk
        dv/dk = dt*d(a_drag)/dk
        dk/dk = 1
        """
        F = np.eye(7)
        vel = self.x[3:6]
        k = max(0.0, self.x[6])
        v_mag = np.linalg.norm(vel)

        # d(a_drag)/dv = -k * (|v|*I + v*v^T/|v|) for |v| > eps
        # d(a_drag)/dk = -|v|*v
        if v_mag > 0.01:
            I3 = np.eye(3)
            vvT = np.outer(vel, vel)
            da_dv = -k * (v_mag * I3 + vvT / v_mag)
            da_dk = -v_mag * vel
        else:
            da_dv = np.zeros((3, 3))
            da_dk = np.zeros(3)

        # dp/dv
        F[0:3, 3:6] = dt * np.eye(3) + 0.5 * dt * dt * da_dv
        # dv/dv
        F[3:6, 3:6] = np.eye(3) + dt * da_dv
        # dp/dk
        F[0:3, 6] = 0.5 * dt * dt * da_dk
        # dv/dk
        F[3:6, 6] = dt * da_dk

        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise: standard kinematic Q for pos/vel + random walk for k."""
        I3 = np.eye(3)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = np.zeros((7, 7))
        Q[0:3, 0:3] = (dt4 / 4.0) * I3
        Q[0:3, 3:6] = (dt3 / 2.0) * I3
        Q[3:6, 0:3] = (dt3 / 2.0) * I3
        Q[3:6, 3:6] = dt2 * I3
        Q[0:6, 0:6] *= self.sigma_a ** 2

        # Drag coefficient random walk
        Q[6, 6] = self.sigma_k ** 2 * dt
        return Q

    def _build_R(self, z_depth: float) -> np.ndarray:
        """Calibrated measurement noise."""
        z = max(0.5, abs(z_depth))
        Rx = 0.000005 * (z ** 0.96)
        Ry = 0.000006 * (z ** 2.11)
        Rz = 0.000046 * (z ** 2.85)
        return np.diag([Rx, Ry, Rz])

    def predict(self, dt: float):
        if dt <= 0:
            return
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = self._predict_state(dt)
        # Clamp k_drag >= 0
        self.x[6] = max(0.0, self.x[6])
        self.P = F @ self.P @ F.T + Q

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        if not self._initialized:
            self.x[0:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._diagnostics['innovation'] = np.zeros(3)
            self._diagnostics['S'] = np.eye(3)
            self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P)[:6], np.zeros(3)])
            return self.get_state()

        # H = [I3, 0_3x3, 0_3x1] -> 3x7
        H = np.zeros((3, 7))
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
            I_KH = np.eye(7) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            # Clamp k_drag >= 0
            self.x[6] = max(0.0, self.x[6])

        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P)[:6], np.zeros(3)])

        return self.get_state()

    def get_state(self) -> FilterState:
        # Compute current acceleration (gravity + drag)
        vel = self.x[3:6]
        k = max(0.0, self.x[6])
        v_mag = np.linalg.norm(vel)
        if v_mag > 0.01:
            drag_accel = -k * v_mag * vel
        else:
            drag_accel = np.zeros(3)
        accel = self.g_vec + drag_accel

        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
            ax=accel[0], ay=accel[1], az=accel[2],
        )
