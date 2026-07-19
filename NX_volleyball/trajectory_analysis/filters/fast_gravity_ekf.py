"""Fast-convergence Gravity EKF with two-phase parameter scheduling.

Key improvements over standard gravity_ekf_6d:
1. Multi-frame differential velocity initialization (first 3 frames)
2. Boost phase: aggressive R/Q for rapid convergence (frames 3-15)
3. Steady phase: standard parameters for smooth tracking
4. Speed-limit divergence protection with parabolic re-initialization
"""

import numpy as np
from .base import FilterBase, FilterState


class FastGravityEKF(FilterBase):
    """6-state gravity EKF with fast-start two-phase parameter scheduling.

    State: [x, y, z, vx, vy, vz]
    """

    PHASE_INIT = 0    # Collecting first observations for velocity init
    PHASE_BOOST = 1   # Aggressive parameters for fast convergence
    PHASE_STEADY = 2  # Normal operation

    def __init__(self,
                 # Steady-state params
                 sigma_a: float = 5.0,
                 R_base: float = 0.015,
                 noise_exponent: float = 2.85,
                 innovation_gate: float = 25.0,
                 # Boost phase params
                 boost_R_base: float = 0.005,
                 boost_sigma_a: float = 50.0,
                 boost_gate: float = 100.0,
                 boost_frames: int = 10,
                 # Init params
                 init_frames: int = 5,
                 # Divergence protection
                 max_speed: float = 15.0,
                 # Physics
                 gravity_vec: list = None,
                 focal: float = 727.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = "FastGravityEKF"

        # Steady params
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate

        # Boost params
        self.boost_R_base = boost_R_base
        self.boost_sigma_a = boost_sigma_a
        self.boost_gate = boost_gate
        self.boost_frames = boost_frames

        # Init
        self.init_frames = init_frames
        self.max_speed = max_speed
        self.focal = focal

        if gravity_vec is not None:
            self.g_vec = np.array(gravity_vec, dtype=float)
        else:
            self.g_vec = np.array([0.0, 9.81, 0.0])

        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100.0
        self.phase = self.PHASE_INIT
        self.frame_count = 0
        self.init_buffer = []  # [(timestamp_relative, obs_xyz)]
        self._last_dt = 1.0 / 60.0
        self._cumulative_dt = 0.0

    def _get_params(self):
        """Get current R_base, sigma_a, gate based on phase and frame count."""
        if self.phase == self.PHASE_BOOST:
            # Linear interpolation from boost to steady over boost_frames
            progress = min(1.0, (self.frame_count - self.init_frames) / self.boost_frames)
            r = self.boost_R_base * (1 - progress) + self.R_base * progress
            sa = self.boost_sigma_a * (1 - progress) + self.sigma_a * progress
            gate = self.boost_gate * (1 - progress) + self.innovation_gate * progress
            return r, sa, gate
        return self.R_base, self.sigma_a, self.innovation_gate

    def _init_velocity(self):
        """Estimate initial velocity from init_buffer using least-squares parabola fit."""
        if len(self.init_buffer) < 2:
            return np.zeros(3)

        # Fit: obs(t) = p0 + v0*t + 0.5*g*t^2
        # Rearrange: obs(t) - 0.5*g*t^2 = p0 + v0*t
        # Least squares for [p0, v0] per axis
        times = np.array([b[0] for b in self.init_buffer])
        positions = np.array([b[1] for b in self.init_buffer])

        t_ref = times[-1]  # Reference to last frame
        dts = times - t_ref  # All negative or zero

        A = np.column_stack([np.ones(len(dts)), dts])

        pos0 = np.zeros(3)
        vel0 = np.zeros(3)
        for axis in range(3):
            b = positions[:, axis] - 0.5 * self.g_vec[axis] * dts**2
            result = np.linalg.lstsq(A, b, rcond=None)
            pos0[axis] = result[0][0]
            vel0[axis] = result[0][1]

        return vel0

    def _build_Q(self, dt: float, sigma_a: float) -> np.ndarray:
        I3 = np.eye(3)
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (dt4 / 4.0) * I3
        Q[0:3, 3:6] = (dt3 / 2.0) * I3
        Q[3:6, 0:3] = (dt3 / 2.0) * I3
        Q[3:6, 3:6] = dt2 * I3
        Q *= sigma_a ** 2
        return Q

    def _build_R(self, z_depth: float, r_base: float) -> np.ndarray:
        z_clamped = max(1.0, abs(z_depth))
        Rz = r_base * (z_clamped ** self.noise_exponent)
        Rxy = Rz * (z_clamped ** 2) / (self.focal ** 2) + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def _predict_state(self, dt: float) -> np.ndarray:
        x_pred = np.zeros(6)
        x_pred[0] = self.x[0] + self.x[3] * dt + 0.5 * self.g_vec[0] * dt**2
        x_pred[1] = self.x[1] + self.x[4] * dt + 0.5 * self.g_vec[1] * dt**2
        x_pred[2] = self.x[2] + self.x[5] * dt + 0.5 * self.g_vec[2] * dt**2
        x_pred[3] = self.x[3] + self.g_vec[0] * dt
        x_pred[4] = self.x[4] + self.g_vec[1] * dt
        x_pred[5] = self.x[5] + self.g_vec[2] * dt
        return x_pred

    def predict(self, dt: float):
        if dt <= 0:
            return
        self._last_dt = dt
        self._cumulative_dt += dt

        if self.phase == self.PHASE_INIT:
            return  # Don't predict during init collection

        _, sigma_a, _ = self._get_params()
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        Q = self._build_Q(dt, sigma_a)

        self.x = self._predict_state(dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        obs = np.array([obs_x, obs_y, obs_z])
        self.frame_count += 1

        # Phase: INIT — collect observations for velocity estimation
        if self.phase == self.PHASE_INIT:
            self.init_buffer.append((self._cumulative_dt, obs.copy()))

            if len(self.init_buffer) >= self.init_frames:
                # Initialize EKF with fitted velocity
                vel0 = self._init_velocity()
                self.x[0:3] = obs
                self.x[3:6] = vel0
                # Set P: position well-known, velocity VERY uncertain
                # Large P_vel ensures strong velocity corrections during BOOST
                self.P = np.eye(6)
                self.P[0:3, 0:3] *= 0.05  # position: ~0.22m std
                self.P[3:6, 3:6] *= 25.0  # velocity: 5.0 m/s std
                self.phase = self.PHASE_BOOST
            else:
                # Not enough frames yet, just store
                self.x[0:3] = obs

            self._diagnostics['innovation'] = np.zeros(3)
            self._diagnostics['S'] = np.eye(3)
            self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])
            return self.get_state()

        # Phase: BOOST or STEADY — standard EKF update
        r_base, _, gate = self._get_params()

        # Transition from BOOST to STEADY
        if self.phase == self.PHASE_BOOST:
            if self.frame_count - self.init_frames >= self.boost_frames:
                self.phase = self.PHASE_STEADY

        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)

        R = self._build_R(obs_z, r_base)
        y = obs - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= gate:
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Divergence protection: speed limit
        speed = np.linalg.norm(self.x[3:6])
        if speed > self.max_speed:
            # Re-init from recent observation with zero velocity
            self.x[0:3] = obs
            self.x[3:6] = np.zeros(3)
            self.P[3:6, 3:6] = np.eye(3) * 25.0
            self.phase = self.PHASE_BOOST
            self.frame_count = self.init_frames  # restart boost countdown

        self._diagnostics['innovation'] = y.copy()
        self._diagnostics['S'] = S.copy()
        self._diagnostics['P_diag'] = np.concatenate([np.diag(self.P), np.zeros(3)])

        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
            ax=self.g_vec[0], ay=self.g_vec[1], az=self.g_vec[2],
        )
