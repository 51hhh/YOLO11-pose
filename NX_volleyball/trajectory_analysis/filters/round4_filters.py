"""Round 4 filter explorations:
A. fallback_v2: Improved outlier handling (gate=25, progressive P inflate)
B. adaptive_q_one_euro: Q-adaptive EKF + One-Euro post-smooth
C. confidence_ekf: det_confidence modulates R
D. imm_2model: Simplified 2-model IMM (static + gravity)
E. robust_bounce_ekf: Outlier-robust EKF with bounce state machine
"""

import numpy as np
from .base import FilterBase, FilterState


class FallbackV2(FilterBase):
    """Improved gravity EKF fallback with:
    - Gate = 25 (aligned with best EKF configs)
    - Progressive P inflation per consecutive outlier frame
    - Velocity-only update during outlier mode (trust direction, not position)
    """

    def __init__(self, sigma_a=5.0, R_base=0.015, noise_exponent=2.85,
                 innovation_gate=25.0, P_inflate_per_frame=1.5,
                 max_consecutive_outliers=5,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.P_inflate_per_frame = P_inflate_per_frame
        self.max_outliers = max_consecutive_outliers
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self._initialized = False
        self._outlier_count = 0

    def _build_Q(self, dt):
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
        return Q

    def _build_R(self, oz):
        z_c = max(1.0, abs(oz))
        Rz = self.R_base * (z_c ** self.noise_exponent)
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def predict(self, dt: float):
        self.x[3:] += self.g_vec * dt
        self.x[:3] += self.x[3:] * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + self._build_Q(dt)

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._outlier_count = 0
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            # Normal update
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
            self._outlier_count = 0
        else:
            # Outlier: inflate P progressively
            self._outlier_count += 1
            self.P *= self.P_inflate_per_frame

            if self._outlier_count >= self.max_outliers:
                # Force re-acquisition
                S_new = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S_new)
                self.x += K @ y
                IKH = np.eye(6) - K @ H
                self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
                self._outlier_count = 0

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S
        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )


class AdaptiveQOneEuro(FilterBase):
    """Gravity EKF with Q-adaptive estimation + One-Euro post-smoothing.

    Key difference from adaptive_ekf: estimates Q scaling factor (not R),
    then applies One-Euro on output for final polish.
    """

    def __init__(self, sigma_a=5.0, R_base=0.3, noise_exponent=2.85,
                 innovation_gate=25.0, window_size=20,
                 min_cutoff=0.005, beta=0.0,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.window_size = window_size
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self._initialized = False
        self.Q_scale = 1.0
        self._innovations = []
        self._S_history = []
        # One-Euro state
        self._oe_x = None
        self._oe_dx = np.zeros(3)
        self._prev_t = None

    def _build_Q(self, dt):
        q = (self.sigma_a * self.Q_scale) ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
        return Q

    def _build_R(self, oz):
        z_c = max(1.0, abs(oz))
        Rz = self.R_base * (z_c ** self.noise_exponent)
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def _adapt_Q(self):
        """Adapt Q scale from innovation covariance mismatch."""
        if len(self._innovations) < self.window_size:
            return
        innov = np.array(self._innovations[-self.window_size:])
        C_innov = np.mean([np.outer(v, v) for v in innov], axis=0)
        S_mean = np.mean(self._S_history[-self.window_size:], axis=0)
        # If innovations are larger than predicted S, Q is too small
        ratio = np.trace(C_innov) / (np.trace(S_mean) + 1e-10)
        # Smooth update
        self.Q_scale = 0.9 * self.Q_scale + 0.1 * np.clip(ratio, 0.1, 5.0)

    def _one_euro_alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _apply_one_euro(self, pos, dt):
        if self._oe_x is None:
            self._oe_x = pos.copy()
            return pos.copy()

        # Derivative
        d_cutoff = 1.0
        a_d = self._one_euro_alpha(d_cutoff, dt)
        dx = (pos - self._oe_x) / dt
        self._oe_dx = a_d * dx + (1 - a_d) * self._oe_dx

        # Adaptive cutoff
        speed = np.linalg.norm(self._oe_dx)
        cutoff = self.min_cutoff + self.beta * speed

        a = self._one_euro_alpha(cutoff, dt)
        self._oe_x = a * pos + (1 - a) * self._oe_x
        return self._oe_x.copy()

    def predict(self, dt: float):
        self.x[3:] += self.g_vec * dt
        self.x[:3] += self.x[3:] * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + self._build_Q(dt)
        self._prev_dt = dt

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._prev_dt = 1.0 / 60.0
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        self._innovations.append(y.copy())
        self._S_history.append(S.copy())
        self._adapt_Q()

        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Apply One-Euro on EKF output
        dt = getattr(self, '_prev_dt', 1.0/60.0)
        smoothed = self._apply_one_euro(self.x[:3], dt)

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S
        # Store smoothed position for output
        self._smoothed_pos = smoothed
        return self.get_state()

    def get_state(self) -> FilterState:
        pos = getattr(self, '_smoothed_pos', self.x[:3])
        return FilterState(
            x=pos[0], y=pos[1], z=pos[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )


class ConfidenceEKF(FilterBase):
    """Gravity EKF with detection confidence modulating R.

    R_effective = R_base / (confidence + eps)^gamma
    Low confidence → large R → trust model more.
    + One-Euro post-smoothing.
    """

    def __init__(self, sigma_a=5.0, R_base=0.3, noise_exponent=2.85,
                 innovation_gate=25.0, gamma=1.0, eps=0.1,
                 min_cutoff=0.005, beta=0.0,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.gamma = gamma
        self.eps = eps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self._initialized = False
        self._confidence = 0.9
        self._oe_x = None
        self._oe_dx = np.zeros(3)
        self._prev_dt = 1.0 / 60.0

    def _build_Q(self, dt):
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
        return Q

    def _build_R(self, oz, confidence):
        z_c = max(1.0, abs(oz))
        conf_scale = 1.0 / (confidence + self.eps) ** self.gamma
        Rz = self.R_base * (z_c ** self.noise_exponent) * conf_scale
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def _one_euro_alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _apply_one_euro(self, pos, dt):
        if self._oe_x is None:
            self._oe_x = pos.copy()
            return pos.copy()
        d_cutoff = 1.0
        a_d = self._one_euro_alpha(d_cutoff, dt)
        dx = (pos - self._oe_x) / dt
        self._oe_dx = a_d * dx + (1 - a_d) * self._oe_dx
        speed = np.linalg.norm(self._oe_dx)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._one_euro_alpha(cutoff, dt)
        self._oe_x = a * pos + (1 - a) * self._oe_x
        return self._oe_x.copy()

    def predict(self, dt: float):
        self.x[3:] += self.g_vec * dt
        self.x[:3] += self.x[3:] * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + self._build_Q(dt)
        self._prev_dt = dt

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        # Standard update (confidence set externally via set_confidence)
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z, self._confidence)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        dt = self._prev_dt
        smoothed = self._apply_one_euro(self.x[:3], dt)
        self._smoothed_pos = smoothed

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S
        return self.get_state()

    def set_confidence(self, confidence: float):
        """Set detection confidence for next update."""
        self._confidence = confidence

    def process_segment(self, frames) -> np.ndarray:
        """Override to pass confidence per frame."""
        self.reset()
        n = len(frames)
        results = np.zeros((n, 9))

        first = frames[0]
        self.set_confidence(first.det_confidence)
        self.update(first.obs_x, first.obs_y, first.obs_z)
        results[0] = self.get_state().as_array()

        for i in range(1, n):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            frame_gap = curr_frame.frame_id - prev_frame.frame_id
            total_dt = curr_frame.timestamp - prev_frame.timestamp

            if total_dt <= 0 or frame_gap <= 0:
                self.set_confidence(curr_frame.det_confidence)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            elif frame_gap > 1:
                step_dt = total_dt / frame_gap
                for _ in range(frame_gap):
                    self.predict(step_dt)
                self.set_confidence(curr_frame.det_confidence)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            else:
                self.predict(total_dt)
                self.set_confidence(curr_frame.det_confidence)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)

            results[i] = state.as_array()
        return results

    def get_state(self) -> FilterState:
        pos = getattr(self, '_smoothed_pos', self.x[:3])
        return FilterState(
            x=pos[0], y=pos[1], z=pos[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )


class IMM2Model(FilterBase):
    """Simplified 2-model IMM: Static + Gravity flight.

    Removes the high-maneuver model which adds noise.
    Transition probabilities tuned for volleyball.
    """

    def __init__(self, R_base=0.015, noise_exponent=2.85,
                 innovation_gate=25.0,
                 sigma_static=0.3, sigma_flight=5.0,
                 p_stay_static=0.97, p_stay_flight=0.97,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.sigma = [sigma_static, sigma_flight]
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.n_models = 2
        self.TPM = np.array([
            [p_stay_static, 1 - p_stay_static],
            [1 - p_stay_flight, p_stay_flight],
        ])
        self.reset()

    def reset(self):
        self.x = [np.zeros(6) for _ in range(self.n_models)]
        self.P = [np.eye(6) for _ in range(self.n_models)]
        for p in self.P:
            p[3:, 3:] *= 10.0
        self.mu = np.array([0.5, 0.5])
        self._initialized = False

    def _make_Q(self, sigma_a, dt):
        q = sigma_a ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
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
            xj = self.x[j].copy()
            xj[3:] += self.g_vec * dt
            xj[:3] += xj[3:] * dt
            self.x[j] = xj
            self.P[j] = F @ self.P[j] @ F.T + self._make_Q(self.sigma[j], dt)

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            for j in range(self.n_models):
                self.x[j][:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        R = self._compute_R(obs_z)
        z = np.array([obs_x, obs_y, obs_z])

        # IMM Mixing
        c_bar = self.TPM.T @ self.mu
        mu_mix = np.zeros((self.n_models, self.n_models))
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_mix[i, j] = self.TPM[i, j] * self.mu[i] / (c_bar[j] + 1e-30)

        x_mixed, P_mixed = [], []
        for j in range(self.n_models):
            xm = sum(mu_mix[i, j] * self.x[i] for i in range(self.n_models))
            x_mixed.append(xm)
            Pm = np.zeros((6, 6))
            for i in range(self.n_models):
                diff = self.x[i] - xm
                Pm += mu_mix[i, j] * (self.P[i] + np.outer(diff, diff))
            P_mixed.append(Pm)

        # Model update
        likelihoods = np.zeros(self.n_models)
        for j in range(self.n_models):
            self.x[j] = x_mixed[j]
            self.P[j] = P_mixed[j]
            y = z - H @ self.x[j]
            S = H @ self.P[j] @ H.T + R
            S_inv = np.linalg.inv(S)
            maha2 = y @ S_inv @ y
            det_S = np.linalg.det(S)
            likelihoods[j] = np.exp(-0.5 * maha2) / (np.sqrt((2*np.pi)**3 * det_S) + 1e-30)

            if maha2 <= self.innovation_gate:
                K = self.P[j] @ H.T @ S_inv
                self.x[j] += K @ y
                IKH = np.eye(6) - K @ H
                self.P[j] = IKH @ self.P[j] @ IKH.T + K @ R @ K.T

        # Mode probability update
        self.mu = c_bar * likelihoods
        mu_sum = np.sum(self.mu)
        if mu_sum > 1e-30:
            self.mu /= mu_sum
        else:
            self.mu = np.ones(self.n_models) / self.n_models

        self._diagnostics['innovation'] = z - H @ self._combined_state()
        self._diagnostics['S'] = np.eye(3)
        return self.get_state()

    def _combined_state(self):
        return sum(self.mu[j] * self.x[j] for j in range(self.n_models))

    def get_state(self) -> FilterState:
        x = self._combined_state()
        return FilterState(
            x=x[0], y=x[1], z=x[2],
            vx=x[3], vy=x[4], vz=x[5],
        )


class RobustBounceEKF(FilterBase):
    """Outlier-robust gravity EKF with bounce state machine.

    States: TRACKING → OUTLIER → BOUNCE_DETECT → REACQUIRE

    - Normal: standard EKF update
    - Outlier (maha² > gate): skip update, inflate P
    - If vy reverses direction for N consecutive frames after outlier: bounce detected
    - Bounce: reverse vy, inflate P_vy, resume tracking
    """

    def __init__(self, sigma_a=5.0, R_base=0.015, noise_exponent=2.85,
                 innovation_gate=25.0, bounce_confirm_frames=2,
                 restitution=0.75, P_bounce_inflate=50.0,
                 max_outlier_frames=5,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.bounce_confirm = bounce_confirm_frames
        self.restitution = restitution
        self.P_bounce = P_bounce_inflate
        self.max_outlier = max_outlier_frames
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self._initialized = False
        self._state = 'TRACKING'  # TRACKING, OUTLIER
        self._outlier_count = 0
        self._vy_before_outlier = 0.0
        self._vy_reversal_count = 0

    def _build_Q(self, dt):
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
        return Q

    def _build_R(self, oz):
        z_c = max(1.0, abs(oz))
        Rz = self.R_base * (z_c ** self.noise_exponent)
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def predict(self, dt: float):
        self.x[3:] += self.g_vec * dt
        self.x[:3] += self.x[3:] * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + self._build_Q(dt)

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self._initialized = True
            self._state = 'TRACKING'
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if self._state == 'TRACKING':
            if maha2 <= self.innovation_gate:
                K = self.P @ H.T @ S_inv
                self.x += K @ y
                IKH = np.eye(6) - K @ H
                self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
            else:
                # Enter outlier mode
                self._state = 'OUTLIER'
                self._outlier_count = 1
                self._vy_before_outlier = self.x[4]
                self.P *= 1.5

        elif self._state == 'OUTLIER':
            self._outlier_count += 1

            # Check if vy has reversed (bounce signature)
            if self.x[4] * self._vy_before_outlier < 0:
                self._vy_reversal_count += 1
            else:
                self._vy_reversal_count = 0

            if self._vy_reversal_count >= self.bounce_confirm:
                # Bounce detected! Reverse vy with restitution
                self.x[4] = -self._vy_before_outlier * self.restitution
                self.P[4, 4] = self.P_bounce
                self._state = 'TRACKING'
                self._outlier_count = 0
                self._vy_reversal_count = 0
                # Try to update with current obs
                y2 = z - H @ self.x
                S2 = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S2)
                self.x += K @ y2
                IKH = np.eye(6) - K @ H
                self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

            elif self._outlier_count >= self.max_outlier:
                # Give up outlier mode, force reacquire
                self.P *= 2.0
                S_new = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S_new)
                self.x += K @ y
                IKH = np.eye(6) - K @ H
                self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
                self._state = 'TRACKING'
                self._outlier_count = 0
                self._vy_reversal_count = 0
            else:
                # Stay in outlier, inflate P
                self.P *= 1.3

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S
        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.x[0], y=self.x[1], z=self.x[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )


class ConfidenceV2(FilterBase):
    """Confidence-modulated R + Velocity-driven One-Euro (V2 architecture).

    Combines:
    - det_confidence scales R (low confidence → trust physics more)
    - EKF velocity directly drives One-Euro cutoff (not position derivative)
    - beta=0 + low min_cutoff (fixed LPF, proven optimal in Round 3)
    """

    def __init__(self, sigma_a=5.0, R_base=1.5, noise_exponent=2.85,
                 innovation_gate=25.0, gamma=1.0, eps=0.1,
                 min_cutoff=0.005, beta=0.0,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.gamma = gamma
        self.eps = eps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self._initialized = False
        self._confidence = 0.9
        self.smooth_pos = np.zeros(3)

    def _build_Q(self, dt):
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
        return Q

    def _build_R(self, oz, confidence):
        z_c = max(1.0, abs(oz))
        conf_scale = 1.0 / (confidence + self.eps) ** self.gamma
        Rz = self.R_base * (z_c ** self.noise_exponent) * conf_scale
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def predict(self, dt: float):
        self.x[3:] += self.g_vec * dt
        self.x[:3] += self.x[3:] * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + self._build_Q(dt)
        self._last_dt = dt

    def set_confidence(self, confidence: float):
        self._confidence = confidence

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self.smooth_pos = self.x[:3].copy()
            self._initialized = True
            self._last_dt = 1.0 / 60.0
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z, self._confidence)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Velocity-driven One-Euro (V2 architecture)
        dt = getattr(self, '_last_dt', 1.0/60.0)
        ekf_vel = self.x[3:6]
        speed = np.abs(ekf_vel)
        cutoff = self.min_cutoff + self.beta * speed
        a = np.array([self._alpha(c, dt) for c in cutoff])
        self.smooth_pos = a * self.x[:3] + (1 - a) * self.smooth_pos

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S
        return self.get_state()

    def process_segment(self, frames) -> np.ndarray:
        """Override to pass confidence per frame."""
        self.reset()
        n = len(frames)
        results = np.zeros((n, 9))

        first = frames[0]
        self.set_confidence(first.det_confidence)
        self.update(first.obs_x, first.obs_y, first.obs_z)
        results[0] = self.get_state().as_array()

        for i in range(1, n):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            frame_gap = curr_frame.frame_id - prev_frame.frame_id
            total_dt = curr_frame.timestamp - prev_frame.timestamp

            if total_dt <= 0 or frame_gap <= 0:
                self.set_confidence(curr_frame.det_confidence)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            elif frame_gap > 1:
                step_dt = total_dt / frame_gap
                for _ in range(frame_gap):
                    self.predict(step_dt)
                self.set_confidence(curr_frame.det_confidence)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            else:
                self.predict(total_dt)
                self.set_confidence(curr_frame.det_confidence)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)

            results[i] = state.as_array()
        return results

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.smooth_pos[0], y=self.smooth_pos[1], z=self.smooth_pos[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )


class AdaptiveQV2(FilterBase):
    """Q-adaptive EKF + Velocity-driven One-Euro (V2 architecture).

    Combines:
    - Innovation-based Q scaling (adapts process noise online)
    - EKF velocity drives One-Euro cutoff
    """

    def __init__(self, sigma_a=5.0, R_base=1.5, noise_exponent=2.85,
                 innovation_gate=25.0, window_size=20,
                 min_cutoff=0.005, beta=0.0,
                 focal=727.0, gravity=9.81, gravity_vec=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma_a = sigma_a
        self.R_base = R_base
        self.noise_exponent = noise_exponent
        self.innovation_gate = innovation_gate
        self.window_size = window_size
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.focal = focal
        self.g_vec = np.array(gravity_vec if gravity_vec else [0, 9.81, 0])
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0
        self._initialized = False
        self.Q_scale = 1.0
        self._innovations = []
        self._S_history = []
        self.smooth_pos = np.zeros(3)

    def _build_Q(self, dt):
        q = (self.sigma_a * self.Q_scale) ** 2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt**4 / 4
            Q[i, i+3] = q * dt**3 / 2
            Q[i+3, i] = q * dt**3 / 2
            Q[i+3, i+3] = q * dt**2
        return Q

    def _build_R(self, oz):
        z_c = max(1.0, abs(oz))
        Rz = self.R_base * (z_c ** self.noise_exponent)
        Rxy = Rz * z_c**2 / self.focal**2 + 0.001
        return np.diag([Rxy, Rxy, Rz])

    def _adapt_Q(self):
        if len(self._innovations) < self.window_size:
            return
        innov = np.array(self._innovations[-self.window_size:])
        C_innov = np.mean([np.outer(v, v) for v in innov], axis=0)
        S_mean = np.mean(self._S_history[-self.window_size:], axis=0)
        ratio = np.trace(C_innov) / (np.trace(S_mean) + 1e-10)
        self.Q_scale = 0.9 * self.Q_scale + 0.1 * np.clip(ratio, 0.1, 5.0)

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def predict(self, dt: float):
        self.x[3:] += self.g_vec * dt
        self.x[:3] += self.x[3:] * dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.P = F @ self.P @ F.T + self._build_Q(dt)
        self._last_dt = dt

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        if not self._initialized:
            self.x[:3] = [obs_x, obs_y, obs_z]
            self.smooth_pos = self.x[:3].copy()
            self._initialized = True
            self._last_dt = 1.0 / 60.0
            return self.get_state()

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        z = np.array([obs_x, obs_y, obs_z])
        R = self._build_R(obs_z)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R

        self._innovations.append(y.copy())
        self._S_history.append(S.copy())
        self._adapt_Q()

        S_inv = np.linalg.inv(S)
        maha2 = y @ S_inv @ y

        if maha2 <= self.innovation_gate:
            K = self.P @ H.T @ S_inv
            self.x += K @ y
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        # Velocity-driven One-Euro (V2 architecture)
        dt = getattr(self, '_last_dt', 1.0/60.0)
        ekf_vel = self.x[3:6]
        speed = np.abs(ekf_vel)
        cutoff = self.min_cutoff + self.beta * speed
        a = np.array([self._alpha(c, dt) for c in cutoff])
        self.smooth_pos = a * self.x[:3] + (1 - a) * self.smooth_pos

        self._diagnostics['innovation'] = y
        self._diagnostics['S'] = S
        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.smooth_pos[0], y=self.smooth_pos[1], z=self.smooth_pos[2],
            vx=self.x[3], vy=self.x[4], vz=self.x[5],
        )
