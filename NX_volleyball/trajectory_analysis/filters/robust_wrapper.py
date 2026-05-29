"""Robust engineering wrapper for any trajectory filter.

Implements techniques from ball-tracking literature:
1. Observation outlier pre-filter (velocity-based plausibility + median)
2. Tracking state machine (INIT → TRACKING → COAST → RESET)
3. Hard state reset on re-acquisition (not gradual P inflation)
4. Speed limit protection with full reset
"""

import numpy as np
from collections import deque
from .base import FilterBase, FilterState


class RobustWrapper(FilterBase):
    """Wraps any FilterBase with engineering robustness layers.
    
    Parameters:
        inner_filter: The filter to wrap (must be already constructed)
        max_speed: Speed limit (m/s), reset if exceeded
        max_jump: Max observation jump per frame (m), reject if exceeded
        median_window: Sliding window size for median pre-filter (0=disabled)
        coast_limit: Max consecutive predict-only frames before reset
        init_frames: Frames to collect before starting filter
        physics_init: Use gravity-model parabola fit for initial velocity
        gravity_vec: Gravity vector for physics init
    """

    STATE_INIT = 0
    STATE_TRACKING = 1
    STATE_COAST = 2

    def __init__(self,
                 inner_filter: FilterBase,
                 max_speed: float = 15.0,
                 max_jump: float = 2.0,
                 median_window: int = 3,
                 coast_limit: int = 5,
                 init_frames: int = 3,
                 physics_init: bool = True,
                 gravity_vec: list = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.inner = inner_filter
        self.name = f"robust_{inner_filter.name}"
        self.max_speed = max_speed
        self.max_jump = max_jump
        self.median_window = median_window
        self.coast_limit = coast_limit
        self.init_frames_count = init_frames
        self.physics_init = physics_init
        
        if gravity_vec is not None:
            self.g_vec = np.array(gravity_vec, dtype=float)
        else:
            self.g_vec = np.array([0.0, 9.81, 0.0])
        
        self.reset()

    def reset(self):
        self.inner.reset()
        self.state = self.STATE_INIT
        self.frame_idx = 0
        self.coast_count = 0
        self.init_buffer = []  # [(cumulative_dt, obs_xyz)]
        self._obs_history = deque(maxlen=max(self.median_window, 3))
        self._last_accepted_obs = None
        self._cumulative_dt = 0.0
        self._last_state = FilterState()

    def predict(self, dt: float):
        self._cumulative_dt += dt
        if self.state != self.STATE_INIT:
            self.inner.predict(dt)

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        obs = np.array([obs_x, obs_y, obs_z])
        self.frame_idx += 1
        
        # --- Phase: INIT ---
        if self.state == self.STATE_INIT:
            self.init_buffer.append((self._cumulative_dt, obs.copy()))
            self._obs_history.append(obs.copy())
            self._last_accepted_obs = obs.copy()
            
            if len(self.init_buffer) >= self.init_frames_count:
                self._do_init()
                self.state = self.STATE_TRACKING
            else:
                # Return raw observation as state during init
                self._last_state = FilterState(
                    x=obs[0], y=obs[1], z=obs[2])
                self._diagnostics['innovation'] = np.zeros(3)
                self._diagnostics['S'] = np.eye(3)
                self._diagnostics['P_diag'] = np.zeros(9)
                return self._last_state
            
            self._last_state = self.inner.get_state()
            return self._last_state
        
        # --- Outlier rejection ---
        is_outlier = self._check_outlier(obs)
        
        if is_outlier:
            # Coast: don't update, predict only
            self.coast_count += 1
            if self.coast_count >= self.coast_limit:
                # Too many consecutive outliers → hard reset
                self._hard_reset(obs)
            self._last_state = self.inner.get_state()
            return self._last_state
        
        # --- Observation accepted ---
        self.coast_count = 0
        
        # Median pre-filter
        filtered_obs = self._median_filter(obs)
        
        # Update inner filter
        state = self.inner.update(filtered_obs[0], filtered_obs[1], filtered_obs[2])
        
        # Speed limit check
        speed = np.linalg.norm(state.velocity())
        if speed > self.max_speed:
            self._hard_reset(obs)
            state = self.inner.get_state()
        
        self._last_accepted_obs = obs.copy()
        self._last_state = state
        self._diagnostics = self.inner.get_diagnostics()
        return state

    def get_state(self) -> FilterState:
        if self.state == self.STATE_INIT:
            return self._last_state
        return self.inner.get_state()

    def _check_outlier(self, obs: np.ndarray) -> bool:
        """Check if observation is an outlier based on jump distance."""
        if self._last_accepted_obs is None:
            return False
        
        jump = np.linalg.norm(obs - self._last_accepted_obs)
        
        # Dynamic threshold: max_jump scales with time since last accepted obs
        # Allow larger jumps if we've been coasting (ball might have moved)
        effective_limit = self.max_jump * (1.0 + 0.5 * self.coast_count)
        
        return jump > effective_limit

    def _median_filter(self, obs: np.ndarray) -> np.ndarray:
        """Apply sliding window median filter to observation."""
        self._obs_history.append(obs.copy())
        
        if self.median_window <= 1 or len(self._obs_history) < self.median_window:
            return obs
        
        window = np.array(list(self._obs_history)[-self.median_window:])
        return np.median(window, axis=0)

    def _do_init(self):
        """Initialize inner filter using collected buffer."""
        self.inner.reset()
        
        if self.physics_init and len(self.init_buffer) >= 3:
            # Physics-based initialization: fit parabola to get velocity
            vel0 = self._fit_initial_velocity()
            
            # Feed all buffered frames to inner filter
            for i, (t, obs) in enumerate(self.init_buffer):
                if i > 0:
                    dt = self.init_buffer[i][0] - self.init_buffer[i-1][0]
                    if dt > 0:
                        self.inner.predict(dt)
                self.inner.update(obs[0], obs[1], obs[2])
            
            # Override velocity if inner filter has state vector
            if hasattr(self.inner, 'x') and len(self.inner.x) >= 6:
                self.inner.x[3:6] = vel0
                # Keep P_vel moderate to allow correction
                if hasattr(self.inner, 'P'):
                    self.inner.P[3:6, 3:6] = np.eye(3) * 9.0
        else:
            # Simple init: just feed frames sequentially
            for i, (t, obs) in enumerate(self.init_buffer):
                if i > 0:
                    dt = self.init_buffer[i][0] - self.init_buffer[i-1][0]
                    if dt > 0:
                        self.inner.predict(dt)
                self.inner.update(obs[0], obs[1], obs[2])

    def _fit_initial_velocity(self) -> np.ndarray:
        """Fit parabola to init_buffer to estimate initial velocity."""
        times = np.array([b[0] for b in self.init_buffer])
        positions = np.array([b[1] for b in self.init_buffer])
        
        # Reference to last frame
        t_ref = times[-1]
        dts = times - t_ref  # negative or zero
        
        # obs(t) = p0 + v0*t + 0.5*g*t²
        # obs(t) - 0.5*g*t² = p0 + v0*t
        A = np.column_stack([np.ones(len(dts)), dts])
        
        vel0 = np.zeros(3)
        for axis in range(3):
            b = positions[:, axis] - 0.5 * self.g_vec[axis] * dts**2
            result = np.linalg.lstsq(A, b, rcond=None)
            vel0[axis] = result[0][1]
        
        return vel0

    def _hard_reset(self, obs: np.ndarray):
        """Hard reset: reinitialize filter from current observation."""
        self.inner.reset()
        self.inner.update(obs[0], obs[1], obs[2])
        self._last_accepted_obs = obs.copy()
        self._obs_history.clear()
        self._obs_history.append(obs.copy())
        self.coast_count = 0
        # Reset to TRACKING (not INIT) for speed - single frame restart
        self.state = self.STATE_TRACKING
