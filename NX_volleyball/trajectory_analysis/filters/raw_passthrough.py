"""Raw passthrough filter — baseline with numerical differentiation."""

import numpy as np
from .base import FilterBase, FilterState


class RawPassthrough(FilterBase):
    """Baseline filter: directly outputs observations with numerical velocity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RawPassthrough"
        self.reset()

    def reset(self):
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._acc = np.zeros(3)
        self._prev_pos = None
        self._prev_vel = None
        self._prev_dt = None
        self._initialized = False

    def predict(self, dt: float):
        # For raw passthrough, predict just propagates last known state
        if self._initialized and dt > 0:
            self._pos = self._pos + self._vel * dt

    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        new_pos = np.array([obs_x, obs_y, obs_z])

        if self._initialized and self._prev_pos is not None:
            # Estimate dt from history or use a default
            dt = self._prev_dt if self._prev_dt and self._prev_dt > 0 else 1.0 / 60.0
            self._vel = (new_pos - self._prev_pos) / dt

            if self._prev_vel is not None:
                self._acc = (self._vel - self._prev_vel) / dt

        self._prev_vel = self._vel.copy()
        self._prev_pos = self._pos.copy()
        self._pos = new_pos
        self._initialized = True

        # Update diagnostics
        self._diagnostics['innovation'] = np.zeros(3)
        self._diagnostics['S'] = np.eye(3)
        self._diagnostics['P_diag'] = np.zeros(9)

        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self._pos[0], y=self._pos[1], z=self._pos[2],
            vx=self._vel[0], vy=self._vel[1], vz=self._vel[2],
            ax=self._acc[0], ay=self._acc[1], az=self._acc[2],
        )

    def process_segment(self, frames) -> np.ndarray:
        """Override to track dt properly for numerical differentiation."""
        self.reset()
        n = len(frames)
        results = np.zeros((n, 9))

        first = frames[0]
        self._pos = np.array([first.obs_x, first.obs_y, first.obs_z])
        self._initialized = True
        self._prev_pos = self._pos.copy()
        results[0] = self.get_state().as_array()

        for i in range(1, n):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            dt = curr_frame.timestamp - prev_frame.timestamp
            if dt <= 0:
                dt = 1.0 / 60.0

            self._prev_dt = dt
            new_pos = np.array([curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z])

            # Numerical differentiation
            self._vel = (new_pos - self._prev_pos) / dt
            if self._prev_vel is not None:
                self._acc = (self._vel - self._prev_vel) / dt

            self._prev_vel = self._vel.copy()
            self._prev_pos = self._pos.copy()
            self._pos = new_pos

            results[i] = self.get_state().as_array()

        return results
