"""One-Euro Filter - speed-adaptive low-pass filter.

Reference: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems" (CHI 2012)

Key idea: cutoff frequency increases with speed → low latency during fast
motion, high smoothing when stationary.
"""

import numpy as np
from .base import FilterBase, FilterState


class OneEuroFilter(FilterBase):
    """3D One-Euro filter with independent axes."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_hat = np.zeros(3)
        self.dx_hat = np.zeros(3)
        self.initialized = False
        self.last_t = 0.0

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def predict(self, dt: float):
        pass  # One-Euro doesn't have a predict step

    def update(self, obs_x, obs_y, obs_z) -> FilterState:
        x = np.array([obs_x, obs_y, obs_z])

        if not self.initialized:
            self.x_hat = x.copy()
            self.dx_hat = np.zeros(3)
            self.initialized = True
            self.last_t = 0.0
            return self.get_state()

        dt = 1.0 / 60.0  # Will be overridden by process_segment timing

        # Estimate derivatives
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_hat) / dt
        self.dx_hat = a_d * dx + (1 - a_d) * self.dx_hat

        # Adaptive cutoff based on speed
        speed = np.abs(self.dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter position
        a = np.array([self._alpha(c, dt) for c in cutoff])
        self.x_hat = a * x + (1 - a) * self.x_hat

        return self.get_state()

    def get_state(self) -> FilterState:
        return FilterState(
            x=self.x_hat[0], y=self.x_hat[1], z=self.x_hat[2],
            vx=self.dx_hat[0], vy=self.dx_hat[1], vz=self.dx_hat[2],
        )

    def process_segment(self, frames) -> np.ndarray:
        """Override to pass actual dt to update."""
        self.reset()
        n = len(frames)
        results = np.zeros((n, 9))

        first = frames[0]
        self.x_hat = np.array([first.obs_x, first.obs_y, first.obs_z])
        self.dx_hat = np.zeros(3)
        self.initialized = True
        results[0] = self.get_state().as_array()

        for i in range(1, n):
            dt = frames[i].timestamp - frames[i-1].timestamp
            if dt <= 0:
                dt = 1.0 / 60.0

            x = np.array([frames[i].obs_x, frames[i].obs_y, frames[i].obs_z])

            # Derivative estimation
            a_d = self._alpha(self.d_cutoff, dt)
            dx = (x - self.x_hat) / dt
            self.dx_hat = a_d * dx + (1 - a_d) * self.dx_hat

            # Adaptive cutoff
            speed = np.abs(self.dx_hat)
            cutoff = self.min_cutoff + self.beta * speed

            # Smooth position
            a = np.array([self._alpha(c, dt) for c in cutoff])
            self.x_hat = a * x + (1 - a) * self.x_hat

            results[i] = self.get_state().as_array()

        return results
