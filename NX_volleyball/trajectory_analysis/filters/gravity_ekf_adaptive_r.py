"""Gravity EKF with confidence-adaptive R(z)."""

import numpy as np
from .gravity_ekf_6d import GravityEKF6D
from .base import FilterState


class GravityEKFAdaptiveR(GravityEKF6D):
    """6-state gravity EKF with detection confidence scaling R.

    R = R_calibrated / (confidence + 0.1)
    Low confidence => large R => filter trusts prediction more.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GravityEKF_AdaptiveR"
        self._current_confidence = 0.5

    def _build_R(self, z_depth: float) -> np.ndarray:
        z = max(0.5, abs(z_depth))
        Rx = 0.000005 * (z ** 0.96)
        Ry = 0.000006 * (z ** 2.11)
        Rz = 0.000046 * (z ** 2.85)
        conf = getattr(self, '_current_confidence', 0.5)
        scale = 1.0 / (conf + 0.1)
        return np.diag([Rx * scale, Ry * scale, Rz * scale])

    def process_segment(self, frames) -> np.ndarray:
        """Override to pass per-frame detection confidence."""
        self.reset()
        n = len(frames)
        results = np.zeros((n, 9))

        first = frames[0]
        self._current_confidence = first.det_confidence
        self.update(first.obs_x, first.obs_y, first.obs_z)
        results[0] = self.get_state().as_array()

        for i in range(1, n):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            frame_gap = curr_frame.frame_id - prev_frame.frame_id
            total_dt = curr_frame.timestamp - prev_frame.timestamp

            if total_dt <= 0 or frame_gap <= 0:
                self._current_confidence = curr_frame.det_confidence
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            elif frame_gap > 1:
                step_dt = total_dt / frame_gap
                for _ in range(frame_gap):
                    self.predict(step_dt)
                self._current_confidence = curr_frame.det_confidence
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            else:
                self.predict(total_dt)
                self._current_confidence = curr_frame.det_confidence
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)

            results[i] = state.as_array()

        return results
