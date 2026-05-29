"""Abstract base class for all filters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class FilterState:
    """Filter output state."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    def acceleration(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z,
                         self.vx, self.vy, self.vz,
                         self.ax, self.ay, self.az])


class FilterBase(ABC):
    """Abstract base class for trajectory filters."""

    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self._diagnostics = {
            'innovation': np.zeros(3),
            'S': np.eye(3),
            'P_diag': np.zeros(9),
        }

    @abstractmethod
    def reset(self):
        """Reset filter to initial state."""
        pass

    @abstractmethod
    def predict(self, dt: float):
        """Predict step (time propagation only)."""
        pass

    @abstractmethod
    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        """Update step with measurement. Returns current state."""
        pass

    @abstractmethod
    def get_state(self) -> FilterState:
        """Get current filter state."""
        pass

    def get_diagnostics(self) -> dict:
        """Get diagnostics from last update (innovation, S, P_diag)."""
        return self._diagnostics.copy()

    def process_segment(self, frames) -> np.ndarray:
        """Process a segment of frames, handling dropped frames.
        
        For frame_id gaps > 1, intermediate predict-only steps are inserted.
        
        Args:
            frames: List of Frame objects (from loader.py).
            
        Returns:
            (N, 9) array of [x,y,z,vx,vy,vz,ax,ay,az] for each frame.
        """
        self.reset()
        n = len(frames)
        results = np.zeros((n, 9))
        
        # Initialize with first observation
        first = frames[0]
        self.update(first.obs_x, first.obs_y, first.obs_z)
        results[0] = self.get_state().as_array()

        for i in range(1, n):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            
            frame_gap = curr_frame.frame_id - prev_frame.frame_id
            total_dt = curr_frame.timestamp - prev_frame.timestamp
            
            if total_dt <= 0 or frame_gap <= 0:
                # Duplicate timestamp/frame_id: skip predict, just update
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            elif frame_gap > 1:
                # Dropped frames: distribute dt evenly across intermediate steps
                step_dt = total_dt / frame_gap
                for _ in range(frame_gap):
                    self.predict(step_dt)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
            else:
                self.predict(total_dt)
                state = self.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)

            results[i] = state.as_array()

        return results
