"""Metrics module for trajectory evaluation."""

from .jitter import compute_sigma_pos_by_distance, compute_sigma_vel, compute_drift_rate
from .latency import compute_phase_lag, compute_direction_change_delay, compute_settle_time
from .stability import compute_jerk_energy, compute_continuity, compute_physics_r2
from .consistency import compute_nis, compute_innovation_acf, compute_P_boundedness
