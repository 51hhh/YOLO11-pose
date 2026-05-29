"""Jitter/noise metrics for filtered trajectories."""

import numpy as np
from typing import List, Tuple


def compute_sigma_pos_by_distance(filtered_xyz: np.ndarray, obs_z: np.ndarray,
                                  z_bins: List[float]) -> dict:
    """Compute position standard deviation binned by depth distance.
    
    Uses local detrending (high-pass) to isolate jitter from signal.
    
    Args:
        filtered_xyz: (N, 3) filtered positions.
        obs_z: (N,) depth values for binning.
        z_bins: Bin edges for distance grouping.
    
    Returns:
        Dict with bin labels as keys and sigma_xyz arrays as values.
    """
    results = {}
    n = len(filtered_xyz)
    
    if n < 5:
        return results

    # High-pass: subtract local mean (window=5) to isolate jitter
    window = min(5, n // 2)
    if window < 2:
        window = 2
    
    # Compute residuals from local linear trend
    residuals = np.zeros_like(filtered_xyz)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local_mean = filtered_xyz[lo:hi].mean(axis=0)
        residuals[i] = filtered_xyz[i] - local_mean

    # Bin by distance
    bin_edges = [0.0] + list(z_bins) + [np.inf]
    for j in range(len(bin_edges) - 1):
        lo_z, hi_z = bin_edges[j], bin_edges[j + 1]
        mask = (obs_z >= lo_z) & (obs_z < hi_z)
        count = mask.sum()
        
        if count >= 3:
            bin_residuals = residuals[mask]
            sigma = bin_residuals.std(axis=0)
            label = f"{lo_z:.1f}-{hi_z:.1f}m"
            results[label] = {
                'sigma_xyz': sigma,
                'sigma_total': np.linalg.norm(sigma),
                'count': count
            }

    return results


def compute_sigma_vel(filtered_results: np.ndarray) -> dict:
    """Compute velocity jitter statistics.
    
    Args:
        filtered_results: (N, 9) array [x,y,z,vx,vy,vz,ax,ay,az].
    
    Returns:
        Dict with velocity statistics.
    """
    velocities = filtered_results[:, 3:6]
    n = len(velocities)
    
    if n < 5:
        return {'sigma_vel': np.zeros(3), 'sigma_total': 0.0}

    # Compute jitter as difference of consecutive velocities
    dv = np.diff(velocities, axis=0)
    sigma_dv = dv.std(axis=0)

    return {
        'sigma_vel': sigma_dv,
        'sigma_total': np.linalg.norm(sigma_dv),
        'mean_speed': np.linalg.norm(velocities, axis=1).mean(),
        'max_speed': np.linalg.norm(velocities, axis=1).max(),
    }


def compute_drift_rate(filtered_xyz: np.ndarray, timestamps: np.ndarray,
                       window: int = 60) -> dict:
    """Compute position drift rate using windowed linear regression.
    
    Args:
        filtered_xyz: (N, 3) filtered positions.
        timestamps: (N,) timestamps.
        window: Window size for smoothing.
    
    Returns:
        Dict with drift metrics.
    """
    n = len(filtered_xyz)
    if n < window:
        return {'max_drift_rate': 0.0, 'mean_drift_rate': 0.0}

    # Compute smoothed trajectory
    smoothed = np.zeros_like(filtered_xyz)
    half_w = window // 2
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        smoothed[i] = filtered_xyz[lo:hi].mean(axis=0)

    # Drift = difference between filtered and smoothed
    drift = filtered_xyz - smoothed
    drift_mag = np.linalg.norm(drift, axis=1)

    # Drift rate = drift per unit time
    dt_total = timestamps[-1] - timestamps[0]
    if dt_total <= 0:
        return {'max_drift_rate': 0.0, 'mean_drift_rate': 0.0}

    return {
        'max_drift_rate': drift_mag.max(),
        'mean_drift_rate': drift_mag.mean(),
        'drift_std': drift_mag.std(),
    }
