"""Stability metrics for filtered trajectories."""

import numpy as np


def compute_jerk_energy(filtered_xyz: np.ndarray, timestamps: np.ndarray) -> dict:
    """Compute jerk (derivative of acceleration) energy.

    Lower jerk energy = smoother trajectory.

    Args:
        filtered_xyz: (N, 3) filtered positions.
        timestamps: (N,) timestamps.

    Returns:
        Dict with jerk energy metrics.
    """
    n = len(filtered_xyz)
    if n < 4:
        return {'jerk_energy': 0.0, 'jerk_rms': 0.0}

    # Compute dt array
    dt = np.diff(timestamps)
    dt[dt <= 0] = 1.0 / 60.0

    # Velocity (central differences where possible)
    vel = np.zeros_like(filtered_xyz)
    vel[1:-1] = (filtered_xyz[2:] - filtered_xyz[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None]
    vel[0] = (filtered_xyz[1] - filtered_xyz[0]) / dt[0]
    vel[-1] = (filtered_xyz[-1] - filtered_xyz[-2]) / dt[-1]

    # Acceleration
    acc = np.zeros_like(filtered_xyz)
    acc[1:-1] = (vel[2:] - vel[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None]
    acc[0] = (vel[1] - vel[0]) / dt[0]
    acc[-1] = (vel[-1] - vel[-2]) / dt[-1]

    # Jerk
    jerk = np.zeros_like(filtered_xyz)
    dt2 = np.diff(timestamps)
    dt2[dt2 <= 0] = 1.0 / 60.0
    if n > 3:
        jerk[1:-1] = (acc[2:] - acc[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None]
        jerk[0] = (acc[1] - acc[0]) / dt[0]
        jerk[-1] = (acc[-1] - acc[-2]) / dt[-1]

    jerk_mag = np.linalg.norm(jerk, axis=1)
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        duration = 1.0

    jerk_energy = np.trapezoid(jerk_mag ** 2, timestamps) / duration

    return {
        'jerk_energy': float(jerk_energy),
        'jerk_rms': float(np.sqrt(np.mean(jerk_mag ** 2))),
        'jerk_max': float(jerk_mag.max()),
        'jerk_mean': float(jerk_mag.mean()),
    }


def compute_continuity(filtered_xyz: np.ndarray, jump_threshold: float = 0.3) -> dict:
    """Compute trajectory continuity (detect position jumps).

    Args:
        filtered_xyz: (N, 3) filtered positions.
        jump_threshold: Distance threshold for declaring a jump.

    Returns:
        Dict with continuity metrics.
    """
    n = len(filtered_xyz)
    if n < 2:
        return {'jump_count': 0, 'jump_ratio': 0.0, 'max_jump': 0.0}

    # Frame-to-frame distances
    deltas = np.linalg.norm(np.diff(filtered_xyz, axis=0), axis=1)

    jumps = deltas > jump_threshold
    jump_count = int(jumps.sum())

    return {
        'jump_count': jump_count,
        'jump_ratio': jump_count / (n - 1),
        'max_jump': float(deltas.max()),
        'mean_step': float(deltas.mean()),
        'std_step': float(deltas.std()),
    }


def compute_physics_r2(filtered_xyz: np.ndarray, timestamps: np.ndarray,
                       gravity: float = 9.81) -> dict:
    """Compute R² of filtered trajectory against parabolic (gravity) model.

    Fits a parabolic model y(t) = y0 + vy0*t + 0.5*g*t² to the y-component
    and computes goodness of fit.

    Args:
        filtered_xyz: (N, 3) filtered positions.
        timestamps: (N,) timestamps.
        gravity: Expected gravity value (positive, y-down).

    Returns:
        Dict with R² values.
    """
    n = len(filtered_xyz)
    if n < 5:
        return {'r2_y': 0.0, 'r2_xz': 0.0, 'gravity_estimate': 0.0}

    t = timestamps - timestamps[0]
    y = filtered_xyz[:, 1]

    # Fit quadratic to y: y = a + b*t + c*t²
    # Known model: c should be ~0.5*g = ~4.905
    coeffs = np.polyfit(t, y, 2)
    y_fit = np.polyval(coeffs, t)

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_y = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    # Gravity estimate from quadratic coefficient
    gravity_est = 2.0 * coeffs[0]

    # Fit linear to x and z (should be ~constant velocity)
    x = filtered_xyz[:, 0]
    z = filtered_xyz[:, 2]

    coeffs_x = np.polyfit(t, x, 1)
    x_fit = np.polyval(coeffs_x, t)
    ss_res_x = np.sum((x - x_fit) ** 2)
    ss_tot_x = np.sum((x - x.mean()) ** 2)
    r2_x = 1.0 - ss_res_x / ss_tot_x if ss_tot_x > 1e-10 else 1.0

    coeffs_z = np.polyfit(t, z, 1)
    z_fit = np.polyval(coeffs_z, t)
    ss_res_z = np.sum((z - z_fit) ** 2)
    ss_tot_z = np.sum((z - z.mean()) ** 2)
    r2_z = 1.0 - ss_res_z / ss_tot_z if ss_tot_z > 1e-10 else 1.0

    r2_xz = (r2_x + r2_z) / 2.0

    return {
        'r2_y': float(r2_y),
        'r2_xz': float(r2_xz),
        'gravity_estimate': float(gravity_est),
        'gravity_error': float(abs(gravity_est - gravity)),
    }
