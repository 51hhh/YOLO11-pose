"""Latency/phase-lag metrics for filtered trajectories."""

import numpy as np
from scipy import signal


def compute_phase_lag(obs_signal: np.ndarray, filtered_signal: np.ndarray) -> dict:
    """Compute phase lag between observation and filtered signal using cross-correlation.

    Args:
        obs_signal: (N,) observation signal (1D).
        filtered_signal: (N,) filtered signal (1D).

    Returns:
        Dict with lag in samples and estimated time lag.
    """
    n = len(obs_signal)
    if n < 10:
        return {'lag_samples': 0, 'lag_seconds': 0.0}

    # Normalize signals
    obs_norm = obs_signal - obs_signal.mean()
    filt_norm = filtered_signal - filtered_signal.mean()

    obs_std = obs_norm.std()
    filt_std = filt_norm.std()

    if obs_std < 1e-10 or filt_std < 1e-10:
        return {'lag_samples': 0, 'lag_seconds': 0.0}

    obs_norm /= obs_std
    filt_norm /= filt_std

    # Cross-correlation
    corr = np.correlate(obs_norm, filt_norm, mode='full')
    lags = np.arange(-n + 1, n)

    # Only look at positive lags (filtered lags behind obs)
    positive_mask = lags >= 0
    corr_pos = corr[positive_mask]
    lags_pos = lags[positive_mask]

    # Find peak in reasonable range (0 to 10 frames)
    search_range = min(10, len(corr_pos))
    peak_idx = np.argmax(corr_pos[:search_range])
    lag_samples = lags_pos[peak_idx]

    return {
        'lag_samples': int(lag_samples),
        'lag_seconds': lag_samples / 60.0,  # Assuming 60Hz
        'peak_correlation': float(corr_pos[peak_idx] / n),
    }


def compute_direction_change_delay(obs_y: np.ndarray, filtered_y: np.ndarray,
                                   timestamps: np.ndarray) -> dict:
    """Compute delay in detecting direction changes (peaks/troughs in y).

    Args:
        obs_y: (N,) observed y positions.
        filtered_y: (N,) filtered y positions.
        timestamps: (N,) timestamps.

    Returns:
        Dict with direction change delay statistics.
    """
    n = len(obs_y)
    if n < 20:
        return {'mean_delay': 0.0, 'max_delay': 0.0, 'count': 0}

    # Find peaks in observed signal (direction changes)
    # Peaks in y = highest points (ball at top of arc, y is down so these are minima)
    obs_peaks, _ = signal.find_peaks(-obs_y, distance=10, prominence=0.05)
    filt_peaks, _ = signal.find_peaks(-filtered_y, distance=10, prominence=0.05)

    if len(obs_peaks) == 0 or len(filt_peaks) == 0:
        return {'mean_delay': 0.0, 'max_delay': 0.0, 'count': 0}

    # Match each obs peak to nearest filtered peak
    delays = []
    for op in obs_peaks:
        dists = np.abs(filt_peaks - op)
        nearest = filt_peaks[np.argmin(dists)]
        if abs(nearest - op) < 15:  # Within 15 frames
            delay = (timestamps[nearest] - timestamps[op]) if nearest < len(timestamps) and op < len(timestamps) else 0
            delays.append(abs(delay))

    if not delays:
        return {'mean_delay': 0.0, 'max_delay': 0.0, 'count': 0}

    return {
        'mean_delay': float(np.mean(delays)),
        'max_delay': float(np.max(delays)),
        'median_delay': float(np.median(delays)),
        'count': len(delays),
    }


def compute_settle_time(obs_xyz: np.ndarray, filtered_xyz: np.ndarray,
                        timestamps: np.ndarray, event_frames: list = None) -> dict:
    """Compute settle time after disturbances (first frames, jumps).

    Settle time = time until |filtered - obs| < threshold for consecutive frames.

    Args:
        obs_xyz: (N, 3) observed positions.
        filtered_xyz: (N, 3) filtered positions.
        timestamps: (N,) timestamps.
        event_frames: List of frame indices where events occur (default: [0]).

    Returns:
        Dict with settle time statistics.
    """
    n = len(obs_xyz)
    if n < 10:
        return {'mean_settle_time': 0.0, 'initial_settle_time': 0.0}

    if event_frames is None:
        event_frames = [0]

    # Compute error magnitude
    error = np.linalg.norm(filtered_xyz - obs_xyz, axis=1)

    # Threshold: 2 sigma of steady-state error (last half of segment)
    half = n // 2
    if half > 5:
        steady_error = error[half:]
        threshold = steady_error.mean() + 2.0 * steady_error.std()
    else:
        threshold = 0.1

    settle_times = []
    for ef in event_frames:
        if ef >= n:
            continue
        # Find first frame after event where error stays below threshold
        settled = False
        for i in range(ef, min(n - 3, n)):
            if all(error[i:i + 3] < threshold):
                settle_time = timestamps[i] - timestamps[ef]
                settle_times.append(settle_time)
                settled = True
                break
        if not settled:
            # Never settled within segment
            settle_times.append(timestamps[-1] - timestamps[ef])

    return {
        'mean_settle_time': float(np.mean(settle_times)) if settle_times else 0.0,
        'initial_settle_time': float(settle_times[0]) if settle_times else 0.0,
        'max_settle_time': float(np.max(settle_times)) if settle_times else 0.0,
    }
