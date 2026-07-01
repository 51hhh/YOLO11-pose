"""Filter consistency metrics (NIS, ACF, P-boundedness)."""

import numpy as np
from typing import List


def compute_nis(innovations: List[np.ndarray], S_matrices: List[np.ndarray]) -> dict:
    """Compute Normalized Innovation Squared (NIS) statistics.

    NIS = innovation^T @ S^{-1} @ innovation
    For a well-tuned 3D filter, E[NIS] ≈ 3 (dimension of measurement).

    Args:
        innovations: List of (3,) innovation vectors.
        S_matrices: List of (3,3) innovation covariance matrices.

    Returns:
        Dict with NIS statistics.
    """
    if not innovations or not S_matrices:
        return {'mean_nis': 0.0, 'std_nis': 0.0, 'pct_above_95': 0.0}

    nis_values = []
    for innov, S in zip(innovations, S_matrices):
        if innov is None or S is None:
            continue
        try:
            S_inv = np.linalg.inv(S)
            nis = float(innov @ S_inv @ innov)
            nis_values.append(nis)
        except np.linalg.LinAlgError:
            continue

    if not nis_values:
        return {'mean_nis': 0.0, 'std_nis': 0.0, 'pct_above_95': 0.0}

    nis_arr = np.array(nis_values)

    # Chi-squared 95% threshold for 3 DOF ≈ 7.815
    chi2_95 = 7.815

    return {
        'mean_nis': float(nis_arr.mean()),
        'std_nis': float(nis_arr.std()),
        'median_nis': float(np.median(nis_arr)),
        'pct_above_95': float(np.mean(nis_arr > chi2_95) * 100.0),
        'expected_nis': 3.0,  # measurement dimension
        'count': len(nis_values),
    }


def compute_innovation_acf(innovations: List[np.ndarray], max_lag: int = 20) -> dict:
    """Compute autocorrelation of innovation sequence.

    For a well-tuned filter, innovations should be white (zero ACF at lag > 0).

    Args:
        innovations: List of (3,) innovation vectors.
        max_lag: Maximum lag to compute.

    Returns:
        Dict with ACF values and whiteness test result.
    """
    if len(innovations) < max_lag + 5:
        return {'acf': np.zeros(max_lag), 'is_white': True, 'max_acf': 0.0}

    # Stack into (N, 3) array and compute magnitude
    innov_arr = np.array(innovations)
    n = len(innov_arr)

    # Compute ACF for each dimension and average
    acf_dims = []
    for d in range(3):
        seq = innov_arr[:, d]
        seq_centered = seq - seq.mean()
        var = seq_centered.var()
        if var < 1e-12:
            acf_dims.append(np.zeros(max_lag))
            continue

        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean(seq_centered[:-lag] * seq_centered[lag:]) / var
        acf_dims.append(acf)

    acf_avg = np.mean(acf_dims, axis=0)

    # Whiteness test: 95% confidence band is ±1.96/sqrt(N)
    confidence_bound = 1.96 / np.sqrt(n)
    is_white = bool(np.all(np.abs(acf_avg[1:]) < confidence_bound))

    return {
        'acf': acf_avg,
        'is_white': is_white,
        'max_acf': float(np.max(np.abs(acf_avg[1:]))) if len(acf_avg) > 1 else 0.0,
        'confidence_bound': float(confidence_bound),
    }


def compute_P_boundedness(P_history: List[np.ndarray]) -> dict:
    """Check that covariance matrix P remains bounded and positive definite.

    Args:
        P_history: List of P diagonal arrays (from get_diagnostics).

    Returns:
        Dict with boundedness metrics.
    """
    if not P_history:
        return {'is_bounded': True, 'max_P': 0.0, 'min_P': 0.0, 'diverged': False}

    P_arr = np.array(P_history)  # (N, state_dim)

    max_P = float(P_arr.max())
    min_P = float(P_arr.min())

    # Check for divergence (P growing unbounded)
    n = len(P_arr)
    diverged = False
    if n > 10:
        # Compare first quarter to last quarter
        q1 = P_arr[:n // 4].max()
        q4 = P_arr[-n // 4:].max()
        if q4 > q1 * 100 and q4 > 1000:
            diverged = True

    # Check positive definiteness (all diagonal elements positive)
    is_positive = bool(np.all(P_arr >= 0))
    is_bounded = bool(max_P < 1e6) and is_positive and not diverged

    return {
        'is_bounded': is_bounded,
        'is_positive': is_positive,
        'diverged': diverged,
        'max_P': max_P,
        'min_P': min_P,
        'final_P_mean': float(P_arr[-1].mean()) if len(P_arr) > 0 else 0.0,
    }
