"""Main evaluation driver for trajectory analysis."""

import os
import sys
import time
import yaml
import numpy as np
from typing import Dict, List

from loader import load_dataset, Segment
from filters import create_filter, list_filters
from metrics import (
    compute_sigma_pos_by_distance, compute_sigma_vel, compute_drift_rate,
    compute_phase_lag, compute_direction_change_delay, compute_settle_time,
    compute_jerk_energy, compute_continuity, compute_physics_r2,
    compute_nis, compute_innovation_acf, compute_P_boundedness,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_prediction_error(filtered_xyz: np.ndarray, filtered_vel: np.ndarray,
                              obs_xyz: np.ndarray, timestamps: np.ndarray,
                              gravity_vec: np.ndarray, horizon_sec: float) -> dict:
    """Compute prediction error at a given time horizon.
    
    At each frame t, use filter's position + velocity to extrapolate forward
    by horizon_sec using ballistic model (gravity), then compare with actual
    observation at the future frame.
    
    Only evaluates on ballistic arcs: checks that the midpoint between start
    and target lies on the parabola (rejects predictions spanning bounces/catches).
    Also requires minimum displacement (rejects static periods).
    """
    n = len(timestamps)
    errors = []
    
    for i in range(n - 3):
        # Find the future frame closest to timestamps[i] + horizon_sec
        target_time = timestamps[i] + horizon_sec
        j = np.searchsorted(timestamps, target_time)
        if j >= n:
            continue
        if j > 0 and abs(timestamps[j-1] - target_time) < abs(timestamps[j] - target_time):
            j = j - 1
        actual_dt = timestamps[j] - timestamps[i]
        if actual_dt < horizon_sec * 0.5:
            continue
        
        # Minimum displacement check (ball must be moving)
        actual_displacement = np.linalg.norm(obs_xyz[j] - obs_xyz[i])
        if actual_displacement < 0.10:
            continue
        
        # Ballistic consistency check: verify midpoint lies on parabola
        # If a bounce/catch happened between i and j, midpoint won't fit
        if j - i > 2:
            mid = (i + j) // 2
            dt1 = timestamps[mid] - timestamps[i]
            dt2 = timestamps[j] - timestamps[i]
            if dt2 > 0:
                v0_est = (obs_xyz[j] - obs_xyz[i] - 0.5 * gravity_vec * dt2**2) / dt2
                expected_mid = obs_xyz[i] + v0_est * dt1 + 0.5 * gravity_vec * dt1**2
                mid_error = np.linalg.norm(obs_xyz[mid] - expected_mid)
                if mid_error > 0.20:  # midpoint deviates > 20cm from parabola → non-ballistic
                    continue
        
        # Filter must be tracking (position within 1m of observation)
        # Excludes diverged states where prediction is meaningless
        filter_tracking_err = np.linalg.norm(filtered_xyz[i] - obs_xyz[i])
        if filter_tracking_err > 1.0:
            continue
        
        # Ballistic prediction: pos + vel*dt + 0.5*g*dt²
        pos = filtered_xyz[i]
        vel = filtered_vel[i]
        pred = pos + vel * actual_dt + 0.5 * gravity_vec * actual_dt**2
        
        # Compare with actual observation at future frame
        actual = obs_xyz[j]
        err = np.linalg.norm(pred - actual)
        errors.append(err)
    
    if not errors:
        return {'rmse': 0.0, 'median': 0.0, 'p90': 0.0, 'n_samples': 0}
    
    errors = np.array(errors)
    return {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
        'n_samples': len(errors),
    }


def compute_prediction_jitter(filtered_xyz: np.ndarray, filtered_vel: np.ndarray,
                               timestamps: np.ndarray, gravity_vec: np.ndarray,
                               horizon_sec: float) -> dict:
    """Compute frame-to-frame jitter in predicted future position.
    
    Measures how stable the predicted landing point is across consecutive frames.
    Low jitter = robot gets consistent target, high jitter = robot oscillates.
    """
    n = len(timestamps)
    predicted_points = []
    
    for i in range(n):
        pos = filtered_xyz[i]
        vel = filtered_vel[i]
        pred = pos + vel * horizon_sec + 0.5 * gravity_vec * horizon_sec**2
        predicted_points.append(pred)
    
    predicted_points = np.array(predicted_points)
    
    # Frame-to-frame change in predicted future point
    diffs = np.diff(predicted_points, axis=0)
    jitters = np.linalg.norm(diffs, axis=1)
    
    if len(jitters) == 0:
        return {'mean_jitter': 1.0, 'max_jitter': 2.0, 'std_jitter': 1.0}
    
    return {
        'mean_jitter': float(np.mean(jitters)),
        'max_jitter': float(np.max(jitters)),
        'std_jitter': float(np.std(jitters)),
    }


def run_filter_on_segment(filter_name: str, segment: Segment, config: dict) -> dict:
    """Run a single filter on a single segment, collecting results and diagnostics."""
    # Get filter params from config
    # For robust_<name>, merge inner filter params + robust params
    if filter_name.startswith('robust_') and filter_name[7:] in config.get('filters', {}):
        inner_name = filter_name[7:]
        filter_params = config.get('filters', {}).get(inner_name, {}).copy()
        # Overlay any robust-specific config
        robust_cfg = config.get('filters', {}).get(filter_name, {})
        filter_params.update(robust_cfg)
    else:
        filter_params = config.get('filters', {}).get(filter_name, {}).copy()
    filter_params['focal'] = config['camera']['focal']
    
    # All gravity-based filters need physics params
    if filter_name not in ('raw_passthrough', 'const_accel_9d'):
        filter_params['gravity'] = config['physics']['gravity']
        filter_params['gravity_vec'] = config['physics']['gravity_vec']

    filt = create_filter(filter_name, **filter_params)

    # Time the processing
    t0 = time.perf_counter()
    results = filt.process_segment(segment.frames)
    elapsed = time.perf_counter() - t0

    # Collect diagnostics by re-running (for innovation/S/P history)
    innovations = []
    S_matrices = []
    P_history = []

    filt.reset()
    frames = segment.frames
    if frames:
        filt.update(frames[0].obs_x, frames[0].obs_y, frames[0].obs_z)
        diag = filt.get_diagnostics()
        innovations.append(diag['innovation'].copy())
        S_matrices.append(diag['S'].copy())
        P_history.append(diag['P_diag'].copy())

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        frame_gap = curr_frame.frame_id - prev_frame.frame_id
        total_dt = curr_frame.timestamp - prev_frame.timestamp

        if total_dt <= 0 or frame_gap <= 0:
            # Duplicate timestamp/frame_id: skip predict, just update
            pass
        elif frame_gap > 1:
            step_dt = total_dt / frame_gap
            for _ in range(frame_gap):
                filt.predict(step_dt)
        else:
            filt.predict(total_dt)
        
        filt.update(curr_frame.obs_x, curr_frame.obs_y, curr_frame.obs_z)
        diag = filt.get_diagnostics()
        innovations.append(diag['innovation'].copy())
        S_matrices.append(diag['S'].copy())
        P_history.append(diag['P_diag'].copy())

    return {
        'results': results,
        'elapsed': elapsed,
        'innovations': innovations,
        'S_matrices': S_matrices,
        'P_history': P_history,
        'frames_per_sec': len(frames) / elapsed if elapsed > 0 else 0,
    }


def compute_all_metrics(filter_output: dict, segment: Segment, config: dict) -> dict:
    """Compute all metric categories for one filter on one segment."""
    results = filter_output['results']
    filtered_xyz = results[:, 0:3]
    filtered_vel = results[:, 3:6]
    obs_xyz = segment.obs_xyz
    obs_z = segment.obs_z_depth
    timestamps = segment.timestamps
    z_bins = config['evaluation']['z_bins']
    window = config['evaluation']['smoothing_window']
    jump_threshold = config['evaluation']['jump_threshold']
    gravity = config['physics']['gravity']
    gravity_vec = np.array(config['physics']['gravity_vec'], dtype=float)

    # Jitter metrics
    jitter_by_dist = compute_sigma_pos_by_distance(filtered_xyz, obs_z, z_bins)
    sigma_vel = compute_sigma_vel(results)
    drift = compute_drift_rate(filtered_xyz, timestamps, window)

    # Latency metrics
    phase_lag = compute_phase_lag(obs_xyz[:, 1], filtered_xyz[:, 1])
    dir_delay = compute_direction_change_delay(obs_xyz[:, 1], filtered_xyz[:, 1], timestamps)
    settle = compute_settle_time(obs_xyz, filtered_xyz, timestamps)
    # Tracking error: measures responsiveness using median (robust to outliers)
    tracking_errors = np.sqrt(np.sum((filtered_xyz - obs_xyz)**2, axis=1))
    tracking_rmse = float(np.median(tracking_errors))

    # Prediction metrics: use filter velocity to extrapolate and compare with future obs
    prediction_05 = compute_prediction_error(filtered_xyz, filtered_vel, obs_xyz,
                                              timestamps, gravity_vec, horizon_sec=0.5)
    prediction_10 = compute_prediction_error(filtered_xyz, filtered_vel, obs_xyz,
                                              timestamps, gravity_vec, horizon_sec=1.0)

    # Prediction jitter: frame-to-frame variation in predicted landing point
    pred_jitter = compute_prediction_jitter(filtered_xyz, filtered_vel, timestamps,
                                             gravity_vec, horizon_sec=1.0)

    # Stability metrics
    jerk = compute_jerk_energy(filtered_xyz, timestamps)
    continuity = compute_continuity(filtered_xyz, jump_threshold)
    physics = compute_physics_r2(filtered_xyz, timestamps, gravity)

    # Consistency metrics
    nis = compute_nis(filter_output['innovations'], filter_output['S_matrices'])
    acf = compute_innovation_acf(filter_output['innovations'])
    p_bound = compute_P_boundedness(filter_output['P_history'])

    return {
        'jitter': {
            'by_distance': jitter_by_dist,
            'velocity': sigma_vel,
            'drift': drift,
        },
        'latency': {
            'phase_lag': phase_lag,
            'direction_delay': dir_delay,
            'settle_time': settle,
            'tracking_rmse': tracking_rmse,
        },
        'prediction': {
            'error_05s': prediction_05,
            'error_10s': prediction_10,
            'pred_jitter': pred_jitter,
        },
        'stability': {
            'jerk': jerk,
            'continuity': continuity,
            'physics': physics,
        },
        'consistency': {
            'nis': nis,
            'acf': {k: v for k, v in acf.items() if k != 'acf'},
            'P_boundedness': p_bound,
        },
        'performance': {
            'elapsed_sec': filter_output['elapsed'],
            'frames_per_sec': filter_output['frames_per_sec'],
        },
    }


def normalize_score(value: float, best: float, worst: float) -> float:
    """Normalize a metric to [0, 1] where 1 is best."""
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (worst - value) / (worst - best)))


def compute_composite_score(metrics: dict) -> dict:
    """Compute weighted composite score from metrics.
    
    New scoring focused on prediction accuracy for ball catching:
    Score = 0.35*Prediction@0.5s + 0.30*Prediction@1.0s + 0.20*Smoothness + 0.15*Tracking
    
    Prediction: how well filter's velocity enables future position extrapolation.
    Smoothness: output jitter (for stable predictions, not wild swings).
    Tracking: position accuracy relative to observation (penalizes lag).
    """
    # Prediction scores: lower RMSE is better
    pred_05_rmse = metrics['prediction']['error_05s'].get('rmse', 1.0)
    pred_10_rmse = metrics['prediction']['error_10s'].get('rmse', 2.0)
    pred_jitter = metrics['prediction']['pred_jitter'].get('mean_jitter', 1.0)
    pred_05_n = metrics['prediction']['error_05s'].get('n_samples', 0)
    pred_10_n = metrics['prediction']['error_10s'].get('n_samples', 0)
    
    # Normalize prediction errors: 0 error → 1.0, 1m error → ~0.5, 2m → ~0.33
    pred_05_score = 1.0 / (1.0 + pred_05_rmse / 0.3) if pred_05_n > 0 else None
    pred_10_score = 1.0 / (1.0 + pred_10_rmse / 0.8) if pred_10_n > 0 else None
    
    # Prediction jitter: lower is better
    pred_jitter_score = 1.0 / (1.0 + pred_jitter / 0.2)

    # Smoothness: lower sigma_vel is better (output doesn't jitter)
    sigma_vel_total = metrics['jitter']['velocity'].get('sigma_total', 1.0)
    smoothness_score = normalize_score(sigma_vel_total, 0.0, 2.0)

    # Tracking: position follows observation closely (penalizes extreme lag)
    tracking_rmse = metrics['latency'].get('tracking_rmse', 0.5)
    tracking_score = normalize_score(tracking_rmse, 0.0, 0.5)

    # Weighted composite - adapt weights based on whether prediction data is available
    has_prediction = pred_05_score is not None or pred_10_score is not None
    
    if has_prediction:
        # Dynamic segment: prediction-focused scoring
        p05 = pred_05_score if pred_05_score is not None else 0.5
        p10 = pred_10_score if pred_10_score is not None else 0.5
        composite = (0.30 * p05 + 
                     0.25 * p10 + 
                     0.20 * pred_jitter_score +
                     0.15 * smoothness_score + 
                     0.10 * tracking_score)
    else:
        # Static segment: no prediction available, use smoothness + tracking
        composite = (0.40 * pred_jitter_score +
                     0.35 * smoothness_score + 
                     0.25 * tracking_score)

    return {
        'composite': float(composite),
        'pred_05_score': float(pred_05_score) if pred_05_score is not None else 0.0,
        'pred_10_score': float(pred_10_score) if pred_10_score is not None else 0.0,
        'pred_jitter_score': float(pred_jitter_score),
        'smoothness_score': float(smoothness_score),
        'tracking_score': float(tracking_score),
        # Keep old names for CSV compatibility
        'jitter_score': float(smoothness_score),
        'latency_score': float(tracking_score),
        'stability_score': float(pred_05_score) if pred_05_score is not None else float(pred_jitter_score),
    }


def evaluate_all(config: dict) -> dict:
    """Run full evaluation pipeline."""
    data_dir = os.path.join(os.path.dirname(__file__), config['data_dir'])
    data_dir = os.path.abspath(data_dir)

    print(f"Loading data from: {data_dir}")

    # Load segments by prefix
    all_results = {}
    filter_names = list_filters()

    for prefix in ['0', '1', '2']:
        segments = load_dataset(data_dir, prefix)
        if not segments:
            print(f"  No segments found for prefix '{prefix}', trying without prefix...")
            continue
        
        print(f"  Prefix '{prefix}': {len(segments)} segments "
              f"({sum(s.length for s in segments)} total frames)")

        for filt_name in filter_names:
            key = f"{prefix}_{filt_name}"
            seg_results = []

            for seg_idx, segment in enumerate(segments):
                output = run_filter_on_segment(filt_name, segment, config)
                metrics = compute_all_metrics(output, segment, config)
                score = compute_composite_score(metrics)
                
                seg_results.append({
                    'segment_idx': seg_idx,
                    'source_file': segment.source_file,
                    'length': segment.length,
                    'metrics': metrics,
                    'score': score,
                })

            all_results[key] = seg_results

    return all_results


def print_summary(all_results: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Aggregate scores by filter
    filter_scores = {}
    for key, seg_results in all_results.items():
        parts = key.split('_', 1)
        prefix = parts[0]
        filt_name = parts[1]

        if filt_name not in filter_scores:
            filter_scores[filt_name] = []

        for sr in seg_results:
            filter_scores[filt_name].append(sr['score']['composite'])

    print(f"\n{'Filter':<25} {'Mean Score':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'N Segments'}")
    print("-" * 80)

    for filt_name in list_filters():
        if filt_name in filter_scores and filter_scores[filt_name]:
            scores = np.array(filter_scores[filt_name])
            print(f"{filt_name:<25} {scores.mean():<12.4f} {scores.std():<10.4f} "
                  f"{scores.min():<10.4f} {scores.max():<10.4f} {len(scores)}")

    # Detailed breakdown for best filter
    print("\n" + "-" * 80)
    print("DETAILED BREAKDOWN (per filter, averaged across segments)")
    print("-" * 80)

    for filt_name in list_filters():
        jitter_scores = []
        latency_scores = []
        stability_scores = []
        fps_list = []

        for key, seg_results in all_results.items():
            if key.endswith(filt_name):
                for sr in seg_results:
                    jitter_scores.append(sr['score']['jitter_score'])
                    latency_scores.append(sr['score']['latency_score'])
                    stability_scores.append(sr['score']['stability_score'])
                    fps_list.append(sr['metrics']['performance']['frames_per_sec'])

        if jitter_scores:
            print(f"\n  {filt_name}:")
            print(f"    Jitter:    {np.mean(jitter_scores):.4f}")
            print(f"    Latency:   {np.mean(latency_scores):.4f}")
            print(f"    Stability: {np.mean(stability_scores):.4f}")
            print(f"    Perf:      {np.mean(fps_list):.0f} frames/sec")


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    
    print("Trajectory Analysis Evaluation Framework")
    print("=" * 50)
    print(f"Filters: {list_filters()}")
    print(f"Config: {config_path}")
    print()

    results = evaluate_all(config)
    print_summary(results)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save scores to CSV
    scores_path = os.path.join(results_dir, "scores.csv")
    with open(scores_path, 'w') as f:
        f.write("prefix,filter,segment,source_file,length,composite,jitter,latency,stability,fps\n")
        for key, seg_results in results.items():
            parts = key.split('_', 1)
            prefix = parts[0]
            filt_name = parts[1]
            for sr in seg_results:
                f.write(f"{prefix},{filt_name},{sr['segment_idx']},{sr['source_file']},"
                        f"{sr['length']},{sr['score']['composite']:.4f},"
                        f"{sr['score']['jitter_score']:.4f},"
                        f"{sr['score']['latency_score']:.4f},"
                        f"{sr['score']['stability_score']:.4f},"
                        f"{sr['metrics']['performance']['frames_per_sec']:.0f}\n")

    print(f"\nResults saved to: {scores_path}")


if __name__ == "__main__":
    main()
