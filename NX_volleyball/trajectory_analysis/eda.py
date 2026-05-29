"""Exploratory Data Analysis for volleyball trajectory data."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml

from loader import load_csv, segment_frames, load_dataset, Segment, Frame
from typing import List


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_all_frames(data_dir: str) -> List[Frame]:
    """Load all detection frames from all CSVs."""
    all_frames = []
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        frames = load_csv(filepath)
        all_frames.extend(frames)
    return all_frames


def analyze_frame_rate(segments: List[Segment], results_dir: str):
    """Analyze frame rate distribution across segments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_dts = []
    all_gaps = []

    for seg in segments:
        frames = seg.frames
        for i in range(1, len(frames)):
            dt = frames[i].timestamp - frames[i - 1].timestamp
            gap = frames[i].frame_id - frames[i - 1].frame_id
            all_dts.append(dt)
            all_gaps.append(gap)

    all_dts = np.array(all_dts)
    all_gaps = np.array(all_gaps)

    # dt histogram
    ax = axes[0, 0]
    valid_dts = all_dts[(all_dts > 0) & (all_dts < 0.2)]
    ax.hist(valid_dts * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(1000 / 60, color='r', linestyle='--', label='16.67ms (60Hz)')
    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Frame Interval Distribution')
    ax.legend()

    # Frame rate over time
    ax = axes[0, 1]
    fps_instantaneous = 1.0 / all_dts[all_dts > 0.001]
    ax.plot(fps_instantaneous[:500], alpha=0.7, linewidth=0.5)
    ax.axhline(60, color='r', linestyle='--', label='60Hz')
    ax.set_xlabel('Frame pair index')
    ax.set_ylabel('Instantaneous FPS')
    ax.set_title('Instantaneous Frame Rate (first 500)')
    ax.legend()
    ax.set_ylim(0, 120)

    # Gap histogram
    ax = axes[1, 0]
    ax.hist(all_gaps, bins=range(1, min(20, int(all_gaps.max()) + 2)),
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Frame ID gap')
    ax.set_ylabel('Count')
    ax.set_title('Frame ID Gap Distribution')
    ax.axvline(5, color='r', linestyle='--', label='Segment break threshold')
    ax.legend()

    # Segment lengths
    ax = axes[1, 1]
    lengths = [seg.length for seg in segments]
    ax.bar(range(len(lengths)), lengths, alpha=0.7)
    ax.set_xlabel('Segment index')
    ax.set_ylabel('Length (frames)')
    ax.set_title(f'Segment Lengths (N={len(segments)}, total={sum(lengths)})')
    ax.axhline(np.mean(lengths), color='r', linestyle='--', label=f'mean={np.mean(lengths):.0f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'frame_rate_analysis.png'), dpi=150)
    plt.close()
    print("  Saved: frame_rate_analysis.png")


def analyze_depth_comparison(frames: List[Frame], results_dir: str):
    """Compare z_mono vs z_stereo."""
    z_mono = np.array([f.z_mono for f in frames])
    z_stereo = np.array([f.z_stereo for f in frames])
    obs_z = np.array([f.obs_z for f in frames])
    confidence = np.array([f.stereo_conf for f in frames])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter: mono vs stereo
    ax = axes[0, 0]
    valid = (z_mono > 0) & (z_stereo > 0) & (z_mono < 15) & (z_stereo < 15)
    ax.scatter(z_mono[valid], z_stereo[valid], alpha=0.3, s=2)
    max_z = max(z_mono[valid].max(), z_stereo[valid].max())
    ax.plot([0, max_z], [0, max_z], 'r--', label='y=x')
    ax.set_xlabel('z_mono (m)')
    ax.set_ylabel('z_stereo (m)')
    ax.set_title('Mono vs Stereo Depth')
    ax.legend()

    # Difference histogram
    ax = axes[0, 1]
    diff = z_stereo[valid] - z_mono[valid]
    ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('z_stereo - z_mono (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'Depth Difference (mean={diff.mean():.3f}, std={diff.std():.3f})')

    # obs_z histogram
    ax = axes[1, 0]
    valid_obs = obs_z[(obs_z > 0) & (obs_z < 15)]
    ax.hist(valid_obs, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('obs_z (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'Final Depth Distribution (mean={valid_obs.mean():.2f}m)')

    # Stereo confidence vs depth
    ax = axes[1, 1]
    valid_conf = (confidence > 0) & valid
    if valid_conf.sum() > 0:
        ax.scatter(obs_z[valid_conf], confidence[valid_conf], alpha=0.3, s=2)
        ax.set_xlabel('obs_z (m)')
        ax.set_ylabel('Stereo Confidence')
        ax.set_title('Stereo Confidence vs Depth')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'depth_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: depth_comparison.png")


def analyze_variance_by_distance(segments: List[Segment], config: dict, results_dir: str):
    """Analyze position variance binned by distance."""
    z_bins = config['evaluation']['z_bins']
    bin_edges = [0.0] + z_bins + [15.0]

    all_obs = []
    for seg in segments:
        all_obs.append(seg.obs_xyz)
    
    if not all_obs:
        return
    
    obs_all = np.vstack(all_obs)
    obs_z = obs_all[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels_list = []
    variances_x = []
    variances_y = []
    variances_z = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (obs_z >= lo) & (obs_z < hi)
        count = mask.sum()
        if count < 10:
            continue

        # Local variance (high-pass residual)
        subset = obs_all[mask]
        # Use rolling window to compute local variance
        window = min(5, count // 2)
        if window < 2:
            continue
        
        residuals = np.zeros_like(subset)
        for j in range(len(subset)):
            lo_j = max(0, j - window)
            hi_j = min(len(subset), j + window + 1)
            residuals[j] = subset[j] - subset[lo_j:hi_j].mean(axis=0)

        label = f"{lo:.0f}-{hi:.0f}m\n(n={count})"
        labels_list.append(label)
        variances_x.append(residuals[:, 0].std())
        variances_y.append(residuals[:, 1].std())
        variances_z.append(residuals[:, 2].std())

    x_pos = range(len(labels_list))
    
    axes[0].bar(x_pos, variances_x, alpha=0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels_list, fontsize=8)
    axes[0].set_ylabel('σ_x (m)')
    axes[0].set_title('X Position Noise vs Distance')

    axes[1].bar(x_pos, variances_y, alpha=0.7, color='orange')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels_list, fontsize=8)
    axes[1].set_ylabel('σ_y (m)')
    axes[1].set_title('Y Position Noise vs Distance')

    axes[2].bar(x_pos, variances_z, alpha=0.7, color='green')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(labels_list, fontsize=8)
    axes[2].set_ylabel('σ_z (m)')
    axes[2].set_title('Z Position Noise vs Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'variance_by_distance.png'), dpi=150)
    plt.close()
    print("  Saved: variance_by_distance.png")


def analyze_acf(segments: List[Segment], results_dir: str):
    """Compute and plot autocorrelation of observation noise."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    max_lag = 30

    # Use longest segment
    if not segments:
        return
    longest = max(segments, key=lambda s: s.length)
    obs = longest.obs_xyz
    n = len(obs)

    if n < max_lag + 10:
        return

    # Detrend with polynomial
    t = np.arange(n)
    labels = ['X', 'Y', 'Z']
    
    for dim in range(3):
        signal = obs[:, dim]
        # Remove trend (3rd order poly)
        coeffs = np.polyfit(t, signal, 3)
        trend = np.polyval(coeffs, t)
        residual = signal - trend
        
        # Compute ACF
        res_centered = residual - residual.mean()
        var = res_centered.var()
        if var < 1e-12:
            continue
        
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean(res_centered[:-lag] * res_centered[lag:]) / var

        ax = axes[dim]
        ax.bar(range(max_lag), acf, alpha=0.7)
        ax.axhline(1.96 / np.sqrt(n), color='r', linestyle='--', alpha=0.5)
        ax.axhline(-1.96 / np.sqrt(n), color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag (frames)')
        ax.set_ylabel('ACF')
        ax.set_title(f'Observation Noise ACF - {labels[dim]}')
        ax.set_ylim(-0.5, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'observation_acf.png'), dpi=150)
    plt.close()
    print("  Saved: observation_acf.png")


def analyze_bounce_detection(segments: List[Segment], config: dict, results_dir: str):
    """Annotate potential bounce points in trajectories."""
    gravity = config['physics']['gravity']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Find segments with potential bounces (y reversal)
    bounce_segments = []
    for seg in segments:
        obs_y = seg.obs_xyz[:, 1]
        if len(obs_y) < 20:
            continue
        # Look for local minima in y (ball at lowest point before bounce)
        # In y-down frame, bounce = local maximum in y
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(obs_y, distance=10, prominence=0.1)
        if len(peaks) > 0:
            bounce_segments.append((seg, peaks))

    if not bounce_segments:
        # Just plot first few trajectories
        for seg in segments[:3]:
            t = seg.timestamps - seg.timestamps[0]
            axes[0].plot(t, seg.obs_xyz[:, 1], alpha=0.7, label=seg.source_file[:20])
            axes[1].plot(t, seg.obs_xyz[:, 2], alpha=0.7)
    else:
        # Plot segments with bounce annotations
        for seg, peaks in bounce_segments[:3]:
            t = seg.timestamps - seg.timestamps[0]
            obs_y = seg.obs_xyz[:, 1]
            
            axes[0].plot(t, obs_y, alpha=0.7, label=seg.source_file[:20])
            axes[0].scatter(t[peaks], obs_y[peaks], color='red', zorder=5, s=50, marker='v')

            obs_z = seg.obs_xyz[:, 2]
            axes[1].plot(t, obs_z, alpha=0.7)
            axes[1].scatter(t[peaks], obs_z[peaks], color='red', zorder=5, s=50, marker='v')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Y position (m, down)')
    axes[0].set_title('Y Trajectory with Bounce Candidates (red markers)')
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Z position (m, forward)')
    axes[1].set_title('Z Trajectory at Bounce Points')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'bounce_detection.png'), dpi=150)
    plt.close()
    print("  Saved: bounce_detection.png")


def analyze_gravity_calibration(segments: List[Segment], config: dict, results_dir: str):
    """Estimate gravity from parabolic segments (prefix '1' = toss/throw)."""
    gravity_true = config['physics']['gravity']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    gravity_estimates = []
    
    for seg in segments:
        obs = seg.obs_xyz
        t = seg.timestamps - seg.timestamps[0]
        n = len(t)
        
        if n < 15:
            continue

        # Fit quadratic to y: y = y0 + vy0*t + 0.5*g*t²
        coeffs = np.polyfit(t, obs[:, 1], 2)
        g_est = 2.0 * coeffs[0]
        
        # Only consider reasonable estimates
        if 5.0 < g_est < 15.0:
            gravity_estimates.append(g_est)
            
            # Plot first few fits
            if len(gravity_estimates) <= 4:
                ax_idx = len(gravity_estimates) - 1
                ax = axes[ax_idx // 2, ax_idx % 2]
                
                y_fit = np.polyval(coeffs, t)
                ax.plot(t, obs[:, 1], 'b.', markersize=2, label='Observed')
                ax.plot(t, y_fit, 'r-', linewidth=2, label=f'Fit (g={g_est:.2f})')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'{seg.source_file[:25]} | g_est={g_est:.2f} m/s²')
                ax.legend()
                ax.invert_yaxis()

    # Fill remaining axes with summary
    if len(gravity_estimates) < 4:
        for idx in range(len(gravity_estimates), 4):
            ax = axes[idx // 2, idx % 2]
            ax.set_visible(False)

    if gravity_estimates:
        # Add text summary
        g_arr = np.array(gravity_estimates)
        fig.suptitle(f'Gravity Calibration: mean={g_arr.mean():.3f} ± {g_arr.std():.3f} m/s² '
                     f'(true={gravity_true:.2f}, N={len(g_arr)} segments)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gravity_calibration.png'), dpi=150)
    plt.close()
    print(f"  Saved: gravity_calibration.png (N={len(gravity_estimates)} estimates)")
    
    if gravity_estimates:
        g_arr = np.array(gravity_estimates)
        print(f"    Gravity estimate: {g_arr.mean():.3f} ± {g_arr.std():.3f} m/s² "
              f"(error: {abs(g_arr.mean() - gravity_true):.3f})")


def analyze_3d_trajectories(segments: List[Segment], results_dir: str):
    """Plot 3D trajectory overview."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(segments))))
    
    for i, seg in enumerate(segments[:10]):
        obs = seg.obs_xyz
        color = colors[i % len(colors)]
        ax.plot(obs[:, 0], obs[:, 2], obs[:, 1],
                color=color, alpha=0.7, linewidth=0.8,
                label=f'{seg.source_file[:15]}({seg.length})')
        # Mark start
        ax.scatter(obs[0, 0], obs[0, 2], obs[0, 1],
                   color=color, s=30, marker='o')

    ax.set_xlabel('X (right)')
    ax.set_ylabel('Z (forward)')
    ax.set_zlabel('Y (down)')
    ax.set_title('3D Trajectory Overview (first 10 segments)')
    ax.legend(fontsize=7, loc='upper left')
    ax.invert_zaxis()  # Y is down

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '3d_trajectories.png'), dpi=150)
    plt.close()
    print("  Saved: 3d_trajectories.png")


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    config = load_config(config_path)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['data_dir'])
    data_dir = os.path.abspath(data_dir)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "eda")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Exploratory Data Analysis - Volleyball Trajectory")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {results_dir}")

    # Load all frames
    print("\nLoading data...")
    all_frames = get_all_frames(data_dir)
    print(f"  Total detection frames: {len(all_frames)}")

    # Load segments
    segments = load_dataset(data_dir, "")
    print(f"  Total segments (min 10 frames): {len(segments)}")
    print(f"  Total segment frames: {sum(s.length for s in segments)}")

    # Stats
    if all_frames:
        obs_z = np.array([f.obs_z for f in all_frames])
        valid_z = obs_z[(obs_z > 0) & (obs_z < 15)]
        print(f"  Depth range: {valid_z.min():.2f} - {valid_z.max():.2f} m "
              f"(mean={valid_z.mean():.2f})")

    print("\nGenerating plots...")

    # 1. Frame rate analysis
    print("\n[1/6] Frame rate analysis...")
    analyze_frame_rate(segments, results_dir)

    # 2. Depth comparison
    print("[2/6] Depth comparison...")
    analyze_depth_comparison(all_frames, results_dir)

    # 3. Variance by distance
    print("[3/6] Variance by distance...")
    analyze_variance_by_distance(segments, config, results_dir)

    # 4. ACF analysis
    print("[4/6] Autocorrelation analysis...")
    analyze_acf(segments, results_dir)

    # 5. Bounce detection
    print("[5/6] Bounce detection...")
    analyze_bounce_detection(segments, config, results_dir)

    # 6. Gravity calibration (use prefix '1' segments for throws)
    print("[6/6] Gravity calibration...")
    segments_1 = load_dataset(data_dir, "1")
    if not segments_1:
        segments_1 = segments  # fallback to all
    analyze_gravity_calibration(segments_1, config, results_dir)

    # 7. 3D trajectory overview
    print("\n[Bonus] 3D trajectory overview...")
    analyze_3d_trajectories(segments, results_dir)

    print(f"\nEDA complete! Results in: {results_dir}")


if __name__ == "__main__":
    main()
