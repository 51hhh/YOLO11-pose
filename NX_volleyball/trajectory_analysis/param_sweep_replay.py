#!/usr/bin/env python3
"""Parameter sweep for replay node robustness settings."""
import sys, csv, numpy as np, glob
sys.path.insert(0, '.')
from filters import create_filter

gravity_vec = np.array([0.0, 9.81, 0.0])
csvs = sorted(glob.glob('../data/raw_observation_data_1_*.csv') + 
              glob.glob('../data/raw_observation_data_2_*.csv'))
print(f'Testing on {len(csvs)} CSV files')

def load_frames(path):
    frames = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_det = int(row['has_detection']) == 1
            x = float(row.get('obs_x', 0) or 0)
            y = float(row.get('obs_y', 0) or 0)
            z = float(row.get('obs_z', 0) or 0)
            frames.append({'ts': float(row['timestamp']), 'det': has_det, 'x': x, 'y': y, 'z': z})
    return frames

def test_config(all_frames_list, filter_name, outlier_gate, init_frames, gap_thresh, **fkw):
    total_accepted = 0
    total_rej_zero = 0
    total_rej_outlier = 0
    total_segments = 0
    pred_errors = {0.1: [], 0.2: [], 0.3: []}
    
    for frames in all_frames_list:
        filt = create_filter(filter_name, **fkw)
        filt.reset()
        init_buf = []
        filter_init = False
        last_ts = None
        positions = []
        
        for frame in frames:
            if not frame['det']:
                continue
            obs = np.array([frame['x'], frame['y'], frame['z']])
            if np.linalg.norm(obs) < 0.01:
                total_rej_zero += 1
                continue
            if last_ts is not None and (frame['ts'] - last_ts) > gap_thresh:
                filt.reset()
                init_buf.clear()
                filter_init = False
                total_segments += 1
            if not filter_init:
                init_buf.append((frame['ts'], obs.copy()))
                if len(init_buf) >= init_frames:
                    t0 = init_buf[0][0]
                    p0 = init_buf[0][1]
                    A_rows, b_rows = [], []
                    for i in range(1, len(init_buf)):
                        dt_i = init_buf[i][0] - t0
                        if dt_i < 0.005:
                            continue
                        b_i = init_buf[i][1] - p0 - 0.5 * gravity_vec * dt_i**2
                        A_rows.append(dt_i * np.eye(3))
                        b_rows.append(b_i)
                    if len(A_rows) >= 2:
                        filt.reset()
                        for i, (ts, ob) in enumerate(init_buf):
                            if i > 0:
                                dt = ts - init_buf[i-1][0]
                                if dt > 0:
                                    filt.predict(dt)
                            filt.update(ob[0], ob[1], ob[2])
                        filter_init = True
                        last_ts = init_buf[-1][0]
                else:
                    filt.update(obs[0], obs[1], obs[2])
                    last_ts = frame['ts']
                    continue
                if not filter_init:
                    continue
            dt = frame['ts'] - last_ts
            if dt > 0:
                filt.predict(dt)
            state = filt.get_state()
            pred_pos = np.array([state.x, state.y, state.z])
            if np.linalg.norm(obs - pred_pos) > outlier_gate:
                total_rej_outlier += 1
                last_ts = frame['ts']
                continue
            state = filt.update(obs[0], obs[1], obs[2])
            last_ts = frame['ts']
            total_accepted += 1
            pos = np.array([state.x, state.y, state.z])
            vel = np.array([state.vx, state.vy, state.vz])
            positions.append((frame['ts'], pos, vel, obs))
        
        # Evaluate prediction error
        for i in range(len(positions)):
            ts_i, pos_i, vel_i, obs_i = positions[i]
            for j in range(i+1, len(positions)):
                ts_j, _, _, obs_j = positions[j]
                dt = ts_j - ts_i
                if dt > 0.35:
                    break
                pred = pos_i + vel_i * dt + 0.5 * gravity_vec * dt**2
                err = np.linalg.norm(pred - obs_j)
                if 0.08 <= dt <= 0.12:
                    pred_errors[0.1].append(err)
                elif 0.18 <= dt <= 0.22:
                    pred_errors[0.2].append(err)
                elif 0.28 <= dt <= 0.32:
                    pred_errors[0.3].append(err)
    
    r01 = np.sqrt(np.mean(np.array(pred_errors[0.1])**2)) if pred_errors[0.1] else float('nan')
    r02 = np.sqrt(np.mean(np.array(pred_errors[0.2])**2)) if pred_errors[0.2] else float('nan')
    r03 = np.sqrt(np.mean(np.array(pred_errors[0.3])**2)) if pred_errors[0.3] else float('nan')
    acc = total_accepted / max(1, total_accepted + total_rej_outlier) * 100
    return acc, r01, r02, r03, total_segments, total_rej_outlier, len(pred_errors[0.1])

all_frames = [load_frames(c) for c in csvs]

# === Part 1: IMM robustness parameter sweep ===
print("\n=== IMM Robustness Parameter Sweep ===")
print(f"{'Gate':<6} {'Init':<5} {'Gap':<6} {'Acc%':<7} {'RMSE01':<8} {'RMSE02':<8} {'RMSE03':<8} {'Segs':<5} {'Rej':<5}")
print('=' * 65)

best = (999, None)
for gate in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for init_n in [3, 4, 5, 6]:
        for gap in [0.1, 0.15, 0.2, 0.3, 0.5]:
            acc, r01, r02, r03, segs, rout, n01 = test_config(
                all_frames, 'imm', gate, init_n, gap,
                R_base=0.003, sigma_flight=10.0, sigma_maneuver=80.0)
            if r02 < best[0]:
                best = (r02, (gate, init_n, gap, acc, r01, r03, segs, rout))
            # Only print interesting rows
            if acc > 90 and not np.isnan(r02):
                print(f"{gate:<6.1f} {init_n:<5} {gap:<6.2f} {acc:<7.1f} {r01:<8.4f} {r02:<8.4f} {r03:<8.4f} {segs:<5} {rout:<5}")

print(f"\nBest RMSE@0.2: {best[0]:.4f}")
print(f"  Gate={best[1][0]}, Init={best[1][1]}, Gap={best[1][2]}")
print(f"  Acc={best[1][3]:.1f}%, RMSE@0.1={best[1][4]:.4f}, RMSE@0.3={best[1][5]:.4f}")
print(f"  Segments={best[1][6]}, Rejected={best[1][7]}")

# === Part 2: Compare filters with best robustness params ===
print("\n\n=== Filter Comparison (best robustness params) ===")
best_gate, best_init, best_gap = best[1][0], best[1][1], best[1][2]
print(f"Using: gate={best_gate}, init={best_init}, gap={best_gap}")
print(f"{'Filter':<25} {'Acc%':<7} {'RMSE01':<8} {'RMSE02':<8} {'RMSE03':<8} {'Segs':<5}")
print('=' * 60)

filter_configs = [
    ('imm', dict(R_base=0.003, sigma_flight=10.0, sigma_maneuver=80.0)),
    ('imm', dict(R_base=0.003, sigma_flight=10.0, sigma_maneuver=50.0)),
    ('imm', dict(R_base=0.005, sigma_flight=10.0, sigma_maneuver=80.0)),
    ('gravity_ekf_6d', dict(R_base=0.003)),
    ('fast_gravity_ekf', {}),
]

for fname, fkw in filter_configs:
    label = f"{fname}({','.join(f'{k}={v}' for k,v in fkw.items())})"[:24]
    acc, r01, r02, r03, segs, rout, n01 = test_config(
        all_frames, fname, best_gate, best_init, best_gap, **fkw)
    print(f"{label:<25} {acc:<7.1f} {r01:<8.4f} {r02:<8.4f} {r03:<8.4f} {segs:<5}")
