#!/usr/bin/env python3
"""Analyze static ball measurement from trajectory_data.csv"""
import csv, statistics
from collections import Counter

data = []
with open('trajectory_data.csv') as f:
    for row in csv.DictReader(f):
        data.append(row)

# Filter track 0 (main ball)
t0 = [r for r in data if r['track_id'] == '0']
print(f'Total frames: {len(data)}, Track0: {len(t0)}')

zm_vals = [float(r['z_mono']) for r in t0 if float(r['z_mono']) > 0]
zs_vals = [float(r['z_stereo']) for r in t0 if float(r['z_stereo']) > 0]
z_kf = [float(r['z']) for r in t0]
methods = [int(r['depth_method']) for r in t0]

print(f'z_mono > 0: {len(zm_vals)}, z_stereo > 0: {len(zs_vals)}')
if t0:
    print(f'Stereo hit rate: {len(zs_vals)/len(t0)*100:.1f}%')

if zm_vals:
    print(f'z_mono   mean={statistics.mean(zm_vals):.4f} std={statistics.stdev(zm_vals):.4f} min={min(zm_vals):.4f} max={max(zm_vals):.4f}')
if zs_vals:
    print(f'z_stereo mean={statistics.mean(zs_vals):.4f} std={statistics.stdev(zs_vals):.4f} min={min(zs_vals):.4f} max={max(zs_vals):.4f}')
if z_kf:
    print(f'z_kalman mean={statistics.mean(z_kf):.4f} std={statistics.stdev(z_kf):.4f} min={min(z_kf):.4f} max={max(z_kf):.4f}')

# ratio zs/zm for frames where both are valid
paired = [(float(r['z_mono']), float(r['z_stereo'])) for r in t0 if float(r['z_mono']) > 0 and float(r['z_stereo']) > 0]
if paired:
    ratios = [zs/zm for zm, zs in paired]
    print(f'zs/zm ratio: mean={statistics.mean(ratios):.4f} std={statistics.stdev(ratios):.4f} N={len(ratios)}')

# method distribution
mc = Counter(methods)
print(f'Methods: {dict(mc)}')

# x, y stability for track 0
x_vals = [float(r['x']) for r in t0]
y_vals = [float(r['y']) for r in t0]
if x_vals:
    print(f'x mean={statistics.mean(x_vals):.4f} std={statistics.stdev(x_vals):.4f}')
    print(f'y mean={statistics.mean(y_vals):.4f} std={statistics.stdev(y_vals):.4f}')

# All tracks summary
tracks = sorted(set(r['track_id'] for r in data))
print(f'\nAll tracks: {tracks}')
for tid in tracks:
    td = [r for r in data if r['track_id'] == tid]
    zk = [float(r['z']) for r in td]
    print(f'  Track {tid}: {len(td)} frames, z_mean={statistics.mean(zk):.2f}')

# Temporal analysis: first 500 vs last 500 frames of track0
if len(t0) > 1000:
    early_zm = [float(r['z_mono']) for r in t0[:500] if float(r['z_mono']) > 0]
    late_zm = [float(r['z_mono']) for r in t0[-500:] if float(r['z_mono']) > 0]
    early_zs = [float(r['z_stereo']) for r in t0[:500] if float(r['z_stereo']) > 0]
    late_zs = [float(r['z_stereo']) for r in t0[-500:] if float(r['z_stereo']) > 0]
    if early_zm and late_zm:
        print(f'\nTemporal drift check:')
        print(f'  Early 500: zm={statistics.mean(early_zm):.4f}, zs={statistics.mean(early_zs):.4f if early_zs else "N/A"}')
        print(f'  Late  500: zm={statistics.mean(late_zm):.4f}, zs={statistics.mean(late_zs):.4f if late_zs else "N/A"}')
