#!/usr/bin/env python3
"""Analyze 4m IVW fusion test data"""
import csv, sys, math

f = '/home/nvidia/NX_volleyball/stereo_3d_pipeline/trajectory_data.csv'
rows = []
with open(f) as fp:
    reader = csv.DictReader(fp)
    for r in reader:
        zm = float(r['z_mono'])
        zs = float(r['z_stereo'])
        zk = float(r['z'])
        m = int(r['depth_method'])
        if zm > 0:
            rows.append((zm, zs, zk, m))

n = len(rows)
if n == 0:
    print("No valid data"); sys.exit(1)

# Basic stats
zms = [r[0] for r in rows]
zss = [r[1] for r in rows if r[1] > 0]
zks = [r[2] for r in rows]
methods = [r[3] for r in rows]

def stats(arr, name):
    if not arr: return
    mn = sum(arr)/len(arr)
    std = math.sqrt(sum((x-mn)**2 for x in arr)/len(arr))
    print(f"{name}: n={len(arr)}, mean={mn:.4f}, std={std:.4f}, min={min(arr):.4f}, max={max(arr):.4f}")

stats(zms, "z_mono")
stats(zss, "z_stereo") 
stats(zks, "z_kalman")

# Method distribution
from collections import Counter
mc = Counter(methods)
print(f"\nMethod distribution: {dict(mc)}")
print(f"  0=mono: {mc.get(0,0)}, 1=stereo: {mc.get(1,0)}, 2=fusion: {mc.get(2,0)}")

# Stereo hit rate
stereo_valid = len([r for r in rows if r[1] > 0])
print(f"\nStereo hit rate: {stereo_valid}/{n} = {stereo_valid/n*100:.1f}%")

# Bias ratio (where both valid)
both = [(r[0], r[1]) for r in rows if r[1] > 0]
if both:
    ratios = [zs/zm for zm, zs in both]
    avg_ratio = sum(ratios)/len(ratios)
    print(f"zs/zm ratio: {avg_ratio:.4f} (bias={1-avg_ratio:.4f})")

# Fusion vs mono comparison (for method=2 frames)
fusion_frames = [(r[0], r[1], r[2]) for r in rows if r[3] == 2]
if fusion_frames:
    print(f"\n--- Fusion frames (method=2): {len(fusion_frames)} ---")
    fzm = [f[0] for f in fusion_frames]
    fzs = [f[1] for f in fusion_frames]
    fzk = [f[2] for f in fusion_frames]
    stats(fzm, "  z_mono(fusion)")
    stats(fzs, "  z_stereo(fusion)")
    stats(fzk, "  z_kalman(fusion)")
    
# Mono-only frames
mono_frames = [(r[0], r[1], r[2]) for r in rows if r[3] == 0]
if mono_frames:
    print(f"\n--- Mono frames (method=0): {len(mono_frames)} ---")
    mzk = [f[2] for f in mono_frames]
    stats(mzk, "  z_kalman(mono)")
