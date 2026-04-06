#!/usr/bin/env python3
"""Analyze calibration: decompose XYZ, compute Euclidean vs Z distance."""
import csv, math

xs, ys, zs, zms = [], [], [], []
with open('trajectory_data.csv', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            zm = float(row.get('z_mono', '0'))
            if z > 0.1:
                xs.append(x); ys.append(y); zs.append(z)
                if zm > 0.1: zms.append(zm)
        except (ValueError, KeyError):
            pass

def stats(arr):
    n = len(arr)
    mn = sum(arr) / n
    sd = (sum((a - mn)**2 for a in arr) / max(n-1,1))**0.5
    return mn, sd

# 3D position (from Kalman)
mx, sx = stats(xs)
my, sy = stats(ys)
mz, sz = stats(zs)

# Euclidean distance per frame
dists = [math.sqrt(x**2 + y**2 + z**2) for x,y,z in zip(xs,ys,zs)]
md, sd = stats(dists)

# z_mono raw
mzm, szm = stats(zms)

print(f"=== 1m Calibration Report ===")
print(f"Kalman 3D position:")
print(f"  x: mean={mx:.4f} std={sx:.4f}")
print(f"  y: mean={my:.4f} std={sy:.4f}")
print(f"  z: mean={mz:.4f} std={sz:.4f}")
print(f"  Euclidean: mean={md:.4f} std={sd:.4f}")
print()
print(f"z_mono raw:")
print(f"  mean={mzm:.4f} std={szm:.4f}")
print()
print(f"--- Interpretation ---")
print(f"True Z (ruler): 1.0000m")
print(f"Mono measures Euclidean distance (D), not pure Z")
print(f"  D = sqrt(x^2 + y^2 + z^2) = {md:.4f}")
print(f"  cos(theta) = Z/D = {mz/md:.4f}")
print(f"  => z_mono ~ D, so z_mono overestimates Z by +{(mzm/1.0 - 1)*100:.2f}%")
print(f"  => After cos correction: z_corrected = {mzm * (mz/md):.4f} (error={((mzm * mz/md)/1.0 - 1)*100:.2f}%)")
