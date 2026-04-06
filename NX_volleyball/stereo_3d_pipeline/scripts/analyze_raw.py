#!/usr/bin/env python3
"""Analyze z_mono and z_stereo raw readings for calibration."""
import csv

zms, zss = [], []
with open('trajectory_data.csv', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            zm = float(row.get('z_mono', '0'))
            zs = float(row.get('z_stereo', '-1'))
            if zm > 0.1: zms.append(zm)
            if zs > 0.1: zss.append(zs)
        except ValueError:
            pass

def stats(arr, label):
    n = len(arr)
    if n == 0:
        print(f"{label}: no data")
        return
    mn = sum(arr) / n
    sd = (sum((x - mn)**2 for x in arr) / max(n - 1, 1))**0.5
    print(f"{label}: n={n}, mean={mn:.4f}, std={sd:.4f}, min={min(arr):.4f}, max={max(arr):.4f}")

stats(zms, "z_mono")
stats(zss, "z_stereo")
