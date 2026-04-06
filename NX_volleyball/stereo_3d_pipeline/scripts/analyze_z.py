#!/usr/bin/env python3
"""Analyze z_kalman from trajectory CSV (works with old and new formats)."""
import csv, sys

zk = []
with open('trajectory_data.csv', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            z = float(row['z'])
            if z > 0.5:
                zk.append(z)
        except (ValueError, KeyError):
            pass

n = len(zk)
if n == 0:
    print("No valid z data"); sys.exit(1)

mn = sum(zk) / n
sd = (sum((x - mn)**2 for x in zk) / (n - 1))**0.5
print(f"n={n}, z_kalman: mean={mn:.4f}, std={sd:.4f}")

for s, e in [(0,50),(50,100),(100,200),(200,300),(500,550),(1000,1050)]:
    w = zk[s:e]
    if len(w) > 5:
        wm = sum(w) / len(w)
        ws = (sum((x - wm)**2 for x in w) / (len(w) - 1))**0.5
        print(f"  [{s}:{e}] mean={wm:.4f} std={ws:.4f}")
