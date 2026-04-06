#!/usr/bin/env python3
import csv, sys, glob, os

files = sorted(glob.glob('trajectory_calibrate_*.csv')) + sorted(glob.glob('trajectory_cal2_*.csv')) + ['trajectory_data.csv']
for fn in files:
    if not os.path.exists(fn):
        continue
    with open(fn) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    stereo_valid = sum(1 for r in rows if float(r['z_stereo']) > 0)
    mono_valid = sum(1 for r in rows if float(r['z_mono']) > 0)
    methods = {}
    for r in rows:
        m = r['depth_method']
        methods[m] = methods.get(m, 0) + 1
    print(f"{fn}: total={total}, mono={mono_valid}, stereo={stereo_valid}, methods={methods}")
