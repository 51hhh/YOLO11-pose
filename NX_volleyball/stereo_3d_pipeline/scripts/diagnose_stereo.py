#!/usr/bin/env python3
import csv, math, io, sys

f = sys.argv[1] if len(sys.argv) > 1 else 'trajectory_data.csv'
with open(f, 'rb') as fp:
    raw = fp.read().replace(b'\x00', b'')
rows = []
for r in csv.DictReader(io.StringIO(raw.decode('utf-8', 'replace'))):
    try:
        zm = float(r['z_mono']); zs = float(r['z_stereo'])
        zk = float(r['z']); m = int(r['depth_method'])
        if zm > 0: rows.append((zm, zs, zk, m))
    except: pass

n = len(rows)
print(f"Total frames: {n}")
stereo_ok = sum(1 for r in rows if r[1] > 0)
print(f"Stereo valid: {stereo_ok}/{n} ({100*stereo_ok/n:.1f}%)")

zs_valid = [r[1] for r in rows if r[1] > 0]
if zs_valid:
    mn = sum(zs_valid)/len(zs_valid)
    sd = math.sqrt(sum((x-mn)**2 for x in zs_valid)/len(zs_valid))
    outliers_2sd = sum(1 for x in zs_valid if abs(x-mn) > 2*sd)
    outliers_3sd = sum(1 for x in zs_valid if abs(x-mn) > 3*sd)
    print(f"z_stereo: mean={mn:.4f}, std={sd:.4f}")
    print(f"  >2sigma: {outliers_2sd} ({100*outliers_2sd/len(zs_valid):.1f}%)")
    print(f"  >3sigma: {outliers_3sd} ({100*outliers_3sd/len(zs_valid):.1f}%)")
    print(f"  min={min(zs_valid):.3f}, max={max(zs_valid):.3f}, range={max(zs_valid)-min(zs_valid):.3f}")
    zs_sorted = sorted(zs_valid)
    for p in [5, 25, 50, 75, 95]:
        idx = int(p/100 * len(zs_sorted))
        print(f"  P{p}: {zs_sorted[idx]:.3f}")

ratios = [r[1]/r[0] for r in rows if r[1] > 0]
mn_r = sum(ratios)/len(ratios)
sd_r = math.sqrt(sum((x-mn_r)**2 for x in ratios)/len(ratios))
print(f"\nzs/zm ratio: mean={mn_r:.4f}, std={sd_r:.4f}")
for label, start in [("first100", 0), ("mid", n//2), ("last100", max(0, n-100))]:
    seg = [r[1]/r[0] for r in rows[start:start+100] if r[1] > 0]
    if seg:
        print(f"  {label}: ratio={sum(seg)/len(seg):.4f}")

print("\nMono/Stereo noise over time (100-frame windows):")
for start in range(0, n, 200):
    end = min(start+100, n)
    if end - start < 20: continue
    seg_m = [r[0] for r in rows[start:end]]
    mn_m = sum(seg_m)/len(seg_m); sd_m = math.sqrt(sum((x-mn_m)**2 for x in seg_m)/len(seg_m))
    seg_s = [r[1] for r in rows[start:end] if r[1] > 0]
    sd_s = 0; mn_s = 0
    if seg_s:
        mn_s = sum(seg_s)/len(seg_s); sd_s = math.sqrt(sum((x-mn_s)**2 for x in seg_s)/len(seg_s))
    ratio_seg = [r[1]/r[0] for r in rows[start:end] if r[1] > 0]
    ratio_m = sum(ratio_seg)/len(ratio_seg) if ratio_seg else 0
    print(f"  [{start:4d}:{end:4d}] m_std={sd_m:.4f} s_std={sd_s:.4f} m_mean={mn_m:.3f} s_mean={mn_s:.3f} ratio={ratio_m:.4f}")
