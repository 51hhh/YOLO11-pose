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
        if zm > 0:
            rows.append((zm, zs, zk, m))
    except:
        pass

n = len(rows)
zms = [r[0] for r in rows]
zss = [r[1] for r in rows if r[1] > 0]
zks = [r[2] for r in rows]

def stats(arr):
    mn = sum(arr) / len(arr)
    sd = math.sqrt(sum((x - mn) ** 2 for x in arr) / len(arr))
    return mn, sd

mn_m, sd_m = stats(zms)
mn_s, sd_s = stats(zss)
mn_k, sd_k = stats(zks)

print(f"n={n}")
print(f"z_mono:   mean={mn_m:.4f}, std={sd_m:.4f}")
print(f"z_stereo: mean={mn_s:.4f}, std={sd_s:.4f}")
print(f"z_kalman: mean={mn_k:.4f}, std={sd_k:.4f}")
print(f"Overall k/m ratio: {sd_k / sd_m:.3f}")

rs = [zs / zm for zm, zs in [(r[0], r[1]) for r in rows if r[1] > 0]]
print(f"zs/zm: {sum(rs) / len(rs):.4f}")

print("\nSliding window (w=50):")
w = 50
for start in [0, 50, 100, 200, 500, 800, 1000]:
    end = min(start + w, n)
    if end - start < 20:
        continue
    seg_k = zks[start:end]
    seg_m = zms[start:end]
    _, dk = stats(seg_k)
    _, dm = stats(seg_m)
    r = dk / dm if dm > 1e-5 else 999
    print(f"  [{start:4d}:{end:4d}] k_std={dk:.4f}  m_std={dm:.4f}  ratio={r:.3f}")
