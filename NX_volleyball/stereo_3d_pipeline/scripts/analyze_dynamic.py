#!/usr/bin/env python3
"""动态抛球轨迹质量分析"""
import csv, sys, math

rows = []
with open('trajectory_data.csv') as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"总帧数: {len(rows)}")

# 按track_id分组
tracks = {}
for r in rows:
    tid = int(float(r['track_id']))
    tracks.setdefault(tid, []).append(r)

print(f"轨迹数: {len(tracks)}")
for tid, frames in sorted(tracks.items(), key=lambda x: -len(x[1])):
    if len(frames) < 5:
        continue
    zs = [float(f['z']) for f in frames]
    z_min, z_max = min(zs), max(zs)
    
    # 速度平滑度: vz的帧间变化
    vzs = [float(f['vz']) for f in frames]
    dvz = [abs(vzs[i+1]-vzs[i]) for i in range(len(vzs)-1)]
    dvz_mean = sum(dvz)/len(dvz) if dvz else 0
    
    # 方法统计
    methods = [int(float(f['depth_method'])) for f in frames]
    m_counts = {0:0, 1:0, 2:0}
    for m in methods:
        m_counts[m] = m_counts.get(m, 0) + 1
    
    # z_kalman vs z_mono 对比
    z_kal = [float(f['z']) for f in frames]
    z_mono = [float(f['z_mono']) for f in frames if float(f['z_mono']) > 0.1]
    
    mono_std = 0
    if len(z_mono) > 1:
        mono_mean = sum(z_mono)/len(z_mono)
        mono_std = math.sqrt(sum((z-mono_mean)**2 for z in z_mono)/len(z_mono))
    
    method_str = f"M:{m_counts[0]} B:{m_counts[2]} S:{m_counts[1]}"
    print(f"  Track {tid}: {len(frames)}帧, z=[{z_min:.2f},{z_max:.2f}]m, "
          f"dvz_mean={dvz_mean:.3f}m/s², {method_str}")

# 全局z_kalman平滑度
print("\n--- Kalman 平滑度 ---")
zs = [float(r['z']) for r in rows]
dz = [abs(zs[i+1]-zs[i]) for i in range(len(zs)-1)]
dz_ok = [d for d in dz if d < 2.0]  # 排除轨迹切换
print(f"帧间|Δz| mean={sum(dz_ok)/len(dz_ok):.4f}m, max={max(dz_ok):.4f}m (排除>2m跳变)")

# stereo质量
print("\n--- Stereo 质量 ---")
stereo_frames = [(float(r['z_mono']), float(r['z_stereo'])) for r in rows 
                  if float(r['z_stereo']) > 0.1 and float(r['z_mono']) > 0.1]
if stereo_frames:
    diffs = [abs(zm-zs)/zm for zm, zs in stereo_frames]
    print(f"同帧 |zm-zs|/zm: mean={sum(diffs)/len(diffs)*100:.1f}%, max={max(diffs)*100:.1f}%, n={len(stereo_frames)}")
    # 按距离分段
    for lo, hi in [(1,2), (2,3), (3,4), (4,5)]:
        seg = [(zm,zs) for zm,zs in stereo_frames if lo <= zm < hi]
        if seg:
            d = [abs(zm-zs)/zm for zm,zs in seg]
            print(f"  {lo}-{hi}m: n={len(seg)}, |zm-zs|/zm={sum(d)/len(d)*100:.1f}%")
