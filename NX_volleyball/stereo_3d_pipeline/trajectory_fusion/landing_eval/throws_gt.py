"""Build landing ground truth from throw segments.

Pipeline (all causal-safe: this is offline GT construction, models never
see it):
1. Stream d0-corrected bbox observations per throw.
2. Split into flight windows by detecting impacts: local maxima of camera-y
   (falling = y increases in y-down camera frame) with velocity reversal.
3. Fit quadratic per-axis on each flight window -> acceleration vector.
   Windows with good fit vote for the gravity direction; |a| should be
   near 9.81 (validates depth scale end-to-end).
4. Fit ground plane through impact points.
5. Emit events JSON: per impact, time + 3D point + owning flight window.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import THROWS, stream_rows, observation

D0 = json.load(open(os.path.join(os.path.dirname(__file__), "d0_fit.json")))
FB_FIT, D0_PX = D0["fB_fit"], D0["d0"]


def series(rid):
    ts, ps = [], []
    for row in stream_rows(rid):
        obs = observation(row, d0=D0_PX)
        if obs is None:
            continue
        t, p, _ = obs
        # convert depth to fitted fB scale
        ts.append(t)
        ps.append(p)
    return np.array(ts), np.array(ps)


def detect_impacts(ts, ps, min_gap_s=0.25, min_drop=0.35):
    """Impact = local max of y (camera y-down) that is a real reversal.

    min_drop: y must fall by this much (m) on both sides of the peak to
    count (rejects jitter). Returns list of index of peak.
    """
    y = ps[:, 1]
    n = len(y)
    impacts = []
    i = 1
    while i < n - 1:
        # local max in a small neighborhood
        lo = max(0, i - 3)
        hi = min(n, i + 4)
        if y[i] == y[lo:hi].max() and y[i] > y[lo] and y[i] > y[hi - 1]:
            # check real drop on both sides within 0.4s
            left = y[(ts >= ts[i] - 0.45) & (ts < ts[i])]
            right = y[(ts > ts[i]) & (ts <= ts[i] + 0.45)]
            if len(left) >= 3 and len(right) >= 3 and \
               y[i] - left.min() > min_drop and y[i] - right.min() > min_drop:
                if not impacts or ts[i] - ts[impacts[-1]] > min_gap_s:
                    impacts.append(i)
                    i += 5
                    continue
        i += 1
    return impacts


def flight_windows(ts, impacts, n):
    """Windows between impacts: [start_idx, end_idx(=impact idx)]"""
    bounds = [0] + [i + 1 for i in impacts]
    wins = []
    for k, imp in enumerate(impacts):
        s = bounds[k]
        if imp - s >= 8 and ts[imp] - ts[s] >= 0.35:
            wins.append((s, imp))
    return wins


def fit_accel(ts, ps, s, e):
    """Quadratic fit per axis on window; return accel vec + rms residual."""
    t = ts[s:e + 1] - ts[s]
    if len(t) < 8 or t[-1] < 0.3:
        return None, None
    A = np.stack([np.ones_like(t), t, 0.5 * t * t], axis=1)
    acc = np.zeros(3)
    rms = 0.0
    for ax in range(3):
        coef, res, *_ = np.linalg.lstsq(A, ps[s:e + 1, ax], rcond=None)
        acc[ax] = coef[2]
        pred = A @ coef
        rms += float(np.mean((pred - ps[s:e + 1, ax]) ** 2))
    return acc, math.sqrt(rms / 3)


def main():
    all_events = {}
    grav_votes = []
    for rid in THROWS:
        ts, ps = series(rid)
        if len(ts) < 20:
            all_events[rid] = {"n_obs": int(len(ts)), "events": []}
            continue
        impacts = detect_impacts(ts, ps)
        wins = flight_windows(ts, impacts, len(ts))
        events = []
        for s, e in wins:
            acc, rms = fit_accel(ts, ps, s, e)
            ev = {"start_idx": int(s), "impact_idx": int(e),
                  "t_start": float(ts[s]), "t_impact": float(ts[e]),
                  "p_impact": [float(v) for v in ps[e]],
                  "fit_rms": rms if rms is None else float(rms),
                  "accel": None if acc is None else [float(v) for v in acc]}
            if acc is not None and rms is not None and rms < 0.35:
                a = float(np.linalg.norm(acc))
                ev["accel_norm"] = a
                if 6.0 < a < 14.0:
                    grav_votes.append(acc)
            events.append(ev)
        all_events[rid] = {"n_obs": int(len(ts)), "events": events}
        print(f"{rid}: obs={len(ts)} impacts={len(impacts)} windows={len(wins)} "
              + " ".join(f"|a|={e.get('accel_norm', float('nan')):.2f}" for e in events))

    if not grav_votes:
        print("NO gravity votes — aborting")
        return
    G = np.array(grav_votes)
    g_dir = np.median(G, axis=0)
    g_norm = float(np.linalg.norm(g_dir))
    g_hat = g_dir / g_norm
    norms = np.linalg.norm(G, axis=1)
    print(f"\ngravity votes: {len(G)}  |g| median={np.median(norms):.3f} m/s^2 "
          f"(MAD {np.median(np.abs(norms-np.median(norms))):.3f})")
    print(f"g_hat (camera frame) = [{g_hat[0]:+.4f} {g_hat[1]:+.4f} {g_hat[2]:+.4f}]")

    # ground plane from impact points: h = -g_hat . p, plane h ~= const
    pts = []
    for rid, seg in all_events.items():
        for ev in seg["events"]:
            pts.append(ev["p_impact"])
    pts = np.array(pts)
    h = -(pts @ g_hat)
    h0 = float(np.median(h))
    print(f"impact points: {len(pts)}  height h spread: median={h0:.3f} "
          f"MAD={float(np.median(np.abs(h - h0))):.3f} m")

    out = {"g_hat": [float(v) for v in g_hat], "g_norm": float(np.median(norms)),
           "ground_h": h0, "d0": D0_PX, "fB_fit": FB_FIT,
           "segments": all_events}
    with open(os.path.join(os.path.dirname(__file__), "throws_gt.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote throws_gt.json")


if __name__ == "__main__":
    main()
