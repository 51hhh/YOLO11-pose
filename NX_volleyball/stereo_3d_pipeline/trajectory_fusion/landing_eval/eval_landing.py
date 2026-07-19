"""Landing prediction evaluation harness.

Feeds observations frame-by-frame (causal), scores predictions inside
each flight window against the GT impact point of that window.

Metrics per model:
- 准 accuracy : in-plane landing error by lead-time bin
- 快 speed    : per window, longest lead time from which prediction error
                stays < THRESH until impact ("lock-on horizon", larger=faster)
- 稳 stability: median frame-to-frame |Δlanding|; availability rate
- 实时 runtime: mean per-frame update wall time (ms)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import THROWS, stream_rows, observation
from models import make_model, G_HAT

GT = json.load(open(os.path.join(os.path.dirname(__file__), "throws_gt.json")))
D0_PX = GT["d0"]
BINS = [(0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.5)]
THRESH = 0.5  # m, lock-on threshold


def inplane_err(pred, gt):
    d = np.asarray(pred) - np.asarray(gt)
    d = d - np.dot(d, G_HAT) * G_HAT  # project onto ground plane
    return float(np.linalg.norm(d))


def eval_model(name, runs=THROWS):
    model = make_model(name)
    recs = []          # (run, win_id, lead, err, t)
    jitters = []
    n_updates = 0
    n_valid = 0
    wall = 0.0
    for rid in runs:
        seg = GT["segments"].get(rid)
        if not seg or not seg["events"]:
            continue
        events = seg["events"]
        model.reset()
        obs_idx = -1
        prev_land = None
        for row in stream_rows(rid):
            obs = observation(row, d0=D0_PX)
            if obs is None:
                continue
            obs_idx += 1
            t, p, q = obs
            t0 = time.perf_counter()
            out = model.update(t, p, q)
            wall += time.perf_counter() - t0
            n_updates += 1
            # find owning flight window by time (robust to observer changes)
            ev = next((e for e in events
                       if e["t_start"] <= t <= e["t_impact"]), None)
            if ev is None:
                prev_land = None
                continue
            lead = ev["t_impact"] - t
            if lead < 0.05:
                continue
            if out is None:
                prev_land = None
                continue
            n_valid += 1
            err = inplane_err(out["landing"], ev["p_impact"])
            recs.append((rid, ev["impact_idx"], lead, err))
            if prev_land is not None:
                jitters.append(float(np.linalg.norm(
                    np.asarray(out["landing"]) - prev_land)))
            prev_land = np.asarray(out["landing"])

    # accuracy by lead bin
    acc = {}
    for lo, hi in BINS:
        es = [e for _, _, l, e in recs if lo <= l < hi]
        acc[f"{lo}-{hi}s"] = (float(np.median(es)), len(es)) if es else (None, 0)
    # lock-on horizon per window
    horizons = []
    for key in {(r, w) for r, w, _, _ in recs}:
        wrecs = sorted([(l, e) for r, w, l, e in recs
                        if (r, w) == key])  # ascending lead
        h = 0.0
        for l, e in wrecs:                  # find max lead with all-below
            if e < THRESH:
                h = l
            else:
                break
        horizons.append(h)
    summary = {
        "model": name,
        "acc_by_lead": acc,
        "lockon_median_s": float(np.median(horizons)) if horizons else 0.0,
        "lockon_p25_s": float(np.percentile(horizons, 25)) if horizons else 0.0,
        "windows": len(horizons),
        "jitter_median_m": float(np.median(jitters)) if jitters else None,
        "valid_rate": n_valid / max(1, n_updates),
        "ms_per_frame": 1000.0 * wall / max(1, n_updates),
    }
    return summary, recs


def fmt(s):
    a = s["acc_by_lead"]
    cells = " ".join(
        f"{k}:{v[0]:.2f}m(n={v[1]})" if v[0] is not None else f"{k}:-"
        for k, v in a.items())
    return (f"{s['model']:<14} {cells}  lock-on(med/p25)={s['lockon_median_s']:.2f}/"
            f"{s['lockon_p25_s']:.2f}s  jitter={s['jitter_median_m'] and round(s['jitter_median_m'],3)}m "
            f"valid={s['valid_rate']:.0%} {s['ms_per_frame']:.3f}ms/f")


if __name__ == "__main__":
    names = sys.argv[1:] or ["ca_kalman", "polyfit", "ekf_nodrag_t",
                             "ekf_drag_t", "ekf_cd"]
    out = {}
    for name in names:
        s, _ = eval_model(name)
        out[name] = s
        print(fmt(s), flush=True)
    with open(os.path.join(os.path.dirname(__file__), "results_physics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
