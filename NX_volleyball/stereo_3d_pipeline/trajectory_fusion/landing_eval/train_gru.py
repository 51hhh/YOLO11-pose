"""M4: tiny GRU residual corrector on top of the best physics EKF.

Formulation: at each in-window frame, the EKF (cd=0.10, q_vel=1.5,
sigma_d=0.4, nu=12) outputs a landing prediction. The GRU sees causal
per-frame features and predicts a 2D in-plane correction to that landing.
If the GRU outputs 0 it exactly reproduces the EKF.

Honest protocol for 15 windows / 13 segments: leave-one-segment-out CV.
Per-fold training is capped by epochs and wall-clock; whole script under
external timeout. CPU torch only.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import THROWS, stream_rows, observation
from models import EkfDrag, G_HAT

GT = json.load(open(os.path.join(os.path.dirname(__file__), "throws_gt.json")))
D0_PX = GT["d0"]
GROUND_H = GT["ground_h"]

# in-plane orthonormal basis
_tmp = np.array([0.0, 0.0, 1.0])
E1 = np.cross(G_HAT, _tmp)
E1 /= np.linalg.norm(E1)
E2 = np.cross(G_HAT, E1)

EKF_KW = dict(cd=0.10, cd_state=False, q_vel=1.5, sigma_d_px=0.4, nu=12.0)
BINS = [(0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.5)]
TRAIN_TIME_CAP_S = 90.0
EPOCHS = 300
HIDDEN = 16


def inplane(vec):
    return np.array([np.dot(vec, E1), np.dot(vec, E2)])


def build_sequences():
    """Per flight window: list of (features[T,F], ekf_land[T,3],
    target_resid[T,2], lead[T], run_id)."""
    seqs = []
    for rid in THROWS:
        seg = GT["segments"].get(rid)
        if not seg or not seg["events"]:
            continue
        ekf = EkfDrag(**EKF_KW)
        obs_idx = -1
        cur = None  # active window record
        for row in stream_rows(rid):
            obs = observation(row, d0=D0_PX)
            if obs is None:
                continue
            obs_idx += 1
            t, p, _ = obs
            out = ekf.update(t, p)
            ev = next((e for e in seg["events"]
                       if e["start_idx"] <= obs_idx < e["impact_idx"]), None)
            if ev is None:
                cur = None
                continue
            if cur is None or cur["ev"] is not ev:
                cur = {"ev": ev, "feat": [], "land": [], "resid": [], "lead": []}
                seqs.append((rid, cur))
            if out is None:
                continue
            lead = ev["t_impact"] - t
            if lead < 0.05:
                continue
            land = np.asarray(out["landing"])
            v = ekf.x[3:6]
            h = -float(np.dot(G_HAT, p)) - GROUND_H
            feat = [
                h / 3.0,
                p[2] / 10.0,                       # range
                float(np.dot(v, G_HAT)) / 10.0,    # vertical speed
                float(np.dot(v, E1)) / 10.0,
                float(np.dot(v, E2)) / 10.0,
                (out["t_impact"] - t) / 1.5,       # predicted time-to-land
                float(np.linalg.norm(inplane(land - p))) / 10.0,
            ]
            resid = inplane(np.asarray(ev["p_impact"]) - land)
            cur["feat"].append(feat)
            cur["land"].append(land)
            cur["resid"].append(resid)
            cur["lead"].append(lead)
    out_seqs = []
    for rid, c in seqs:
        if len(c["feat"]) >= 5:
            out_seqs.append({
                "run": rid,
                "feat": np.array(c["feat"], dtype=np.float32),
                "land": np.array(c["land"], dtype=np.float32),
                "resid": np.array(c["resid"], dtype=np.float32),
                "lead": np.array(c["lead"], dtype=np.float32),
                "p_impact": np.array(c["ev"]["p_impact"], dtype=np.float32),
            })
    return out_seqs


class TinyGRU(nn.Module):
    def __init__(self, nf, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y)


def train_fold(train_seqs, nf):
    torch.manual_seed(0)
    model = TinyGRU(nf)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(EPOCHS):
        if time.time() - t0 > TRAIN_TIME_CAP_S:
            break
        tot = 0.0
        for s in train_seqs:
            x = torch.from_numpy(s["feat"]).unsqueeze(0)
            y = torch.from_numpy(s["resid"]).unsqueeze(0)
            pred = model(x)
            loss = nn.functional.huber_loss(pred, y, delta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
    return model


def main():
    t_all = time.time()
    seqs = build_sequences()
    runs = sorted({s["run"] for s in seqs})
    nf = seqs[0]["feat"].shape[1]
    print(f"windows={len(seqs)} runs={len(runs)} features={nf}")

    recs_ekf, recs_gru = [], []
    infer_times = []
    for held in runs:
        train = [s for s in seqs if s["run"] != held]
        test = [s for s in seqs if s["run"] == held]
        model = train_fold(train, nf)
        model.eval()
        with torch.no_grad():
            for s in test:
                x = torch.from_numpy(s["feat"]).unsqueeze(0)
                t0 = time.perf_counter()
                corr = model(x)[0].numpy()
                infer_times.append((time.perf_counter() - t0) / len(s["feat"]))
                for i in range(len(s["feat"])):
                    gt2 = inplane(s["p_impact"])
                    base = inplane(s["land"][i])
                    # fade correction near impact: EKF is already sharp there
                    tti = float(s["feat"][i][5]) * 1.5  # un-normalize
                    w = min(1.0, max(0.0, (tti - 0.15) / 0.35))
                    e_ekf = float(np.linalg.norm(base - gt2))
                    e_gru = float(np.linalg.norm(base + w * corr[i] - gt2))
                    recs_ekf.append((float(s["lead"][i]), e_ekf))
                    recs_gru.append((float(s["lead"][i]), e_gru))
        print(f"fold {held}: done ({time.time()-t_all:.0f}s elapsed)", flush=True)

    def table(recs, name):
        cells = []
        for lo, hi in BINS:
            es = [e for l, e in recs if lo <= l < hi]
            cells.append(f"{lo}-{hi}s:{np.median(es):.2f}m(n={len(es)})" if es else f"{lo}-{hi}s:-")
        print(f"{name:<12} " + " ".join(cells))

    table(recs_ekf, "ekf (base)")
    table(recs_gru, "gru resid")
    print(f"gru inference: {1000*np.median(infer_times):.3f} ms/frame (torch cpu)")
    json.dump({
        "ekf": recs_ekf, "gru": recs_gru,
        "infer_ms": float(1000 * np.median(infer_times)),
    }, open(os.path.join(os.path.dirname(__file__), "results_gru.json"), "w"))


if __name__ == "__main__":
    main()
