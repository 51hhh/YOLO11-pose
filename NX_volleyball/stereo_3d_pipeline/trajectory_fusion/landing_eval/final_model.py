"""Final deployable causal landing predictor: tuned EKF + TinyGRU residual.

- Trains the GRU on ALL windows (deployment artifact gru_final.pt);
  honest generalization numbers come from train_gru.py's LOO CV.
- CausalLandingPredictor: strict per-frame streaming API, GRU hidden state
  carried across frames, reset on temporal gaps / upward velocity flips
  (bounce); correction faded near impact.
- Measures full metrics (accuracy bins, lock-on, jitter, ms/frame,
  in-sample caveat applies to accuracy but not to timing/stability).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import models as M
import eval_landing as E
from train_gru import (build_sequences, train_fold, TinyGRU, EKF_KW,
                       inplane, E1, E2, GROUND_H)
from models import EkfDrag, G_HAT

CKPT = os.path.join(os.path.dirname(__file__), "gru_final.pt")


def train_final():
    seqs = build_sequences()
    nf = seqs[0]["feat"].shape[1]
    t0 = time.time()
    model = train_fold(seqs, nf)
    dt = time.time() - t0
    torch.save({"state": model.state_dict(), "nf": nf, "hidden": 16,
                "ekf_kw": EKF_KW}, CKPT)
    print(f"trained final GRU on {len(seqs)} windows in {dt:.0f}s -> gru_final.pt")
    return model, nf


class CausalLandingPredictor:
    """Streaming EKF + GRU residual. update(t, p, q) -> dict|None."""

    def __init__(self, ckpt=CKPT):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = TinyGRU(blob["nf"], blob["hidden"])
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.reset()

    def reset(self):
        self.ekf = EkfDrag(**EKF_KW)
        self.h = None
        self.prev_vdown = None
        self.prev_t = None

    def update(self, t, p, q=None):
        if self.prev_t is not None and (t - self.prev_t) > 0.5:
            self.h = None  # temporal gap: reset GRU memory
        self.prev_t = t
        out = self.ekf.update(t, p, q)
        if out is None:
            return None
        v = self.ekf.x[3:6]
        vdown = float(np.dot(v, G_HAT))
        # bounce: downward -> strongly upward flip resets GRU memory
        if self.prev_vdown is not None and self.prev_vdown > 1.0 and vdown < -1.0:
            self.h = None
        self.prev_vdown = vdown
        land = np.asarray(out["landing"])
        hgt = -float(np.dot(G_HAT, np.asarray(p))) - GROUND_H
        tti = out["t_impact"] - t
        feat = torch.tensor([[[
            hgt / 3.0,
            p[2] / 10.0,
            vdown / 10.0,
            float(np.dot(v, E1)) / 10.0,
            float(np.dot(v, E2)) / 10.0,
            tti / 1.5,
            float(np.linalg.norm(inplane(land - np.asarray(p)))) / 10.0,
        ]]], dtype=torch.float32)
        with torch.no_grad():
            y, self.h = self.net.gru(feat, self.h)
            corr = self.net.head(y)[0, 0].numpy()
        w = min(1.0, max(0.0, (tti - 0.15) / 0.35))
        corr = np.clip(corr, -1.0, 1.0) * w      # bounded correction
        land_corr = land + corr[0] * E1 + corr[1] * E2
        return {"landing": land_corr, "t_impact": out["t_impact"]}


if __name__ == "__main__":
    if not os.path.exists(CKPT) or "--retrain" in sys.argv:
        train_final()
    # restore default bbox observer (obs_variants may have patched it)
    import data
    E.observation = data.observation
    E.make_model = lambda n: CausalLandingPredictor()
    s, _ = E.eval_model("m4_stream")
    s["model"] = "M4 stream"
    print(E.fmt(s))
    json.dump(s, open(os.path.join(os.path.dirname(__file__),
                                   "results_m4_stream.json"), "w"), indent=1)
