"""Observation variants: per-method (fB,d0)-calibrated observers and
fallback chains. Evaluates landing metrics per observation policy on the
tuned physics EKF.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import data
from data import stream_rows, f, FX, FY, CX, CY

# per-method calibration fitted on static buckets (depth_eval.fit_d0)
METHODS = {
    "bbox": {"disp": "disparity_bbox_center", "u": "left_bbox_cx",
             "v": "left_bbox_cy", "fB": 1493.085, "d0": -4.143},
    "circle": {"disp": "disparity_circle_center", "u": "left_circle_cx",
               "v": "left_circle_cy", "fB": 1487.8, "d0": -3.88},
    "xfeat": {"disp": "disparity_roi_neural_xfeat", "u": "left_bbox_cx",
              "v": "left_bbox_cy", "fB": 1560.9, "d0": -11.13},
}


def make_observer(chain):
    """chain: list of method names, first usable wins."""
    def obs(row, d0=None):
        t = f(row, "timestamp")
        if t <= 0:
            return None
        for name in chain:
            m = METHODS[name]
            disp = f(row, m["disp"])
            u, v = f(row, m["u"]), f(row, m["v"])
            if disp <= 0 or u < 0 or v < 0:
                continue
            denom = disp - m["d0"]
            if denom <= 1.0:
                continue
            z = m["fB"] / denom
            x = (u - CX) * z / FX
            y = (v - CY) * z / FY
            return t, (x, y, z), {"method": name}
        return None
    return obs


if __name__ == "__main__":
    import models as M
    import eval_landing as E

    variants = {
        "bbox_only": ["bbox"],
        "circle_only": ["circle"],
        "bbox>circle>xfeat": ["bbox", "circle", "xfeat"],
        "circle>bbox>xfeat": ["circle", "bbox", "xfeat"],
    }
    out = {}
    for vname, chain in variants.items():
        E.observation = make_observer(chain)
        M.make_model = lambda n: M.EkfDrag(cd=0.10, cd_state=False,
                                           q_vel=1.5, sigma_d_px=0.4, nu=12.0)
        E.make_model = M.make_model
        s, _ = E.eval_model("x")
        s["model"] = vname
        out[vname] = s
        print(E.fmt(s), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                     "results_obs_variants.json"), "w"), indent=1)
