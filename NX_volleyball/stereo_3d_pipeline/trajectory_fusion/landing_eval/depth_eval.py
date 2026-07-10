"""Depth accuracy evaluation (评价标准 1: 深度准).

Fits disparity zero offset d0 on static known-z buckets (3-11m, excluding
known-bad segments), reports per-bucket bias/MAD/RMS before/after
correction, plus leave-one-bucket-out validation of the fit.

Model: disp = fB * (1/z) + d0  (linear in 1/z, analytic LS)
       z_corr = fB_fit / (disp - d0)
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data import (STATIC_BUCKETS, EXCLUDE_FROM_FIT, LABEL_SUSPECT_BUCKETS,
                  FB, stream_rows, f)

FIELD = "disparity_bbox_center"


def collect(field=FIELD):
    """known_z -> list of median disparity per segment (robust per-seg stat)."""
    by_bucket = {}
    for kz, ids in STATIC_BUCKETS.items():
        for rid in ids:
            if rid in EXCLUDE_FROM_FIT:
                continue
            disps = [d for d in (f(r, field) for r in stream_rows(rid)) if d > 0]
            if len(disps) < 500:
                continue
            by_bucket.setdefault(kz, []).append(statistics.median(disps))
    return by_bucket


def fit_d0(by_bucket, exclude=frozenset(), fit_fb=True):
    xs, ys = [], []
    for kz, meds in by_bucket.items():
        if kz in exclude or kz in LABEL_SUSPECT_BUCKETS:
            continue
        for d in meds:
            xs.append(1.0 / kz)
            ys.append(d)
    n = len(xs)
    if fit_fb:
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        fb = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        d0 = (sy - fb * sx) / n
    else:
        fb = FB
        d0 = sum(y - fb * x for x, y in zip(xs, ys)) / n
    return fb, d0


def residual_table(by_bucket, fb, d0):
    rows = []
    for kz in sorted(by_bucket):
        raw = [FB / d for d in by_bucket[kz]]
        cor = [fb / (d - d0) for d in by_bucket[kz]]
        rows.append((kz,
                     statistics.median(raw) - kz,
                     statistics.median(cor) - kz))
    return rows


def main():
    by_bucket = collect()
    fb, d0 = fit_d0(by_bucket)
    _, d0_fixed = fit_d0(by_bucket, fit_fb=False)
    print(f"fit (fB free) : fB={fb:.3f} (calib {FB:.3f}, {100*(fb/FB-1):+.2f}%)  d0={d0:+.3f}px")
    print(f"fit (fB fixed): d0={d0_fixed:+.3f}px")

    print(f"\n{'kz':>5} {'raw bias':>10} {'corrected':>10}")
    for kz, rb, cb in residual_table(by_bucket, fb, d0):
        tag = " (label suspect, not fit)" if kz in LABEL_SUSPECT_BUCKETS else ""
        print(f"{kz:>5} {rb:>+9.3f}m {cb:>+9.3f}m{tag}")

    # leave-one-bucket-out: fit without bucket k, evaluate on k
    print("\nleave-one-bucket-out (corrected bias on held-out bucket):")
    errs = []
    for kz in sorted(by_bucket):
        if kz in LABEL_SUSPECT_BUCKETS:
            continue
        fb_l, d0_l = fit_d0(by_bucket, exclude={kz})
        cor = [fb_l / (d - d0_l) for d in by_bucket[kz]]
        e = statistics.median(cor) - kz
        errs.append(abs(e))
        print(f"  {kz:>5}: {e:+.3f}m  (fit d0={d0_l:+.2f})")
    print(f"  held-out |bias|: median {statistics.median(errs):.3f}m  max {max(errs):.3f}m")

    out = {"fB_fit": fb, "d0": d0, "d0_fb_fixed": d0_fixed, "field": FIELD,
           "fit_buckets": sorted(k for k in by_bucket if k not in LABEL_SUSPECT_BUCKETS)}
    with open(os.path.join(os.path.dirname(__file__), "d0_fit.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote d0_fit.json")


if __name__ == "__main__":
    main()
