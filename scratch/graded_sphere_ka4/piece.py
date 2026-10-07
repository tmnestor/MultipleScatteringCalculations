"""Self-convergence of a piece = variant A minus variant B (or A alone), coefficient (ka)^p.
Usage: python piece.py A B|- rc"""
import math
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).parent / "runs"
a_var, b_var, rc = sys.argv[1:4]
suf = sys.argv[4] if len(sys.argv) > 4 else ""


def load(v):
    fs = [p for p in R.glob(f"{v}_{rc}_*.npz") if (p.stem.endswith(suf) if suf else "_J" not in p.stem)]
    return {int(p.stem.split("_")[2]): np.load(p) for p in fs}


A = load(a_var)
B = load(b_var) if b_var != "-" else None
ns = sorted(set(A) & (set(B) if B else set(A)))
ref = A[ns[0]]
for p in range(2, min(8, ref["P"].shape[0])):
    lead = max(np.abs(ref["exP"][2]).max(), np.abs(ref["exS"][2]).max()) * 10.0**-2
    size = max(np.abs(ref["exP"][p]).max(), np.abs(ref["exS"][p]).max()) * 10.0**-p
    norm = size if size > 1e-12 * lead else lead
    vals = [np.concatenate([(A[n][m][p] - (B[n][m][p] if B else 0)).ravel() for m in "PS"]) * 10.0**-p / norm for n in ns]
    d = [np.abs(vals[i + 1] - vals[i]).max() for i in range(len(ns) - 1)]
    # for an error C h^r, d_i = C (h_i^r - h_{i+1}^r): fit r from consecutive ratios numerically
    orders = []
    for i in range(len(d) - 1):
        n0, n1, n2 = ns[i : i + 3]
        f = lambda r: (n0**-r - n1**-r) / (n1**-r - n2**-r) - d[i] / d[i + 1]
        lo, hi = 0.1, 12.0
        if f(lo) * f(hi) > 0:
            orders.append(float("nan")); continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        orders.append(mid)
    print(f"(ka)^{p} |piece| {np.abs(vals[-1]).max():.2e}  diffs " + " ".join(f"{x:.2e}" for x in d)
          + "  orders " + " ".join(f"{o:.2f}" for o in orders))
