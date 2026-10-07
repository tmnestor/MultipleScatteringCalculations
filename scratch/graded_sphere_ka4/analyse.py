"""Successive differences and errors of saved far-field series. Usage: python analyse.py variant rc"""
import math
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).parent / "runs"
A = 10.0
var, rc = sys.argv[1], sys.argv[2]
suf = sys.argv[3] if len(sys.argv) > 3 else ""
files = sorted([p for p in R.glob(f"{var}_{rc}_*{suf}.npz") if p.stem.endswith(suf) and (suf or "_J" not in p.stem)], key=lambda p: int(p.stem.split("_")[2]))
ns = [int(p.stem.split("_")[2]) for p in files]
d = [np.load(p) for p in files]


def coef(z, p, key):
    return z[key][p] * A**-p


lead = max(np.abs(coef(d[0], 2, "exP")).max(), np.abs(coef(d[0], 2, "exS")).max())
for p in range(2, min(8, d[0]["P"].shape[0])):
    size = max(np.abs(coef(d[0], p, "exP")).max(), np.abs(coef(d[0], p, "exS")).max())
    norm = size if size > 1e-12 * lead else lead
    errs = [max(np.abs(coef(z, p, "P") - coef(z, p, "exP")).max(), np.abs(coef(z, p, "S") - coef(z, p, "exS")).max()) / norm for z in d]
    diffs = [max(np.abs(coef(d[i + 1], p, m) - coef(d[i], p, m)).max() for m in "PS") / norm for i in range(len(d) - 1)]
    o_err = [math.log(errs[i] / errs[i + 1]) / math.log(ns[i + 1] / ns[i]) for i in range(len(d) - 1)]
    o_dif = [math.log(diffs[i] / diffs[i + 1]) / math.log(ns[i + 2] / ns[i]) * 1 for i in range(len(diffs) - 1)]
    print(f"(ka)^{p} err " + " ".join(f"{e:.2e}" for e in errs) + " | ord " + " ".join(f"{o:.2f}" for o in o_err))
    print(f"      diff " + " ".join(f"{e:.2e}" for e in diffs) + " | ~ord " + " ".join(f"{o:.2f}" for o in o_dif))
