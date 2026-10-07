"""Density-only body: exact (ka)^p = exact Born + exact F2 (at p = 4), against the saved model runs."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import born_test as bt  # noqa: E402
import dens2  # noqa: E402

fs, gs = bt.fs, bt.gs
from cubic_scattering import MaterialContrast  # noqa: E402

gs.CORE = 0.1 * gs.RADIUS
gs.set_profile("smoothstep")
fs.CONTRAST = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=fs.hier.CONTRAST.Drho)
bt.CONTRAST = fs.CONTRAST
bt.OBS = fs.obs_points(gs.R_FAR, gs.THETA)
ex = bt.exact_born()
f2 = dens2.exact_f2() * gs.POL
dirs = bt.OBS / np.linalg.norm(bt.OBS, axis=1)[:, None]
for o, rh in enumerate(dirs):
    for mode, speed in (("P", fs.REF.alpha), ("S", fs.REF.beta)):
        pref = 1.0 / (4.0 * math.pi * fs.REF.rho * speed**2)
        amp = rh * (rh @ f2) if mode == "P" else f2 - rh * (rh @ f2)
        ex[mode][4, o] += pref * amp
R = Path(__file__).parent / "runs"
for p in (2, 4, 6):
    size = max(np.abs(ex[m][p]).max() for m in "PS")
    row = []
    for n in (4, 6, 8, 10, 12):
        f = R / f"rhoonly_1_{n}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        err = max(np.abs(z[m][p] - ex[m][p]).max() for m in "PS") / size
        row.append(f"n={n}:{err:.2e}")
    print(f"(ka)^{p}", " ".join(row))
