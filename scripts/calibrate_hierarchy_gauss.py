#!/usr/bin/env python3
"""Calibrate the Gauss order of the moment hierarchy's cell-to-cell tables against a tolerance.

The table between two cells, T[i, n, D, W](o) = int_cube (d_D G_in)(o - xi) xi^W d xi, has a smooth
integrand (the field point is the centre of another cell, at least half a cell outside the source cube),
and a tensor Gauss rule converges fast. For each Chebyshev distance and tolerance this finds the least
number of points per axis whose error, against 40 points, stays below the tolerance at that order and
every higher one, over every D and W of the third-gradient system (|D|, |W| <= 4), k_S d = 0.05, 1 and 3,
and offsets on the axes, the face and the body diagonals. The result is the table
``measure_graded_sphere_gradient_hierarchy.GAUSS_POINTS``.

Run:  python scripts/calibrate_hierarchy_gauss.py   (about two minutes)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.graded_voxel import derivatives as gd  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
SIDE = 2.0
OFFSETS = {
    1: [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
    2: [(2, 0, 0), (2, 1, 1), (2, 2, 2)],
    3: [(3, 0, 0), (3, 2, 1)],
    4: [(4, 0, 0), (4, 3, 2)],
    6: [(6, 0, 0)],
    8: [(8, 0, 0)],
}
TOLS = (1e-8, 1e-10, 1e-12)


def main() -> int:
    d_list = list(gd.multi_indices(4))
    w_list = list(gd.multi_indices(4))
    for reach, offsets in OFFSETS.items():
        need = dict.fromkeys(TOLS, 0)
        for ks in (0.05, 1.0, 3.0):
            omega = ks * REF.beta / SIDE
            for off in offsets:
                o = SIDE * np.array(off, float)
                ref = gd.moment_table(o, SIDE, omega, REF, d_list, w_list, 40)
                scale = np.abs(ref).max()
                errs = {
                    n: np.abs(gd.moment_table(o, SIDE, omega, REF, d_list, w_list, n) - ref).max() / scale
                    for n in range(4, 29, 2)
                }
                for t in TOLS:
                    ok = [n for n in errs if all(errs[m] < t for m in errs if m >= n)]
                    need[t] = max(need[t], min(ok) if ok else 99)
        print(f"reach {reach}: points per axis needed " + ", ".join(f"tol {t:g}: {need[t]}" for t in TOLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
