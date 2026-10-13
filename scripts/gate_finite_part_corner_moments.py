"""Gate: corner-box moments of the finite-part kernels, Euler's identity and Duffy pyramids against 25 digits.

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (13 October: K1 blocks).
Paper: LatexPDFs/InterfaceCouplingIntegrals, Eqs. (loghomogeneous), (eulerlog), Table "verification".

int_box x^n f over a box with the singular point at a vertex, f = d^alpha Phi_j (interface_image), by
_corner_moments (Euler's identity with -int x^n g) and _corner_pyramids (Duffy, exact rho^e log rho),
against an mpmath evaluation over the same three pyramids with tanh-sinh in the radial coordinate (no
homogeneity, no log identity).  Families: Phi1..Phi4 with and without a log term, two static-type controls and
Phi0; monomials (0,0,0), (1,0,2), (2,1,1).  A monomial with 3 + |n| + d <= 0 has no integral: both routes must
refuse it.  Measured 13 October 2026: 26 cases, worst 8.3e-16 (Euler), 4.3e-16 (pyramids); the divergent
case refused by both.  About 5 CPU-hours: run in workers, ... k n (k of n), each appending to its own
log in CORNER_LOG_DIR (default .); a rerun skips the cases already logged.

Run:  PYTHONPATH=. python scripts/gate_finite_part_corner_moments.py [k n] > $CORNER_LOG_DIR/corner_k.log
"""

import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

from cubic_scattering import interface_image as ii

mp.mp.dps = 25
FAMS = [
    (1, (0, 0, 0)),
    (2, (0, 0, 0)),
    (3, (0, 2, 0)),
    (3, (1, 0, 0)),
    (4, (2, 0, 0)),
    (4, (1, 1, 0)),
    (3, (3, 0, 0)),
    (4, (2, 2, 0)),
    (0, (1, 0, 1)),
]
BOX = (
    (0, 2),
    (-2, 0),
    (0, 2),
)  # corner at the origin, the singular point; a negative lateral side too
MONOS = [(0, 0, 0), (1, 0, 2), (2, 1, 1)]


def reference(j, alpha, mono):
    expr = ii._potential(j)
    expr = (
        sp.diff(expr, ii._RX, alpha[0], ii._RY, alpha[1], ii._ZETA, alpha[2])
        if sum(alpha)
        else expr
    )
    fm = sp.lambdify((ii._RX, ii._RY, ii._ZETA), expr, "mpmath")
    far = [lo if hi == 0 else hi for lo, hi in BOX]
    vol = abs(far[0] * far[1] * far[2])
    tot = mp.mpf(0)
    for k in range(3):  # pyramid k: t_k = rho, t_others = rho v
        o = [i for i in range(3) if i != k]

        def integrand(rho, v1, v2, k=k, o=o):
            t = [None] * 3
            t[k], t[o[0]], t[o[1]] = rho, rho * v1, rho * v2
            s = [far[i] * t[i] for i in range(3)]
            return rho**2 * s[0] ** mono[0] * s[1] ** mono[1] * s[2] ** mono[2] * fm(*s)

        tot += mp.quad(integrand, [0, 1], [0, 1], [0, 1])
    return vol * tot


RUNS = Path(os.environ.get("CORNER_LOG_DIR", "."))
#: worker k of n (argv): takes every n-th of the cases not yet logged; each worker appends to its own log,
#: and every worker reads all logs, so a restart resumes after the last finished case.
WORKER, N_WORKERS = (
    (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
)
done = "".join(f.read_text() for f in sorted(RUNS.glob("corner*.log")))


def finished(j, alpha, mono):
    """The case is already in a log (a restart resumes after it)."""
    return any(
        ln.startswith(f"Phi{j} d^{alpha} ") and f"x^{mono}:" in ln
        for ln in done.splitlines()
    )


TODO = [(j, a, m) for j, a in FAMS for m in MONOS if not finished(j, a, m)]
MINE = set(TODO[WORKER::N_WORKERS])
if os.environ.get("ONLY_J") is not None:  # one potential only, e.g. a rerun of a family
    MINE = {c for c in TODO if c[0] == int(os.environ["ONLY_J"])}
worst_e, worst_p = 0.0, 0.0
for j, alpha in FAMS:
    if not any((j, alpha, m) in MINE for m in MONOS):
        continue
    f = ii.kernel_function(j, alpha)
    g = ii.kernel_log_coefficient(j, alpha)
    d = ii.kernel_degree(j, alpha)
    deg = (2, 1, 2)
    t0 = time.time()
    euler = ii._corner_moments(f, d, BOX, deg, 40, g)
    # pyramids: weights are the monomials themselves (Poly in s about the corner)
    ws = [
        {
            m: tuple(Fraction(int(i == m)) for i in range(m + 1))
            for m in range(deg[ax] + 1)
        }
        for ax in range(3)
    ]
    # a monomial with 3 + |n| + d <= 0 has no integral at the singular corner: both routes must refuse it
    valid = [m for m in MONOS if 3 + sum(m) + d > 0]
    for m in MONOS:
        if m not in valid and (j, alpha, m) in MINE:
            refused = np.isnan(euler[m])
            try:
                ii._corner_pyramids(f, d, BOX, ws, [m], 40, g)
            except ValueError:
                refused = refused and True
            else:
                refused = False
            print(
                f"Phi{j} d^{alpha} (d={d}) x^{m}: divergent (3 + |n| + d = {3 + sum(m) + d}); "
                f"refused by Euler and pyramids: {refused}",
                flush=True,
            )
    pyr = ii._corner_pyramids(f, d, BOX, ws, valid, 40, g)
    for mono, pv in zip(valid, pyr):
        if (j, alpha, mono) not in MINE:
            continue
        ref = reference(j, alpha, mono)
        ee = abs(euler[mono] - float(ref)) / abs(float(ref))
        pe = abs(pv - float(ref)) / abs(float(ref))
        worst_e, worst_p = max(worst_e, ee), max(worst_p, pe)
        print(
            f"Phi{j} d^{alpha} (d={d}, log term {'yes' if g else 'no'}) x^{mono}: Euler {ee:.1e}, pyramids {pe:.1e}",
            flush=True,
        )
print(f"worst: Euler {worst_e:.1e}, pyramids {worst_p:.1e}")
