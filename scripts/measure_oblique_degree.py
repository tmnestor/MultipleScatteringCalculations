#!/usr/bin/env python3
"""The voxel of degree p at oblique incidence and for incident S: order against the exact layer.

The scheme is that of ``crosscheck_first_moment_voxel.Scheme`` (Bloch coupling by Poisson summation, every
integral by quadrature) with the voxel basis the Legendre products of total degree at most p in (z, x, y):
1 function for p = 0, 4 for p = 1, 10 for p = 2.  The exact answer is the reflection coefficient of the
homogeneous layer from the package's Kennett recursion, an independent algorithm.  Compared is the
same-type coefficient (R_PP for incident P, R_SS for SV, R_SH for SH), which does not depend on how the
waves are normalised; a sign of +-1 between the two conventions is accepted.

Measured, for each incident wave and angle: the relative error on n = 1, 2, 4 planes for p = 0, 1, 2, and
the order between successive grids, against 2p + 2.

Run:  conda run -n seismic python -u scripts/measure_oblique_degree.py [omega] [p_max]
SI units; e^{-i omega t}; z down.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crosscheck_first_moment_voxel import Scheme  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack, kennett_layers  # noqa: E402

ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
CONTRAST = {"dlambda": 2e9, "dmu": 1e9, "drho": 100.0}
D_LAYER = 2.0
BASES = {0: [1], 1: [1, 2, 3, 4], 2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
CASES = (("P", 0.0), ("P", 20.0), ("SV", 20.0), ("SH", 20.0), ("SH", 40.0))
NS = (1, 2, 4)


def exact(incident: str, theta: float, omega: float) -> complex:
    """The same-type reflection coefficient of the layer, by Kennett's recursion."""
    mu, lam = RHO * BETA**2, RHO * (ALPHA**2 - 2 * BETA**2)
    m1, l1, r1 = mu + CONTRAST["dmu"], lam + CONTRAST["dlambda"], RHO + CONTRAST["drho"]
    stack = LayerStack(
        [
            IsotropicLayer(ALPHA, BETA, RHO, 100.0),
            IsotropicLayer(np.sqrt((l1 + 2 * m1) / r1), np.sqrt(m1 / r1), r1, D_LAYER),
            IsotropicLayer(ALPHA, BETA, RHO, np.inf),
        ]
    )
    slowness = np.sin(np.deg2rad(theta)) / (ALPHA if incident == "P" else BETA)
    res = kennett_layers(stack, slowness, np.array([omega]))
    return {
        "P": complex(res.RD_psv[0][0, 0]),
        "SV": complex(res.RD_psv[0][1, 1]),
        "SH": complex(res.RD_sh[0]),
    }[incident]


def scheme(incident: str, theta: float, omega: float, n: int, p: int, p_max: int) -> complex:
    """The same-type reflection coefficient of n planes of voxels of degree p."""
    ref = ReferenceMedium(ALPHA, BETA, RHO)
    kw = omega / (ALPHA if incident == "P" else BETA)
    kx = kw * np.sin(np.deg2rad(theta))
    amp_s, amp_p = Scheme(
        ref, CONTRAST, D_LAYER, omega, kx, n, BASES[p], p_max if theta else 0, incident
    ).solve()
    gz = np.sqrt(kw**2 - kx**2)
    if incident == "P":  # upgoing P is polarised along its wavevector (-gz, kx, 0)
        return complex(amp_p @ np.array([-gz, kx, 0.0]) / kw)
    if incident == "SV":  # upgoing SV: in the (z, x) plane, normal to (-gz, kx, 0)
        return complex(amp_s @ np.array([kx, gz, 0.0]) / kw)
    return complex(amp_s[2])


def main() -> int:
    omega = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0
    p_max = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.perf_counter()
    print(f"omega = {omega}: k_P D = {omega / ALPHA * D_LAYER:.2f}, k_S D = {omega / BETA * D_LAYER:.2f}")
    print(f"lattice orders |g| <= {p_max}; planes n = {NS}; error of the same-type reflection coefficient")
    oks = []
    for incident, theta in CASES:
        ex = exact(incident, theta, omega)
        for p in (0, 1, 2):
            errs = []
            for n in NS:
                r = scheme(incident, theta, omega, n, p, p_max)
                errs.append(min(abs(r - ex), abs(r + ex)) / abs(ex))
            orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(NS) - 1)]
            ok = abs(orders[-1] - (2 * p + 2)) < 0.35
            oks.append(ok)
            print(
                f"  {incident:2s} {theta:4.0f} deg  p = {p}: "
                + "  ".join(f"{e:.2e}" for e in errs)
                + "  orders "
                + " ".join(f"{o:.2f}" for o in orders)
                + f"  (predicted {2 * p + 2}) {'PASS' if ok else 'FAIL'}"
                + f"  [{time.perf_counter() - t0:4.0f} s]",
                flush=True,
            )
    print(f"{sum(oks)}/{len(oks)} orders as predicted")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
