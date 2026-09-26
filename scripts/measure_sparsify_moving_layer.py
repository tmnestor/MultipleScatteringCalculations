#!/usr/bin/env python3
"""MEASUREMENT, not a gate: approximate the sweep's Schur inverses by a moving absorbing layer.

WHY.  ``measure_sparsify_sweep.py``: the sparsified system H is block-tridiagonal
in depth; its exact LU is the Riccati sweep with Schur complements
S_z = D_z - L_z S_{z-1}^{-1} U_{z-1}, and S_z -- the discrete Dirichlet-to-Neumann
map of the half-domain above plane z -- can be neither dropped nor banded.  Liu
& Ying (SISC 2018) approximate S_z^{-1} by a small LOCAL problem: the last b
planes down to z, with an absorbing layer standing in for everything above.

THE APPROXIMATION.  T_z v := solve the sub-system of H on planes z-b+1 .. z
(clipped at the domain top), right-hand side v on plane z and zero elsewhere,
and read plane z.  When the window reaches the top plane nothing is cut off
and T_z is exact (Liu & Ying, their eq. 20).  Otherwise the cut is closed:
  truncated  -- nothing added: the cut is a reflecting boundary;
  sponge     -- n_pad planes prepended, built from the contrast-free system's
                interior blocks (the background, as Liu & Ying's PML region is
                background), with an imaginary diagonal shift
                i sigma_k c, sigma_k = sigma ((k+1)/n_pad)^2 ramping away from
                the window, c the median |diagonal| of the background block.
A sponge is the crude absorber; Liu & Ying's PML stencils, fitted to
complex-stretched plane waves, are the refined one.  sigma is SCANNED and every
value reported -- it is not tuned to the answer.

CHECK: b = N (full history, never cut) must reproduce exact LU.

RESULT (2026-09-27).  The check passes at 4^3 and 8^3.  The sponge never helps:
at 8^3 with a 4-plane pad, over sigma = 0.03 ... 1, it matches or worsens plain
truncation (b = 4, sigma = 0.1 fails to converge even at rho 0.69).  Window
depth does not converge either.  GMRES iterations, truncated / best sponge:
    rho 0.063:  b=1 56/52   b=2 31/31   b=3 24/24   b=4 23/26    (exact 6)
    rho 0.687:  b=1 59/56   b=2 35/35   b=3 29/30   b=4 31/31    (exact 11)
    rho 2.480:  b=1 73/70   b=2 48/48   b=3 47/51   b=4 56/60    (exact 23)
    rho 6.412:  b=1 185/193; b=2 and b=3 fail for every sigma     (exact 50)
(b = 3 sigma = 1 and b = 4 at rho 6.4 were not run: stopped once every b >= 2
case at that contrast had failed.)  A diagonal shift reflects when weak and
decouples when strong; neither absorbs.  A window that is cut introduces
resonances rather than approximating the Dirichlet-to-Neumann map.

Run:  conda run -n seismic python scripts/measure_sparsify_moving_layer.py [N] [n_pad]
SI units, whole space, as measure_sparsify_preconditioner.py.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs, splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import measure_sparsify_preconditioner as mp  # noqa: E402
from scripts.measure_sparsify_sweep import blocks  # noqa: E402

N = mp.N
N_PAD = int(sys.argv[2]) if len(sys.argv) > 2 else 2
T_START = time.perf_counter()


def stage(label: str) -> None:
    """Print one timed stage."""
    print(f"  [{time.perf_counter() - T_START:6.0f} s] {label}", flush=True)


def moving_layer_sweep(d: list, lo: list, up: list, b: int, pad: dict | None):
    """The Riccati sweep with S_z^{-1} replaced by a windowed, optionally padded, local solve.

    Args:
        d: Diagonal blocks of H.
        lo: Sub-diagonal blocks.
        up: Super-diagonal blocks.
        b: Window depth in planes (b >= N: never cut, exact).
        pad: None (truncated) or {"d": pad diagonal blocks (list, outermost first),
            "lo": background sub-diagonal block, "up": background super-diagonal block}.

    Returns:
        Callable f -> approximate H^{-1} f.
    """
    n = len(d)
    m = d[0].shape[0]
    solvers = []
    for z in range(n):
        top = max(0, z - b + 1)
        planes = list(range(top, z + 1))
        diag = [d[k] for k in planes]
        low = [None] + [lo[k] for k in planes[1:]]
        upp = [up[k] for k in planes[:-1]] + [None]
        if top > 0 and pad is not None:
            diag = list(pad["d"]) + diag
            low = [None] + [pad["lo"]] * len(pad["d"]) + low[1:]
            upp = [pad["up"]] * len(pad["d"]) + upp
        nb = len(diag)
        rows = []
        for i in range(nb):
            row = [None] * nb
            row[i] = diag[i]
            if i > 0:
                row[i - 1] = low[i]
            if i < nb - 1:
                row[i + 1] = upp[i]
            rows.append(row)
        solvers.append((splu(sp.bmat(rows, format="csc")), nb))

    def t_apply(z: int, v: NDArray) -> NDArray:
        lu, nb = solvers[z]
        rhs = np.zeros(nb * m, dtype=complex)
        rhs[-m:] = v
        return lu.solve(rhs)[-m:]

    def solve(f: NDArray) -> NDArray:
        fb = f.reshape(n, m).astype(complex)
        y = np.zeros_like(fb)
        for z in range(n):
            y[z] = t_apply(z, fb[z] - (lo[z] @ y[z - 1] if z else 0.0))
        for z in range(n - 2, -1, -1):
            y[z] = y[z] - t_apply(z, up[z] @ y[z + 1])
        return y.ravel()

    return solve


def sponge(h_ref: sp.csc_matrix, sigma: float) -> dict:
    """Background blocks of an interior plane, with a ramped imaginary diagonal shift.

    Args:
        h_ref: The contrast-free sparsified system.
        sigma: Peak shift, in units of the median |diagonal| of the background block.

    Returns:
        {"d": [outermost ... innermost pad diagonal], "lo": ..., "up": ...}.
    """
    d, lo, up = blocks(h_ref, N)
    zi = N // 2
    c = float(np.median(np.abs(d[zi].diagonal())))
    eye = sp.identity(d[zi].shape[0], dtype=complex, format="csc")
    pads = [d[zi] + 1j * sigma * ((k + 1) / N_PAD) ** 2 * c * eye for k in range(N_PAD)]
    return {"d": pads[::-1], "lo": lo[zi], "up": up[zi]}


def main() -> int:
    """GMRES iterations: exact LU; window b truncated / sponge-padded, for several sigma.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"MOVING ABSORBING LAYER: {N}^3 voxels, whole space, invariant stencils, pad {N_PAD} planes")
    print("=" * 78)
    g = mp.dense_g0()
    types = mp.stencils_invariant(mp.kernel_window(8), 7)
    sten = mp.place_invariant(types)
    row_scale = np.tile(np.r_[np.ones(3), np.full(6, mp.PITCH)], N**3)
    size = g.shape[0]
    h_ref, _ = mp.assemble(g, np.zeros((size, size)), sten, row_scale)
    stage("dense G0, invariant stencils, contrast-free system")
    rng = np.random.default_rng(1)
    rhs: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    sigmas = (0.03, 0.1, 0.3, 1.0)
    pads = {s: sponge(h_ref, s) for s in sigmas}
    variants: list[tuple[str, int | None, float | None]] = [
        ("exact LU", None, None),
        (f"b={N} (check)", N, None),
    ]
    for depth in (1, 2, 3, 4):
        variants.append((f"b={depth} trunc", depth, None))
        variants += [(f"b={depth} sp{s:g}", depth, s) for s in sigmas]
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = mp.t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(eigs(g @ t, k=1, which="LM", return_eigenvectors=False))[0])
        h, p = mp.assemble(g, t, sten, row_scale)
        d, lo, up = blocks(h, N)
        print(f"\n  strength {strength:g}: rho {rho:.3f}; none {mp.run_gmres(a, rhs, None)}", flush=True)
        for name, b, s in variants:
            solve = (
                splu(h).solve
                if b is None
                else moving_layer_sweep(d, lo, up, b, None if s is None else pads[s])
            )
            its = mp.run_gmres(a, rhs, lambda v, sv=solve, pp=p: sv(pp @ v))
            print(f"    {name:14s} {its:5d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
