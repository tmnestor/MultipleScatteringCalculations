#!/usr/bin/env python3
"""MEASUREMENT, not a gate: replace the exact solve of the sparsified system by a thesis-style sweep.

WHY.  ``measure_sparsify_preconditioner.py`` preconditions (I - G0 T0) with
M r = H^{-1} P r, H the sparse 27-voxel system, solved EXACTLY by sparse LU.
Exact LU does not scale.  The thesis's own preconditioner (Ch. 6, Alg. 6.4,
Omega~ = I + P~_F Delta C) is a Riccati up/down sweep that keeps transmission
and the forward-scattered phases and IGNORES back- and side-scattering.

THE SWEEPS.  H is block-tridiagonal in depth planes (a 27-voxel stencil couples
z-1, z, z+1).  Exact block LU is the Riccati sweep

    S_1 = H_11,  S_z = H_zz - H_{z,z-1} S_{z-1}^{-1} H_{z-1,z}     (downsweep)
    back-substitution with S_z^{-1} H_{z,z+1}                       (upsweep)

and the variants measured:
  exact Riccati      -- the full Schur complement.  Mathematically exact LU, so
                        it MUST reproduce the LU iteration counts: the check on
                        the sweep code.
  one-way            -- S_z ~ H_zz: the Schur term, which carries what planes
                        above REFLECT back, is dropped (thesis: ignore
                        backscattering).  A down sweep then an up sweep
                        (symmetric block Gauss-Seidel over planes).
  one-way, nested    -- the same, with each plane solve H_zz itself replaced by
                        the same one-way sweep over x-lines: the full
                        alternating-direction scheme (up/down, left/right).
                        Every solve is one line of 9 N unknowns.

Also: ``ref Schur`` -- the Schur terms of the contrast-free system (T0 = 0),
once per background, the closest analogue of the thesis's P~_F; ``band r`` --
the exact recursion with each S_z truncated to lateral radius r (sparse).

RESULT (2026-09-27).  GMRES iterations at 8^3, whole space:
    rho      none  exact LU  exact Riccati  one-way  nested  ref Schur  band 1  band 2  band 3
    0.063      7       6          6            56       63       8        35      22      13
    0.687     23      11         11            59       66      23        37      25      17
    2.480     99      23         23            73       87      58        50      40      32
    6.412    998      50         50           185      279     332       229     279     302
The exact Riccati sweep reproduces exact LU (the check).  Every cheap
approximation loses most of the gain at strong contrast.  Dropping the Schur
term is WORSE than no preconditioner at weak contrast: S_z is the discrete
Dirichlet-to-Neumann map of the half-domain above the plane, and dropping it
makes every plane a reflector.  The reference impedance ignores the
heterogeneity's own backscatter, which dominates at strong contrast.  A lateral
band of the DtN converges slowly in r and is not even monotone at rho 6.4 -- the
DtN map is laterally NONLOCAL, which is why Liu & Ying approximate S^{-1} with a
moving PML rather than truncate it.

Run:  conda run -n seismic python scripts/measure_sparsify_sweep.py [N]
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

N = mp.N  # set by the first argument, as in measure_sparsify_preconditioner
T_START = time.perf_counter()


def stage(label: str) -> None:
    """Print one timed stage."""
    print(f"  [{time.perf_counter() - T_START:6.0f} s] {label}", flush=True)


def blocks(h: sp.csc_matrix, n_blk: int) -> tuple[list, list, list]:
    """Diagonal, sub- and super-diagonal blocks of a block-tridiagonal matrix.

    Args:
        h: The matrix; dof order puts each block contiguous.
        n_blk: Number of blocks.

    Returns:
        (D, L, U): D[k] = H_kk, L[k] = H_{k,k-1} (k >= 1), U[k] = H_{k,k+1} (k < n-1).
    """
    m = h.shape[0] // n_blk
    hc = h.tocsr()
    sl = [slice(k * m, (k + 1) * m) for k in range(n_blk)]
    d = [hc[sl[k], sl[k]].tocsc() for k in range(n_blk)]
    lo = [None] + [hc[sl[k], sl[k - 1]].tocsc() for k in range(1, n_blk)]
    up = [hc[sl[k], sl[k + 1]].tocsc() for k in range(n_blk - 1)] + [None]
    # Nothing may couple beyond the neighbouring block, or the sweep is not exact.
    for k in range(n_blk):
        for j in range(n_blk):
            if abs(j - k) > 1 and hc[sl[k], sl[j]].count_nonzero():
                msg = (
                    f"block ({k}, {j}) is nonzero: the matrix is not block-tridiagonal.\n"
                    "  Where: scripts/measure_sparsify_sweep.py, blocks\n"
                    "  Valid: a stencil coupling only neighbouring planes (27-voxel)\n"
                    "  Fix:   check the dof ordering (plane-major) or the stencil reach."
                )
                raise ValueError(msg)
    return d, lo, up


def riccati_exact(d: list, lo: list, up: list):
    """Exact block LU as a Riccati sweep, with dense Schur complements.

    Args:
        d: Diagonal blocks.
        lo: Sub-diagonal blocks.
        up: Super-diagonal blocks.

    Returns:
        Callable f -> H^{-1} f.
    """
    n = len(d)
    s_lu = []
    s_inv_up: list = []
    for k in range(n):
        s = d[k].toarray()
        if k:
            s = s - lo[k] @ s_inv_up[k - 1]
        lu = np.linalg.inv(s)
        s_lu.append(lu)
        s_inv_up.append(lu @ up[k].toarray() if k < n - 1 else None)
    m = d[0].shape[0]

    def solve(f: NDArray) -> NDArray:
        fb = f.reshape(n, m)
        y = np.zeros_like(fb)
        for k in range(n):  # downsweep
            y[k] = s_lu[k] @ (fb[k] - (lo[k] @ y[k - 1] if k else 0.0))
        for k in range(n - 2, -1, -1):  # upsweep
            y[k] = y[k] - s_inv_up[k] @ y[k + 1]
        return y.ravel()

    return solve


def riccati_with_schur(d: list, lo: list, up: list, corr: list):
    """Block LU sweep with GIVEN Schur corrections: S_z = H_zz - corr[z].

    Args:
        d: Diagonal blocks of the system being preconditioned.
        lo: Its sub-diagonal blocks.
        up: Its super-diagonal blocks.
        corr: Dense Schur corrections per block (corr[0] = 0).

    Returns:
        Callable f -> approximate H^{-1} f.
    """
    n = len(d)
    s_inv = [np.linalg.inv(d[k].toarray() - corr[k]) for k in range(n)]
    s_inv_up = [s_inv[k] @ up[k].toarray() if k < n - 1 else None for k in range(n)]
    m = d[0].shape[0]

    def solve(f: NDArray) -> NDArray:
        fb = f.reshape(n, m)
        y = np.zeros_like(fb, dtype=complex)
        for k in range(n):
            y[k] = s_inv[k] @ (fb[k] - (lo[k] @ y[k - 1] if k else 0.0))
        for k in range(n - 2, -1, -1):
            y[k] = y[k] - s_inv_up[k] @ y[k + 1]
        return y.ravel()

    return solve


def schur_corrections(d: list, lo: list, up: list, band: NDArray | None = None) -> list:
    """The Riccati Schur terms L_z S_{z-1}^{-1} U_{z-1}, optionally banded laterally.

    With ``band`` (a boolean plane mask), each S_z is truncated to it before the
    next step: a sparse, laterally local impedance -- the scalable approximation.

    Args:
        d: Diagonal blocks.
        lo: Sub-diagonal blocks.
        up: Super-diagonal blocks.
        band: Optional lateral mask for S_z.

    Returns:
        Dense corrections per block, corr[0] = 0.
    """
    n = len(d)
    corr = [np.zeros(d[0].shape, dtype=complex)]
    s = d[0].toarray()
    for k in range(1, n):
        if band is not None:
            s = np.where(band, s, 0.0)
        c = lo[k] @ np.linalg.solve(s, up[k - 1].toarray())
        if band is not None:
            c = np.where(band, c, 0.0)
        corr.append(np.asarray(c))
        s = d[k].toarray() - corr[k]
    return corr


def lateral_band(r: int) -> NDArray:
    """Plane mask: dof pairs whose voxels are within lateral distance r (|dx|, |dy| <= r)."""
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    xs = np.repeat(x.ravel(), 9)
    ys = np.repeat(y.ravel(), 9)
    return (np.abs(xs[:, None] - xs[None, :]) <= r) & (np.abs(ys[:, None] - ys[None, :]) <= r)


def one_way(d_solve: list, lo: list, up: list, d: list):
    """Symmetric block Gauss-Seidel: Schur complement S_z replaced by H_zz.

    M^{-1} = (D + U)^{-1} D (D + L)^{-1}: a down sweep, then an up sweep.

    Args:
        d_solve: Per-block solvers for H_kk (callables).
        lo: Sub-diagonal blocks.
        up: Super-diagonal blocks.
        d: Diagonal blocks (for the middle D).

    Returns:
        Callable f -> M^{-1} f.
    """
    n = len(d)
    m = d[0].shape[0]

    def solve(f: NDArray) -> NDArray:
        fb = f.reshape(n, m).astype(complex)
        y = np.zeros_like(fb)
        for k in range(n):  # down: (D + L) y = f
            y[k] = d_solve[k](fb[k] - (lo[k] @ y[k - 1] if k else 0.0))
        w = np.array([d[k] @ y[k] for k in range(n)])
        u = np.zeros_like(fb)
        for k in range(n - 1, -1, -1):  # up: (D + U) u = D y
            u[k] = d_solve[k](w[k] - (up[k] @ u[k + 1] if k < n - 1 else 0.0))
        return u.ravel()

    return solve


def nested_plane_solver(h_zz: sp.csc_matrix):
    """One-way sweep over the x-lines of one plane, each line solved exactly.

    Args:
        h_zz: A plane block (dof order x, y, component within the plane).

    Returns:
        Callable f -> approximate H_zz^{-1} f.
    """
    d, lo, up = blocks(h_zz, N)
    lus = [splu(b).solve for b in d]
    return one_way(lus, lo, up, d)


def main() -> int:
    """GMRES iterations: exact LU / exact Riccati / one-way / nested one-way.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"SWEEPING THE SPARSIFIED SYSTEM: {N}^3 voxels, whole space, invariant stencils")
    print("=" * 78)
    g = mp.dense_g0()
    stage("dense G0")
    types = mp.stencils_invariant(mp.kernel_window(8), 7)
    sten = mp.place_invariant(types)
    row_scale = np.tile(np.r_[np.ones(3), np.full(6, mp.PITCH)], N**3)
    stage("invariant stencils")
    size = g.shape[0]
    rng = np.random.default_rng(1)
    b: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    # The REFERENCE impedance: the Schur terms of the contrast-free system (T0 = 0),
    # computed once per background -- the thesis's P~_F keeps the reference medium's
    # transmission and reflection and ignores only the heterogeneity's backscatter.
    h_ref, _ = mp.assemble(g, np.zeros((size, size)), sten, row_scale)
    corr_ref = schur_corrections(*blocks(h_ref, N))
    bands = {r: lateral_band(r) for r in (1, 2, 3)}
    stage("reference Schur terms (contrast-free, once)")
    cols = (
        "rho",
        "none",
        "exact LU",
        "exact Riccati",
        "one-way",
        "nested",
        "ref Schur",
        "band 1",
        "band 2",
        "band 3",
    )
    widths = (7, 6, 9, 14, 8, 7, 10, 7, 7, 7)
    print(f"\n  {'strength':>8} " + " ".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True)))
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = mp.t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(eigs(g @ t, k=1, which="LM", return_eigenvectors=False))[0])
        h, p = mp.assemble(g, t, sten, row_scale)
        d, lo, up = blocks(h, N)
        solvers = {
            "exact LU": splu(h).solve,
            "exact Riccati": riccati_exact(d, lo, up),
            "one-way": one_way([splu(x).solve for x in d], lo, up, d),
            "nested": one_way([nested_plane_solver(x) for x in d], lo, up, d),
            "ref Schur": riccati_with_schur(d, lo, up, corr_ref),
            "band 1": riccati_with_schur(d, lo, up, schur_corrections(d, lo, up, bands[1])),
            "band 2": riccati_with_schur(d, lo, up, schur_corrections(d, lo, up, bands[2])),
            "band 3": riccati_with_schur(d, lo, up, schur_corrections(d, lo, up, bands[3])),
        }
        row = [mp.run_gmres(a, b, None)]
        for name, solve in solvers.items():
            row.append(mp.run_gmres(a, b, lambda v, s=solve, pp=p: s(pp @ v)))
            stage(f"strength {strength:g}: {name} done")
        cells = [f"{rho:7.3f}"] + [f"{v:{w}d}" for v, w in zip(row, widths[1:], strict=True)]
        print(f"  {strength:8g} " + " ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
