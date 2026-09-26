#!/usr/bin/env python3
"""MEASUREMENT, not a gate: does a Liu-Ying sparsified system precondition the 3-D Foldy-Lax solve?

WHY.  ``measure_sparsify_stencil_accuracy.py`` showed 27-voxel stencils annihilate
the elastic far field to ~1e-3 at 8 points per S wavelength.  Annihilation is
necessary, not sufficient: the stencils must assemble into a sparse system that
approximates the physics well enough to cut GMRES iterations, above all at
strong contrast, where the unpreconditioned solve is expensive.

THE SYSTEM is the solver's own: (I - G0 T0) psi = psi_inc, G0 assembled densely
through ``directional_sweeps.apply_g0_3d`` (so its self-term conventions are the
solver's), T0 physical cube T-matrices (``compute_slab_tmatrices``), a distinct
contrast per voxel, scaled up to push the spectral radius of G0 T0 past 1.

THE PRECONDITIONER, following Liu & Ying (SISC 2018) with two idealisations
that make it an UPPER BOUND on what sparsification can deliver:
  * per voxel i, 9 stencils alpha_i = the 9 smallest left singular vectors of
    G0[mu_i, mu_i^c], fitted against EVERY other voxel of this domain (no PML
    approximation of the exterior);
  * the sparse system H is solved EXACTLY by sparse LU (no sweep).
Row i:  alpha_i^H psi_mu - beta_i^H T0 psi_mu = alpha_i^H r_mu,  beta_i^H = alpha_i^H G0[mu, mu].
Applied as a RIGHT preconditioner, M r = H^{-1} P r, P r = alpha^H r_mu.

CONTROL: H truncated -- I - G0 restricted to the 27-voxel neighbourhood, no
stencils.  The gap between the two is what the annihilation buys.

RESULT (2026-09-26, ``variants`` mode).  The smallest-singular stencils FAIL:
their stencil operator P is singular (cond 2e18 at 6^3) and GMRES never
converges, although |PA - H|/|PA| is 6e-4 -- annihilation is necessary, not
sufficient.  CENTRE-ANCHORED stencils (``stencils_centred``: own block = I,
neighbours by regularised least squares) make P invertible (cond 3e3) and cut
iterations; at 6^3, none / 27-voxel truncation / centred (lam = 1e-3):
    rho 0.06: 7 / 7 / 5      rho 0.57: 20 / 16 / 9
    rho 1.98: 61 / 36 / 13   rho 4.99: 269 / 142 / 28
TRANSLATION-INVARIANT stencils (``invariant`` mode: 27 neighbourhood types,
each fitted ONCE against a fixed far window R = 7, independent of the domain)
keep nearly all of it.  none / per-domain / invariant:
    6^3:  rho 0.06: 7/5/5   0.57: 20/9/10   1.98: 61/13/16   4.99: 269/28/34
    8^3:  rho 0.06: 7/6/6   0.69: 23/10/11  2.48: 99/20/23   6.41: 998/43/50
Still idealised: H solved by exact LU (no sweep), whole-space background.

Run:  conda run -n seismic python scripts/measure_sparsify_preconditioner.py [N] [variants|invariant]
SI units (m, Pa, kg/m3); real omega, as the end-to-end gate.
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigs, gmres, splu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.directional_sweeps import SweepGrid3D, apply_g0_3d, build_g0_cache_3d  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6  # voxels per axis
A_HALF = 50.0  # cube half-width, m; pitch 100 m
PITCH = 2 * A_HALF
OM = 2 * np.pi * REF.beta / (8 * PITCH)  # 8 points per S wavelength
TOL = 1e-8


def dense_g0() -> NDArray:
    """G0 as a dense matrix, column by column through the solver's own apply_g0_3d.

    Returns:
        Shape (9 N^3, 9 N^3), index (z, x, y, component).
    """
    grid = SweepGrid3D(n_z=N, n_x=N, n_y=N, pitch=PITCH)
    cache = build_g0_cache_3d(grid, REF, OM)
    size = 9 * N**3
    g = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        g[:, c] = apply_g0_3d(e.reshape(N, N, N, 9), cache).ravel()
    return g


def t_blocks(strength: float) -> NDArray:
    """Physical cube T-matrices, a distinct contrast per voxel, times ``strength``.

    Args:
        strength: Contrast multiplier on (2 GPa, 1 GPa, 100 kg/m3).

    Returns:
        Shape (N, N, N, 9, 9).
    """
    rng = np.random.default_rng(20260926)
    scale = strength * (0.5 + rng.random((N, N, N)))
    mat = SlabMaterial(Dlambda=2.0e9 * scale, Dmu=1.0e9 * scale, Drho=100.0 * scale, ref=REF)
    return compute_slab_tmatrices(SlabGeometry(M=N, N_z=N, a=A_HALF), mat, OM)


def neighbourhoods() -> list[NDArray]:
    """For each voxel, the flat indices of its (up to) 27-voxel neighbourhood."""
    out = []
    for z, x, y in itertools.product(range(N), repeat=3):
        nb = [
            (zz * N + xx) * N + yy
            for zz, xx, yy in itertools.product(
                range(max(z - 1, 0), min(z + 2, N)),
                range(max(x - 1, 0), min(x + 2, N)),
                range(max(y - 1, 0), min(y + 2, N)),
            )
        ]
        out.append(np.array(nb))
    return out


def dofs(vox: NDArray) -> NDArray:
    """The 9 degrees of freedom of each voxel in ``vox``."""
    return (9 * vox[:, None] + np.arange(9)[None, :]).ravel()


def stencils(g: NDArray, nbs: list[NDArray], row_scale: NDArray) -> tuple[list, float]:
    """Per voxel, 9 stencils annihilating G0 from every voxel outside its neighbourhood.

    Args:
        g: Dense G0.
        nbs: Neighbourhoods.
        row_scale: Per-dof row scaling (strain rows x pitch), applied before fitting.

    Returns:
        (list of (mu dofs, alpha of shape (9|mu|, 9)), worst annihilation ratio).
    """
    out, worst = [], 0.0
    all_dofs = np.arange(g.shape[0])
    for nb in nbs:
        mu = dofs(nb)
        far = np.setdiff1d(all_dofs, mu)
        k = row_scale[mu, None] * g[np.ix_(mu, far)]
        # Thin: k is (9|mu|) x (far dofs) with far >> rows, so U is already square.
        # full_matrices=True would also build the unused far x far factor -- 85 min.
        u, s, _ = np.linalg.svd(k, full_matrices=False)
        alpha = u[:, -9:]
        # ratio of what the 9 kept stencils leave to the block's largest response
        worst = max(worst, float(np.linalg.norm(alpha.conj().T @ k, 2) / s[0]))
        out.append((mu, alpha))
    return out, worst


def stencils_centred(
    g: NDArray, nbs: list[NDArray], row_scale: NDArray, lam: float = 0.0
) -> tuple[list, float]:
    """Per voxel, 9 stencils with the voxel's OWN block pinned to the identity.

    The smallest-singular-vector stencils of ``stencils`` are not independent
    across voxels: P came out singular (cond 1.9e16 at 4^3) and the
    preconditioner diverged.  Anchoring row i on voxel i, as a finite-difference
    stencil is anchored on its centre, makes P = I + (neighbour blocks).  With
    alpha = [I; X] over (centre, others) the far leakage is K_c + X^H K_o, and

        X^H = -K_c K_o^H (K_o K_o^H + lam I)^{-1}      (lam = 0: least squares).

    Args:
        g: Dense G0.
        nbs: Neighbourhoods.
        row_scale: Per-dof row scaling, applied before fitting.
        lam: Tikhonov weight on X, relative to the largest eigenvalue of K_o K_o^H.

    Returns:
        (list of (mu dofs, alpha of shape (9|mu|, 9)), worst leakage |alpha^H K| / |K_c|).
    """
    out, worst = [], 0.0
    all_dofs = np.arange(g.shape[0])
    for i, nb in enumerate(nbs):
        order = np.r_[np.flatnonzero(nb == i), np.flatnonzero(nb != i)]  # centre first
        mu = dofs(nb[order])
        far = np.setdiff1d(all_dofs, mu)
        k = row_scale[mu, None] * g[np.ix_(mu, far)]
        k_c, k_o = k[:9], k[9:]
        gram = k_o @ k_o.conj().T
        reg = lam * float(np.linalg.eigvalsh(gram)[-1]) * np.eye(gram.shape[0])
        xh = -np.linalg.solve((gram + reg).T, (k_c @ k_o.conj().T).T).T  # (9, 9|mu|-9)
        alpha = np.vstack([np.eye(9), xh.conj().T])
        worst = max(worst, float(np.linalg.norm(alpha.conj().T @ k, 2) / np.linalg.norm(k_c, 2)))
        out.append((mu, alpha))
    return out, worst


def assemble(g: NDArray, t: NDArray, sten: list, row_scale: NDArray) -> tuple[sp.csc_matrix, sp.csr_matrix]:
    """The sparse system H and the stencil operator P.

    Args:
        g: Dense G0.
        t: Block-diagonal T0 as a dense matrix.
        sten: Output of ``stencils``.
        row_scale: Per-dof row scaling.

    Returns:
        (H, P), both (9 N^3, 9 N^3).
    """
    size = g.shape[0]
    h = sp.lil_matrix((size, size), dtype=complex)
    p = sp.lil_matrix((size, size), dtype=complex)
    for i, (mu, alpha) in enumerate(sten):
        ah = alpha.conj().T * row_scale[mu][None, :]  # alpha^H D
        beta_h = ah @ g[np.ix_(mu, mu)]
        rows = slice(9 * i, 9 * i + 9)
        h[rows, mu] = ah - beta_h @ t[np.ix_(mu, mu)]
        p[rows, mu] = ah
    return h.tocsc(), p.tocsr()


def run_gmres(a: NDArray, b: NDArray, prec) -> int:
    """Right-preconditioned GMRES iterations to TOL on the TRUE residual.

    Args:
        a: The dense system matrix.
        b: Right-hand side.
        prec: Callable r -> M r, or None.

    Returns:
        Iterations; -1 if not converged within 3 x size.
    """
    size = a.shape[0]
    m = (lambda v: v) if prec is None else prec
    op = LinearOperator((size, size), matvec=lambda y: a @ m(y), dtype=complex)
    its: list[float] = []  # one entry per inner (Krylov) iteration
    # scipy's maxiter counts RESTART CYCLES, not iterations: restart = maxiter = 2000
    # allowed 4 million inner steps and made every divergent case look like a hang.
    restart = min(size, 500)
    y, info = gmres(
        op,
        b,
        rtol=TOL,
        restart=restart,
        maxiter=-(-3 * size // restart),
        callback=its.append,
        callback_type="pr_norm",
    )
    true = float(np.linalg.norm(a @ m(y) - b) / np.linalg.norm(b))
    return len(its) if info == 0 and true < 10 * TOL else -1


def main() -> int:
    """Iterations with and without the sparsifying preconditioner, over contrast.

    Returns:
        0.
    """
    print("=" * 78)
    print("SPARSIFY-AND-SOLVE PRECONDITIONER ON THE 3-D FOLDY-LAX SYSTEM")
    ka = OM / REF.alpha * A_HALF
    print(f"  {N}^3 voxels ({9 * N**3} unknowns), pitch {PITCH:.0f} m, ka = {ka:.3f}")
    print("  8 points per S wavelength")
    print("=" * 78)
    t0 = time.perf_counter()
    g = dense_g0()
    print(f"  dense G0 assembled in {time.perf_counter() - t0:.0f} s")
    nbs = neighbourhoods()
    row_scale = np.tile(np.r_[np.ones(3), np.full(6, PITCH)], N**3)
    t0 = time.perf_counter()
    sten, worst = stencils(g, nbs, row_scale)
    print(f"  stencils fitted in {time.perf_counter() - t0:.0f} s; worst annihilation ratio {worst:.2e}")

    rng = np.random.default_rng(1)
    b = rng.standard_normal(9 * N**3) + 1j * rng.standard_normal(9 * N**3)
    size = 9 * N**3
    trunc_mask = np.zeros((size, size), dtype=bool)
    for i, nb in enumerate(nbs):
        trunc_mask[np.ix_(np.arange(9 * i, 9 * i + 9), dofs(nb))] = True

    head = ("strength", "rho(G0 T0)", "none", "truncated", "sparsified", "|PA-H|/|PA|")
    print(f"\n  {head[0]:>8} {head[1]:>11} {head[2]:>6} {head[3]:>10} {head[4]:>11} {head[5]:>12}")
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        gt = g @ t
        a = np.eye(size) - gt
        clock = time.perf_counter()

        def stage(label: str, value: object, start: float = clock) -> None:
            print(f"      [{time.perf_counter() - start:6.0f} s] {label}: {value}", flush=True)

        rho = float(np.abs(eigs(gt, k=1, which="LM", return_eigenvectors=False))[0])
        stage(f"strength {strength:g}, rho(G0 T0)", f"{rho:.3f}")
        h, p = assemble(g, t, sten, row_scale)
        pa = p @ a
        err = float(np.linalg.norm(pa - h.toarray()) / np.linalg.norm(pa))
        stage("sparsification error |PA - H|/|PA|", f"{err:.2e}")
        lu_h = splu(h)
        lu_tr = splu(sp.csc_matrix(np.where(trunc_mask, a, 0.0)))
        n_none = run_gmres(a, b, None)
        stage("GMRES, no preconditioner", n_none)
        n_tr = run_gmres(a, b, lu_tr.solve)
        stage("GMRES, 27-voxel truncation", n_tr)
        n_sp = run_gmres(a, b, lambda r, lu=lu_h, pp=p: lu.solve(pp @ r))
        stage("GMRES, sparsified", n_sp)
        print(f"  {strength:8.0f} {rho:11.3f} {n_none:6d} {n_tr:10d} {n_sp:11d} {err:12.2e}", flush=True)
    return 0


def kernel_window(r: int) -> NDArray:
    """G0 blocks at every offset within |d|_inf <= r, from the solver's own apply_g0_3d.

    Args:
        r: Window half-width, in pitches.

    Returns:
        Shape (2r+1, 2r+1, 2r+1, 9, 9): [dz+r, dx+r, dy+r] = G0(receiver - source = d).
    """
    n = 2 * r + 1
    cache = build_g0_cache_3d(SweepGrid3D(n_z=n, n_x=n, n_y=n, pitch=PITCH), REF, OM)
    ker = np.zeros((n, n, n, 9, 9), dtype=complex)
    for c in range(9):
        e = np.zeros((n, n, n, 9), dtype=complex)
        e[r, r, r, c] = 1.0  # source at the window centre: receiver index = offset + r
        ker[..., c] = apply_g0_3d(e, cache)
    return ker


def stencils_invariant(
    ker: NDArray, r: int, lam: float = 1e-3
) -> dict[tuple[int, ...], tuple[list[tuple[int, ...]], NDArray]]:
    """Centre-anchored stencils fitted ONCE per neighbourhood type, not per voxel.

    A type is (s_z, s_x, s_y), each -1 (voxel on the low face), 0 (interior) or
    +1 (high face).  Near offsets are those of {-1,0,1}^3 the type allows; the far
    set is every other offset within |d|_inf <= r that the type allows (a voxel on
    a low face has nothing below it) -- Liu & Ying's union of translated
    complements, independent of the domain it will be used on.

    Args:
        ker: ``kernel_window(r + 1)``: a near offset (|o| <= 1) minus a far one
            (|f| <= r) reaches r + 1.
        r: Half-width of the far set.
        lam: Tikhonov weight, as ``stencils_centred``.

    Returns:
        {type: (near offsets, centre first; alpha of shape (9 |near|, 9))}.
    """
    d9 = np.r_[np.ones(3), np.full(6, PITCH)]
    c = r + 1  # index of offset 0 in ker
    if ker.shape[0] != 2 * c + 1:
        msg = (
            f"kernel window has half-width {(ker.shape[0] - 1) // 2}, expected {c}.\n"
            "  Where: scripts/measure_sparsify_preconditioner.py, stencils_invariant(ker=...)\n"
            f"  Valid: kernel_window({c}) -- near minus far offsets reach r + 1\n"
            "  Fix:   build the window one pitch wider than the far set."
        )
        raise ValueError(msg)
    out = {}
    for typ in itertools.product((-1, 0, 1), repeat=3):

        def allowed(o: tuple[int, ...], lim: int, t: tuple[int, ...] = typ) -> bool:
            return all(
                (-lim if s >= 0 else 0) <= v <= (lim if s <= 0 else 0) for v, s in zip(o, t, strict=True)
            )

        near = [o for o in itertools.product((-1, 0, 1), repeat=3) if allowed(o, 1)]
        near.sort(key=lambda o: o != (0, 0, 0))  # centre first
        far = [
            o
            for o in itertools.product(range(-r, r + 1), repeat=3)
            if allowed(o, r) and max(map(abs, o)) > 1
        ]
        k = np.zeros((9 * len(near), 9 * len(far)), dtype=complex)
        for i, o in enumerate(near):
            for j, f in enumerate(far):
                dz, dx, dy = (o[0] - f[0] + c, o[1] - f[1] + c, o[2] - f[2] + c)
                k[9 * i : 9 * i + 9, 9 * j : 9 * j + 9] = d9[:, None] * ker[dz, dx, dy]
        k_c, k_o = k[:9], k[9:]
        gram = k_o @ k_o.conj().T
        reg = lam * float(np.linalg.eigvalsh(gram)[-1]) * np.eye(gram.shape[0])
        xh = -np.linalg.solve((gram + reg).T, (k_c @ k_o.conj().T).T).T
        out[typ] = (near, np.vstack([np.eye(9), xh.conj().T]))
    return out


def place_invariant(sten_types: dict) -> list:
    """Apply the per-type stencils at every voxel of the N^3 domain.

    Args:
        sten_types: ``stencils_invariant`` output.

    Returns:
        The ``stencils`` list format: per voxel, (mu dofs, alpha).
    """
    out = []
    for z, x, y in itertools.product(range(N), repeat=3):
        typ = tuple(-1 if v == 0 else (1 if v == N - 1 else 0) for v in (z, x, y))
        near, alpha = sten_types[typ]
        vox = np.array([((z + a) * N + (x + b)) * N + (y + c) for a, b, c in near])
        out.append((dofs(vox), alpha))
    return out


def compare_invariant(r: int = 7) -> int:
    """Per-domain stencils (the idealised best case) against translation-invariant ones.

    Args:
        r: Window half-width for the invariant fit.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"TRANSLATION-INVARIANT STENCILS AT {N}^3 ({9 * N**3} unknowns), window R = {r}")
    print("=" * 78)
    clock = time.perf_counter()
    g = dense_g0()
    nbs = neighbourhoods()
    row_scale = np.tile(np.r_[np.ones(3), np.full(6, PITCH)], N**3)
    per_domain, _ = stencils_centred(g, nbs, row_scale, 1e-3)
    print(f"  [{time.perf_counter() - clock:5.0f} s] dense G0 and per-domain stencils", flush=True)
    types = stencils_invariant(kernel_window(r + 1), r)
    invariant = place_invariant(types)
    print(f"  [{time.perf_counter() - clock:5.0f} s] 27 invariant stencil types fitted once", flush=True)
    size = g.shape[0]
    rng = np.random.default_rng(1)
    b: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    cols = ("rho", "none", "per-domain", "invariant", "|PA-H|/|PA| inv")
    print(
        f"\n  {'strength':>8} "
        + " ".join(f"{c:>{w}}" for c, w in zip(cols, (7, 6, 11, 10, 16), strict=True))
    )
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(eigs(g @ t, k=1, which="LM", return_eigenvectors=False))[0])
        row = [run_gmres(a, b, None)]
        for sten in (per_domain, invariant):
            h, p = assemble(g, t, sten, row_scale)
            lu_h = splu(h)
            row.append(run_gmres(a, b, lambda v, lu=lu_h, pp=p: lu.solve(pp @ v)))
        pa = p @ a
        err = float(np.linalg.norm(pa - h.toarray()) / np.linalg.norm(pa))
        print(f"  {strength:8g} {rho:7.3f} {row[0]:6d} {row[1]:11d} {row[2]:10d} {err:16.2e}", flush=True)
    return 0


def compare_variants() -> int:
    """Smallest-singular versus centre-anchored stencils: conditioning, spectrum, iterations.

    Small lattices only (dense SVDs and eigenvalues of the preconditioned operator).

    Returns:
        0.
    """
    print("=" * 78)
    print(f"STENCIL VARIANTS AT {N}^3 ({9 * N**3} unknowns), 8 points per S wavelength")
    print("=" * 78)
    g = dense_g0()
    nbs = neighbourhoods()
    row_scale = np.tile(np.r_[np.ones(3), np.full(6, PITCH)], N**3)
    size = g.shape[0]
    variants = {
        "svd (smallest 9)": stencils(g, nbs, row_scale),
        "centred, lam 0": stencils_centred(g, nbs, row_scale, 0.0),
        "centred, lam 1e-6": stencils_centred(g, nbs, row_scale, 1e-6),
        "centred, lam 1e-3": stencils_centred(g, nbs, row_scale, 1e-3),
    }
    for name, (_, worst) in variants.items():
        print(f"  {name:18s} worst far leakage {worst:.2e}", flush=True)
    rng = np.random.default_rng(1)
    b: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    trunc_mask = np.zeros((size, size), dtype=bool)
    for i, nb in enumerate(nbs):
        trunc_mask[np.ix_(np.arange(9 * i, 9 * i + 9), dofs(nb))] = True
    cols = ("cond P", "cond H", "|PA-H|/|PA|", "|lam| min", "GMRES")
    widths = (9, 9, 12, 10, 6)
    head = f"    {'variant':18s} " + " ".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True))
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(np.linalg.eigvals(g @ t)).max())
        n_none = run_gmres(a, b, None)
        n_tr = run_gmres(a, b, splu(sp.csc_matrix(np.where(trunc_mask, a, 0.0))).solve)
        print(f"\n  strength {strength:g}: rho(G0 T0) {rho:.3f}; GMRES none {n_none}, truncation {n_tr}")
        print(head)
        for name, (sten, _) in variants.items():
            h, p = assemble(g, t, sten, row_scale)
            hd, pd = h.toarray(), p.toarray()
            err = float(np.linalg.norm(pd @ a - hd) / np.linalg.norm(pd @ a))
            ev = np.linalg.eigvals(a @ np.linalg.solve(hd, pd))
            lu_h = splu(h)
            n_sp = run_gmres(a, b, lambda r, lu=lu_h, pp=p: lu.solve(pp @ r))
            print(
                f"    {name:18s} {np.linalg.cond(pd):9.1e} {np.linalg.cond(hd):9.1e} {err:12.2e} "
                f"{np.abs(ev).min():10.1e} {n_sp:6d}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    if "invariant" in sys.argv:
        raise SystemExit(compare_invariant())
    raise SystemExit(compare_variants() if "variants" in sys.argv else main())
