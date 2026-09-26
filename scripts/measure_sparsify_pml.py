#!/usr/bin/env python3
"""MEASUREMENT, not a gate: Liu & Ying's absorbing layer, elastic, for the sweep's Schur inverses.

WHY.  The sweep of the sparsified system needs S_z^{-1}, the inverse of the
Dirichlet-to-Neumann map of the half-domain above plane z
(``measure_sparsify_sweep.py``).  Windowing it with a reflecting cut, or a
sponge of background blocks with an imaginary diagonal shift, fails
(``measure_sparsify_moving_layer.py``).  Liu & Ying close the window instead
with a perfectly matched layer whose stencils ANNIHILATE COMPLEX-STRETCHED PLANE
WAVES (SISC 2018, Sec. 2.2.2).

THE PAD.  n_pad planes above the window top, at virtual depths d = p, 2p, ...,
closed by zero beyond the last.  Stretched vertical coordinate
z~ = z - i sigma(d), sigma(d) = (C / k_S) (d / eta)^2, eta = n_pad p: an upgoing
wave e^{-i k_z z} decays as e^{-k_z sigma}, to ~e^{-C} at the far end.  The
fitted waves, for each direction r of a set R: P (polarisation r) and two S,
as the 9-component state (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy,
2e_zx); strain z-derivatives carry dz~/dz = 1 + i sigma'(d).  Stencils are
centre-anchored, fitted per pad plane and lateral type, contrast-independent.
R: Liu & Ying's 26 lattice directions, or a denser 64-direction Fibonacci set.

CHECKS.  Each pad stencil's residual on 200 random directions NOT in R (does it
represent the stretched wave equation or only its training waves?); C = 0 (an
unstretched, reflecting pad) as the baseline; the full window reproducing exact LU.

RESULT (2026-09-27, 6^3).  NEGATIVE, and the reason is structural.
  * GMRES (rho 0.06, exact 5): plain truncation 39; nearly every pad fails or
    needs 45-609 -- the unstretched pad (C = 0) included.
  * ``dtn`` mode, contrast-free system, |T_z - S_z^-1| / |S_z^-1| at the bottom
    plane: truncation 0.78 (b = 1) / 0.59 (b = 2); pads at C = 0 are 10-380x
    off, and the best pad of all (b = 2, 26 directions, 4 planes, C = 10) 0.46.
    The square 78-direction fit does not help.
  * The interior (Green-fitted) stencil annihilates free plane waves to 5e-3,
    yet differs by 18% from a plane-wave-fitted one on the same points: the
    annihilating family is large.  ANCHORING the pad stencil to the interior one
    (``anchor=``; least deviation) does not help either: at C = 0 it is more
    interior planes closed by the same face stencils, and still misses by 10-230x.
    The error comes from DEPTH, not from the pad.
  Reading, NOT yet tested: this H is built with FACE stencils fitted to the
  dense operator's far field, so H alone is a closed cavity and the radiation
  condition lives in the cancellation between H^{-1} and P.  Its Schur
  complements are cavity maps, not Dirichlet-to-Neumann maps, and any window
  breaks the cancellation.  Liu & Ying's H is radiating by construction -- the
  domain padded on every side with PML voxels and PML stencils -- and only then
  can a moving PML approximate its Schur complements.  The next step is to build
  H that way.

Run:  conda run -n seismic python scripts/measure_sparsify_pml.py [N] [dtn [anchored-only]]
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
P = mp.PITCH
K_P, K_S = mp.OM / mp.REF.alpha, mp.OM / mp.REF.beta
D9 = np.r_[np.ones(3), np.full(6, P)]
T_START = time.perf_counter()


def stage(label: str) -> None:
    """Print one timed stage."""
    print(f"  [{time.perf_counter() - T_START:6.0f} s] {label}", flush=True)


def directions(kind: str) -> NDArray:
    """Unit propagation directions: 'lattice' (Liu & Ying's 26) or 'fib64'."""
    if kind == "lattice":
        v = np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)], dtype=float)
    else:
        n_dir = int(kind.removeprefix("fib"))  # fib64, fib78 (78 x 3 modes = 234 = a square fit)
        i = np.arange(n_dir) + 0.5
        phi, cz = np.pi * (1 + 5**0.5) * i, 1 - 2 * i / n_dir
        v = np.stack([cz, np.sqrt(1 - cz**2) * np.cos(phi), np.sqrt(1 - cz**2) * np.sin(phi)], -1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sigma(d: NDArray, c: float, eta: float) -> tuple[NDArray, NDArray]:
    """Stretch sigma(d) and its derivative, d >= 0 the depth into the pad (zero in the window)."""
    dd = np.maximum(d, 0.0)
    return c / K_S * (dd / eta) ** 2, np.where(d > 0, 2 * c / K_S * dd / eta**2, 0.0)


def waves(pts: NDArray, dirs: NDArray, c: float, eta: float) -> NDArray:
    """9-component stretched plane waves at points (z, x, y); column per (direction, mode).

    Args:
        pts: Points, shape (n, 3), z = 0 on the window top, negative into the pad.
        dirs: Directions, shape (m, 3), components (z, x, y).
        c: Stretch strength C.
        eta: Pad thickness.

    Returns:
        Shape (9 n, 3 m), rows scaled by D9 (strain rows x pitch).
    """
    s, ds = sigma(-pts[:, 0], c, eta)
    zt = pts[:, 0] - 1j * s  # stretched z
    jac = 1.0 + 1j * ds  # dz~/dz
    cols = []
    for r in dirs:
        a = np.array([1.0, 0.0, 0.0]) if abs(r[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(r, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(r, e1)
        for k, e in ((K_P, r), (K_S, e1), (K_S, e2)):
            ph = np.exp(1j * k * (r[0] * zt + r[1] * pts[:, 1] + r[2] * pts[:, 2]))
            grad = 1j * k * np.stack([r[0] * jac, np.full_like(jac, r[1]), np.full_like(jac, r[2])], -1)
            gu = grad[:, :, None] * e[None, None, :]  # d_i u_j
            eps = 0.5 * (gu + gu.transpose(0, 2, 1))
            state = np.stack(
                [
                    e[0] * np.ones_like(jac),
                    e[1] * np.ones_like(jac),
                    e[2] * np.ones_like(jac),
                    eps[:, 0, 0],
                    eps[:, 1, 1],
                    eps[:, 2, 2],
                    2 * eps[:, 1, 2],
                    2 * eps[:, 0, 2],
                    2 * eps[:, 0, 1],
                ],
                -1,
            )
            cols.append((ph[:, None] * state * D9[None, :]).ravel())
    return np.array(cols).T


def pad_stencil(
    near: list,
    k: int,
    dirs: NDArray,
    c: float,
    eta: float,
    lam: float = 1e-8,
    anchor: NDArray | None = None,
) -> NDArray:
    """Centre-anchored stencil annihilating stretched waves on a pad voxel's neighbourhood.

    Plane-wave annihilation leaves a large family of stencils (234 coefficients,
    78-234 constraints).  Least-norm picks one that differs by ~18% from the
    interior stencil, which also annihilates plane waves: two discrete media, a
    reflecting junction.  With ``anchor`` (the interior stencil of the same
    type) the pad stencil is the interior one plus the SMALLEST correction that
    annihilates the stretched waves -- equal to the interior stencil at C = 0,
    so the layer is matched by construction.

    Args:
        near: Near offsets (dz, dx, dy), centre first.
        k: Pad plane (1 = innermost), at z = -k p.
        dirs: Direction set.
        c: Stretch strength.
        eta: Pad thickness.
        lam: Tikhonov weight (relative).
        anchor: Interior alpha of the same type and offset order, or None.

    Returns:
        alpha, shape (9 |near|, 9).
    """
    pts = np.array([[(-k + a) * P, bx * P, by * P] for a, bx, by in near])
    w = waves(pts, dirs, c, eta)
    w_c, w_o = w[:9], w[9:]
    # alpha^H w = 0 with alpha = [I; X]:  w_c + X^H w_o = 0
    gram = w_o.conj().T @ w_o
    reg = lam * float(np.linalg.eigvalsh(gram)[-1]) * np.eye(gram.shape[0])
    xh0 = np.zeros((9, w_o.shape[0]), dtype=complex) if anchor is None else anchor[9:].conj().T
    # least-norm DEVIATION from xh0 (from zero when there is no anchor)
    xh = xh0 - (w_c + xh0 @ w_o) @ np.linalg.solve(gram + reg, w_o.conj().T)
    return np.vstack([np.eye(9), xh.conj().T])


def pad_blocks(
    n_pad: int, dirs: NDArray, c: float, interior: dict | None = None
) -> tuple[list, list, list, float]:
    """Pad-plane row blocks (outermost first): (diag, to plane above, to plane below).

    Args:
        n_pad: Pad planes.
        dirs: Direction set.
        c: Stretch strength.
        interior: ``stencils_invariant`` output, to anchor each pad stencil on the
            interior stencil of the same type; None for the unanchored fit.

    Returns:
        (d_pad, lo_pad, up_pad, worst out-of-set residual).
    """
    eta = n_pad * P
    m = 9 * N * N
    d_pad, lo_pad, up_pad = [], [], []
    rng = np.random.default_rng(3)
    test = rng.standard_normal((200, 3))
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    worst = 0.0
    for k in range(n_pad, 0, -1):  # outermost first
        blk = {a: sp.lil_matrix((m, m), dtype=complex) for a in (-1, 0, 1)}
        cache: dict = {}
        for x, y in itertools.product(range(N), repeat=2):
            sx = -1 if x == 0 else (1 if x == N - 1 else 0)
            sy = -1 if y == 0 else (1 if y == N - 1 else 0)
            if (sx, sy) not in cache:
                near = [
                    (a, bx, by)
                    for a, bx, by in itertools.product((-1, 0, 1), repeat=3)
                    if not (k == n_pad and a == -1)  # zero beyond the outermost pad plane
                    and (bx >= 0 if sx < 0 else bx <= 0 if sx > 0 else True)
                    and (by >= 0 if sy < 0 else by <= 0 if sy > 0 else True)
                ]
                near.sort(key=lambda o: o != (0, 0, 0))
                anchor = None
                if interior is not None:
                    sz = -1 if k == n_pad else 0  # the outermost plane has nothing above
                    near_int, anchor = interior[sz, sx, sy]
                    if list(near_int) != near:
                        msg = (
                            f"pad and interior neighbourhoods differ for type {(sz, sx, sy)}.\n"
                            "  Where: scripts/measure_sparsify_pml.py, pad_blocks(interior=...)\n"
                            "  Valid: the same offsets in the same (centre-first) order\n"
                            "  Fix:   build both lists with the same filter and sort."
                        )
                        raise ValueError(msg)
                alpha = pad_stencil(near, k, dirs, c, eta, anchor=anchor)
                pts = np.array([[(-k + a) * P, bx * P, by * P] for a, bx, by in near])
                w_t = waves(pts, test, c, eta)
                res = np.linalg.norm(alpha.conj().T @ w_t) / np.linalg.norm(w_t[:9])
                worst = max(worst, float(res))
                cache[sx, sy] = (near, alpha)
            near, alpha = cache[sx, sy]
            ah = alpha.conj().T * np.tile(D9, len(near))[None, :]
            for j, (a, bx, by) in enumerate(near):
                col = 9 * ((x + bx) * N + (y + by))
                row = 9 * (x * N + y)
                blk[a][row : row + 9, col : col + 9] = ah[:, 9 * j : 9 * j + 9]
        d_pad.append(blk[0].tocsc())
        lo_pad.append(blk[-1].tocsc())  # to the plane above (farther out)
        up_pad.append(blk[1].tocsc())  # to the plane below (inner pad or the window top)
    return d_pad, lo_pad, up_pad, worst


def pml_sweep(d: list, lo: list, up: list, b: int, pad: tuple | None):
    """Riccati sweep with S_z^{-1} replaced by the window solve closed by the pad.

    Args:
        d: Diagonal blocks of H.
        lo: Sub-diagonal blocks.
        up: Super-diagonal blocks.
        b: Window depth in planes.
        pad: (d_pad, lo_pad, up_pad) outermost first, or None.

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
            dp, lp, upd = pad
            # the window top's coupling to the plane above now lands on the innermost pad plane
            low = [None] + list(lp[1:]) + [lo[top]] + low[1:]
            diag = list(dp) + diag
            upp = list(upd) + upp
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


def local_inverse(d: list, lo: list, up: list, z: int, b: int, pad: tuple | None) -> NDArray:
    """Dense T_z: the plane-z block of the windowed (optionally padded) sub-system's inverse."""
    top = max(0, z - b + 1)
    planes = list(range(top, z + 1))
    diag = [d[k] for k in planes]
    low = [None] + [lo[k] for k in planes[1:]]
    upp = [up[k] for k in planes[:-1]] + [None]
    if top > 0 and pad is not None:
        dp, lp, upd = pad
        low = [None] + list(lp[1:]) + [lo[top]] + low[1:]
        diag = list(dp) + diag
        upp = list(upd) + upp
    nb = len(diag)
    rows = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        rows[i][i] = diag[i]
        if i > 0:
            rows[i][i - 1] = low[i]
        if i < nb - 1:
            rows[i][i + 1] = upp[i]
    m = d[0].shape[0]
    rhs = np.zeros((nb * m, m), dtype=complex)
    rhs[-m:] = np.eye(m)
    return splu(sp.bmat(rows, format="csc")).solve(rhs)[-m:]


def dtn_diagnostic() -> int:
    """How well does each closure approximate S_z^{-1}, the exact Schur inverse? Contrast-free system.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"DtN DIAGNOSTIC: |T_z - S_z^-1| / |S_z^-1| at the bottom plane, {N}^3, contrast-free")
    print("=" * 78)
    g = mp.dense_g0()
    types = mp.stencils_invariant(mp.kernel_window(8), 7)
    sten = mp.place_invariant(types)
    size = g.shape[0]
    h, _ = mp.assemble(g, np.zeros((size, size)), sten, np.tile(D9, N**3))
    d, lo, up = blocks(h, N)
    z = N - 1
    exact = local_inverse(d, lo, up, z, N, None)
    scale = np.linalg.norm(exact)
    stage("exact Schur inverse at the bottom plane")
    both = (("free", None), ("anchored", types))
    anchors = both[1:] if "anchored-only" in sys.argv else both
    for b in (1, 2):
        err = np.linalg.norm(local_inverse(d, lo, up, z, b, None) - exact) / scale
        print(f"  b={b} truncated: {err:.3e}", flush=True)
        for label, interior in anchors:
            for kind in ("lattice", "fib78"):
                for n_pad in (2, 4, 6):
                    cells = []
                    for c in (0.0, 2.0, 5.0, 10.0):
                        dp, lp, upd, worst = pad_blocks(n_pad, directions(kind), c, interior)
                        e = np.linalg.norm(local_inverse(d, lo, up, z, b, (dp, lp, upd)) - exact) / scale
                        cells.append(f"C{c:g} {e:.2e}")
                    print(f"  b={b} {label:8s} {kind:7s} pad {n_pad}: " + "  ".join(cells), flush=True)
    return 0


def main() -> int:
    """GMRES iterations with the PML-closed window, scanning C, pad thickness, window, direction set.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"LIU-YING ABSORBING LAYER, ELASTIC: {N}^3 voxels, whole space")
    print("=" * 78)
    g = mp.dense_g0()
    sten = mp.place_invariant(mp.stencils_invariant(mp.kernel_window(8), 7))
    row_scale = np.tile(D9, N**3)
    stage("dense G0 and interior stencils")
    pads = {}
    for kind in ("lattice", "fib64"):
        for n_pad in (2, 4):
            for c in (0.0, 2.0, 5.0, 10.0):
                dp, lp, upd, worst = pad_blocks(n_pad, directions(kind), c)
                pads[kind, n_pad, c] = (dp, lp, upd)
                print(
                    f"    pad {kind:7s} n_pad {n_pad} C {c:4.1f}: out-of-set residual {worst:.2e}",
                    flush=True,
                )
    stage("pad stencils fitted (contrast-independent)")
    size = g.shape[0]
    rng = np.random.default_rng(1)
    rhs: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = mp.t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(eigs(g @ t, k=1, which="LM", return_eigenvectors=False))[0])
        h, p = mp.assemble(g, t, sten, row_scale)
        d, lo, up = blocks(h, N)

        def iters(solve, pp=p, aa=a) -> int:
            """GMRES iterations with M = solve(P .)."""
            return mp.run_gmres(aa, rhs, lambda v: solve(pp @ v))

        exact = iters(splu(h).solve)
        none = mp.run_gmres(a, rhs, None)
        print(f"\n  strength {strength:g}: rho {rho:.3f}; none {none}; exact {exact}")
        print(f"    full window (check) {iters(pml_sweep(d, lo, up, N, None))}", flush=True)
        for b in (1, 2):
            cells = [f"trunc {iters(pml_sweep(d, lo, up, b, None))}"]
            for key, pad in pads.items():
                its = iters(pml_sweep(d, lo, up, b, pad))
                cells.append(f"{key[0][:3]}/{key[1]}/C{key[2]:g} {its}")
            print(f"    b={b}: " + "  ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(dtn_diagnostic() if "dtn" in sys.argv else main())
