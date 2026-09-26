#!/usr/bin/env python3
"""MEASUREMENT, not a gate: a RADIATING sparsified system, built as Liu & Ying build it.

WHY.  ``measure_sparsify_pml.py`` found that no moving absorbing layer closes the
sparsified system H of ``measure_sparsify_preconditioner.py``: against the exact
Schur inverse, even an unstretched pad is 10-380x off, and the error follows
the DEPTH of the truncated system.  The reading, untested there: that H closes
the domain with FACE stencils fitted to the dense far field, so H alone is a
closed cavity -- the radiation condition lives in the cancellation between
H^{-1} and P -- and its Schur complements are cavity maps, not
Dirichlet-to-Neumann maps.  Liu & Ying's H radiates by construction.  This
builds it their way and tests the reading.

THE REDUCTION.  A residual r on the domain D, zero-padded, turns A phi = r
(A = I - G0 T0 on D) into phi_x - sum_{y in D} G0(x - y) T0_y phi_y = r_x on the
whole lattice; T0 vanishes off D, so the restriction to D of its solution is
A^{-1} r.  Its sparse form, on the domain extended by a one-voxel shell and
n_pml absorbing voxels on every side:
  * shell and domain: the INTERIOR centre-anchored stencil at every voxel -- the
    neighbourhood is always full; no face stencils.  alpha^H D phi - beta^H T0 phi
    = alpha^H D r, beta^H = alpha^H D G0[mu, mu];
  * absorbing layer: centre-anchored stencils cancelling plane waves stretched
    along every axis whose layer the voxel is in (x~ = x + i s sigma(d), s the
    outward sign, sigma = (C / k_S)(d / eta)^2), ANCHORED on the interior stencil
    (least deviation) so the layer continues the interior medium;
  * zero beyond the layer.
M r = restriction to D of H~^{-1} f, f the stencil-combined padded residual.

STAGES, each gating the next:
  [1] exact LU of H~ preconditions at least as well as the face-stencil H
      (6^3 whole space: 5 / 10 / 16 / 34 at rho 0.06 / 0.57 / 1.98 / 4.99);
  [2] the hypothesis: on H~ the moving absorbing layer approximates the exact
      Schur inverse FAR better than on the face-stencil H (best there 0.46);
  [3] the moving-layer sweep as the preconditioner.

RESULT (2026-09-27, 4^3).  STAGE [1] FAILS: exact LU of the radiating H~
preconditions WORSE than the face-stencil H at every contrast measured.
GMRES iterations, none / face-stencil H exact / radiating H~ exact:
    rho 0.052:  7 / 5 / 6-7      (best: 4-layer PML, C = 5: 6)
    rho 0.399: 16 / 8 / 10-14    (best: 10)
    rho 1.322: 36 / 11 / 15-20   (best: 15)
(the rho 3.33 row was cut off by the run's 25-minute limit.)  The thicker
layer is always best, consistent with thin layers reflecting.  Stages [2] and
[3] were therefore not run: the cavity reading of measure_sparsify_pml.py is
neither confirmed nor refuted, but this construction gains nothing.  Caveat:
at 4^3 the domain is small against the absorbing layer.

Run:  conda run -n seismic python scripts/measure_sparsify_radiating.py [N] [stage]
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

N = mp.N
P = mp.PITCH
K_P, K_S = mp.OM / mp.REF.alpha, mp.OM / mp.REF.beta
D9 = np.r_[np.ones(3), np.full(6, P)]
R_WIN = 7  # interior stencil's far window; must reach across the domain plus shell
T_START = time.perf_counter()


def stage(label: str) -> None:
    """Print one timed stage."""
    print(f"  [{time.perf_counter() - T_START:6.0f} s] {label}", flush=True)


class Grid:
    """The extended lattice: n_pml | shell | domain N | shell | n_pml along each axis."""

    def __init__(self, n_pml: int) -> None:
        self.n_pml = n_pml
        self.ne = N + 2 + 2 * n_pml
        self.off = n_pml + 1  # extended index of domain index 0

    def depth(self, i: int) -> tuple[int, int]:
        """(signed outward side, layers into the absorbing layer) along one axis; (0, 0) if not in it."""
        if i < self.n_pml:
            return -1, self.n_pml - i
        if i >= self.ne - self.n_pml:
            return 1, i - (self.ne - self.n_pml - 1)
        return 0, 0

    def flat(self, z: int, x: int, y: int) -> int:
        """Flat voxel index, plane-major."""
        return (z * self.ne + x) * self.ne + y


def directions(n_dir: int = 78) -> NDArray:
    """Fibonacci directions (78 x 3 modes = 234: a square fit)."""
    i = np.arange(n_dir) + 0.5
    phi, cz = np.pi * (1 + 5**0.5) * i, 1 - 2 * i / n_dir
    return np.stack([cz, np.sqrt(1 - cz**2) * np.cos(phi), np.sqrt(1 - cz**2) * np.sin(phi)], -1)


def stretched_waves(pts: NDArray, dirs: NDArray, grid: Grid, c: float) -> NDArray:
    """9-component plane waves, stretched along every axis inside the absorbing layer.

    Args:
        pts: Extended-grid integer positions, shape (n, 3), axes (z, x, y).
        dirs: Directions, shape (m, 3).
        grid: The extended lattice.
        c: Stretch strength C.

    Returns:
        Shape (9 n, 3 m), rows scaled by D9.
    """
    eta = max(grid.n_pml, 1) * P
    xt = np.zeros(pts.shape, dtype=complex)  # stretched coordinates
    jac = np.ones(pts.shape, dtype=complex)  # d x~ / d x per axis
    for i, q in enumerate(pts):
        for ax in range(3):
            s, layers = grid.depth(int(q[ax]))
            d = layers * P
            sig, dsig = c / K_S * (d / eta) ** 2, 2 * c / K_S * d / eta**2
            xt[i, ax] = q[ax] * P + 1j * s * sig
            jac[i, ax] = 1.0 + 1j * dsig if layers else 1.0
    cols = []
    for r in dirs:
        a = np.array([1.0, 0.0, 0.0]) if abs(r[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(r, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(r, e1)
        for k, e in ((K_P, r), (K_S, e1), (K_S, e2)):
            ph = np.exp(1j * k * (xt @ r))
            grad = 1j * k * r[None, :] * jac  # d_i, per point
            gu = grad[:, :, None] * e[None, None, :]
            eps = 0.5 * (gu + gu.transpose(0, 2, 1))
            one = np.ones(len(pts), dtype=complex)
            state = np.stack(
                [
                    e[0] * one,
                    e[1] * one,
                    e[2] * one,
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


def anchored_layer_stencil(near: list, centre: NDArray, grid: Grid, c: float, anchor: NDArray) -> NDArray:
    """Interior stencil plus the least correction annihilating the stretched waves."""
    pts = np.array([centre + np.array(o) for o in near])
    w = stretched_waves(pts, directions(), grid, c)
    w_c, w_o = w[:9], w[9:]
    gram = w_o.conj().T @ w_o
    reg = 1e-8 * float(np.linalg.eigvalsh(gram)[-1]) * np.eye(gram.shape[0])
    xh0 = anchor[9:].conj().T
    xh = xh0 - (w_c + xh0 @ w_o) @ np.linalg.solve(gram + reg, w_o.conj().T)
    return np.vstack([np.eye(9), xh.conj().T])


def build(
    grid: Grid, c: float, types: dict, ker: NDArray, t_dom: NDArray
) -> tuple[sp.csc_matrix, sp.csr_matrix]:
    """The radiating sparse system H~ and the operator taking a domain residual to its right-hand side.

    Args:
        grid: The extended lattice.
        c: Stretch strength.
        types: ``stencils_invariant`` output (anchors, by clipping type).
        ker: ``kernel_window(R_WIN + 1)`` (G0 blocks by offset).
        t_dom: T0 blocks on the domain, shape (N, N, N, 9, 9).

    Returns:
        (H~ on the extended lattice, F: domain dofs -> extended right-hand side).
    """
    ne, off, cw = grid.ne, grid.off, R_WIN + 1
    size = 9 * ne**3
    h = sp.lil_matrix((size, size), dtype=complex)
    f = sp.lil_matrix((size, 9 * N**3), dtype=complex)
    layer_cache: dict = {}

    def t_at(q) -> NDArray | None:
        z, x, y = (int(v) - off for v in q)
        return t_dom[z, x, y] if 0 <= z < N and 0 <= x < N and 0 <= y < N else None

    near_full, alpha_int = types[0, 0, 0]
    for q in itertools.product(range(ne), repeat=3):
        qa = np.array(q)
        side = [grid.depth(v) for v in q]
        in_layer = any(layers for _, layers in side)
        clip = tuple(
            -1 if v == 0 else (1 if v == ne - 1 else 0) for v in q
        )  # zero beyond the outermost layer
        row = 9 * grid.flat(*q)
        if not in_layer:
            near, alpha = near_full, alpha_int
            ah = alpha.conj().T * np.tile(D9, len(near))[None, :]
            for j, o in enumerate(near):
                col = 9 * grid.flat(*(qa + np.array(o)))
                h[row : row + 9, col : col + 9] = ah[:, 9 * j : 9 * j + 9]
                # - beta^H T0: beta^H = alpha^H D G0[mu, mu]; the column's T0 multiplies
                tq = t_at(qa + np.array(o))
                if tq is not None:
                    blk = np.zeros((9, 9), dtype=complex)
                    for i2, o2 in enumerate(near):
                        dd = np.array(o2) - np.array(o) + cw
                        blk += ah[:, 9 * i2 : 9 * i2 + 9] @ ker[dd[0], dd[1], dd[2]]
                    h[row : row + 9, col : col + 9] = h[row : row + 9, col : col + 9] - blk @ tq
                # right-hand side: alpha^H D r on the domain part of the neighbourhood
                z, x, y = (int(v) - off for v in qa + np.array(o))
                if 0 <= z < N and 0 <= x < N and 0 <= y < N:
                    dcol = 9 * ((z * N + x) * N + y)
                    f[row : row + 9, dcol : dcol + 9] = ah[:, 9 * j : 9 * j + 9]
        else:
            key = (tuple((s, lay) for s, lay in side), clip)
            if key not in layer_cache:
                near_c, anchor = types[clip]
                layer_cache[key] = (near_c, anchored_layer_stencil(near_c, qa, grid, c, anchor))
            near, alpha = layer_cache[key]
            ah = alpha.conj().T * np.tile(D9, len(near))[None, :]
            for j, o in enumerate(near):
                col = 9 * grid.flat(*(qa + np.array(o)))
                h[row : row + 9, col : col + 9] = ah[:, 9 * j : 9 * j + 9]
    return h.tocsc(), f.tocsr()


def restrict(grid: Grid) -> NDArray:
    """Extended dof indices of the domain dofs, in domain order."""
    idx: list[int] = []
    for z, x, y in itertools.product(range(N), repeat=3):
        v = 9 * grid.flat(z + grid.off, x + grid.off, y + grid.off)
        idx.extend(range(v, v + 9))
    return np.array(idx)


def stage1() -> int:
    """[1] Exact LU of the radiating H~, against no preconditioner.

    Returns:
        0.
    """
    print("=" * 78)
    print(f"RADIATING SPARSIFIED SYSTEM, stage [1]: exact LU, {N}^3 domain")
    print("=" * 78)
    g = mp.dense_g0()
    ker = mp.kernel_window(R_WIN + 1)
    types = mp.stencils_invariant(ker, R_WIN)
    stage("dense G0, kernel window, interior stencils")
    size = g.shape[0]
    rng = np.random.default_rng(1)
    b: NDArray = np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size))
    configs = [(2, 2.0), (2, 5.0), (2, 10.0), (4, 5.0)]
    print("  face-stencil H, exact LU (measure_sparsify_preconditioner): 5 / 10 / 16 / 34")
    for strength in (1.0, 10.0, 30.0, 60.0):
        tb = mp.t_blocks(strength)
        t = sp.block_diag([tb[z, x, y] for z, x, y in itertools.product(range(N), repeat=3)]).toarray()
        a = np.eye(size) - g @ t
        rho = float(np.abs(eigs(g @ t, k=1, which="LM", return_eigenvectors=False))[0])
        cells = [f"none {mp.run_gmres(a, b, None)}"]
        for n_pml, c in configs:
            grid = Grid(n_pml)
            h, fmap = build(grid, c, types, ker, tb)
            lu, idx = splu(h), restrict(grid)
            its = mp.run_gmres(a, b, lambda v, s=lu, fm=fmap, ix=idx: s.solve(fm @ v)[ix])
            cells.append(f"pml{n_pml}/C{c:g} {its}")
        print(f"  rho {rho:6.3f}: " + "  ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(stage1())
