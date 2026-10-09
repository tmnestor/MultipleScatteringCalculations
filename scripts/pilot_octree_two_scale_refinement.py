#!/usr/bin/env python3
"""Refinement of an octree by the two-scale indicator, on the localised feature of the octree paper.

THE OPERATOR ON THE FINEST GRID. Every leaf is a union of finest cells (half-width h0). A leaf's
polynomials re-expand exactly on its finest descendants: the source monomials by ``source_reexpansion``,
the test functions by ``field_reexpansion``. So the Galerkin operator of any tree is

    A psi = M psi - C^T [ K_finest * (D^T E psi) ],

with K_finest the equal-cell blocks of the finest grid applied by FFT (octant storage by reflection
parity, as in ``measure_graded_sphere_legendre_series.py``). It reproduces ``octree.solve_graded_octree``,
whose blocks are the same sums of equal-cell blocks, and it costs one FFT convolution whatever the tree.

THE INDICATOR (docs/2026-10-09-two-scale-galerkin.md). For a tree with solution x_H, the fine tree splits
every leaf larger than h0 into its eight children. With r = b_h - A_h S x_H, the residual of the prolonged
solution, the two-level estimate of the coarse-to-fine error is

    e ~ L r + S A_H^-1 S^T (r - A_h L r),        L the inverse of each fine leaf's self block,

and the share of leaf l in F_h - F_H is the far field of its children's e-sources, plus its own medium
detail radiating with the coarse field. Refine the leaves whose share exceeds tol x the peak far field.

COMPARED with the stored rows of the octree paper's Table (``figures/data_refinement_localised_p1.json``:
uniform grids, the medium-only rule and the two-term rule), on the same body, frequency and error measure.

Run:  python -u scripts/pilot_octree_two_scale_refinement.py [--check] [--tol=1e-4,3e-5,1e-5] [--p=1]
"""

import itertools
import json
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import measure_graded_voxel_resolution as mres  # noqa: E402
import pilot_octree_two_term_refinement as pt  # noqa: E402

from cubic_scattering.graded_voxel.basis import SOURCE_EXPONENTS_QUARTIC, gram_test, monomials, source_expansion  # noqa: E402
from cubic_scattering.graded_voxel.blocks import coupling_block  # noqa: E402
from cubic_scattering.graded_voxel.fft import _transform, signed_permutations, symmetry_reps  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    OctreeResult,
    field_reexpansion,
    octree_far_field,
    refine_leaves,
    solve_graded_octree,
    source_reexpansion,
    uniform_leaves,
)
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients  # noqa: E402
from cubic_scattering.graded_voxel.solver import field_sizes, plane_wave_moments  # noqa: E402
from cubic_scattering.sphere_scattering import _plane_wave_strain_voigt, _voigt_to_tensor  # noqa: E402

REF, CONTRAST, RADIUS = pt.REF, pt.CONTRAST, pt.RADIUS
TOL = 1e-12
CACHE = ROOT / "scripts" / "data" / "octree_finest_blocks"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ------------------------------------------------------------------ the finest-grid blocks
def _block_job(job):
    off, h0, omega, nc, nf = job
    return off, coupling_block(off, h0, omega, REF, nc, nf)


def finest_blocks(n_f, h0, omega, nc, nf, workers=4):
    """K(o) for every canonical offset o (o0 >= o1 >= o2 >= 0) of the n_f grid, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"n{n_f}_h{h0:.6g}_w{omega:.6g}_nc{nc}_nf{nf}.pkl"
    store = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = sorted({tuple(sorted(o, reverse=True)) for o in itertools.product(range(n_f), repeat=3)} - set(store))
    if todo:
        env = {v: os.environ.get(v) for v in THREADS}
        os.environ.update({v: "1" for v in THREADS})
        try:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                store.update(pool.map(_block_job, [(o, h0, omega, nc, nf) for o in todo], chunksize=8))
        finally:
            for v, val in env.items():
                os.environ.pop(v, None) if val is None else os.environ.__setitem__(v, val)
        path.write_bytes(pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL))
    return store


def reflection_signs(nc, nf, na):
    rows, cols = [], []
    for m in range(3):
        q = np.eye(3)
        q[m, m] = -1.0
        t, u, s = symmetry_reps(q, nc, nf)
        rows.append(np.kron(np.diag(t), np.diag(s))[: na * 9])
        cols.append(np.kron(np.diag(u), np.diag(s)))
    return rows, cols


def octant_transform(bo, n, rs, cs):
    """K^(k), k in [0, n)^3, from K(o), o in [0, n)^3; DFT size 2n - 1; each entry's parity per axis."""
    npad = 2 * n - 1
    k = np.arange(n)
    theta = 2.0 * np.pi * np.outer(k, k) / npad
    even = np.where(k[None, :] == 0, 1.0, 2.0 * np.cos(theta))
    odd = np.where(k[None, :] == 0, 0.0, -2.0j * np.sin(theta))
    out = bo.astype(complex)
    for m in range(3):
        par = np.outer(rs[m], cs[m]) > 0
        moved = np.moveaxis(out, m, 0)
        tr = np.tensordot(odd, moved, axes=(1, 0))
        tr[..., par] = np.tensordot(even, moved[..., par], axes=(1, 0))
        out = np.moveaxis(tr, 0, m)
    return out


class Finest:
    """The equal-cell operator of the finest grid, applied by FFT with the octant of its transform."""

    def __init__(self, n_f, h0, omega, p, r):
        self.na, self.nf, _, self.nc = field_sizes(p, r)
        self.n, self.h0 = n_f, h0
        self.rows, self.cols = self.na * 9, self.nc * 9
        canon = finest_blocks(n_f, h0, omega, self.nc, self.nf)
        qs = signed_permutations()
        bo = np.zeros((n_f, n_f, n_f, self.rows, self.cols), dtype=complex)
        for o in itertools.product(range(n_f), repeat=3):
            c = tuple(sorted(o, reverse=True))
            q = next(q for q in qs if np.array_equal(q @ np.array(c, float), np.array(o, float)))
            bo[o] = _transform(q, canon[c])[: self.na].transpose(0, 2, 1, 3).reshape(self.rows, self.cols)
        self.rs, self.cs = reflection_signs(self.nc, self.nf, self.na)
        self.k_oct = octant_transform(bo, n_f, self.rs, self.cs)
        del bo
        npad = 2 * n_f - 1
        self.npad = npad
        self.patterns = []
        for sig in itertools.product((1, -1), repeat=3):
            tgt, src, dr, dc = [], [], np.ones(self.rows), np.ones(self.cols)
            for m, sg in enumerate(sig):
                if sg > 0:
                    tgt.append(np.arange(0, n_f))
                    src.append(np.arange(0, n_f))
                else:
                    tt = np.arange(n_f, npad)
                    tgt.append(tt)
                    src.append(npad - tt)
                    dr, dc = dr * self.rs[m], dc * self.cs[m]
            self.patterns.append((np.ix_(*tgt), np.ix_(*src), dr, dc))

    def conv(self, gidx, src):
        """y at the finest cells gidx (F, 3) from sources src (F, cols) at the same cells: (F, rows)."""
        grid = np.zeros((self.npad,) * 3 + (self.cols,), dtype=complex)
        grid[tuple(gidx.T)] = src
        sh = np.fft.fftn(grid, axes=(0, 1, 2))
        yh = np.zeros((self.npad,) * 3 + (self.rows,), dtype=complex)
        for tgt, srci, dr, dc in self.patterns:
            yh[tgt] = dr * np.einsum("xyzrc,xyzc->xyzr", self.k_oct[srci], sh[tgt] * dc)
        return np.fft.ifftn(yh, axes=(0, 1, 2))[tuple(gidx.T)]


class Tree:
    """A tree's leaves on the finest grid: the Galerkin operator, its local blocks, its right-hand side."""

    def __init__(self, fin: Finest, centres, hs, omega, p, r):
        self.fin, self.c, self.h, self.omega, self.p, self.r = fin, np.asarray(centres), np.asarray(hs), omega, p, r
        na, nf, nc = fin.na, fin.nf, fin.nc
        self.L = len(self.h)
        h0, a = fin.h0, RADIUS
        gidx, owner, cmat, dmat = [], [], [], []
        for leaf, (c, h) in enumerate(zip(self.c, self.h, strict=True)):
            m = int(round(h / h0))
            base = np.rint((c - h + a) / (2 * h0)).astype(int)
            for i in itertools.product(range(m), repeat=3):
                g = base + np.array(i)
                x = -a + h0 * (2 * g + 1)
                s = tuple(round(float(v), 12) for v in (x - c) / h)
                gidx.append(g)
                owner.append(leaf)
                cmat.append(field_reexpansion(nf, round(h0 / h, 12), s)[:na, :na])
                dmat.append(source_reexpansion(nc, round(h0 / h, 12), s))
        self.gidx, self.owner = np.array(gidx), np.array(owner)
        self.cmat, self.dmat = np.array(cmat), np.array(dmat)
        self.delta = [cell_contrast_coefficients(pt.prof, c, float(h), CONTRAST, REF, omega, degree=r)
                      for c, h in zip(self.c, self.h, strict=True)]  # fmt: skip
        self.e = np.array([source_expansion(d, nf)[:nc, :na] for d in self.delta])  # (L, nc, na, 9, 9)
        self.gram = np.array([gram_test(float(h), nf)[:na, :na] for h in self.h])
        selfk = {}
        loc = []
        for h, en, gm in zip(self.h, self.e, self.gram, strict=True):
            key = round(float(h), 9)
            if key not in selfk:
                selfk[key] = coupling_block((0, 0, 0), float(h), omega, REF, nc, nf)[:na]
            loc.append(np.kron(gm, np.eye(9)) - np.einsum("acij,cbjk->aibk", selfk[key], en).reshape(na * 9, na * 9))
        self.local_inv = np.linalg.inv(np.array(loc))
        k_mag = omega / REF.alpha
        amp = np.concatenate([pt.K_HAT.astype(complex), _plane_wave_strain_voigt(pt.K_HAT, pt.K_HAT, k_mag)])
        self.b = np.array([plane_wave_moments(c[None, :], float(h), k_mag * pt.K_HAT, amp, nf)[0, :na]
                           for c, h in zip(self.c, self.h, strict=True)])  # fmt: skip

    def apply(self, psi):
        fin = self.fin
        src = np.einsum("lcbij,lbj->lci", self.e, psi)  # leaf source monomial coefficients (L, nc, 9)
        sf = np.einsum("fcd,fci->fdi", self.dmat, src[self.owner]).reshape(len(self.owner), fin.cols)
        yf = fin.conv(self.gidx, sf).reshape(len(self.owner), fin.na, 9)
        y = np.zeros((self.L, fin.na, 9), dtype=complex)
        np.add.at(y, self.owner, np.einsum("fab,fbi->fai", self.cmat, yf))
        return np.einsum("lab,lbi->lai", self.gram, psi) - y

    def local_solve(self, v):
        return np.einsum("lij,lj->li", self.local_inv, v.reshape(self.L, -1)).reshape(v.shape)

    def solve(self, rhs):
        dim = rhs.size
        a = LinearOperator((dim, dim), matvec=lambda v: self.apply(v.reshape(rhs.shape)).ravel(), dtype=complex)
        m = LinearOperator((dim, dim), matvec=lambda v: self.local_solve(v.reshape(rhs.shape)).ravel(), dtype=complex)
        sol, info = gmres(a, rhs.ravel(), x0=m.matvec(rhs.ravel()), M=m, rtol=TOL, atol=0.0, restart=200, maxiter=40)
        assert info == 0, info
        return sol.reshape(rhs.shape)

    def result(self, psi):
        full = np.zeros((self.L, self.fin.nf, 9), dtype=complex)
        full[:, : self.fin.na] = psi
        return OctreeResult(self.c, self.h, self.omega, REF, np.array(self.delta), full, self.p, self.r)


def far_leaves(tree: Tree, psi, dirs, n_gauss=6):
    """Each leaf's far field at r = 1 (P then S, directions x components), shape (L, 2 * D * 3)."""
    x, w = leggauss(n_gauss)
    xi = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    ww = np.einsum("i,j,k->ijk", w, w, w).ravel()
    ms = monomials(SOURCE_EXPONENTS_QUARTIC[: tree.fin.nc], xi)
    src = np.einsum("lcbij,lbj->lci", tree.e, psi)
    nodes = np.einsum("cg,lci->lgi", ms, src) * (ww[None, :, None] * tree.h[:, None, None] ** 3)
    pts = tree.c[:, None, :] + tree.h[:, None, None] * xi[None]
    forces = nodes[..., :3]
    sig = np.array([_voigt_to_tensor(v) for v in nodes[..., 3:].reshape(-1, 6)]).reshape(*nodes.shape[:2], 3, 3)
    kp, ks = tree.omega / REF.alpha, tree.omega / REF.beta
    out_p, out_s = [], []
    for rh in dirs:
        proj = pts @ rh
        gp = np.exp(1j * kp * (1.0 - proj)) / (4 * np.pi * REF.rho * REF.alpha**2)
        gsh = np.exp(1j * ks * (1.0 - proj)) / (4 * np.pi * REF.rho * REF.beta**2)
        sr = sig @ rh
        qp = ((forces @ rh) + 1j * kp * (sr @ rh)) * gp
        out_p.append(qp.sum(1)[:, None] * rh[None, :])
        qs = forces + 1j * ks * sr
        qs = (qs - (qs @ rh)[..., None] * rh) * gsh[..., None]
        out_s.append(qs.sum(1))
    return np.concatenate([np.stack(out_p, 1), np.stack(out_s, 1)], 1).reshape(tree.L, -1)


def two_scale_shares(fin, coarse: Tree, x_h, omega, p, r, dirs):
    """Each coarse leaf's estimated share of F_h - F_H (two-level), and the fine tree's leaf count."""
    refine = coarse.h > fin.h0 * (1 + 1e-9)
    fc, fh = refine_leaves(coarse.c, coarse.h, refine)
    fine = Tree(fin, fc, fh, omega, p, r)
    # parent of each fine leaf (refine_leaves keeps the unrefined first, then the children in order)
    parent = np.concatenate([np.where(~refine)[0], np.repeat(np.where(refine)[0], 8)])
    na, nf = fin.na, fin.nf
    pmat = []
    for c, h, par in zip(fc, fh, parent, strict=True):
        if refine[par]:
            s = tuple(round(float(v), 12) for v in (c - coarse.c[par]) / coarse.h[par])
            pmat.append(field_reexpansion(nf, 0.5, s)[:na, :na])
        else:
            pmat.append(np.eye(na))
    pmat = np.array(pmat)

    def prolong(xc):
        return np.einsum("fab,fai->fbi", pmat, xc[parent])

    def restrict(v):
        out = np.zeros((coarse.L, na, 9), dtype=complex)
        np.add.at(out, parent, np.einsum("fab,fbi->fai", pmat, v))
        return out

    psi_f = prolong(x_h)
    res = fine.b - fine.apply(psi_f)
    lr = fine.local_solve(res)
    e = lr + prolong(coarse.solve(restrict(res - fine.apply(lr))))
    far_e = far_leaves(fine, e, dirs)
    far_rad = far_leaves(fine, psi_f, dirs)
    share = np.zeros((coarse.L, far_e.shape[1]), dtype=complex)
    np.add.at(share, parent, far_e + far_rad)
    share -= far_leaves(coarse, x_h, dirs)
    share[~refine] = 0.0  # a leaf at the finest size cannot be refined; its share is not an option
    return share, len(fh)


def true_error(tree: Tree, psi, body, exact, peak):
    u = sum(octree_far_field(tree.result(psi), body.pts / body.rf, body.rf))
    return float(np.abs(u - exact).max() / peak)


def main() -> int:
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    p = int(opts.get("p", 1))
    tols = [float(v) for v in opts.get("tol", "1e-4,3e-5,1e-5").split(",")]
    pt.CORE, pt.WIDTH = 0.1 * RADIUS, 0.3 * RADIUS  # the localised feature of the paper's Table
    ka, h0 = 1.0, RADIUS / 32
    n_f = int(round(RADIUS / h0))
    body = pt.Body(ka, p)
    dirs = body.pts / np.linalg.norm(body.pts, axis=1)[:, None]
    t0 = time.perf_counter()
    print(f"localised feature (core 0.1a, to 0.3a, halo {pt.HALO}), k_S a = {ka}, p = r = {p}, finest h0 = a/32", flush=True)
    fin = Finest(n_f, h0, body.omega, p, p)
    print(f"   finest-grid operator: {n_f}^3 grid   [{time.perf_counter() - t0:.0f} s]", flush=True)
    exact = mres.exact_field(pt.SHAPE, pt.CORE, body.omega, 1.0, body.pts)
    peak = float(np.abs(exact).max())
    if "--check" in sys.argv:
        for name, (cc, hh) in (("uniform n 4", uniform_leaves(RADIUS, 4)),):
            tree = Tree(fin, cc, hh, body.omega, p, p)
            x = tree.solve(tree.b)
            ref = solve_graded_octree(body.omega, REF, CONTRAST, cc, hh, pt.prof, pt.K_HAT, pt.K_HAT, "P", p=p, r=p)
            dx = np.abs(x - ref.psi[:, : fin.na]).max() / np.abs(x).max()
            print(f"   [check] {name}: finest-grid solve vs dense octree solve {dx:.1e}; "
                  f"error {true_error(tree, x, body, exact, peak):.4e}", flush=True)  # fmt: skip
        return 0
    rows = []
    if "theta" in opts:  # Doerfler marking: refine the fewest leaves that hold a fraction theta of sum |eta|
        theta, max_leaves = float(opts["theta"]), int(opts.get("max", 800))
        tols = []
        cc, hh = uniform_leaves(RADIUS, 2)
        it = 0
        while True:
            tree = Tree(fin, cc, hh, body.omega, p, p)
            x = tree.solve(tree.b)
            err = true_error(tree, x, body, exact, peak)
            share, _ = two_scale_shares(fin, tree, x, body.omega, p, p, dirs)
            fpeak = np.abs(far_leaves(tree, x, dirs).sum(0)).max()
            eta = np.abs(share).max(axis=1) / fpeak
            predicted = float(np.abs(share.sum(0)).max() / fpeak)
            sizes = {float(h): int((hh == h).sum()) for h in np.unique(hh)}
            print(f"   theta {theta:g} it {it}: leaves {len(hh):5d}, error {err:.3e}, two-scale difference "
                  f"{predicted:.3e}; {sizes}   [{time.perf_counter() - t0:.0f} s]", flush=True)  # fmt: skip
            rows.append({"theta": theta, "it": it, "cells": len(hh), "error": err, "difference": predicted,
                         "sizes": sizes})  # fmt: skip
            order = np.argsort(-eta)
            cum = np.cumsum(eta[order])
            if cum[-1] <= 0.0:
                break
            flag = np.zeros(len(hh), dtype=bool)
            flag[order[: int(np.searchsorted(cum, theta * cum[-1])) + 1]] = True
            flag &= eta > 0.0
            if len(hh) + 7 * flag.sum() > max_leaves:
                break
            cc, hh = refine_leaves(cc, hh, flag)
            it += 1
    for tol in tols:
        cc, hh = uniform_leaves(RADIUS, 2)
        it = 0
        while True:
            tree = Tree(fin, cc, hh, body.omega, p, p)
            x = tree.solve(tree.b)
            err = true_error(tree, x, body, exact, peak)
            share, n_fine = two_scale_shares(fin, tree, x, body.omega, p, p, dirs)
            fpeak = np.abs(far_leaves(tree, x, dirs).sum(0)).max()
            eta = np.abs(share).max(axis=1) / fpeak
            predicted = float(np.abs(share.sum(0)).max() / fpeak)
            sizes = {float(h): int((hh == h).sum()) for h in np.unique(hh)}
            print(f"   tol {tol:g} it {it}: leaves {len(hh):5d}, error {err:.3e}, two-scale difference {predicted:.3e}; "
                  f"{sizes}   [{time.perf_counter() - t0:.0f} s]", flush=True)  # fmt: skip
            rows.append({"tol": tol, "it": it, "cells": len(hh), "error": err, "difference": predicted, "sizes": sizes})
            flag = eta > tol
            if not flag.any():
                break
            cc, hh = refine_leaves(cc, hh, flag)
            it += 1
    tag = f"_theta{opts['theta']}" if "theta" in opts else ""
    out = ROOT / "scratch" / "two_scale" / f"octree_refinement_localised_p{p}{tag}.json"
    out.write_text(json.dumps({"ka_s": ka, "p": p, "h0": h0, "rows": rows}, indent=2) + "\n")
    print(f"   wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
