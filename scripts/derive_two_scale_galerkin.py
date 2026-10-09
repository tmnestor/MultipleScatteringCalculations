#!/usr/bin/env python3
"""The two-scale formulation of the Galerkin projection, checked on the graded sphere.

(docs/2026-10-09-two-scale-galerkin.md.) A coarse grid of n cells across and the grid of their children
(2n across, every child of every coarse cell kept). V_H ⊂ V_h, with the prolongation S from
``octree.field_reexpansion``. With x_H the coarse solution and r_h = b_h − A_h S x_H the fine residual of
its prolongation, the coarse-to-fine error e = y_h − S x_H solves A_h e = r_h exactly. The residual splits
by the dual projector P = M_h S M_H⁻¹ S^T into

    r_med = P r_h        (driven only by the contrast's detail Δ_h − Δ_H; S^T r_h = K_H (Ẽ − E_H) x_H),
    r_fld = (I − P) r_h  (the multiwavelet detail of the field the coarse solution implies),

and the far field into

    F_h − F_H = R Δ_h A_h⁻¹ r_med + R Δ_h A_h⁻¹ r_fld + R(δΔ ψ_H).

Checked: the split to the solve tolerance; S^T r_h = 0 when the medium has no detail (--uniform: a contrast
constant over the bounding cube, which every grid holds exactly); saturation, F_H − F_exact against
F_H − F_h; and the sizes of the three terms, with local estimates of e that need no fine solve
(block-diagonal inverse of A_h per child cell, and per parent).

Run:  python -u scripts/derive_two_scale_galerkin.py [--p=1] [--r=1] [--ka=0.5] [--scale=1] [--uniform] n1 ...
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

from cubic_scattering.effective_contrasts import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.basis import gram_test, source_expansion  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import offset_blocks, solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.graded_voxel.octree import field_reexpansion  # noqa: E402
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients  # noqa: E402
from cubic_scattering.graded_voxel.solver import GradedVoxelResult, field_sizes, plane_wave_moments  # noqa: E402
from cubic_scattering.sphere_scattering import _plane_wave_strain_voigt, mie_scattered_displacement  # noqa: E402

REF = hier.REF
TOL = 1e-12
CHILD_SHIFTS = [np.array(s) - 0.5 for s in itertools.product((0, 1), repeat=3)]  # xi_parent = xi/2 + s


class GridOperator:
    """A = M - K E on a set of cells of one size, applied by FFT on the (2 n_grid - 1)^3 offsets."""

    def __init__(self, centres, grid_idx, h, delta, n_grid, omega, p, r):
        self.n = len(centres)
        self.na, self.n_field, _, self.n_source = field_sizes(p, r)
        na, nc = self.na, self.n_source
        self.e = np.array([source_expansion(d, self.n_field)[:nc, :na] for d in delta])
        self.blocks = offset_blocks(n_grid, h, omega, REF, nc, self.n_field)
        npad = 2 * n_grid - 1
        rows, cols = na * 9, nc * 9
        kh = np.zeros((rows, cols, npad, npad, npad), dtype=complex)
        for off, blk in self.blocks.items():
            kh[:, :, off[0] % npad, off[1] % npad, off[2] % npad] = blk[:na, :nc].transpose(0, 2, 1, 3).reshape(rows, cols)
        for row in range(rows):
            kh[row] = np.fft.fftn(kh[row], axes=(1, 2, 3))
        self.kh, self.npad, self.rows, self.cols = kh, npad, rows, cols
        self.g = tuple(grid_idx.T)
        self.m9 = gram_test(h, self.n_field)[:na, :na]
        k0 = self.blocks[(0, 0, 0)][:na, :nc]
        self.local = np.array([
            np.kron(self.m9, np.eye(9)) - np.einsum("acij,cbjk->aibk", k0, en).reshape(rows, rows) for en in self.e
        ])  # fmt: skip
        self.local_inv = np.linalg.inv(self.local)

    def kconv(self, psi):
        src = np.einsum("ncbij,nbj->nci", self.e, psi).reshape(self.n, self.cols)
        grid = np.zeros((self.cols, self.npad, self.npad, self.npad), dtype=complex)
        grid[(slice(None), *self.g)] = src.T
        sh = np.fft.fftn(grid, axes=(1, 2, 3)).reshape(self.cols, -1)
        yh = np.einsum("rcf,cf->rf", self.kh.reshape(self.rows, self.cols, -1), sh)
        y = np.fft.ifftn(yh.reshape(self.rows, self.npad, self.npad, self.npad), axes=(1, 2, 3))
        return y[(slice(None), *self.g)].T.reshape(self.n, self.na, 9)

    def apply(self, psi):
        return np.einsum("ab,nbi->nai", self.m9, psi) - self.kconv(psi)

    def solve(self, rhs):
        dim = self.n * self.rows
        a = LinearOperator((dim, dim), matvec=lambda v: self.apply(v.reshape(self.n, self.na, 9)).ravel(), dtype=complex)
        m = LinearOperator((dim, dim), matvec=lambda v: np.einsum("nij,nj->ni", self.local_inv, v.reshape(self.n, -1)).ravel(),
                           dtype=complex)  # fmt: skip
        b = rhs.ravel()
        sol, info = gmres(a, b, x0=m.matvec(b), M=m, rtol=TOL, atol=0.0, restart=200, maxiter=40)
        assert info == 0, info
        return sol.reshape(self.n, self.na, 9)

    def local_solve(self, rhs):
        return np.einsum("nij,nj->ni", self.local_inv, rhs.reshape(self.n, -1)).reshape(self.n, self.na, 9)


def far(centres, grid_idx, h, omega, delta, psi, p, r, dirs, dist):
    na, n_field, _, _ = field_sizes(p, r)
    full = np.zeros((len(centres), n_field, 9), dtype=complex)
    full[:, :na] = psi
    res = GradedVoxelResult(centres, grid_idx, h, omega, REF, delta, full, p, r)
    up, us = graded_far_field(res, dirs, dist, gs.K_HAT, gs.POL, "P", n_gauss=6)
    return np.concatenate([up, us])  # P amplitudes over the directions, then S


def far_cells(centres, grid_idx, h, omega, delta, psi, p, r, dirs, n_gauss=6):
    """Far field at r = 1 of each cell's sources separately, shape (cells, 2 * directions * 3)."""
    from cubic_scattering.graded_voxel.farfield import _node_sources
    from cubic_scattering.sphere_scattering import _voigt_to_tensor

    na, n_field, _, _ = field_sizes(p, r)
    full = np.zeros((len(centres), n_field, 9), dtype=complex)
    full[:, :na] = psi
    res = GradedVoxelResult(centres, grid_idx, h, omega, REF, delta, full, p, r)
    pts, srcs = _node_sources(res, n_gauss)
    g = n_gauss**3
    forces = srcs[:, :3]
    sig = np.array([_voigt_to_tensor(v[3:]) for v in srcs])
    k_p, k_s = omega / REF.alpha, omega / REF.beta
    out_p, out_s = [], []
    for rh in dirs:
        proj = pts @ rh
        gp = np.exp(-1j * k_p * proj) / (4 * np.pi * REF.rho * REF.alpha**2)
        gsh = np.exp(-1j * k_s * proj) / (4 * np.pi * REF.rho * REF.beta**2)
        sr = sig @ rh
        qp = (forces @ rh + 1j * k_p * (sr @ rh)) * gp
        out_p.append((qp.reshape(-1, g).sum(1))[:, None] * rh[None, :])
        qs = forces + 1j * k_s * sr
        qs = (qs - np.outer(qs @ rh, rh)) * gsh[:, None]
        out_s.append(qs.reshape(-1, g, 3).sum(1))
    return np.concatenate([np.stack(out_p, 1), np.stack(out_s, 1)], 1).reshape(len(centres), -1)


def ranking(est, true, frac=0.2):
    """Share of the true total |eta| held by the top `frac` of cells chosen by `est`, and by the best choice."""
    k = max(1, int(round(frac * len(true))))
    t = np.abs(true)
    return t[np.argsort(-np.abs(est))[:k]].sum() / t.sum(), t[np.argsort(-t)[:k]].sum() / t.sum()


def main() -> int:
    args = sys.argv[1:]
    opts = dict(a[2:].split("=") for a in args if a.startswith("--") and "=" in a)
    p, r = int(opts.get("p", 1)), int(opts.get("r", 1))
    ka, scale = float(opts.get("ka", 0.5)), float(opts.get("scale", 1.0))
    uniform = "--uniform" in args
    ladder = [int(a) for a in args if not a.startswith("--")] or [4]
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile("smoothstep")
    c0 = hier.CONTRAST
    contrast = MaterialContrast(Dlambda=scale * c0.Dlambda, Dmu=scale * c0.Dmu, Drho=scale * c0.Drho)
    omega = ka * REF.beta / gs.RADIUS
    obs = obs_points(gs.R_FAR, gs.THETA)
    dirs = obs / np.linalg.norm(obs, axis=1)[:, None]

    def prof(pos) -> float:
        return 1.0 if uniform else float(gs.smoothstep(np.array([np.linalg.norm(pos)]))[0])

    exact = None
    if not uniform:
        n_max = max(8, int(np.ceil(ka + 4 * ka ** (1 / 3) + 6)))
        exact = mie_scattered_displacement(graded_mie_result(omega, gs.RADIUS, gs.CORE, REF, contrast, n_max), obs)
    na, n_field, _, _ = field_sizes(p, r)
    print(f"two-scale Galerkin: p = {p}, r = {r}, k_S a = {ka}, contrast x {scale}, "
          f"{'uniform medium' if uniform else 'graded sphere'}", flush=True)  # fmt: skip
    for n in ladder:
        # coarse
        res_h = solve_graded_sphere_fft(omega, gs.RADIUS, REF, contrast, n, prof, gs.K_HAT, gs.POL, "P", p=p, r=r,
                                        gmres_tol=TOL, max_cycles=60)  # fmt: skip
        cH, gH, H = res_h.centres, res_h.grid_idx, res_h.h
        xH = res_h.psi[:, :na]
        # fine: every child of every coarse cell
        h = H / 2.0
        cf = np.concatenate([cH + 2 * h * s for s in CHILD_SHIFTS])  # child centre = parent + H s
        gf = np.concatenate([2 * gH + (s + 0.5).astype(int) for s in CHILD_SHIFTS])
        nH = len(cH)
        child = [np.arange(d * nH, (d + 1) * nH) for d in range(8)]  # fine indices of child d of each parent
        cmat = [field_reexpansion(n_field, 0.5, tuple(float(v) for v in s))[:na, :na] for s in CHILD_SHIFTS]
        delta_f = np.array([cell_contrast_coefficients(prof, c, h, contrast, REF, omega, degree=r) for c in cf])
        op = GridOperator(cf, gf, h, delta_f, 2 * n, omega, p, r)
        k_mag = omega / REF.alpha
        amp = np.concatenate([gs.POL.astype(complex), _plane_wave_strain_voigt(gs.K_HAT, gs.POL, k_mag)])
        b_h = plane_wave_moments(cf, h, k_mag * gs.K_HAT, amp, n_field)[:, :na]
        y_h = op.solve(b_h)

        def prolong(x):
            out = np.zeros((8 * nH, na, 9), dtype=complex)
            for d in range(8):
                out[child[d]] = np.einsum("ab,nai->nbi", cmat[d], x)
            return out

        def restrict(rf):  # S^T on dual vectors
            return sum(np.einsum("ab,nbi->nai", cmat[d], rf[child[d]]) for d in range(8))

        m_h = gram_test(h, n_field)[:na, :na]
        m_hinv = np.linalg.inv(gram_test(H, n_field)[:na, :na])
        psi_H = prolong(xH)
        r_h = b_h - op.apply(psi_H)
        r_med = np.einsum("ab,nbi->nai", m_h, prolong(np.einsum("ab,nbi->nai", m_hinv, restrict(r_h))))
        r_fld = r_h - r_med
        e_med, e_fld = op.solve(r_med), op.solve(r_fld)
        # far fields at r = 1 (the identity) and at R_FAR (against the exact sphere)
        F = {}
        F["H"] = far(cH, gH, H, omega, res_h.delta, xH, p, r, dirs, 1.0)
        F["h"] = far(cf, gf, h, omega, delta_f, y_h, p, r, dirs, 1.0)
        F["med"] = far(cf, gf, h, omega, delta_f, e_med, p, r, dirs, 1.0)
        F["fld"] = far(cf, gf, h, omega, delta_f, e_fld, p, r, dirs, 1.0)
        F["rad"] = far(cf, gf, h, omega, delta_f, psi_H, p, r, dirs, 1.0) - F["H"]
        F["med_loc"] = far(cf, gf, h, omega, delta_f, op.local_solve(r_med), p, r, dirs, 1.0)
        F["fld_loc"] = far(cf, gf, h, omega, delta_f, op.local_solve(r_fld), p, r, dirs, 1.0)
        # two-level estimate: the local inverse for the detail, one coarse solve for the smooth remainder,
        #   e2 = L r + S A_H^-1 S^T (r - A_h L r)        (no fine solve; one fine matvec)
        op_H = GridOperator(cH, gH, H, res_h.delta, n, omega, p, r)

        def two_level(rv):
            lr = op.local_solve(rv)
            return lr + prolong(op_H.solve(restrict(rv - op.apply(lr))))

        F["med_2l"] = far(cf, gf, h, omega, delta_f, two_level(r_med), p, r, dirs, 1.0)
        F["fld_2l"] = far(cf, gf, h, omega, delta_f, two_level(r_fld), p, r, dirs, 1.0)
        diff = F["h"] - F["H"]
        scale_f = np.abs(F["h"]).max()
        ident = np.abs(diff - F["med"] - F["fld"] - F["rad"]).max() / scale_f
        e_err = np.abs(y_h - psi_H - e_med - e_fld).max() / np.abs(y_h).max()
        print(f"  n {n:2d} -> {2 * n:2d}: coarse cells {nH}, fine {8 * nH};  y_h = S x_H + e_med + e_fld to {e_err:.1e},"
              f"  far-field split to {ident:.1e}", flush=True)  # fmt: skip
        print(f"     |S^T r_h| / |S^T b_h| = {np.abs(restrict(r_h)).max() / np.abs(restrict(b_h)).max():.2e},"
              f"  |r_fld| / |b_h| = {np.abs(r_fld).max() / np.abs(b_h).max():.2e}", flush=True)  # fmt: skip
        rel = {k: np.abs(F[k]).max() / scale_f for k in ("med", "fld", "rad", "med_loc", "fld_loc")}
        print(f"     |F_h - F_H| = {np.abs(diff).max() / scale_f:.3e} of |F_h|:  medium solve {rel['med']:.3e}, "
              f"field solve {rel['fld']:.3e}, medium radiation {rel['rad']:.3e}", flush=True)  # fmt: skip
        loc = np.abs(F["med_loc"] + F["fld_loc"] + F["rad"] - diff).max() / np.abs(diff).max()
        print(f"     local estimate (block-diagonal A_h per child): medium {rel['med_loc']:.3e}, field "
              f"{rel['fld_loc']:.3e}; error of the estimated F_h - F_H {loc:.2e} of it", flush=True)  # fmt: skip
        tl = np.abs(F["med_2l"] + F["fld_2l"] + F["rad"] - diff).max() / np.abs(diff).max()
        print(f"     two-level estimate (local + one coarse solve): medium "
              f"{np.abs(F['med_2l']).max() / scale_f:.3e}, field {np.abs(F['fld_2l']).max() / scale_f:.3e}; "
              f"error of the estimated F_h - F_H {tl:.2e} of it", flush=True)  # fmt: skip
        # the share of each coarse cell: its children's error sources plus its own medium-detail radiation
        def per_parent(cells_fine):
            return sum(cells_fine[child[d]] for d in range(8))

        rad_cells = per_parent(far_cells(cf, gf, h, omega, delta_f, psi_H, p, r, dirs)) - far_cells(
            cH, gH, H, omega, res_h.delta, xH, p, r, dirs)
        eta_true = per_parent(far_cells(cf, gf, h, omega, delta_f, e_med + e_fld, p, r, dirs)) + rad_cells
        eta_est = per_parent(far_cells(cf, gf, h, omega, delta_f, two_level(r_med) + two_level(r_fld), p, r, dirs)) + rad_cells
        # the far field of all cells, summed over the 2 x directions x 3 components: sum of shares = F_h - F_H
        sum_err = np.abs(eta_true.sum(0) - diff.ravel()).max() / np.abs(diff).max()
        nt, ne = np.linalg.norm(eta_true, axis=1), np.linalg.norm(eta_est, axis=1)
        shares = np.abs(ne - nt).sum() / nt.sum()
        # a rule on the medium alone: each coarse cell's contrast detail |Delta_h - Delta_H|, from the children's
        # projected contrast against the parent's re-expanded on them (Frobenius norm over the 9 x 9 operator)
        cfull = [field_reexpansion(len(res_h.delta[0]), 0.5, tuple(float(v) for v in sv)) for sv in CHILD_SHIFTS]
        med = np.zeros(nH)
        for d in range(8):
            dd = delta_f[child[d]] - np.einsum("ab,nbij->naij", cfull[d], res_h.delta)
            med += np.einsum("naij->n", np.abs(dd) ** 2)
        med = np.sqrt(med)
        top_est, top_best = ranking(ne, nt)
        top_med, _ = ranking(med, nt)
        conc = np.searchsorted(np.cumsum(np.sort(nt)[::-1]) / nt.sum(), 0.9) + 1
        print(f"     cell shares: sum = F_h - F_H to {sum_err:.1e}; two-level shares off by {shares:.2e} in total; "
              f"90% of the error in {conc} of {nH} cells", flush=True)  # fmt: skip
        print(f"     top 20% of cells hold {top_best:.3f} of the error; chosen by the two-level indicator {top_est:.3f}, "
              f"by the medium's detail alone {top_med:.3f}", flush=True)  # fmt: skip
        if exact is not None:
            fh = far(cf, gf, h, omega, delta_f, y_h, p, r, obs / gs.R_FAR, gs.R_FAR)
            fH = far(cH, gH, H, omega, res_h.delta, xH, p, r, obs / gs.R_FAR, gs.R_FAR)
            nd = len(obs)
            tot_h, tot_H = fh[:nd] + fh[nd:], fH[:nd] + fH[nd:]
            peak = np.abs(exact).max()
            eH, eh = np.abs(tot_H - exact).max() / peak, np.abs(tot_h - exact).max() / peak
            sat = np.abs((tot_H - exact) - (tot_H - tot_h)).max() / np.abs(tot_H - exact).max()
            print(f"     against the exact sphere: coarse {eH:.3e}, fine {eh:.3e} (order {math.log(eH / eh) / math.log(2):.2f});"
                  f"  F_H - F_exact = F_H - F_h to {sat:.2e} of it", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
