#!/usr/bin/env python3
"""The second-order term of the 3-D voxel scheme against the exact one, angle by angle and wave by wave,
divided by the relative projection error of the medium: the test of the 3-D T2 law.

In the layer the relative error of T2 is the relative projection error E exactly in the long-wave limit
(Paper 1, Appendix G). In three dimensions the strain-strain kernel is not local, and the argument in
Fourier space gives, for cells of constant field and contrast (p = r = 0) and a body whose spectrum is
isotropic (a radial profile),

    (T2 - T2_scheme) / (T2 E)  =  2 - m_ax / m_bar ,

with m(xi) = a : Gamma(xi) : b the static strain Green operator of the background contracted with the
contrast times the outgoing strain (a) and times the incident strain (b), m_bar its average over
directions and m_ax its mean over the three cube axes. The factor depends on the angle and the wave type.

Measured here: the graded sphere of the voxel tests (radius 10 m, core a/2, smoothstep), a P wave along x,
stiffness contrast only (the density channel is of higher order in k a), at a low frequency, on uniform
grids. Scheme: psi0 = M^-1 <L, psi_inc>, psi1 = M^-1 sum_n K(o_mn) E_n psi0_n, T2 = far field of psi1.
Exact: T2 = (f(d) + f(-d)) / (2 d^2) from the exact sphere at contrast +-d. Far fields are split into
their P (radial) and S (transverse) parts at 5e8 radii.

Run:  conda run -n seismic python -u scripts/measure_t2_law_3d.py <k_S a> <p> <n ...>
      e.g. ... 0.125 0 4 6 8
"""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.basis import gram_test, source_expansion  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import offset_blocks  # noqa: E402
from cubic_scattering.graded_voxel.site import cell_contrast_coefficients  # noqa: E402
from cubic_scattering.graded_voxel.solver import GradedVoxelResult, plane_wave_moments  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    _plane_wave_strain_voigt,
    mie_scattered_displacement,
)
from cubic_scattering.sphere_scattering_fft import _build_grid_index_map  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST as GATE  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA, profile  # noqa: E402

K = np.array([1.0, 0.0, 0.0])
CONTRAST = MaterialContrast(GATE.Dlambda, GATE.Dmu, 0.0)  # stiffness only


# ------------------------------------------------------------------ the p = 0 prediction
def strain_green_m(xi: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """a : Gamma(xi) : b for the static strain Green operator of the isotropic background."""
    mu = REF.rho * REF.beta**2
    lam = REF.rho * REF.alpha**2 - 2 * mu
    xi = xi / np.linalg.norm(xi)
    c = (lam + mu) / (mu * (lam + 2 * mu))
    return float(xi @ ((a @ b + b @ a) / 2) @ xi / mu - c * (xi @ a @ xi) * (xi @ b @ xi))


def strain_green_m_bar(a: np.ndarray, b: np.ndarray) -> float:
    """The average of a : Gamma(xi) : b over directions, in closed form (AdaptiveOctree_T2Factor.wl (c))."""
    mu = REF.rho * REF.beta**2
    lam = REF.rho * REF.alpha**2 - 2 * mu
    c = (lam + mu) / (mu * (lam + 2 * mu))
    return float(
        np.trace((a @ b + b @ a) / 2) / (3 * mu) - c * (np.trace(a) * np.trace(b) + 2 * np.sum(a * b)) / 15
    )


def predicted_factor(e_out: np.ndarray) -> float:
    """2 - m_ax / m_bar for an outgoing strain e_out and the incident P strain along x."""

    def dc(e):
        return CONTRAST.Dlambda * np.trace(e) * np.eye(3) + 2 * CONTRAST.Dmu * e

    a, b = dc(e_out), dc(np.outer(K, K))
    m_bar = strain_green_m_bar(a, b)
    m_ax = np.mean([strain_green_m(e, a, b) for e in np.eye(3)])
    return 2.0 - m_ax / m_bar


# ------------------------------------------------------------------ the relative projection error
def projection_error(centres: np.ndarray, h: float, p: int) -> float:
    """sum over cells of int (f - Pi_p f)^2 over int f^2, Pi_p onto Legendre products of total degree p."""
    x1, w1 = leggauss(10)
    pts = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    basis = [np.ones(len(pts))] + ([pts[:, i] for i in range(3)] if p >= 1 else [])
    norms = [np.sum(wts * b * b) for b in basis]
    num = den = 0.0
    for c in centres:
        f = np.array([profile(c + h * q) for q in pts])
        proj = sum(np.sum(wts * f * b) / nb * b for b, nb in zip(basis, norms, strict=True))
        num += np.sum(wts * (f - proj) ** 2)
        den += np.sum(wts * f * f)
    return num / den


def split(u: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The P (radial) and S (transverse) parts of far-field displacements u at unit directions dirs."""
    up = np.einsum("ni,ni->n", u, dirs)[:, None] * dirs
    return up, u - up


def main() -> int:
    ka = float(sys.argv[1])
    p_deg = int(sys.argv[2])
    ns = [int(a) for a in sys.argv[3:]]
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    d = 1e-3

    def exact(scale: float) -> np.ndarray:
        c = MaterialContrast(scale * CONTRAST.Dlambda, scale * CONTRAST.Dmu, 0.0)
        return mie_scattered_displacement(graded_mie_result(omega, RADIUS, RADIUS / 2, REF, c, 12), pts)

    fp, fm = exact(d), exact(-d)
    t2p_ex, t2s_ex = split((fp + fm) / (2 * d * d), dirs)

    tangents = np.stack([-dirs[:, 1], dirs[:, 0], np.zeros(len(dirs))], 1)
    # the closed-form limit is derived for constant cells only; for p = 1 the column is left empty
    nan = [float("nan")] * len(dirs)
    pred_p = [predicted_factor(np.outer(n, n)) for n in dirs] if p_deg == 0 else nan
    pred_s = (
        [
            predicted_factor(0.5 * (np.outer(n, t) + np.outer(t, n)))
            for n, t in zip(dirs, tangents, strict=True)
        ]
        if p_deg == 0
        else nan
    )

    kp = omega / REF.alpha
    amp = np.concatenate([K.astype(complex), _plane_wave_strain_voigt(K, K, kp)])
    print(
        f"T2 law in 3-D: k_S a = {ka}, p = r = {p_deg}, stiffness contrast, graded sphere core a/2",
        flush=True,
    )
    for n in ns:
        h = RADIUS / n
        grid, centres, _ = _build_grid_index_map(
            RADIUS, n, lambda q, _h=h: np.linalg.norm(q) < RADIUS + np.sqrt(3) * _h
        )
        delta = np.array(
            [cell_contrast_coefficients(profile, c, h, CONTRAST, REF, omega, p_deg) for c in centres]
        )
        e_ops = np.array([source_expansion(dd) for dd in delta])
        minv = np.linalg.inv(gram_test(h))
        psi0 = np.einsum("ab,nbi->nai", minv, plane_wave_moments(centres, h, kp * K, amp))
        if p_deg == 0:
            psi0[:, 1:] = 0.0
        src0 = np.einsum("ncbij,nbj->nci", e_ops, psi0)
        blocks = offset_blocks(n, h, omega, REF)
        rhs = np.zeros((len(centres), 4, 9), complex)
        for m in range(len(centres)):
            for q in range(len(centres)):
                off = tuple(int(v) for v in grid[m] - grid[q])
                rhs[m] += np.einsum("acij,cj->ai", blocks[off], src0[q])
        psi1 = np.einsum("ab,nbi->nai", minv, rhs)
        if p_deg == 0:
            psi1 = np.zeros_like(psi1)
            psi1[:, 0] = rhs[:, 0] / gram_test(h)[0, 0]
        res = GradedVoxelResult(centres, grid, h, omega, REF, delta, psi1, p_deg, p_deg)
        up, us = graded_far_field(res, dirs, rf, K, K, "P")
        e_proj = projection_error(centres, h, p_deg)
        print(f"\n n = {n}: {len(centres)} cells, relative projection error E = {e_proj:.4e}", flush=True)
        print("  angle   P: (T2-T2s)/(T2 E)  predicted     S: (T2-T2s)/(T2 E)  predicted")
        for j, th in enumerate(THETA):
            rp = ((t2p_ex[j] - up[j]) @ np.conj(t2p_ex[j])) / (np.vdot(t2p_ex[j], t2p_ex[j]) * e_proj)
            rs = ((t2s_ex[j] - us[j]) @ np.conj(t2s_ex[j])) / (np.vdot(t2s_ex[j], t2s_ex[j]) * e_proj)
            print(
                f"  {th:5.2f}   {rp.real: .4f}{rp.imag:+.4f}i   {pred_p[j]: .4f}"
                f"      {rs.real: .4f}{rs.imag:+.4f}i   {pred_s[j]: .4f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
