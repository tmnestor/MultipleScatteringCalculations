"""The lattice coupling of elastic multipoles (L, M, N) for the layer-KKR solve.

WHAT IS COUPLED.  Outgoing multipoles at every lattice site but the origin,
Bloch-phased, re-expanded as regular multipoles at the origin:

    sum_{R != 0} e^{i k.R} F^h_nm(r - R) = sum_{nu mu, F'} G0[(nu mu F'), (n m F)] F'^j_numu(r) ,

with L = grad(z_n Y_nm) at k_P, M = curl(r z_n Y_nm) and N = curl M at k_S,
Y orthonormal with the Condon-Shortley phase -- the same potentials, and so
the same T-matrix, as ``sphere_scattering.mie_tmatrix_psv/sh``.

HOW.  The scalar translation is the Gaunt contraction of the structure
constants (checked on one translation against direct evaluation).  The
toroidal field is M^h_nm = -i L psi_nm with L the angular momentum about its
OWN centre, a fixed combination of psi_{n,m-1..m+1} times Cartesian vectors,
so its lattice sum is scalar lattice sums; N = curl M; and the regular M and N
content is read off by projection on a sphere about the origin.

THE CHECKS:
  * quadrature Gaunt coefficients against sympy's exact ones;
  * the analytic N field against the finite-difference curl of M;
  * ONE translation (a single site, out of plane): every L, M, N source
    re-expanded and compared with its direct evaluation near the origin;
  * the LATTICE, at damped kappa: coupling built from the Ewald structure
    constants against the plain direct lattice sum of the vector fields.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_layer_kkr_coupling.py -v
"""

import numpy as np
import pytest
from sympy.physics.wigner import gaunt as exact_gaunt

from cubic_scattering import layer_kkr as lk
from cubic_scattering.planar_kambe import structure_constants


def test_gaunt_by_quadrature_matches_sympy() -> None:
    tab = lk.GauntTable(6, 8)
    worst = 0.0
    for l1, m1, l2, m2, l3 in (
        (2, 1, 3, -2, 3),
        (4, -3, 6, 5, 6),
        (6, 0, 8, 0, 10),
        (1, 1, 1, -1, 2),
        (5, 2, 7, -4, 4),
    ):
        m3 = -(m1 + m2)
        got = tab.value(l1, m1, l2, m2, l3)
        want = float(exact_gaunt(l1, l2, l3, m1, m2, m3))
        worst = max(worst, abs(got - want))
    assert worst < 1e-14, f"worst Gaunt error {worst:.2e}"


def test_analytic_n_field_is_the_curl_of_m() -> None:
    k = 1.3
    pts = np.random.default_rng(2).normal(size=(25, 3))
    h = 1e-5
    worst = 0.0
    for kind in ("j", "h"):
        for n, m in ((1, 0), (2, -1), (4, 3)):
            jac = np.stack(
                [
                    (lk.m_field(n, m, k, pts + h * e, kind) - lk.m_field(n, m, k, pts - h * e, kind))
                    / (2 * h)
                    for e in np.eye(3)
                ],
                -1,
            )
            curl = np.stack(
                [jac[:, 2, 1] - jac[:, 1, 2], jac[:, 0, 2] - jac[:, 2, 0], jac[:, 1, 0] - jac[:, 0, 1]], -1
            )
            want = lk.n_field(n, m, k, pts, kind)
            worst = max(worst, np.max(np.abs(curl - want)) / np.max(np.abs(want)))
    assert worst < 1e-7, f"worst |curl M - N| = {worst:.2e}"


def _reconstruct(g0: dict, idx: list, src: tuple, k_p: float, k_s: float, pts: np.ndarray) -> np.ndarray:
    """Regular-multipole field of the coupling column for one source."""
    out = np.zeros((len(pts), 3), dtype=complex)
    for fam_t, (nu, mu) in idx:
        c = g0.get(((fam_t, nu, mu), src), 0.0)
        if c == 0.0:
            continue
        if fam_t == "L":
            out += c * lk.l_field(nu, mu, k_p, pts, "j")
        elif fam_t == "M":
            out += c * lk.m_field(nu, mu, k_s, pts, "j")
        else:
            out += c * lk.n_field(nu, mu, k_s, pts, "j")
    return out


def _direct(fam: str, n: int, m: int, k_p: float, k_s: float, pts: np.ndarray) -> np.ndarray:
    if fam == "L":
        return lk.l_field(n, m, k_p, pts, "h")
    if fam == "M":
        return lk.m_field(n, m, k_s, pts, "h")
    return lk.n_field(n, m, k_s, pts, "h")


def test_one_translation_reconstructs_every_family() -> None:
    """A single site at R (out of plane): its outgoing L, M, N fields near the origin."""
    k_p, k_s = 0.9, 1.5
    rvec = np.array([2.3, -1.1, 0.8])
    # The addition theorem converges geometrically in |r|/|R| (~0.28 at these
    # points), so 16 extra orders leave 1e-7 of truncation and 22 leave 1e-10;
    # a projection radius of 0.8 keeps j_nu(k_S rho0) clear of underflow at
    # nu ~ 25, where rho0 = 0.4 costs three digits.  Measured, not assumed.
    nmax, extra = 3, 22
    q_needed = nmax + extra + nmax
    dp = lk.single_site_constants(k_p, rvec, q_needed)
    ds = lk.single_site_constants(k_s, rvec, q_needed)
    g0, idx = lk.vector_coupling(dp, ds, k_p, k_s, nmax, rho0=0.8, n_eval_extra=extra)
    pts = np.random.default_rng(4).normal(size=(30, 3)) * 0.25
    worst = 0.0
    for fam in ("L", "M", "N"):
        for n in range(1, nmax + 1):
            for m in range(-n, n + 1):
                want = _direct(fam, n, m, k_p, k_s, pts - rvec)
                got = _reconstruct(g0, lk.full_index(extra + nmax), (fam, n, m), k_p, k_s, pts)
                worst = max(worst, np.max(np.abs(got - want)) / np.max(np.abs(want)))
    assert worst < 1e-8, f"worst single-translation reconstruction error {worst:.2e}"


def test_damped_lattice_matches_the_direct_vector_sum() -> None:
    """Ewald-based coupling against the plain damped lattice sum of the vector fields."""
    a_l, kpar = 2.0, np.array([0.2, 0.1])
    k_p, k_s = 0.9 + 0.3j, 1.5 + 0.3j
    nmax, extra = 3, 22
    qmax = nmax + extra + nmax
    dp = structure_constants(k_p, qmax, a_l, kpar)
    ds = structure_constants(k_s, qmax, a_l, kpar)
    g0, _ = lk.vector_coupling(dp, ds, k_p, k_s, nmax, rho0=0.8, n_eval_extra=extra)
    pts = np.random.default_rng(6).normal(size=(12, 3)) * 0.2
    sites = [(i, j) for i in range(-40, 41) for j in range(-40, 41) if (i, j) != (0, 0)]
    worst = 0.0
    for fam in ("L", "M", "N"):
        for n, m in ((1, 0), (2, 1), (3, -2)):
            want = np.zeros((len(pts), 3), dtype=complex)
            for i, j in sites:
                rv = np.array([a_l * i, a_l * j, 0.0])
                want += np.exp(1j * (kpar[0] * rv[0] + kpar[1] * rv[1])) * _direct(
                    fam, n, m, k_p, k_s, pts - rv
                )
            got = _reconstruct(g0, lk.full_index(extra + nmax), (fam, n, m), k_p, k_s, pts)
            worst = max(worst, np.max(np.abs(got - want)) / np.max(np.abs(want)))
    assert worst < 1e-7, f"worst damped-lattice error {worst:.2e}"


@pytest.mark.parametrize("fam", ["L", "M", "N"])
def test_p_and_s_do_not_mix(fam: str) -> None:
    """In a homogeneous background L couples only to L, and M, N only to M, N."""
    rvec = np.array([2.0, 0.5, -0.7])
    dp = lk.single_site_constants(0.9, rvec, 12)
    ds = lk.single_site_constants(1.5, rvec, 12)
    g0, _ = lk.vector_coupling(dp, ds, 0.9, 1.5, 2, rho0=0.4, n_eval_extra=6)
    for (tgt, src), val in g0.items():
        if src[0] == fam and abs(val) > 0:
            assert (tgt[0] == "L") == (fam == "L"), f"{src} couples into {tgt}"
