#!/usr/bin/env python3
"""The propagator moment of the first-order system, assembled laterally.

SUPERSEDED BY scripts/gate_first_order_schwinger.py.  DO NOT READ THE NUMBER
THIS PRINTS.  Its assembly carries two faults recorded in the second revision
banner of the plan: it works in the THESIS basis while the contrast operator
works in the PAPER basis, and it loops once over the contrast terms, so it
computes <J6 psi, Gamma DeltaA psi> where the Schwinger object needs
<J6 psi, DeltaA Gamma DeltaA psi>.  It fails by a factor of order 1e5 and is
kept only because ``tri`` and ``amat`` live here and are imported by the gate
that supersedes it -- and because the claim in the HOW THE PIECES MEET note
below, that no factor of k appears, is the false step worth keeping on record.

Task 1 of docs/plans/2026-09-18-first-order-propagator-moment.md, final step.

This computes the object itself rather than a demonstration of the method:

    [DeltaC G DeltaC]_kl = < J6 psi_k | DeltaA Gamma DeltaA | psi_l >

entirely from the first-order system -- Gamma from the spectral projectors of A,
the vertical integral in closed form, the lateral integral numerically -- and
compares it with the same object built from the Kelvin route in
scripts/gate_first_order_tmatrix.py.  In a whole-space background the two must
agree; that is the point of doing whole-space first.

HOW THE PIECES MEET
-------------------
Each trial q-field is affine, psi_i(x) = q0_i + qg[i,a] x_a with a = 0,1 lateral
and a = 2 depth.  Its lateral transform at fixed depth is therefore

    psihat_i(k, z) = A_i(k) + B_i(k) z ,
    A_i = q0_i F + qg[i,0] F1 + qg[i,1] F2 ,   B_i = qg[i,2] F ,

with F the box form factor and F1, F2 its derivatives -- the transforms of
x1 and x2 over the square.  Pairing two such fields through Gamma and using the
triangular z integrals T[m,n] gives

    sum_{m,n in {0,1}}  C^phi_{i,m}(-k)  K^{mn}_ij(k)  C^psi_{j,n}(k) .

The contrast operator enters through its transferred terms: a term carrying a
derivative on one side replaces that side's affine weight by a CONSTANT, since
d_alpha of an affine function is constant.  That is why no factor of k appears
and the lateral integral converges.

Run:  conda run -n seismic python scripts/gate_first_order_lateral_moment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from scripts.gate_first_order_tmatrix import (  # noqa: E402
    coupling_first_order,
    delta_a_terms,
    propagator_moment,
    qfield_of,
)

J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def amat(ref: ReferenceMedium, omega: complex, k1: float, k2: float) -> np.ndarray:
    """The 6x6 system matrix in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumber, first component.
        k2: Lateral wavenumber, second component.

    Returns:
        Shape (6, 6) complex.
    """
    lam, mu = ref.lam, ref.mu
    kc = lam + 2.0 * mu
    gam, aa, bb = lam / kc, 1.0 / kc, 1.0 / mu
    zet, chi = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    rw2 = ref.rho * omega**2
    j = 1j
    return np.array(
        [
            [0, -j * gam * k1, -j * gam * k2, aa, 0, 0],
            [-j * k1, 0, 0, 0, bb, 0],
            [-j * k2, 0, 0, 0, 0, bb],
            [-rw2, 0, 0, 0, -j * k1, -j * k2],
            [0, -rw2 + zet * k1**2 + mu * k2**2, k1 * k2 * (chi + mu), -j * k1 * gam, 0, 0],
            [0, k1 * k2 * (chi + mu), -rw2 + zet * k2**2 + mu * k1**2, -j * k2 * gam, 0, 0],
        ],
        dtype=np.complex128,
    )


def _tri_series(m: int, n: int, q: np.ndarray, a: float) -> np.ndarray:
    """T[m,n](q) as a power series in q, for use where the closed form cancels.

    Substituting u = z - z' and expanding the exponential,

        T[m,n](q) = sum_j (-q)^j / j!  int_0^{2a} u^j J_{mn}(u) du ,

    with J the inner integral over z' at fixed u.  Every u-integral is
    elementary, and with b = 2a the coefficients are

        (0,0): b^2 / (j+1)(j+2)
        (1,0): b^3 / 2(j+2)(j+3)
        (1,1): b^4 [ 1/12(j+1) - 1/4(j+2) + 1/6(j+4) ] .

    The series is entire, so it is valid for complex q (propagating modes give
    imaginary q); it is used only where it is also the accurate branch.

    Args:
        m: Power of z on the receiver side.
        n: Power of z' on the source side.
        q: Decay rates, any shape.
        a: Cube half-width.

    Returns:
        Same shape as q.
    """
    b = 2.0 * a
    x = -b * q
    acc = np.zeros_like(q, dtype=np.complex128)
    term = np.ones_like(q, dtype=np.complex128)
    for j in range(40):
        if (m, n) == (0, 0):
            c = b**2 / ((j + 1.0) * (j + 2.0))
        elif (m, n) in {(1, 0), (0, 1)}:
            c = b**3 / (2.0 * (j + 2.0) * (j + 3.0))
        else:
            c = b**4 * (1.0 / (12.0 * (j + 1.0)) - 1.0 / (4.0 * (j + 2.0)) + 1.0 / (6.0 * (j + 4.0)))
        acc = acc + c * term
        term = term * x / (j + 1.0)
    return -acc if (m, n) == (0, 1) else acc


def tri(m: int, n: int, q: np.ndarray, a: float) -> np.ndarray:
    """T[m,n](q), the triangular z integral.

    The closed forms below are exact but CANCEL badly as q -> 0: the numerator of
    the (1,1) form is O((aq)^5) while its individual terms are O(1), so at
    aq ~ 0.025 it has already lost about eight digits.  That is not an academic
    corner -- it is the whole propagating window k < omega/beta, and it was
    measured breaking the reciprocity of the z-integrated kernel at 1.3e-6.
    Below |2aq| = 3 the series of ``_tri_series`` is used instead, where no
    cancellation occurs.

    Args:
        m: Power of z on the receiver side.
        n: Power of z' on the source side.
        q: Decay rates, any shape.
        a: Cube half-width.

    Returns:
        Same shape as q.
    """
    q = np.asarray(q)
    small = np.abs(2.0 * a * q) < 3.0
    safe = np.where(small, 1.0, q)
    e = np.exp(-2.0 * a * safe)
    if (m, n) == (0, 0):
        big = (2.0 * a * safe - 1.0 + e) / safe**2
    elif (m, n) == (1, 0):
        big = (a * safe - 1.0 + (1.0 + a * safe) * e) / safe**3
    elif (m, n) == (0, 1):
        big = -(a * safe - 1.0 + (1.0 + a * safe) * e) / safe**3
    else:
        big = (3.0 - 3.0 * (1.0 + a * safe) ** 2 * e + a**2 * safe**2 * (2.0 * a * safe - 3.0)) / (
            3.0 * safe**4
        )
    return np.where(small, _tri_series(m, n, q, a), big)


def amat_batch(ref: ReferenceMedium, omega: complex, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """The system matrix over a grid of lateral wavenumbers.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumbers, shape (N,).
        k2: Lateral wavenumbers, shape (N,).

    Returns:
        Shape (N, 6, 6) complex.
    """
    lam, mu = ref.lam, ref.mu
    kc = lam + 2.0 * mu
    gam, aa, bb = lam / kc, 1.0 / kc, 1.0 / mu
    zet, chi = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    rw2 = ref.rho * omega**2
    n = k1.shape[0]
    a6 = np.zeros((n, 6, 6), dtype=np.complex128)
    j = 1j
    a6[:, 0, 1], a6[:, 0, 2], a6[:, 0, 3] = -j * gam * k1, -j * gam * k2, aa
    a6[:, 1, 0], a6[:, 1, 4] = -j * k1, bb
    a6[:, 2, 0], a6[:, 2, 5] = -j * k2, bb
    a6[:, 3, 0], a6[:, 3, 4], a6[:, 3, 5] = -rw2, -j * k1, -j * k2
    a6[:, 4, 1] = -rw2 + zet * k1**2 + mu * k2**2
    a6[:, 4, 2], a6[:, 4, 3] = k1 * k2 * (chi + mu), -j * k1 * gam
    a6[:, 5, 1] = k1 * k2 * (chi + mu)
    a6[:, 5, 2] = -rw2 + zet * k2**2 + mu * k1**2
    a6[:, 5, 3] = -j * k2 * gam
    return a6


def kernels_batch(ref: ReferenceMedium, omega: complex, k1: np.ndarray, k2: np.ndarray, a: float) -> dict:
    """The four z-integrated kernels over a grid, vectorised.

    A scalar loop calling ``numpy.linalg.eig`` once per node cannot reach the
    resolution this integral needs: the integrand is O(1e8) while the answer is
    O(1e7), so the cancellation demands many nodes.  ``eig`` batches over stacked
    arrays, which is what makes the required grid affordable.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumbers, shape (N,).
        k2: Lateral wavenumbers, shape (N,).
        a: Cube half-width.

    Returns:
        Dict keyed by (m, n) with (N, 6, 6) complex values.
    """
    k = np.maximum(np.hypot(k1, k2), 1e-12)
    n = k1.shape[0]
    sd = np.ones((n, 6), dtype=np.complex128)
    sd[:, 3:] = k[:, None]
    a6 = amat_batch(ref, omega, k1, k2)
    scaled = a6 / sd[:, :, None] * sd[:, None, :]
    ev, rv = np.linalg.eig(scaled)
    lv = np.linalg.inv(rv)
    dn = np.real(ev) < 0.0
    out = {}
    for m in (0, 1):
        for nn in (0, 1):
            w = np.where(dn, tri(m, nn, -ev, a), -tri(nn, m, ev, a))
            acc = np.einsum("nij,nj,njl->nil", rv, w, lv)
            out[(m, nn)] = acc * sd[:, :, None] / sd[:, None, :]
    return out


def kernels(ref: ReferenceMedium, omega: complex, k1: float, k2: float, a: float) -> dict:
    """The four z-integrated kernels K^{mn}(k), from the spectral form of Gamma.

    The eigen-decomposition is done on the SCALED matrix, since on the unscaled
    one the up- and down-going eigenvectors become nearly parallel as k grows.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        k1: Lateral wavenumber, first component.
        k2: Lateral wavenumber, second component.
        a: Cube half-width.

    Returns:
        Dict keyed by (m, n) with (6, 6) complex values.
    """
    k = max(np.hypot(k1, k2), 1e-12)
    s = np.diag([1.0, 1.0, 1.0, k, k, k]).astype(np.complex128)
    sinv = np.diag([1.0, 1.0, 1.0, 1.0 / k, 1.0 / k, 1.0 / k]).astype(np.complex128)
    ev, rv = np.linalg.eig(sinv @ amat(ref, omega, k1, k2) @ s)
    lv = np.linalg.inv(rv)  # rows are left eigenvectors
    dn = np.real(ev) < 0.0
    out = {}
    for m in (0, 1):
        for n in (0, 1):
            acc = np.zeros((6, 6), dtype=np.complex128)
            for i in range(6):
                outer = np.outer(rv[:, i], lv[i, :])
                if dn[i]:
                    acc += tri(m, n, -ev[i], a) * outer
                else:
                    acc -= tri(n, m, ev[i], a) * outer
            out[(m, n)] = s @ acc @ sinv
    return out


def _cluster_average(ev: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Replace near-degenerate eigenvalues by their cluster mean.

    ``k_zS`` is doubly degenerate -- SV and SH share it -- so ``eig`` splits a
    two-dimensional eigenspace arbitrarily.  The individual outer products
    ``v_i w_i^t`` are then ill conditioned even though their sum, the block
    projector, is not.  Slightly unequal eigenvalues weight them differently and
    the conditioning leaks into the answer at ~1e-7.  Giving every member of a
    cluster the same weight collapses the sum back onto the projector.

    Args:
        ev: Eigenvalues, shape (..., 6).
        tol: Relative separation below which two eigenvalues are one cluster.

    Returns:
        Same shape, cluster members replaced by their mean.
    """
    out = np.array(ev, dtype=np.complex128, copy=True)
    flat = out.reshape(-1, out.shape[-1])
    for row in flat:
        scale = max(np.max(np.abs(row)), 1e-300)
        used = np.zeros(row.shape[0], dtype=bool)
        for i in range(row.shape[0]):
            if used[i]:
                continue
            grp = np.abs(row - row[i]) < tol * scale
            if grp.sum() > 1:
                row[grp] = row[grp].mean()
            used |= grp
    return out


def form_factors(k1: float, k2: float, a: float) -> tuple[complex, complex, complex]:
    """Lateral transforms over the square of 1, x1 and x2.

    F = int e^{-i k.x} dx over [-a,a]^2, and F1, F2 the same with a factor x1 or
    x2.  Since x e^{-ikx} = i d/dk e^{-ikx}, these are i dF/dk.

    Args:
        k1: Lateral wavenumber, first component.
        k2: Lateral wavenumber, second component.
        a: Cube half-width.

    Returns:
        (F, F1, F2).
    """

    def g(k: float) -> complex:
        return 2.0 * a if abs(k) < 1e-12 else 2.0 * np.sin(k * a) / k

    def dg(k: float) -> complex:
        if abs(k) < 1e-9:
            return 0.0
        return 2.0 * (a * np.cos(k * a) / k - np.sin(k * a) / k**2)

    f = g(k1) * g(k2)
    return f, 1j * dg(k1) * g(k2), 1j * g(k1) * dg(k2)


def weight_coeffs(q0: np.ndarray, qg: np.ndarray, comp: int, d: int, ff: tuple) -> tuple:
    """The (z^0, z^1) lateral coefficients of one component, possibly differentiated.

    A term of the transferred contrast operator that carries a derivative
    replaces the affine weight by a constant, because d_alpha of an affine
    function is constant -- which is why no factor of k enters.

    Args:
        q0: Constant part of the q-field, shape (6,).
        qg: Gradient of the q-field, shape (6, 3).
        comp: Component index.
        d: 0 for the field, 1 or 2 for d/dx_1 or d/dx_2.
        ff: (F, F1, F2) form factors.

    Returns:
        (coefficient of z^0, coefficient of z^1).
    """
    f, f1, f2 = ff
    if d != 0:
        return qg[comp, d - 1] * f, 0.0 + 0.0j
    return q0[comp] * f + qg[comp, 0] * f1 + qg[comp, 1] * f2, qg[comp, 2] * f


def integrand(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kk: int,
    ll: int,
    k1: float,
    k2: float,
) -> complex:
    """One entry of the lateral integrand at a given wavenumber.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kk: Test trial index, 0..8.
        ll: Trial index, 0..8.
        k1: Lateral wavenumber, first component.
        k2: Lateral wavenumber, second component.

    Returns:
        The integrand value.
    """
    lam_t, mu_t = ref.lam + con.Dlambda, ref.mu + con.Dmu
    p0k, pgk = qfield_of(kk, lam_t, mu_t, omega)
    p0l, pgl = qfield_of(ll, lam_t, mu_t, omega)
    ffm = form_factors(-k1, -k2, a)  # test side at -k
    ffp = form_factors(k1, k2, a)  # trial side at +k
    kern = kernels(ref, omega, k1, k2, a)
    terms = delta_a_terms(ref.lam, ref.mu, ref.rho, con, omega)

    tot = 0.0 + 0.0j
    for row, col, coef, dphi, dpsi in terms:
        if row < 3:
            idx, sgn = row + 3, -1.0
        else:
            idx, sgn = row - 3, 1.0
        ck = weight_coeffs(p0k, pgk, idx, dphi, ffm)
        cl = weight_coeffs(p0l, pgl, col, dpsi, ffp)
        for m in (0, 1):
            for n in (0, 1):
                if ck[m] == 0.0 or cl[n] == 0.0:
                    continue
                tot += sgn * coef * ck[m] * kern[(m, n)][idx, col] * cl[n]
    return tot


def lateral(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kk: int,
    ll: int,
    lam_cut: float,
    n: int = 400,
) -> complex:
    """The lateral integral of one entry over the square |k_1|, |k_2| < lam_cut.

    CARTESIAN, not polar.  The box form factor is separable, F = g(k1) g(k2),
    and near the k1 axis g(k2) stays at 2a over an angular width 1/(k a) --
    0.025 rad at lam_cut = 80.  A polar product rule with any affordable number
    of angular nodes cannot resolve that, and reports an integral that grows
    with the cutoff instead of converging.  On a Cartesian grid each sinc is
    resolved by its own one-dimensional rule.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kk: Test trial index.
        ll: Trial index.
        lam_cut: Cutoff on each axis.
        n: Nodes per axis.

    Returns:
        The truncated integral, including the 1/(2 pi)^2.
    """
    ks = (np.arange(n) + 0.5) * 2.0 * lam_cut / n - lam_cut
    dk = 2.0 * lam_cut / n
    tot = 0.0 + 0.0j
    for k1 in ks:
        for k2 in ks:
            tot += integrand(ref, omega, con, a, kk, ll, k1, k2)
    return tot * dk * dk / (2.0 * np.pi) ** 2


def lateral_fast(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kk: int,
    ll: int,
    lam_cut: float,
    n: int,
) -> complex:
    """The lateral integral, vectorised over the whole grid.

    Cartesian, because the form factor is separable and a polar rule cannot
    resolve its near-axis structure.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kk: Test trial index.
        ll: Trial index.
        lam_cut: Cutoff on each axis.
        n: Nodes per axis.

    Returns:
        The truncated integral, including 1/(2 pi)^2.
    """
    ax = (np.arange(n) + 0.5) * 2.0 * lam_cut / n - lam_cut
    dk = 2.0 * lam_cut / n
    k1 = np.repeat(ax, n)
    k2 = np.tile(ax, n)

    def g(k: np.ndarray) -> np.ndarray:
        out = np.full_like(k, 2.0 * a, dtype=np.complex128)
        nz = np.abs(k) > 1e-12
        out[nz] = 2.0 * np.sin(k[nz] * a) / k[nz]
        return out

    def dg(k: np.ndarray) -> np.ndarray:
        out = np.zeros_like(k, dtype=np.complex128)
        nz = np.abs(k) > 1e-9
        out[nz] = 2.0 * (a * np.cos(k[nz] * a) / k[nz] - np.sin(k[nz] * a) / k[nz] ** 2)
        return out

    g1, g2, d1, d2 = g(k1), g(k2), dg(k1), dg(k2)
    fp = g1 * g2
    f1p, f2p = 1j * d1 * g2, 1j * g1 * d2
    # F is even, F1 and F2 are odd, so the test side at -k flips only F1, F2
    fm, f1m, f2m = fp, -f1p, -f2p

    lam_t, mu_t = ref.lam + con.Dlambda, ref.mu + con.Dmu
    p0k, pgk = qfield_of(kk, lam_t, mu_t, omega)
    p0l, pgl = qfield_of(ll, lam_t, mu_t, omega)
    kern = kernels_batch(ref, omega, k1, k2, a)

    def coeffs(q0: np.ndarray, qg: np.ndarray, comp: int, d: int, side: str) -> tuple:
        f, fa, fb = (fm, f1m, f2m) if side == "test" else (fp, f1p, f2p)
        if d != 0:
            return qg[comp, d - 1] * f, None
        return q0[comp] * f + qg[comp, 0] * fa + qg[comp, 1] * fb, qg[comp, 2] * f

    tot = np.zeros(k1.shape[0], dtype=np.complex128)
    for row, col, coef, dphi, dpsi in delta_a_terms(ref.lam, ref.mu, ref.rho, con, omega):
        idx, sgn = (row + 3, -1.0) if row < 3 else (row - 3, 1.0)
        ck = coeffs(p0k, pgk, idx, dphi, "test")
        cl = coeffs(p0l, pgl, col, dpsi, "trial")
        for m in (0, 1):
            if ck[m] is None:
                continue
            for nn in (0, 1):
                if cl[nn] is None:
                    continue
                tot += sgn * coef * ck[m] * kern[(m, nn)][:, idx, col] * cl[nn]
    return complex(np.sum(tot) * dk * dk / (2.0 * np.pi) ** 2)


def main() -> int:
    """Assemble one entry from the first-order system and compare with Kelvin.

    Returns:
        0 if every check passes, 1 otherwise.
    """
    print("=" * 78)
    print("  The propagator moment, assembled from the first-order system")
    print("=" * 78)

    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    a = 0.5
    omega = 60.0 * (1.0 + 0.02j)
    con = MaterialContrast(0.0, 1.0e9, 0.0)  # pure shear

    # Kelvin route: [DeltaC G DeltaC], the same object
    dc = coupling_first_order(ref.lam, ref.mu, ref.rho, con, omega, a) / (1j * omega)
    g = propagator_moment(ref, 60.0, a)
    target = dc @ g @ dc

    print(f"\n  medium alpha={ref.alpha}, beta={ref.beta}, rho={ref.rho}, a={a}")
    print(f"  omega = {omega}, contrast Dmu = {con.Dmu:.3e}")
    print("\n--- one entry, both routes ---------------------------------------")
    for kk, ll, name in ((6, 6, "e12-e12 (in-plane shear)"),):
        cuts = [30.0, 60.0, 120.0]
        vals = [lateral(ref, omega, con, a, kk, ll, c) for c in cuts]
        rich = 2.0 * vals[-1] - vals[-2]  # 1/Lambda extrapolation
        print(f"  {name}")
        for c, v in zip(cuts, vals, strict=True):
            print(f"     cutoff {c:6.1f}   first-order {v:+.8e}")
        print(f"     extrapolated       first-order {rich:+.8e}")
        print(f"     Kelvin route                   {target[kk, ll]:+.8e}")
        rel = abs(rich - target[kk, ll]) / abs(target[kk, ll])
        print(f"     relative difference            {rel:.3e}")
        report(f"{name}: the two routes agree", rel < 5e-2)

    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok in _PASS if ok)
    print(f"  {n_ok} passed, {len(_PASS) - n_ok} failed")
    print("=" * 78)
    return 0 if n_ok == len(_PASS) else 1


def rot6(th: np.ndarray) -> np.ndarray:
    """Azimuthal rotation on (u_z, u_x, u_y, T_zz, T_xz, T_yz).

    Args:
        th: Angles, shape (M,).

    Returns:
        Shape (M, 6, 6) real.
    """
    c, s = np.cos(th), np.sin(th)
    r = np.zeros((th.shape[0], 6, 6))
    r[:, 0, 0] = r[:, 3, 3] = 1.0
    for b in (1, 4):
        r[:, b, b] = r[:, b + 1, b + 1] = c
        r[:, b, b + 1] = -s
        r[:, b + 1, b] = s
    return r


def lateral_polar(
    ref: ReferenceMedium,
    omega: complex,
    con: MaterialContrast,
    a: float,
    kk: int,
    ll: int,
    lam_cut: float,
    nr: int,
    nth: int,
) -> complex:
    """The lateral integral with the angular reduction.

    For an isotropic medium ``A(k cos th, k sin th) = R(th) A(k,0) R(th)^t``, so
    the kernel at any azimuth is a rotation of the one at ``th = 0``.  The
    eigen-decomposition is therefore needed on a RADIAL grid only -- ``nr`` of
    them rather than ``nr x nth``.  Computing once and rotating also removes the
    angle-to-angle inconsistency separate decompositions suffer, since ``k_zS``
    is doubly degenerate and ``eig`` splits that eigenspace arbitrarily.

    Args:
        ref: Background medium.
        omega: Angular frequency, complex.
        con: Material contrast.
        a: Cube half-width.
        kk: Test trial index.
        ll: Trial index.
        lam_cut: Radial cutoff.
        nr: Radial nodes.
        nth: Angular nodes.

    Returns:
        The truncated integral, including 1/(2 pi)^2.
    """
    kr = (np.arange(nr) + 0.5) * lam_cut / nr
    th = (np.arange(nth) + 0.5) * 2.0 * np.pi / nth
    dkr, dth = lam_cut / nr, 2.0 * np.pi / nth

    k0 = kernels_batch(ref, omega, kr, np.zeros_like(kr), a)
    rr = rot6(th)

    lam_t, mu_t = ref.lam + con.Dlambda, ref.mu + con.Dmu
    p0k, pgk = qfield_of(kk, lam_t, mu_t, omega)
    p0l, pgl = qfield_of(ll, lam_t, mu_t, omega)
    terms = delta_a_terms(ref.lam, ref.mu, ref.rho, con, omega)

    def gg(k: np.ndarray) -> np.ndarray:
        out = np.full(k.shape, 2.0 * a, dtype=np.complex128)
        nz = np.abs(k) > 1e-12
        out[nz] = 2.0 * np.sin(k[nz] * a) / k[nz]
        return out

    def dgg(k: np.ndarray) -> np.ndarray:
        out = np.zeros(k.shape, dtype=np.complex128)
        nz = np.abs(k) > 1e-9
        out[nz] = 2.0 * (a * np.cos(k[nz] * a) / k[nz] - np.sin(k[nz] * a) / k[nz] ** 2)
        return out

    k1g = kr[:, None] * np.cos(th)[None, :]
    k2g = kr[:, None] * np.sin(th)[None, :]
    g1, g2 = gg(k1g), gg(k2g)
    fp = g1 * g2
    f1p, f2p = 1j * dgg(k1g) * g2, 1j * g1 * dgg(k2g)

    tot = np.zeros((nr, nth), dtype=np.complex128)
    for m in (0, 1):
        for nn in (0, 1):
            # Only a few (idx, col) entries are ever needed.  The full rotated
            # kernel would be (nr, nth, 6, 6) -- gigabytes -- so each required
            # entry is contracted on its own into an (nr, nth) array.
            need = {(row + 3 if row < 3 else row - 3, col) for row, col, *_ in terms}
            ent = {
                (i, j): np.einsum("ta,ral,tl->rt", rr[:, i, :], k0[(m, nn)], rr[:, j, :]) for (i, j) in need
            }
            for row, col, coef, dphi, dpsi in terms:
                idx, sgn = (row + 3, -1.0) if row < 3 else (row - 3, 1.0)
                if dphi != 0:
                    ck = pgk[idx, dphi - 1] * fp if m == 0 else None
                elif m == 0:
                    ck = p0k[idx] * fp - pgk[idx, 0] * f1p - pgk[idx, 1] * f2p
                else:
                    ck = pgk[idx, 2] * fp
                if dpsi != 0:
                    cl = pgl[col, dpsi - 1] * fp if nn == 0 else None
                elif nn == 0:
                    cl = p0l[col] * fp + pgl[col, 0] * f1p + pgl[col, 1] * f2p
                else:
                    cl = pgl[col, 2] * fp
                if ck is None or cl is None:
                    continue
                tot += sgn * coef * ck * ent[(idx, col)] * cl
    return complex(np.sum(tot * kr[:, None]) * dkr * dth / (2.0 * np.pi) ** 2)


if __name__ == "__main__":
    sys.exit(main())
