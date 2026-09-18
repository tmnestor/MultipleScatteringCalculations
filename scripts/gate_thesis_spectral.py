#!/usr/bin/env python3
"""The thesis spectral representation of A, implemented and gated.

ANCHOR: Nestor (1996), Chapter 2, GRepresentations.tex --
    (Akdef)   the Fourier-domain system matrix A(k_x, k_y)
    (ATdef)   A^T(-k_x,-k_y) J_6 + J_6 A(k_x,k_y) = 0
    (eigDef)  Lambda = i diag[k_zP, k_zS, k_zH, -k_zP, -k_zS, -k_zH]
    (kzcDef)  the branch of k_zc, defined piecewise
    (specA)   A = D_z Lambda D_z^-1
    (Peigen), (SVeigen), (SHeigen)  the eigenvectors, analytically
    (p.371)   D_z^-1(k) = -i J_6 D_z^T(-k) J_6

WHY THIS REPLACES THE NUMERICAL ROUTE
-------------------------------------
Everything above is closed form.  Building Gamma from it uses NO numerical
linear algebra at all -- no eigendecomposition, no inversion, no balancing, no
branch classification -- and that removes four hazards that were found the hard
way while working from the operator form instead:

  * the up/down split.  (kzcDef) defines the branch piecewise, real root when
    propagating and i sqrt(k^2 - K^2) when evanescent, and (eigDef) orders the
    modes [+P,+S,+H,-P,-S,-H].  Classifying numerically by sign(Re(ev)) is
    meaningless in the propagating window and gave a reciprocity residual of
    1.245 there.
  * the degenerate shear eigenspace.  The thesis notes k_zS = k_zH and keeps
    them labelled apart, and its quasi-SV / quasi-SH basis is chosen to be
    NON-DEGENERATE at k_x = k_y = 0 -- explicitly, and in contrast to the
    conventional seismological decomposition.  numpy's eig has no such choice
    and returns an arbitrary mixture, which produced an intermittent factor of
    four in a mode-converted coefficient.
  * the conditioning of the eigenvectors, which motivated a scalar scaling that
    turned out to be in the wrong direction, and then a Parlett--Reinsch
    balancing.  Neither is needed if the eigenvectors are not computed.
  * the inversion of the eigenvector matrix, replaced by the symplectic
    identity above.

Run:  conda run -n seismic python scripts/gate_thesis_spectral.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402

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


def amat_thesis(ref: ReferenceMedium, omega: complex, kx: float, ky: float) -> np.ndarray:
    """A(k_x, k_y) in the thesis basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).

    Transcribed from (Akdef).  The shorthand gamma, a, b, zeta, chi is LOCAL to
    that equation -- the thesis says so in its own footnote -- and is not the
    gamma, zeta, chi of the List of Symbols, which are the Chapter 5
    material-contrast ratios.

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

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
            [0, -j * gam * kx, -j * gam * ky, aa, 0, 0],
            [-j * kx, 0, 0, 0, bb, 0],
            [-j * ky, 0, 0, 0, 0, bb],
            [-rw2, 0, 0, 0, -j * kx, -j * ky],
            [0, -rw2 + zet * kx**2 + mu * ky**2, kx * ky * (chi + mu), -j * kx * gam, 0, 0],
            [0, kx * ky * (chi + mu), -rw2 + zet * ky**2 + mu * kx**2, -j * ky * gam, 0, 0],
        ],
        dtype=np.complex128,
    )


def khat_c(c: float, omega: complex, ky: float) -> complex:
    """K_hat_c(k_y), the in-plane wavenumber after the y transform.  (Kdef).

    Args:
        c: Wave speed.
        omega: Angular frequency.
        ky: Lateral wavenumber, y.

    Returns:
        Real when propagating, i sqrt(...) when evanescent.
    """
    w2 = (omega / c) ** 2
    if abs(ky) < abs(np.sqrt(w2)):
        return complex(np.sqrt(w2 - ky**2))
    return complex(1j * np.sqrt(ky**2 - w2))


def kz_c(c: float, omega: complex, kx: float, ky: float) -> complex:
    """k_{z,c}(k_x, k_y), branch defined piecewise.  (kzcDef).

    This is the whole of the up/down question, settled analytically: the
    positive root is taken in both regimes, and (eigDef) then assigns the first
    three columns to downgoing and the last three to upgoing.  No sign of a
    numerically computed real part is consulted anywhere.

    Args:
        c: Wave speed.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        Real when propagating, i sqrt(...) when evanescent.
    """
    kh = khat_c(c, omega, ky)
    if abs(kx) < abs(kh):
        return complex(np.sqrt(kh**2 - kx**2))
    return complex(1j * np.sqrt(kx**2 - kh**2))


def dz_columns(ref: ReferenceMedium, omega: complex, kx: float, ky: float) -> np.ndarray:
    """D_z, the eigen matrix, from (Peigen), (SVeigen), (SHeigen).

    Columns are ordered [+P, +S, +H, -P, -S, -H] per (eigDef), with + downgoing.
    The normalisation factors epsilon are left at unity here and fixed by
    ``dz_normalised``.

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        Shape (6, 6) complex.
    """
    rho, b2 = ref.rho, ref.beta**2
    zp = kz_c(ref.alpha, omega, kx, ky)
    zs = kz_c(ref.beta, omega, kx, ky)
    zh = kz_c(ref.beta, omega, kx, ky)
    kh_h = khat_c(ref.beta, omega, ky)
    out = np.zeros((6, 6), dtype=np.complex128)
    for col, sgn in ((0, +1.0), (3, -1.0)):
        out[:, col] = [
            sgn * 1j * zp,
            1j * kx,
            1j * ky,
            rho * (2.0 * b2 * kx**2 + 2.0 * b2 * ky**2 - omega**2),
            -sgn * 2.0 * rho * b2 * kx * zp,
            -sgn * 2.0 * rho * b2 * ky * zp,
        ]
    for col, sgn in ((1, +1.0), (4, -1.0)):
        out[:, col] = [
            1j * kx,
            -sgn * 1j * zs,
            0.0,
            -sgn * 2.0 * rho * b2 * kx * zs,
            rho * (omega**2 - 2.0 * b2 * kx**2 - b2 * ky**2),
            -rho * b2 * kx * ky,
        ]
    for col, sgn in ((2, +1.0), (5, -1.0)):
        out[:, col] = [
            -sgn * ky * zh,
            -kx * ky,
            kh_h**2,
            2j * ky * rho * (b2 * kx**2 + b2 * ky**2 - omega**2),
            -sgn * 2j * rho * b2 * kx * ky * zh,
            sgn * 1j * zh * rho * (omega**2 - 2.0 * b2 * ky**2),
        ]
    return out


def lam_diag(ref: ReferenceMedium, omega: complex, kx: float, ky: float) -> np.ndarray:
    """Lambda, the diagonal eigenvalue array.  (eigDef).

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        Shape (6,) complex.
    """
    zp = kz_c(ref.alpha, omega, kx, ky)
    zs = kz_c(ref.beta, omega, kx, ky)
    return 1j * np.array([zp, zs, zs, -zp, -zs, -zs], dtype=np.complex128)


def dz_inverse_symplectic(dz_m: np.ndarray) -> np.ndarray:
    """D_z^-1(k) = -i J_6 D_z^T(-k) J_6, with no inversion.

    Args:
        dz_m: D_z evaluated at MINUS the wavenumber.

    Returns:
        Shape (6, 6) complex.
    """
    return -1j * J6 @ dz_m.T @ J6


def dz_normalised(ref: ReferenceMedium, omega: complex, kx: float, ky: float) -> tuple:
    """D_z with the epsilon factors fixed so the symplectic inverse is exact.

    The thesis says the epsilons are "chosen so that D_z^-1 takes a simple
    form", and gives that form as -i J6 D_z^T(-k) J6.  That fixes them: with the
    epsilons at unity the product of the stated inverse with D_z is DIAGONAL,
    and each epsilon^2 is the reciprocal of the corresponding diagonal entry.
    So they are read off rather than guessed, and the check that the product is
    diagonal in the first place is itself a test of the thesis eigenvectors.

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        (D_z, D_z^-1, off_diagonal_residual).
    """
    raw = dz_columns(ref, omega, kx, ky)
    raw_m = dz_columns(ref, omega, -kx, -ky)
    prod = dz_inverse_symplectic(raw_m) @ raw
    diag = np.diag(prod)
    off = float(np.max(np.abs(prod - np.diag(diag))) / max(float(np.max(np.abs(diag))), 1e-300))
    # The SAME epsilon scales D_z at +k and at -k: the identity then reads
    #   [-i J6 (Dz(-k) E)^T J6] (Dz(k) E) = E diag(d) E = E^2 diag(d),
    # so epsilon_i^2 d_i = 1 fixes every factor at once.
    eps = 1.0 / np.sqrt(diag.astype(np.complex128))
    dz_m = raw * eps[None, :]
    inv = dz_inverse_symplectic(raw_m * eps[None, :])
    return dz_m, inv, off


def gamma_thesis(ref: ReferenceMedium, omega: complex, kx: float, ky: float, dz: float) -> np.ndarray:
    """Gamma(k; dz), the propagator, from the thesis spectral representation.

    ``[d_z - A] G = I_6 delta(z - z')`` (FLdef), so Gamma jumps by the identity
    across dz = 0 and is built from the downgoing modes below the source and the
    upgoing ones above:

        Gamma(dz > 0) = + sum_{+} exp(lambda_i dz) P_i
        Gamma(dz < 0) = - sum_{-} exp(lambda_i dz) P_i

    The split is by COLUMN INDEX -- (eigDef) puts the downgoing modes first --
    so there is no classification to get wrong, and no numerical quantity whose
    sign has to be consulted.  Compare the operator-form route, where the sign
    of Re(lambda) is meaningless inside the propagating window.

    Args:
        ref: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.
        dz: Depth offset, receiver minus source.

    Returns:
        Shape (6, 6) complex.
    """
    dz_m, inv, _ = dz_normalised(ref, omega, kx, ky)
    lam = lam_diag(ref, omega, kx, ky)
    sel = range(3) if dz > 0.0 else range(3, 6)
    sgn = 1.0 if dz > 0.0 else -1.0
    out = np.zeros((6, 6), dtype=np.complex128)
    for i in sel:
        out += sgn * np.exp(lam[i] * dz) * np.outer(dz_m[:, i], inv[i, :])
    return out


def _radial_panels(edges: list[float], nper: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes and weights over a set of radial panels.

    Args:
        edges: Increasing panel boundaries.
        nper: Nodes per panel.

    Returns:
        (nodes, weights).
    """
    t, w = np.polynomial.legendre.leggauss(nper)
    nodes, wts = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        nodes.append(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
        wts.append(0.5 * (hi - lo) * w)
    return np.concatenate(nodes), np.concatenate(wts)


def kelvin_static(ref: ReferenceMedium, r: np.ndarray) -> np.ndarray:
    """The static Kelvin tensor in the thesis (z, x, y) ordering.

    Args:
        ref: Medium.
        r: Separation vector (z, x, y).

    Returns:
        Shape (3, 3).
    """
    nu = ref.lam / (2.0 * (ref.lam + ref.mu))
    rr = float(np.linalg.norm(r))
    n = r / rr
    return ((3.0 - 4.0 * nu) * np.eye(3) + np.outer(n, n)) / (16.0 * np.pi * ref.mu * (1.0 - nu) * rr)


def main() -> int:
    """Gate the thesis spectral representation.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  The thesis spectral representation of A -- Nestor (1996) Ch.2")
    print("=" * 74)
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    omega = 60.0
    rng = np.random.default_rng(1996)

    print("")
    print("--- 1: D_z^-1 from the symplectic identity, with no inversion -----")
    worst_off, worst_id = 0.0, 0.0
    for _ in range(8):
        kx, ky = rng.uniform(-0.05, 0.05, size=2)
        dz_m, inv, off = dz_normalised(ref, omega, kx, ky)
        worst_off = max(worst_off, off)
        worst_id = max(worst_id, float(np.max(np.abs(inv @ dz_m - np.eye(6)))))
    print(f"    worst OFF-DIAGONAL of  -i J6 Dz^T(-k) J6 Dz  = {worst_off:.3e}")
    print(f"    worst | Dz^-1 Dz - I | after normalisation   = {worst_id:.3e}")
    report("the stated inverse is diagonal on the thesis eigenvectors", worst_off < 1e-10)
    report("D_z^-1 D_z = I with the epsilons read off, no inversion used", worst_id < 1e-8)

    print("")
    print("--- 2: A = D_z Lambda D_z^-1, against (Akdef) ---------------------")
    worst_a = 0.0
    for _ in range(8):
        kx, ky = rng.uniform(-0.05, 0.05, size=2)
        dz_m, inv, _ = dz_normalised(ref, omega, kx, ky)
        rebuilt = dz_m @ np.diag(lam_diag(ref, omega, kx, ky)) @ inv
        target = amat_thesis(ref, omega, kx, ky)
        worst_a = max(worst_a, float(np.max(np.abs(rebuilt - target)) / np.max(np.abs(target))))
    print(f"    worst relative entry difference = {worst_a:.3e}")
    report("the analytic eigen-system reproduces A (specA)", worst_a < 1e-9)

    print("")
    print("--- 3: the quasi-Hamiltonian relation (ATdef) ---------------------")
    worst_h = 0.0
    for _ in range(8):
        kx, ky = rng.uniform(-0.05, 0.05, size=2)
        ap = amat_thesis(ref, omega, kx, ky)
        am = amat_thesis(ref, omega, -kx, -ky)
        res = float(np.max(np.abs(am.T @ J6 + J6 @ ap)))
        worst_h = max(worst_h, res / float(np.finfo(float).eps * np.max(np.abs(ap))))
    print(f"    worst residual, in units of the arithmetic floor = {worst_h:.2f}")
    report("A^T(-k) J6 + J6 A(k) = 0, as the thesis states it", worst_h < 8.0)

    print("")
    print("--- 4: the shear basis is NON-DEGENERATE at k = 0 -----------------")
    print("    The thesis chooses quasi-SV and quasi-SH precisely so that they")
    print("    stay independent where the conventional decomposition does not.")
    dz0, _, _ = dz_normalised(ref, omega, 0.0, 0.0)
    s_cols = dz0[:, [1, 2]]
    ang = abs(np.vdot(s_cols[:, 0], s_cols[:, 1])) / (
        np.linalg.norm(s_cols[:, 0]) * np.linalg.norm(s_cols[:, 1])
    )
    print(f"    |<+S | +H>| / (|+S||+H|) at k_x = k_y = 0 = {ang:.3e}")
    report("quasi-SV and quasi-SH remain independent at k = 0", ang < 1e-10)

    print("")
    print("--- 5: reciprocity of Gamma, with no classification involved ------")
    worst_r = 0.0
    for _ in range(5):
        kx, ky = rng.uniform(-3.0, 3.0, size=2)
        gp = gamma_thesis(ref, omega, kx, ky, 0.37)
        gm = gamma_thesis(ref, omega, -kx, -ky, -0.37)
        worst_r = max(worst_r, float(np.max(np.abs(gm.T - J6 @ gp @ J6)) / np.max(np.abs(gp))))
    print(f"    worst | Gamma(-k; z', z)^T - J6 Gamma(k; z, z') J6 | = {worst_r:.3e}")
    report("Gamma is reciprocal, from the analytic modes alone", worst_r < 1e-10)

    print("")
    print("--- 6: Gamma inverted laterally IS the Green's tensor -------------")
    print("    In the thesis basis a force f_j enters row 3+j with a MINUS sign")
    print("    -- d_z T_zz = ... - f_z -- and u comes out in rows 0..2, so")
    print("    G_ij = -Gamma[i, 3+j] directly, with no velocity factor at all.")
    dzs = 0.5
    ka, kb = omega / ref.alpha, omega / ref.beta
    gref = kelvin_static(ref, np.array([dzs, 0.0, 0.0]))
    rel = 1.0
    gnum = np.zeros((3, 3), dtype=complex)
    for nr, nth in ((300, 64), (600, 96)):
        kr, wr = _radial_panels([0.0, ka, kb, 4.0 / dzs, 20.0 / dzs, 80.0 / dzs], nr // 5)
        th = 2.0 * np.pi * np.arange(nth) / nth
        acc = np.zeros((6, 6), dtype=complex)
        for ir, kk in enumerate(kr):
            for t in th:
                acc += (wr[ir] * kk * 2.0 * np.pi / nth) * gamma_thesis(
                    ref, omega, float(kk * np.cos(t)), float(kk * np.sin(t)), dzs
                )
        gnum = -acc[:3, 3:] / (2.0 * np.pi) ** 2
        rel = float(np.max(np.abs(gnum.real - gref)) / np.max(np.abs(gref)))
        print(f"    nr={nr:4d} nth={nth:3d}   worst relative difference = {rel:.3e}")
    print(f"    G_zz first-order {gnum[0, 0].real: .10e}   analytic {gref[0, 0]: .10e}")
    print(f"    G_xx first-order {gnum[1, 1].real: .10e}   analytic {gref[1, 1]: .10e}")
    report("the thesis Gamma reproduces the Kelvin tensor", rel < 2e-3)
    rad = float(omega * (1.0 / ref.alpha**3 + 2.0 / ref.beta**3) / (12.0 * np.pi * ref.rho))
    relr = float(abs(gnum[0, 0].imag - rad) / rad)
    print(f"    Im G_zz {gnum[0, 0].imag: .6e}   w(1/a^3+2/b^3)/12 pi rho {rad: .6e}  rel {relr:.3e}")
    report("and carries the radiation reaction, with no epsilon prescription", relr < 5e-3)

    print("")
    print("=" * 74)
    ok = sum(1 for _, passed in _PASS if passed)
    print(f"  {ok}/{len(_PASS)} checks passed")
    for label, passed in _PASS:
        if not passed:
            print(f"    FAILED: {label}")
    print("=" * 74)
    return 0 if ok == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
