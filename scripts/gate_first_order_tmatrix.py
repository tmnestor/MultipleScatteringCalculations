#!/usr/bin/env python3
"""The single-site T-matrix of the first-order matrix-vector wave equation.

This is the object the whole line of work was for:

    T = DeltaC (I - G DeltaC)^-1

on the nine-parameter affine trial space (three rigid translations, six uniform
strains), where DeltaC is the projected contrast coupling and G the self
propagator moment of the cube.

WHAT IS NEW HERE, AND WHAT IS BORROWED
--------------------------------------
New: ``DeltaC_eff``, the coupling obtained by projecting the first-order
contrast operator ``DeltaA`` with the symplectic pairing, test = J6 x trial.
Derived in ``Mathematica/FirstOrderContrastOperator.wl``; rebuilt here
INDEPENDENTLY from ``DeltaA`` and gated against that script's exported values,
because two implementations agreeing is the evidence standard and one
implementation agreeing with itself is not.

Borrowed: the propagator moment.  The first-order route changes the coupling,
not the Green's function.

THE SELF MOMENTS ARE INTEGRATED, NOT QUOTED
-------------------------------------------
The second moment's integrand goes as 1/r^3, which is not absolutely integrable
in three dimensions.  The usual remedy is to quote a tabulated Eshelby tensor
and bolt on a delta-function term by hand; forget the delta and the
depolarisation comes out with the WRONG SIGN.  This package's own standard
rejects that route, and an earlier version of this file took it anyway.

The static moments are now read from ``Mathematica/CubeSelfMomentExport.wl``,
which computes them on the moment engine as distributions paired with the cube
indicator -- the derivatives peeled onto the faces, so the r=0 content arrives
inside a surface integral, with no delta to add and no table to consult.  The
export is itself gated on a trace identity derived independently of the engine.

The package's tabulated route is still evaluated here, but only as a
CROSS-CHECK: section 2 shows the two agree, which verifies the table rather
than using it.  The dynamic (smooth radiation) part carries no table -- it is
already exact polynomial integration -- and is taken from
``_compute_ABC_polynomial`` with its tabulated static piece removed.

THE CALIBRATION, WHICH FIXES EVERY NORMALISATION
------------------------------------------------
Run the same assembly with the ORDINARY Navier coupling in place of
``DeltaC_eff``.  It must reproduce ``compute_cube_tmatrix``'s four
amplification factors exactly.  That pins the volume factors, the engineering
doubling on the shear slots and the sign of G against existing validated code,
with no freedom left to tune.  Only then is the coupling swapped.

WHAT THIS GATE DOES NOT DO
--------------------------
It does not claim the first-order T is better.  Two of the three effects that
separate it from T9 -- the Galerkin-vs-collocation projection and the O_h
symmetry breaking of the coupling -- are not resolved by any single number, so
the comparison below is reported CHANNEL BY CHANNEL and no winner is declared.
The propagator used is the single (collocation) moment; the J6 pairing makes
the scheme Galerkin, so the consistent pairing would use the double moment.
That mismatch is stated in the output, not hidden.

Run:  conda run -n seismic python scripts/gate_first_order_tmatrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    G0_CUBE,
    MaterialContrast,
    ReferenceMedium,
    _static_eshelby_ABC,
    compute_cube_tmatrix,
)

REF_JSON = ROOT / "Mathematica" / "first_order_coupling_reference.json"
CUBE_MOMENTS_JSON = ROOT / "Mathematica" / "cube_self_moments.json"

# parameter order, shared with the Mathematica derivation
PARAMS = ("c1", "c2", "c3", "e11", "e22", "e33", "e12", "e13", "e23")
# representative index pair of each strain parameter
STRAIN_IJ = {3: (0, 0), 4: (1, 1), 5: (2, 2), 6: (0, 1), 7: (0, 2), 8: (1, 2)}

J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description of the check.
        ok: Whether it passed.
    """
    _PASS.append((label, ok))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


# ---------------------------------------------------------------------------
# The nine affine trial functions
# ---------------------------------------------------------------------------


def strain_of(p: int) -> np.ndarray:
    """Unit strain tensor of trial parameter ``p`` (zero for the rigid ones).

    Args:
        p: Parameter index, 0..8.

    Returns:
        Shape (3, 3) symmetric.
    """
    e = np.zeros((3, 3))
    if p >= 3:
        i, j = STRAIN_IJ[p]
        e[i, j] = 1.0
        e[j, i] = 1.0
    return e


def qfield_of(p: int, lam_t: float, mu_t: float, omega: complex) -> tuple[np.ndarray, np.ndarray]:
    """The q-field carried by trial parameter ``p``, as an affine function.

    q = (-tau_13, -tau_23, -tau_33, v1, v2, v3) with v = -i omega u and the
    stresses built from the TOTAL moduli, since this is the field inside the
    scatterer.  Every component is affine, so it is returned as a constant part
    and a gradient.

    Args:
        p: Parameter index, 0..8.
        lam_t: Total lambda inside the scatterer.
        mu_t: Total mu inside the scatterer.
        omega: Angular frequency, may be complex.

    Returns:
        (q0, qg) with q0 shape (6,) and qg shape (6, 3), q_i(x) = q0_i + qg_ia x_a.
    """
    q0 = np.zeros(6, dtype=complex)
    qg = np.zeros((6, 3), dtype=complex)
    eps = strain_of(p)
    if p < 3:
        q0[3 + p] = -1j * omega
    else:
        tau = lam_t * np.trace(eps) * np.eye(3) + 2.0 * mu_t * eps
        q0[:3] = -tau[:, 2]
        qg[3:, :] = -1j * omega * eps
    return q0, qg


def _affine(q0: np.ndarray, qg: np.ndarray, i: int, d: int) -> tuple[complex, np.ndarray]:
    """Component ``i`` of an affine field, optionally differentiated.

    Args:
        q0: Constant part, shape (6,).
        qg: Gradient, shape (6, 3).
        i: Component index.
        d: 0 for the field itself, 1 or 2 for d/dx_1 or d/dx_2.

    Returns:
        (constant, gradient) of the resulting affine scalar.
    """
    if d == 0:
        return q0[i], qg[i, :].copy()
    return qg[i, d - 1], np.zeros(3, dtype=complex)


def _cube_inner(f: tuple[complex, np.ndarray], g: tuple[complex, np.ndarray], a: float) -> complex:
    """Integral of the product of two affine scalars over the cube [-a, a]^3.

    Uses int 1 = V, int x_i = 0 and int x_i x_j = delta_ij V a^2 / 3.

    Args:
        f: (constant, gradient) of the first factor.
        g: (constant, gradient) of the second factor.
        a: Cube half-width.

    Returns:
        The integral.
    """
    vol = (2.0 * a) ** 3
    return complex(f[0] * g[0] * vol + np.dot(f[1], g[1]) * vol * a**2 / 3.0)


# ---------------------------------------------------------------------------
# The first-order coupling, rebuilt from DeltaA
# ---------------------------------------------------------------------------


def _moduli(lam: float, mu: float) -> dict[str, float]:
    """Auxiliary moduli of the first-order system.

    Args:
        lam: Lame lambda.
        mu: Shear modulus.

    Returns:
        gam, a, b, nu1, nu2.
    """
    kc = lam + 2.0 * mu
    return {
        "gam": lam / kc,
        "a": 1.0 / kc,
        "b": 1.0 / mu,
        "nu1": 4.0 * mu * (lam + mu) / kc,
        "nu2": 2.0 * mu * lam / kc,
    }


def delta_a_terms(lam: float, mu: float, rho: float, con: MaterialContrast, omega: complex) -> list:
    """The terms of DeltaA after the by-parts transfer.

    Each term is ``(row, col, coef, d_phi, d_psi)``: the derivative index on the
    test function and on the trial function, with 0 meaning none.  The transfer
    moves a LEFTMOST derivative onto the test function, which is what keeps the
    indicator of the scatterer from being differentiated.

    Blocks A11 and A22 contribute only through their material-dependent entries;
    the entries that carry a bare derivative are identical in both media and
    cancel in the difference.

    Args:
        lam: Background lambda.
        mu: Background mu.
        rho: Background density.
        con: Material contrast.
        omega: Angular frequency, may be complex.

    Returns:
        List of terms.
    """
    bg = _moduli(lam, mu)
    tot = _moduli(lam + con.Dlambda, mu + con.Dmu)
    d = {k: tot[k] - bg[k] for k in bg}
    iw = 1j * omega

    terms: list = []
    # A11: -d_alpha . Delta gamma, transferred onto the test function
    terms.append((0, 2, d["gam"], 1, 0))
    terms.append((1, 2, d["gam"], 2, 0))
    # A22: -Delta gamma . d_beta, derivative already on the trial function
    terms.append((5, 3, -d["gam"], 0, 1))
    terms.append((5, 4, -d["gam"], 0, 2))
    # A21: multiplicative
    terms.append((3, 0, iw * d["b"], 0, 0))
    terms.append((4, 1, iw * d["b"], 0, 0))
    terms.append((5, 2, iw * d["a"], 0, 0))
    # A12: density, multiplicative
    for i in range(3):
        terms.append((i, 3 + i, iw * con.Drho, 0, 0))
    # A12: -1/(i w) d_alpha U_alpha,beta d_beta
    dmu = con.Dmu
    du = {
        (1, 1): np.diag([d["nu1"], dmu, 0.0]),
        (1, 2): np.array([[0.0, d["nu2"], 0.0], [dmu, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        (2, 1): np.array([[0.0, dmu, 0.0], [d["nu2"], 0.0, 0.0], [0.0, 0.0, 0.0]]),
        (2, 2): np.diag([dmu, d["nu1"], 0.0]),
    }
    for (al, be), mat in du.items():
        for i in range(3):
            for k in range(3):
                if mat[i, k] != 0.0:
                    terms.append((i, 3 + k, mat[i, k] / iw, al, be))
    return terms


def coupling_first_order(
    lam: float, mu: float, rho: float, con: MaterialContrast, omega: complex, a: float
) -> np.ndarray:
    """The J6-paired first-order coupling on the affine trial space.

    Args:
        lam: Background lambda.
        mu: Background mu.
        rho: Background density.
        con: Material contrast.
        omega: Angular frequency, may be complex.
        a: Cube half-width.

    Returns:
        Shape (9, 9) complex, symmetric.
    """
    terms = delta_a_terms(lam, mu, rho, con, omega)
    fields = [qfield_of(p, lam + con.Dlambda, mu + con.Dmu, omega) for p in range(9)]
    out = np.zeros((9, 9), dtype=complex)
    for k in range(9):
        for m in range(9):
            tot = 0.0 + 0.0j
            for row, col, coef, dphi, dpsi in terms:
                # sum_i J6[i, row] phi_i  is  -phi_{row+3} (row < 3) or +phi_{row-3}
                if row < 3:
                    idx, sgn = row + 3, -1.0
                else:
                    idx, sgn = row - 3, 1.0
                f = _affine(*fields[k], idx, dphi)
                g = _affine(*fields[m], col, dpsi)
                tot += sgn * coef * _cube_inner(f, g, a)
            out[k, m] = tot
    return out


def to_navier_units(coupling: np.ndarray, omega: complex) -> np.ndarray:
    """Convert DeltaC_eff into the units DeltaC_Navier and this package use.

    The J6 pairing tests against the field q, whose velocity slots carry
    ``v = -i omega u``.  The Navier weak form and this package's local T-matrix
    both test against the DISPLACEMENT.  So DeltaC_eff carries exactly one extra
    factor of ``i omega``: its rigid block is ``i omega^3 Drho V`` where
    ``_sub_cell_tmatrix_9x9`` uses ``omega^2 Drho V``.

    This is a change of units, not of physics, and it cancels out of every
    amplification factor and channel ratio -- which is precisely why it can sit
    unnoticed in a gate whose checks are all ratios.  It does NOT cancel when
    DeltaC_eff is combined with any object built in displacement units, which is
    what makes it worth naming rather than dividing inline.

    Args:
        coupling: DeltaC_eff, shape (9, 9).
        omega: Angular frequency, may be complex.

    Returns:
        The same coupling in displacement units.
    """
    return coupling / (1j * omega)


def coupling_navier(
    lam: float,
    mu: float,
    rho: float,
    con: MaterialContrast,
    omega: complex,
    a: float,
    *,
    dynamic_overlap: bool = True,
) -> np.ndarray:
    """The ordinary Navier weak contrast on the same trial space.

    b_N[phi, u] = int_V [ omega^2 Drho phi.u - eps(phi) : Dc : eps(u) ] dV.

    The ``phi.u`` term contributes to the STRAIN-strain block as well as the
    rigid one, because an affine trial displacement is not constant:
    ``int_V x_a x_b dV = delta_ab V a^2 / 3``.  That is a finite-size dynamic
    coupling between the density contrast and the strain channels, and T9's
    factorised amplification factors do not carry it -- there ``amp_u`` and the
    strain amplifications are independent.  ``dynamic_overlap=False`` drops it,
    which is what makes the calibration against T9 exact rather than merely
    close; see ``main``.

    Args:
        lam: Background lambda (unused; the weak form sees only the contrast).
        mu: Background mu (unused).
        rho: Background density (unused).
        con: Material contrast.
        omega: Angular frequency, may be complex.
        a: Cube half-width.
        dynamic_overlap: Include the density-strain finite-size term.

    Returns:
        Shape (9, 9) complex, symmetric.
    """
    vol = (2.0 * a) ** 3
    out = np.zeros((9, 9), dtype=complex)
    for k in range(9):
        ek = strain_of(k)
        for m in range(9):
            em = strain_of(m)
            if k < 3 and m < 3:
                overlap = vol if k == m else 0.0
            elif k >= 3 and m >= 3:
                overlap = vol * a**2 / 3.0 * float(np.einsum("ia,ia->", ek, em)) if dynamic_overlap else 0.0
            else:
                overlap = 0.0  # odd in x, vanishes on the cube
            stiff = vol * (
                con.Dlambda * np.trace(ek) * np.trace(em)
                + 2.0 * con.Dmu * float(np.einsum("ij,ij->", ek, em))
            )
            out[k, m] = omega**2 * con.Drho * overlap - stiff
    return out


# ---------------------------------------------------------------------------
# The self propagator moment
# ---------------------------------------------------------------------------


def static_moments_integrated(ref: ReferenceMedium, a: float) -> dict[str, complex]:
    """The cube's static self moments, BY INTEGRATION.

    The tempting shortcut is to take these from a tabulated Eshelby tensor.
    This package's own standard rejects that: the integrand of the second
    moment goes as 1/r^3, is not absolutely integrable in three dimensions, and
    a table plus a hand-added delta is exactly the "Eshelby magic" the moment
    engine exists to avoid.  ``Mathematica/CubeSelfMomentExport.wl`` computes
    both moments as distributions paired with the cube indicator, peeling the
    derivatives onto the faces so the r=0 content arrives inside a surface
    integral, and gates the result on a trace identity derived independently of
    the engine.  Its output is read here.

    A, B and C are scale-free; G goes as the square of the cube width.  The
    engine integrates over [-Del/2, Del/2]^3, so Del = 2a.

    Args:
        ref: Background medium.
        a: Cube half-width.

    Returns:
        Keys A, B, C, G.

    Raises:
        SystemExit: never; callers check ``CUBE_MOMENTS_JSON`` first.
        ValueError: if the exported medium does not match ``ref``.
    """
    mom = json.loads(CUBE_MOMENTS_JSON.read_text())
    if not (np.isclose(mom["lam"], ref.lam, rtol=1e-12) and np.isclose(mom["mu"], ref.mu, rtol=1e-12)):
        raise ValueError(
            f"the exported moments are for lam={mom['lam']:.6g}, mu={mom['mu']:.6g} "
            f"but the reference medium has lam={ref.lam:.6g}, mu={ref.mu:.6g}.\n"
            f"Fix: edit numRule in Mathematica/CubeSelfMomentExport.wl to this "
            f"medium and re-run\n"
            f"  wolframscript -file Mathematica/CubeSelfMomentExport.wl"
        )
    scale = (2.0 * a / mom["Del"]) ** 2
    return {"A": mom["A"], "B": mom["B"], "C": mom["C"], "G": mom["G"] * scale}


def static_moments_tabulated(ref: ReferenceMedium, a: float) -> dict[str, complex]:
    """The same static moments from the package's tabulated Eshelby route.

    Used ONLY as a cross-check against the integrated values, never as their
    source.

    Args:
        ref: Background medium.
        a: Cube half-width.

    Returns:
        Keys A, B, C, G.
    """
    a_t, b_t, c_t = _static_eshelby_ABC(ref.alpha, ref.beta, ref.rho)
    a0 = (ref.alpha**2 + ref.beta**2) / (8.0 * np.pi * ref.rho * ref.alpha**2 * ref.beta**2)
    b0 = (ref.alpha**2 - ref.beta**2) / (8.0 * np.pi * ref.rho * ref.alpha**2 * ref.beta**2)
    return {"A": a_t, "B": b_t, "C": c_t, "G": a**2 * (a0 + b0 / 3.0) * G0_CUBE}


def dynamic_moments_integrated(ref: ReferenceMedium, omega: complex, a: float) -> dict[str, complex]:
    """The cube's self moments at finite frequency, BY INTEGRATION.

    The static part is not the whole moment.  Keeping the exponential of the
    Green's tensor gives the radiation corrections, including the imaginary
    part, which is the radiation damping and has no static counterpart at all.
    These come from ``gDyn`` on the same distributional construction, not from a
    separate polynomial route.

    The export is gated on convergence rather than on a bar: A and B approach
    their static values as O(omega^2) while G approaches its as O(omega), since
    the zeroth moment carries the radiation reaction; and the imaginary part of
    G is checked against ``V omega (1/alpha^3 + 2/beta^3)/(12 pi rho)``, which
    the engine knows nothing about.

    Args:
        ref: Background medium.
        omega: Angular frequency, may be complex.
        a: Cube half-width.

    Returns:
        Keys A, B, C, G, complex.

    Raises:
        ValueError: if the export was made at different parameters.
    """
    mom = json.loads(CUBE_MOMENTS_JSON.read_text())
    want = {"omega": omega, "alpha": ref.alpha, "beta": ref.beta, "Del": 2.0 * a}
    for key, val in want.items():
        if not np.isclose(mom[key], val, rtol=1e-12):
            raise ValueError(
                f"the dynamic moments were exported at {key}={mom[key]:.6g} but this "
                f"gate uses {key}={val:.6g}.\n"
                f"Fix: edit dynRule/numRule in Mathematica/CubeSelfMomentExport.wl "
                f"and re-run\n  wolframscript -file Mathematica/CubeSelfMomentExport.wl"
            )
    return {k: complex(mom[f"{k}_dyn_re"], mom[f"{k}_dyn_im"]) for k in ("A", "B", "C", "G")}


def propagator_moment(ref: ReferenceMedium, omega: complex, a: float) -> np.ndarray:
    """G, the self propagator moment of the cube, in the nine-parameter basis.

    The STATIC part comes from the moment engine, by integration.  The dynamic
    (smooth radiation) part is already computed by exact polynomial integration
    in this package and carries no table, so it is taken from there by removing
    that route's tabulated static piece and substituting the integrated one.

    Built so that ``G @ coupling_navier`` reproduces this package's T9 operator:
    the displacement block carries Gamma0 and the strain block the Eshelby
    moment S, with the volume factor and the engineering doubling of the shear
    slots undone.  The calibration in ``main`` is what verifies that.

    Args:
        ref: Background medium.
        omega: Angular frequency, may be complex.
        a: Cube half-width.

    Returns:
        Shape (9, 9) complex.
    """
    dyn = dynamic_moments_integrated(ref, omega, a)
    ac, bc, cc, g0 = dyn["A"], dyn["B"], dyn["C"], dyn["G"]
    vol = (2.0 * a) ** 3

    def itens(i: int, j: int, k: int, m: int) -> complex:
        iso = ac * (i == j) * (k == m) + bc * ((i == k) * (j == m) + (i == m) * (j == k))
        return iso + cc * (i == j == k == m)

    def stens(m: int, n: int, j: int, k: int) -> complex:
        return 0.5 * (itens(m, j, k, n) + itens(n, j, k, m))

    # S in the parameter basis: (S : eps_q)_{ij(p)}
    s_par = np.zeros((6, 6), dtype=complex)
    for p in range(3, 9):
        i, j = STRAIN_IJ[p]
        for q in range(3, 9):
            eq = strain_of(q)
            s_par[p - 3, q - 3] = sum(stens(i, j, r, s) * eq[r, s] for r in range(3) for s in range(3))
    # undo the engineering doubling: eps_p : X = w_p X_{ij(p)}
    w = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    g = np.zeros((9, 9), dtype=complex)
    g[:3, :3] = (g0 / vol) * np.eye(3)
    g[3:, 3:] = -(1.0 / vol) * s_par @ la.inv(w)
    return g


def tmatrix(coupling: np.ndarray, g: np.ndarray) -> np.ndarray:
    """T = DeltaC (I - G DeltaC)^-1.

    Args:
        coupling: Shape (9, 9) contrast coupling.
        g: Shape (9, 9) propagator moment.

    Returns:
        Shape (9, 9) complex.
    """
    return coupling @ la.inv(np.eye(9) - g @ coupling)


# channel projectors in the strain sub-basis (e11, e22, e33, e12, e13, e23)
CH_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0)
CH_DIAG = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
CH_OFF = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # e12, in-plane shear
# e13 is O_h-equivalent to e12 for an isotropic cube in an isotropic background,
# so the two MUST amplify identically.  The first-order coupling touches e13 and
# leaves e12 alone, so this pair is where the broken symmetry becomes visible in
# the T-matrix itself rather than only in the coupling.
CH_OFF_XZ = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # e13, out-of-plane shear


def channel_amp(coupling: np.ndarray, g: np.ndarray, vec: np.ndarray) -> complex:
    """Amplification (I - G DeltaC)^-1 along one strain channel.

    Args:
        coupling: Contrast coupling.
        g: Propagator moment.
        vec: Channel vector in the six strain parameters.

    Returns:
        The Rayleigh quotient along that channel.
    """
    amp = la.inv(np.eye(9) - g @ coupling)
    full = np.zeros(9, dtype=complex)
    full[3:] = vec
    return complex(full @ amp @ full) / complex(full @ full)


def main() -> int:
    """Gate the coupling, calibrate the assembly, then build the first-order T.

    Returns:
        0 if every check passes, 1 otherwise.
    """
    print("=" * 78)
    print("  The single-site T-matrix of the first-order matrix-vector wave equation")
    print("=" * 78)

    # ---- 1. the coupling, against the independent symbolic derivation -----
    print("\n--- 1: coupling rebuilt in Python vs the symbolic derivation ------")
    if not REF_JSON.exists():
        print(f"  reference missing: {REF_JSON}")
        print("  run: wolframscript -file Mathematica/FirstOrderContrastOperator.wl")
        return 1
    ref_data = json.loads(REF_JSON.read_text())
    lam_r, mu_r = ref_data["lam"], ref_data["mu"]
    rho_r, w_r, a_r = ref_data["rho"], ref_data["omega"], ref_data["a"]

    worst_eff = worst_nav = 0.0
    for case in ref_data["cases"]:
        con = MaterialContrast(case["dlam"], case["dmu"], case["drho"])
        tgt_eff = np.array(case["dCeff_re"]) + 1j * np.array(case["dCeff_im"])
        tgt_nav = np.array(case["dCnav_re"]) + 1j * np.array(case["dCnav_im"])
        got_eff = coupling_first_order(lam_r, mu_r, rho_r, con, w_r, a_r)
        got_nav = coupling_navier(lam_r, mu_r, rho_r, con, w_r, a_r)
        worst_eff = max(worst_eff, float(la.norm(got_eff - tgt_eff) / la.norm(tgt_eff)))
        worst_nav = max(worst_nav, float(la.norm(got_nav - tgt_nav) / la.norm(tgt_nav)))
    print(f"  worst relative difference, DeltaC_eff : {worst_eff:.3e}")
    print(f"  worst relative difference, DeltaC_Nav : {worst_nav:.3e}")
    report("Python DeltaC_eff matches the symbolic derivation", worst_eff < 1e-10)
    report("Python DeltaC_Navier matches the symbolic derivation", worst_nav < 1e-10)

    # ---- 2. the self moments: INTEGRATED, with the table demoted ---------
    print("\n--- 2: the cube's self moments, by integration --------------------")
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    omega, a = 60.0, 0.5
    if not CUBE_MOMENTS_JSON.exists():
        print(f"  missing: {CUBE_MOMENTS_JSON}")
        print("  run: wolframscript -file Mathematica/CubeSelfMomentExport.wl")
        return 1
    integ = static_moments_integrated(ref, a)
    dynn = dynamic_moments_integrated(ref, omega, a)
    tab = static_moments_tabulated(ref, a)
    print("  the second moment's integrand is O(1/r^3): NOT absolutely integrable.")
    print("  the engine pairs it as a distribution with 1_V and peels the")
    print("  derivatives onto the cube faces, so the r=0 content arrives inside")
    print("  a surface integral -- no table, and no delta added by hand.")
    print(f"  {'moment':8s} {'integrated':>22s} {'tabulated':>22s} {'rel':>10s}")
    worst_mom = 0.0
    for key in ("A", "B", "C", "G"):
        rel = abs(integ[key] - tab[key]) / abs(integ[key])
        worst_mom = max(worst_mom, float(rel))
        print(f"  {key:8s} {integ[key]:22.12e} {tab[key]:22.12e} {rel:10.2e}")
    report("the tabulated Eshelby route REPRODUCES the integrated moments", worst_mom < 1e-12)
    print("  => the table is verified by the integration, not used in place of it;")
    print("     the assembly below reads the integrated values.")

    # ---- 3. calibration: the Navier coupling must reproduce T9 -----------
    print("\n--- 3: CALIBRATION -- Navier coupling must reproduce T9 -----------")
    cases = [
        ("pure shear", MaterialContrast(0.0, 1.0e9, 0.0)),
        ("moderate", MaterialContrast(2.0e9, 1.0e9, 100.0)),
    ]
    g = propagator_moment(ref, omega, a)

    worst_cal = 0.0
    for name, con in cases:
        t9 = compute_cube_tmatrix(omega, a, ref, con)
        nav = coupling_navier(ref.lam, ref.mu, ref.rho, con, omega, a, dynamic_overlap=False)
        amp = la.inv(np.eye(9) - g @ nav)
        got = {
            "u": complex(amp[0, 0]),
            "theta": channel_amp(nav, g, CH_TRACE),
            "e_diag": channel_amp(nav, g, CH_DIAG),
            "e_off": channel_amp(nav, g, CH_OFF),
        }
        want = {
            "u": complex(t9.amp_u),
            "theta": complex(t9.amp_theta),
            "e_diag": complex(t9.amp_e_diag),
            "e_off": complex(t9.amp_e_off),
        }
        print(f"  {name}")
        for key in ("u", "theta", "e_diag", "e_off"):
            rel = abs(got[key] - want[key]) / abs(want[key])
            worst_cal = max(worst_cal, rel)
            print(f"     amp_{key:7s} assembled {got[key]:.9g}   T9 {want[key]:.9g}   rel {rel:.2e}")
    # Tolerance note.  When both routes used the SAME moments this agreed to
    # machine precision.  It no longer does, because the moments are now taken
    # from the engine's own dynamic series (nord = 6, reaching r^5) while the
    # package's route carries eight Taylor terms.  The residual is that
    # truncation difference and nothing else: raising the engine's order from 6
    # to 10 moves Im A by 4e-6 relative, the same size as the gap below.  The
    # bar is set there deliberately rather than loosened until it passed.
    report("the assembly reproduces all four T9 amplification factors", worst_cal < 5e-6)
    print(f"  worst gap {worst_cal:.2e}; this is series truncation (nord 6 vs 8 terms),")
    print("  not a normalisation error -- the static-only substitution was exact.")

    # The term T9's factorisation drops, isolated and sized rather than ignored.
    print("\n  the finite-size density-strain term T9's factorisation omits:")
    worst_dyn = 0.0
    for name, con in cases:
        with_dyn = coupling_navier(ref.lam, ref.mu, ref.rho, con, omega, a)
        no_dyn = coupling_navier(ref.lam, ref.mu, ref.rho, con, omega, a, dynamic_overlap=False)
        d_theta = abs(channel_amp(with_dyn, g, CH_TRACE) - channel_amp(no_dyn, g, CH_TRACE)) / abs(
            channel_amp(no_dyn, g, CH_TRACE)
        )
        worst_dyn = max(worst_dyn, d_theta)
        print(f"     {name:12s} shifts amp_theta by {d_theta:.3e}")
    report("that term is a small correction, not a defect (< 1e-5)", worst_dyn < 1e-5)

    no_rho = MaterialContrast(2.0e9, 1.0e9, 0.0)
    gap = float(
        la.norm(
            coupling_navier(ref.lam, ref.mu, ref.rho, no_rho, omega, a)
            - coupling_navier(ref.lam, ref.mu, ref.rho, no_rho, omega, a, dynamic_overlap=False)
        )
    )
    report("it vanishes identically with Drho", gap == 0.0)

    # ---- 3. the first-order T, channel by channel ------------------------
    print("\n--- 4: the first-order T-matrix, channel by channel ---------------")
    print("  amplification (I - G DeltaC)^-1 along each channel")
    print(f"  {'case':12s} {'channel':9s} {'Navier':>22s} {'first-order':>22s} {'rel diff':>10s}")
    for name, con in cases:
        nav = coupling_navier(ref.lam, ref.mu, ref.rho, con, omega, a)
        eff = coupling_first_order(ref.lam, ref.mu, ref.rho, con, omega, a) / (1j * omega)
        chans = (
            ("theta", CH_TRACE),
            ("e_diag", CH_DIAG),
            ("e12", CH_OFF),
            ("e13", CH_OFF_XZ),
        )
        amps: dict[str, tuple[complex, complex]] = {}
        for label, vec in chans:
            an = channel_amp(nav, g, vec)
            ae = channel_amp(eff, g, vec)
            amps[label] = (an, ae)
            rel = abs(ae - an) / abs(an)
            print(f"  {name:12s} {label:9s} {an:22.9g} {ae:22.9g} {rel:10.3e}")
        an_u = complex(la.inv(np.eye(9) - g @ nav)[0, 0])
        ae_u = complex(la.inv(np.eye(9) - g @ eff)[0, 0])
        rel_u = abs(ae_u - an_u) / abs(an_u)
        print(f"  {name:12s} {'u':9s} {an_u:22.9g} {ae_u:22.9g} {rel_u:10.3e}")
        report(f"{name}: density channel is identical in both routes", rel_u < 1e-12)

        # The symmetry statement, made directly on the T-matrix.
        nav_split = abs(amps["e12"][0] - amps["e13"][0]) / abs(amps["e12"][0])
        eff_split = abs(amps["e12"][1] - amps["e13"][1]) / abs(amps["e12"][1])
        print(
            f"  {name:12s} e12 vs e13 (O_h-equivalent): "
            f"Navier {nav_split:.3e}   first-order {eff_split:.3e}"
        )
        report(f"{name}: the Navier T respects O_h on the e12/e13 pair", nav_split < 1e-12)
        report(f"{name}: the first-order T BREAKS it on that pair", eff_split > 1e-6)
        # A symmetry violation bounds an error without any arbiter: if the two
        # must be equal and differ by delta, the larger error is at least
        # delta/2.  It bounds ONLY the route that breaks the symmetry -- a
        # symmetric approximation can be symmetrically wrong -- so this orders
        # nothing.  Stated because "breaks a symmetry" reads as a verdict.
        print(f"  {name:12s} => lower bound on the FIRST-ORDER error in that channel: {eff_split / 2:.3e}")
        print(f"  {name:12s} => bound on the Navier error from this check: none")

    # ---- 4. Born limit ---------------------------------------------------
    print("\n--- 5: the Born limit must agree exactly --------------------------")
    print("  DeltaC_eff is in VELOCITY units and DeltaC_Navier in DISPLACEMENT")
    print("  units -- see to_navier_units.  The rigid blocks are i w^3 Drho V and")
    print("  w^2 Drho V, a ratio of exactly i w = " + f"{1j * omega:+.1f}.")
    print("  Every other check in this gate is a ratio, so the factor cancels and")
    print("  none of them can see it.  This one is absolute, and does.")
    tiny = MaterialContrast(2.0e3, 1.0e3, 1.0e-4)
    nav = coupling_navier(ref.lam, ref.mu, ref.rho, tiny, omega, a)
    eff = to_navier_units(coupling_first_order(ref.lam, ref.mu, ref.rho, tiny, omega, a), omega)
    rel_born = float(la.norm(eff - nav) / la.norm(nav))
    print(f"  ||DeltaC_eff/(i w) - DeltaC_Nav|| / ||DeltaC_Nav|| at 1e-6 contrast: {rel_born:.3e}")
    report("the two couplings agree to O(Dc) in the Born limit", bool(rel_born < 1e-5))

    # A single small contrast shows agreement; a SCALING shows the residual is
    # the O(Dc^2) difference and not a second, smaller units error hiding under
    # the first.  The ratio between successive rows must be the contrast ratio.
    print("  and the residual is O(Dc^2), not a smaller units slip:")
    prev = None
    ok_scale = True
    for scale in (1.0, 0.1, 0.01, 0.001):
        con_s = MaterialContrast(2.0e9 * scale, 1.0e9 * scale, 100.0 * scale)
        nav_s = coupling_navier(ref.lam, ref.mu, ref.rho, con_s, omega, a)
        eff_s = to_navier_units(coupling_first_order(ref.lam, ref.mu, ref.rho, con_s, omega, a), omega)
        rel_s = float(la.norm(eff_s - nav_s) / la.norm(nav_s))
        rate = "" if prev is None else f"   ratio {prev / rel_s:6.2f}  (expect 10)"
        print(f"    contrast x{scale:<7g} {rel_s:.4e}{rate}")
        if prev is not None and not 9.0 < prev / rel_s < 11.0:
            ok_scale = False
        prev = rel_s
    report("that residual scales as the FIRST power of the contrast", ok_scale)

    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok in _PASS if ok)
    print(f"  {n_ok} passed, {len(_PASS) - n_ok} failed")
    print("=" * 78)
    print("  CAVEAT 1: the propagator moment used is the SINGLE (collocation)")
    print("  one, while the J6 pairing makes the scheme Galerkin.  The gap is")
    print("  measured, not suspected -- gate_first_order_schwinger.py factors it")
    print("  into int_V 1/r / int_V int_V 1/r = 1.2644 and a factor i*omega.")
    print("  The i*omega is NOT open: it is the velocity-vs-displacement units of")
    print("  to_navier_units, closed by check 5 above.  What remains open is the")
    print("  GEOMETRIC factor -- collocation against Galerkin.  Do not read these")
    print("  amplitudes as a consistent Galerkin T.")
    print("  CAVEAT 2: the O_h split lower-bounds the FIRST-ORDER error only.")
    print("  A symmetric approximation can be symmetrically wrong, so this")
    print("  bounds the Navier route not at all.  Which is more accurate is")
    print("  OPEN.  No winner is declared here, and none should be read in.")
    return 0 if n_ok == len(_PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
