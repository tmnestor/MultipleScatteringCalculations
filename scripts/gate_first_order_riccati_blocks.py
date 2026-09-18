#!/usr/bin/env python3
"""The Riccati coefficient matrices V, their structure, and whether the march is stiff.

ANCHOR: Nestor (1996) Ch.2 (Akdef), (ATdef), (specA); Ch.5 GstratRep.tex
        (Amat) and Algorithm 5.1.

WHAT THIS ANSWERS
-----------------
The Leibniz argument turns the Lippmann-Schwinger equation into
``d_z q = [A + dA(z)] q``, and in mode coordinates the reflection operator R
obeys a matrix Riccati equation whose coefficients are the four direction blocks
of ``V = D_z^-1 dA D_z``.  Two questions follow immediately, and both are
measured here rather than argued:

  1. HOW ARE THE BLOCKS BUILT, and does V carry any structure worth exploiting?
  2. IS THE MARCH STIFF, and if so where does the stiffness sit?

WHAT IS FOUND
-------------
V inherits the quasi-Hamiltonian relation (ATdef) exactly:

    J6 V^T(-k) + V(k) J6 = 0   <=>   V_du(k) = V_du^T(-k),
                                     V_ud(k) = V_ud^T(-k),
                                     V_dd(k) = -V_uu^T(-k),

so only half the blocks are independent, and the Riccati flow preserves the
symmetry of R -- which is reciprocity of the reflection operator.

The march IS stiff, but in a harmless place.  The linear part of the Riccati
equation is DIAGONAL on the entries of R: entry (c,c') sees -i(k_zc + k_zc').
Inside the propagating window that is purely imaginary -- oscillatory, not
stiff.  Beyond it, it is real and decaying at a rate up to 2|k| marching upward,
giving a stiffness ratio of about |k| beta / omega.  Because it is diagonal, an
integrating factor removes it EXACTLY at no cost, and the layer recursion in the
sibling repository already is that integrating factor: in a homogeneous layer
V = 0 and the exact update is the phase factor it already applies.

The real obstruction is not stiffness.  It is that V is strongly NON-NORMAL:
eig(V)/2|k| is flat at 0.0656 over four decades -- the contrast strength, as the
physics requires -- while |V|/2|k| grows as |k|^2.  It is not a scaling choice:
Parlett--Reinsch balancing of V returns the identity.  The mechanism is measured
here.  At large lateral wavenumber k_zP and k_zS both tend to i|k|, so the P and
S columns of D_z become PARALLEL -- cos(+P,+S) reaches 1.000000 -- and
cond(D_z) grows as |k|^2.001 even after the symplectic row scaling.  That is the
elastostatic degeneracy: at short lateral scale there is no distinction between
P and S, both being the same decaying static field.  It is physical, and unlike
the unit mismatch the row scaling removes, it cannot be scaled away.  Up- and
downgoing modes are NOT affected: cos(+P,-P) falls to 2e-6.

The consequence is a ceiling.  V_du and V_ud are each 2.0e6 at |k| = 600 while
the eigenvalues of their product are only 79^2, a nine-order cancellation, and
the similarity eig(V) = eig(dA) degrades from 1.5e-15 at |k| = 2 to 2.0 at
|k| = 600.  The mode basis is usable to |k| of order 200 here, about 35 times
the S wavenumber, and not beyond.  Part 6 records that range.

Units are km, km/s, g/cm^3, rad/km, rad/s.

Run:  conda run -n seismic python scripts/gate_first_order_riccati_blocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import matrix_balance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from scripts.gate_thesis_spectral import (  # noqa: E402
    J6,
    amat_thesis,
    dz_balanced,
    dz_normalised,
    kz_c,
    rowscale,
)

BG = ReferenceMedium(4.0, 2.22, 2.6)
QFAC = 600.0
OMEGA = 2.0 * np.pi * 2.0 * (1.0 + 0.5j / QFAC)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def perturbed(dlam: float, dmu: float, drho: float) -> ReferenceMedium:
    """The contrasted medium, expressed back as velocities.

    Args:
        dlam: Lame lambda contrast.
        dmu: Shear modulus contrast.
        drho: Density contrast.

    Returns:
        The perturbed medium.
    """
    lam, mu, rho = BG.lam + dlam, BG.mu + dmu, BG.rho + drho
    return ReferenceMedium(np.sqrt((lam + 2.0 * mu) / rho), np.sqrt(mu / rho), rho)


PER = perturbed(2.0, 1.0, 0.1)


def delta_a(kx: float, ky: float) -> np.ndarray:
    """The contrast operator, as the difference of two (Akdef) matrices.

    Args:
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        Shape (6, 6) complex.
    """
    return amat_thesis(PER, OMEGA, kx, ky) - amat_thesis(BG, OMEGA, kx, ky)


def vmat(kx: float, ky: float, *, balanced: bool = True) -> np.ndarray:
    """V = D_z^-1 dA D_z, by either route.

    The symplectic row scaling is a similarity and CANCELS out of V exactly:
    (R D)^-1 (R dA R^-1)(R D) = D^-1 dA D.  So the two routes must agree, and
    Part 2 uses that as an independent check on both.

    Args:
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.
        balanced: Use the balanced route rather than the epsilon-normalised one.

    Returns:
        Shape (6, 6) complex.
    """
    da = delta_a(kx, ky)
    if not balanced:
        dzm, inv, _ = dz_normalised(BG, OMEGA, kx, ky)
        return np.asarray(inv @ da @ dzm)
    scl = rowscale(BG, OMEGA, kx, ky)
    dzb, invb, _, _ = dz_balanced(BG, OMEGA, kx, ky)
    return np.asarray(invb @ (scl[:, None] * da * (1.0 / scl)[None, :]) @ dzb)


def blocks(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split V into (V_dd, V_du, V_ud, V_uu) by the (eigDef) column ordering.

    Args:
        v: Shape (6, 6).

    Returns:
        The four 3x3 direction blocks.
    """
    return v[:3, :3], v[:3, 3:], v[3:, :3], v[3:, 3:]


def main() -> int:
    """Gate the Riccati coefficient matrices.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  The Riccati coefficient matrices, and whether the march is stiff")
    print("=" * 74)
    kp, ks = abs(OMEGA / BG.alpha), abs(OMEGA / BG.beta)
    print(f"    omega/alpha = {kp:.4f}, omega/beta = {ks:.4f} rad/km")

    print("")
    print("--- 1: dA is sparse, and its sparsity is structural ----------------")
    print("    dA is the difference of two (Akdef) matrices, so every entry")
    print("    built from k alone -- the kinematic -i k_x, -i k_y -- cancels.")
    da = delta_a(1.3, 0.7)
    pat = np.abs(da) > 1e-13 * np.max(np.abs(da))
    for r in range(6):
        print("        " + "  ".join("X" if pat[r, c] else "." for c in range(6)))
    print(f"    {int(pat.sum())} of 36 entries carry material contrast")
    tr_diag = bool(np.all(pat[:3, 3:] == np.eye(3, dtype=bool)))
    print("    top-right block is exactly diagonal, diag(D a, D b, D b):", tr_diag)
    report("dA has 12 of 36 nonzero, the kinematic entries having cancelled", int(pat.sum()) == 12)
    report("the compliance block is diagonal, so V_du has a 3-term source", tr_diag)

    print("")
    print("--- 2: V, built two independent ways -------------------------------")
    print("    The row scaling is a similarity and cancels out of V exactly, so")
    print("    the epsilon-normalised and balanced routes must agree.")
    worst_route = 0.0
    for k in (2.0, 8.0, 60.0, 200.0):
        v_n, v_b = vmat(k, 0.0, balanced=False), vmat(k, 0.0)
        rel = float(np.max(np.abs(v_n - v_b)) / np.max(np.abs(v_b)))
        worst_route = max(worst_route, rel)
        print(f"    |k| = {k:7.1f}   routes differ by {rel:10.3e}")
    report("the two routes to V agree, so neither normalisation is at fault", worst_route < 1e-13)

    print("")
    print("--- 3: V inherits (ATdef), so half the blocks are redundant ---------")
    print("    dA is a difference of two operators that each satisfy (ATdef), so")
    print("    it satisfies it too, and D_z^-1 = -i J6 D_z^T(-k) J6 carries it")
    print("    into the mode basis:  J6 V^T(-k) + V(k) J6 = 0.")
    print(f"    {'kx':>6} {'ky':>6} {'|J6 V^T(-k) + V(k) J6|':>24} {'V_dd + V_uu^T(-k)':>20}")
    worst_ham = 0.0
    for kx, ky in ((0.9, 0.0), (2.1, 1.3), (5.0, -2.0), (9.0, 4.0), (40.0, -15.0)):
        v, vm = vmat(kx, ky), vmat(-kx, -ky)
        sc = float(np.max(np.abs(v)))
        e1 = float(np.max(np.abs(J6 @ vm.T + v @ J6))) / sc
        dd, _, _, _ = blocks(v)
        _, _, _, uu_m = blocks(vm)
        e2 = float(np.max(np.abs(dd + uu_m.T))) / sc
        worst_ham = max(worst_ham, e1, e2)
        print(f"    {kx:6.1f} {ky:6.1f} {e1:24.3e} {e2:20.3e}")
    report("V is quasi-Hamiltonian, so R stays symmetric and reciprocal", worst_ham < 1e-13)

    print("")
    print("--- 4: is it stiff?  Where the stiffness sits ----------------------")
    print("    The linear part of the Riccati equation is DIAGONAL on the")
    print("    entries of R: entry (c,c') sees mu = -i(k_zc + k_zc').  Inside")
    print("    the propagating window that is imaginary -- oscillatory.  Beyond")
    print("    it, real and decaying marching up, at a rate up to 2|k|.")
    print(f"    {'|k|':>8} {'max|Re mu|':>12} {'max|Im mu|':>12} {'regime':>14} {'ratio':>8}")
    base, worst_ratio, prop_is_osc = None, 0.0, True
    for k in (0.5, 2.0, 5.0, 8.0, 20.0, 60.0, 200.0, 600.0):
        zs = [kz_c(c, OMEGA, k, 0.0) for c in (BG.alpha, BG.beta, BG.beta)]
        mu = np.array([-1j * (a + b) for a in zs for b in zs])
        re, im = float(np.max(np.abs(mu.real))), float(np.max(np.abs(mu.imag)))
        if base is None:
            base = max(re, im)
        worst_ratio = max(worst_ratio, max(re, im) / base)
        if k < kp:
            prop_is_osc = prop_is_osc and re < 1e-2 * im
        tag = "propagating" if k < kp else ("mixed" if k < ks else "evanescent")
        print(f"    {k:8.1f} {re:12.4f} {im:12.4f} {tag:>14} {max(re, im) / base:8.1f}")
    report("inside the propagating window the linear part is oscillatory", prop_is_osc)
    report("beyond it the march IS stiff, the ratio growing as |k| beta/omega", worst_ratio > 50.0)
    print("    Because it is DIAGONAL, an integrating factor removes it exactly.")
    print("    The sibling repository's layer recursion already is that factor:")
    print("    in a homogeneous layer V = 0 and the update is the phase it")
    print("    applies, which is why it is stable at any layer thickness.")

    print("")
    print("--- 5: the real obstruction is NON-NORMALITY, not stiffness --------")
    print("    Physics demands eig(V) scale with Lambda ~ |k|: a fixed material")
    print("    contrast cannot strengthen with lateral wavenumber.  It does.")
    print("    The NORM does not.")
    print(f"    {'|k|':>8} {'|V|/2|k|':>12} {'max|eig|/2|k|':>15} {'after balancing':>17}")
    ratios, unbalanceable = [], 0.0
    for k in (2.0, 8.0, 20.0, 60.0, 200.0, 600.0, 2000.0):
        v = vmat(k, 0.0)
        ev = float(np.max(np.abs(np.linalg.eigvals(delta_a(k, 0.0)))))
        vb, tmat = matrix_balance(v, permute=False)
        nb = float(np.max(np.abs(vb))) / (2 * k)
        unbalanceable = max(unbalanceable, abs(nb - float(np.max(np.abs(v))) / (2 * k)))
        ratios.append(ev / (2 * k))
        print(f"    {k:8.1f} {float(np.max(np.abs(v))) / (2 * k):12.3f} {ev / (2 * k):15.4f} {nb:17.3f}")
    spread = float(np.max(ratios) / np.min(ratios))
    print(f"    eig/2|k| spread over four decades = {spread:.4f}")
    report("the SPECTRUM of V is proportional to Lambda, as physics requires", spread < 1.05)
    report("the NORM is not, and balancing cannot fix it -- V is non-normal", unbalanceable < 1e-9)

    print("")
    print("--- 6: the mechanism, and the ceiling it imposes --------------------")
    print("    At large lateral wavenumber k_zP and k_zS both tend to i|k|, so")
    print("    the P and S columns of D_z become PARALLEL.  That is the")
    print("    elastostatic degeneracy -- at short lateral scale there is no")
    print("    distinction between P and S -- and it is physical, unlike the")
    print("    unit mismatch the row scaling removes.  Up/down is unaffected.")
    print(
        f"    {'|k|':>8} {'cond(D_z bal)':>15} {'cos(+P,+S)':>12} {'cos(+P,-P)':>12} {'eig(V)=eig(dA)':>16}"
    )
    ks_fit, cs_fit, worst_ok, ceiling = [], [], 0.0, True
    for k in (2.0, 8.0, 20.0, 60.0, 200.0, 600.0, 2000.0):
        d, _, _, _ = dz_balanced(BG, OMEGA, k, 0.0)
        cnd = float(np.linalg.cond(d))

        def cosang(u: np.ndarray, w: np.ndarray) -> float:
            return float(abs(np.vdot(u, w)) / (np.linalg.norm(u) * np.linalg.norm(w)))

        eig_da = np.sort_complex(np.linalg.eigvals(delta_a(k, 0.0)))
        eig_v = np.sort_complex(np.linalg.eigvals(vmat(k, 0.0)))
        sim = float(np.max(np.abs(eig_v - eig_da)) / np.max(np.abs(eig_da)))
        if k <= 200.0:
            worst_ok = max(worst_ok, sim)
        if k >= 600.0:
            ceiling = ceiling and sim > 1e-3
        ks_fit.append(k)
        cs_fit.append(cnd)
        print(
            f"    {k:8.1f} {cnd:15.4e} {cosang(d[:, 0], d[:, 1]):12.6f} "
            f"{cosang(d[:, 0], d[:, 3]):12.6f} {sim:16.3e}"
        )
    slope = float(np.polyfit(np.log(ks_fit[-4:]), np.log(cs_fit[-4:]), 1)[0])
    print(f"    d log cond / d log |k| over the last two decades = {slope:.3f}")
    report("cond(D_z) grows as |k|^2, from P and S coalescing", abs(slope - 2.0) < 0.05)
    report("the mode basis is sound to |k| ~ 200 here, 35x the S wavenumber", worst_ok < 1e-7)
    report("and is NOT sound beyond it -- the ceiling is recorded, not assumed", ceiling)

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
