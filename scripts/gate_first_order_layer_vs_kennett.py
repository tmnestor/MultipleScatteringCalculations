#!/usr/bin/env python3
"""A single layer in a whole space, from the thesis system matrix, against Kennett.

ANCHOR: Nestor (1996) Ch.2, GRepresentations.tex -- (Akdef), (specA), (kzcDef),
(Peigen), (SVeigen), (SHeigen).  Everything here is built on
``scripts/gate_thesis_spectral.py``, which implements and gates that
representation.

This is the one EXTERNAL arbiter in the first-order development.  Every other
comparison sets two formulations against each other; this one sets the
first-order contrast operator against an independently validated reflectivity
recursion, and against a closed form written out from scratch below.

WHY A LAYER IS THE RIGHT TEST
-----------------------------
For a laterally invariant contrast the operator difference is simply

    DeltaA(z) = [ A(lay; k) - A(ref; k) ] * 1_{[0,h]}(z) ,

a piecewise-constant matrix, obtained by evaluating (Akdef) twice.  There is NO
augmentation, no by-parts transfer, no surface term and no differentiated
indicator -- because A carries no d_z, a jump in the medium is a jump in the
coefficients of an ODE and is harmless.  That is the structural advantage of the
first-order form, and it is exactly the property the cube development had to work
around.  It is also nonlinear in the contrast, so it reaches the RESUMMATION; and
DeltaA carries k explicitly, so scanning slowness tests a k dependence that the
multiplicative density channel of the cube could not.

WHAT THE THESIS BASIS REMOVES
-----------------------------
An earlier version of this gate worked in the paper's velocity--traction basis
and needed three pieces of machinery that are simply absent here:

  * a numerical eigendecomposition, with the balancing required to condition it;
  * a classification of the modes into up- and down-going, which is meaningless
    inside the propagating window where Re(lambda) vanishes to round-off;
  * a restriction to the 4x4 P-SV block, introduced because SV and SH share
    q_S and numpy returns an arbitrary mixture of them -- which gave an SS
    coefficient wrong by a factor of four at one slowness and right at three.

In the thesis representation the eigenvectors are analytic, the columns are
ordered [+P, +S, +H, -P, -S, -H] with + downgoing, and the quasi-SV and quasi-SH
vectors are chosen NON-DEGENERATE.  So the mode wanted is an index, the full 6x6
is used, and SH comes for free.

THE NORMALISATION TRAP
----------------------
Reflection coefficients differ between the displacement and flux bases by
diag(alpha sqrt(eta_P), i beta sqrt(eta_S)), which no per-mode ratio test can
see.  Nothing is compared against Kennett until the convention is pinned at
normal incidence against ``normal_incidence_layer``, derived here by continuity
and owing nothing to any other code.  R is defined throughout as

    R = [u_z of the up-going field at z = 0-] / [u_z of the incident at z = 0-]

on both sides of every comparison, which leaves no sign convention free.  In the
thesis basis u is the FIRST block, so this needs no velocity factor at all.

Run:  conda run -n seismic python scripts/gate_first_order_layer_vs_kennett.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack, kennett_layers  # noqa: E402
from scripts.gate_thesis_spectral import amat_thesis, dz_normalised, lam_diag  # noqa: E402

# Column order of D_z, from (eigDef): the first three are downgoing.
IDX = {"Pd": 0, "Sd": 1, "Hd": 2, "Pu": 3, "Su": 4, "Hu": 5}
# The component that normalises each mode -- the one that survives at k = 0,
# which in the thesis basis (u_z, u_x, u_y, T_zz, T_xz, T_yz) is u_z for P,
# u_x for quasi-SV and u_y for quasi-SH.
NORM = {"P": 0, "S": 1, "H": 2}

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def normal_incidence_layer(
    rho1: float, a1: float, rho2: float, a2: float, h: float, omega: float
) -> complex:
    """Reflection of a normally incident P wave from a layer, by continuity.

    Derived here rather than quoted.  With u the vertical displacement and
    e^{-i omega t}, in each medium u = D e^{ikz} + U e^{-ikz} with k = omega/c,
    and the traction is tau = rho c^2 du/dz = i rho c omega (D e^{ikz} - U
    e^{-ikz}).  Matching u and tau at z = 0 and z = h with no up-going wave below
    gives the two-interface result.  The point of deriving it is that the SIGN is
    then fixed by the derivation and not by a convention.

    Args:
        rho1: Density outside.
        a1: P velocity outside.
        rho2: Density in the layer.
        a2: P velocity in the layer.
        h: Layer thickness.
        omega: Angular frequency.

    Returns:
        U/D in the incident medium, at z = 0.
    """
    z1, z2 = rho1 * a1, rho2 * a2
    r12 = (z1 - z2) / (z1 + z2)
    r23 = (z2 - z1) / (z2 + z1)
    e2 = np.exp(2j * omega * h / a2)
    return complex((r12 + r23 * e2) / (1.0 + r12 * r23 * e2))


def layer_reflection_matrix(
    ref: ReferenceMedium,
    lay: ReferenceMedium,
    h: float,
    omega: complex,
    p: float,
    ncell: int,
    zero_k_contrast: bool = False,
) -> np.ndarray:
    """R for a layer in a whole space, from the thesis system matrix.

    Solves the laterally invariant Lippmann--Schwinger equation

        b(z) = b_inc(z) + int_0^h Gamma(k; z, z') DeltaA b(z') dz'

    by piecewise-constant collocation.  The cell integrals are closed form from
    (specA) and are split at z' = z: the jump across the diagonal is the source
    delta of (FLdef), and quadrature across it would smear the very object that
    defines the propagator.

    Args:
        ref: Background medium.
        lay: Layer medium.
        h: Layer thickness.
        omega: Angular frequency.
        p: Horizontal slowness, so k_x = omega p.
        ncell: Cells across the layer.
        zero_k_contrast: Build DeltaA at k = 0 -- a control, not physics.

    Returns:
        Shape (3, 3) complex, indexed [out, in] over (P, S, H).
    """
    kx = float(np.real(omega) * p)
    kc = 0.0 if zero_k_contrast else kx
    da = amat_thesis(lay, omega, kc, 0.0) - amat_thesis(ref, omega, kc, 0.0)
    dz_m, inv, _ = dz_normalised(ref, omega, kx, 0.0)
    lam = lam_diag(ref, omega, kx, 0.0)
    proj = np.stack([np.outer(dz_m[:, i], inv[i, :]) for i in range(6)])

    edge = np.linspace(0.0, h, ncell + 1)
    zc = 0.5 * (edge[:-1] + edge[1:])

    def kernel_block(z: float, lo: float, hi: float) -> np.ndarray:
        """int_lo^hi Gamma(z - z') dz', exactly, split at z' = z."""
        out = np.zeros((6, 6), dtype=complex)
        for modes, sgn in ((range(3), 1.0), (range(3, 6), -1.0)):
            a, b = (lo, min(hi, z)) if sgn > 0 else (max(lo, z), hi)
            if b <= a:
                continue
            for i in modes:
                e = lam[i]
                val = (b - a) if abs(e) < 1e-300 else (np.exp(e * (z - a)) - np.exp(e * (z - b))) / e
                out += sgn * val * proj[i]
        return out

    big = np.zeros((6 * ncell, 6 * ncell), dtype=complex)
    for m in range(ncell):
        for n in range(ncell):
            big[6 * m : 6 * m + 6, 6 * n : 6 * n + 6] = kernel_block(zc[m], edge[n], edge[n + 1]) @ da
    lhs = np.eye(6 * ncell, dtype=complex) - big

    out = np.zeros((3, 3), dtype=complex)
    ztop = -0.25 * h if h > 0 else -1.0
    for jc, jn in enumerate(("P", "S", "H")):
        idx = IDX[f"{jn}d"]
        vin = dz_m[:, idx] / dz_m[NORM[jn], idx]
        rhs = np.concatenate([vin * np.exp(lam[idx] * z) for z in zc])
        qs = np.linalg.solve(lhs, rhs).reshape(ncell, 6)
        sc = np.zeros(6, dtype=complex)
        for n in range(ncell):
            sc += kernel_block(ztop, edge[n], edge[n + 1]) @ da @ qs[n]
        for ic, iname in enumerate(("P", "S", "H")):
            iu = IDX[f"{iname}u"]
            cu = (inv[iu, :] @ sc) * np.exp(-lam[iu] * ztop)
            out[ic, jc] = cu * dz_m[NORM[iname], iu]
    return out


def layer_reflection_first_order(
    ref: ReferenceMedium,
    lay: ReferenceMedium,
    h: float,
    omega: complex,
    p: float,
    ncell: int,
    zero_k_contrast: bool = False,
) -> complex:
    """The PP entry of the reflection matrix, in the u_z convention.

    Args:
        ref: Background medium.
        lay: Layer medium.
        h: Layer thickness.
        omega: Angular frequency.
        p: Horizontal slowness.
        ncell: Cells across the layer.
        zero_k_contrast: Build DeltaA at k = 0 -- a control, not physics.

    Returns:
        R_PP.
    """
    return complex(layer_reflection_matrix(ref, lay, h, omega, p, ncell, zero_k_contrast)[0, 0])


def layer_reflection_kennett(
    ref: ReferenceMedium, lay: ReferenceMedium, h: float, omega: float, p: float
) -> tuple[np.ndarray, complex]:
    """RD_psv and RD_sh for the same layer, from the Kennett recursion.

    Kennett needs a finite top layer but does NOT reference RD to the top of it:
    RD is unchanged for d = 50, 100, 200, 400, so it already sits at the top of
    the target layer.  Dividing out a two-way phase here -- the natural guess --
    manufactures a spurious slowness-dependent offset.  Measured, not assumed.

    Args:
        ref: Background medium.
        lay: Layer medium.
        h: Layer thickness.
        omega: Angular frequency.
        p: Horizontal slowness.

    Returns:
        (RD_psv as (2, 2), RD_sh).
    """
    stack = LayerStack(
        [
            IsotropicLayer(ref.alpha, ref.beta, ref.rho, 100.0),
            IsotropicLayer(lay.alpha, lay.beta, lay.rho, h),
            IsotropicLayer(ref.alpha, ref.beta, ref.rho, np.inf),
        ]
    )
    res = kennett_layers(stack, p, np.array([omega]))
    return np.asarray(res.RD_psv[0], dtype=complex), complex(res.RD_sh[0])


def main() -> int:
    """Run the layer gate.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  A layer in a whole space: the thesis system matrix against Kennett")
    print("=" * 74)
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    omega = 60.0
    h = 200.0

    print("")
    print("--- 1: normal incidence, against the closed form derived here -----")
    print(f"    background alpha={ref.alpha} beta={ref.beta} rho={ref.rho}, h={h} m")
    print(f"    omega={omega} rad/s, so omega h / alpha = {omega * h / ref.alpha:.3f}")
    for name, lay in (
        ("weak   (+2%)", ReferenceMedium(5100.0, 3060.0, 2550.0)),
        ("medium (+10%)", ReferenceMedium(5500.0, 3300.0, 2750.0)),
        ("strong (+40%)", ReferenceMedium(7000.0, 4200.0, 3500.0)),
    ):
        exact = normal_incidence_layer(ref.rho, ref.alpha, lay.rho, lay.alpha, h, omega)
        got = layer_reflection_first_order(ref, lay, h, omega, 0.0, 160)
        rel = abs(got - exact) / abs(exact)
        print(f"    {name}:  closed form {exact.real:+.6f}{exact.imag:+.6f}i")
        print(f"                  first-order {got.real:+.6f}{got.imag:+.6f}i   rel {rel:.3e}")
        report(f"normal incidence, {name}, matches the closed form", rel < 5e-3)

    print("")
    print("--- 2: convergence in the number of cells -------------------------")
    lay = ReferenceMedium(5500.0, 3300.0, 2750.0)
    exact = normal_incidence_layer(ref.rho, ref.alpha, lay.rho, lay.alpha, h, omega)
    errs = []
    for nc in (20, 40, 80, 160):
        got = layer_reflection_first_order(ref, lay, h, omega, 0.0, nc)
        errs.append(abs(got - exact) / abs(exact))
        print(f"    ncell={nc:4d}   rel {errs[-1]:.3e}")
    rates = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(f"    refinement ratios: {', '.join(f'{r:.2f}' for r in rates)}")
    report("the collocation converges under refinement", errs[-1] < errs[0] / 4.0)

    print("")
    print("--- 3: oblique incidence, against Kennett -------------------------")
    print("    What is CHECKED here is that the three DIAGONAL ratios are")
    print("    constant across angle.  That is not the whole convention: Part 7")
    print("    of ``gate_first_order_impedance_march`` determines it properly and")
    print("    finds a two-sided transform R_th = S_u R_ken S_d^-1 whose P/S part")
    print("    is sqrt(k_zS/k_zP) and so is NOT constant.  The diagonal is.")
    ratios = []
    for p in (0.0, 5e-5, 1.0e-4, 1.5e-4):
        rmat = layer_reflection_matrix(ref, lay, h, omega, p, 120)
        ken, ksh = layer_reflection_kennett(ref, lay, h, omega, p)
        rat = np.array([rmat[0, 0] / ken[0, 0], rmat[1, 1] / ken[1, 1], rmat[2, 2] / ksh], dtype=complex)
        ratios.append(rat)
        print(
            f"    p={p:8.2e} (sin i={p * ref.alpha:.3f})   ratio  PP {rat[0]:+.5f}"
            f"   SS {rat[1]:+.5f}   HH {rat[2]:+.5f}"
        )
    arr = np.array(ratios)
    spread = float(np.max(np.abs(arr - arr[0])))
    print(f"    spread of the diagonal ratios across slowness = {spread:.3e}")
    report("the PP, SS and SH conventions are constant across slowness", spread < 1e-3)
    print(f"    the constants are {arr[0][0]:+.6f}, {arr[0][1]:+.6f}, {arr[0][2]:+.6f}")

    print("")
    print("--- 4: the k dependence of DeltaA is load-bearing -----------------")
    for p in (0.0, 1.0e-4):
        good = layer_reflection_first_order(ref, lay, h, omega, p, 120)
        bad = layer_reflection_first_order(ref, lay, h, omega, p, 120, zero_k_contrast=True)
        rel = abs(bad - good) / abs(good)
        print(
            f"    p={p:8.2e}   with k {good.real:+.6f}{good.imag:+.6f}i"
            f"   without {bad.real:+.6f}{bad.imag:+.6f}i   rel {rel:.3e}"
        )
        if p == 0.0:
            report("NEGATIVE CONTROL: at normal incidence the k terms are inert", rel < 1e-12)
        else:
            report("NEGATIVE CONTROL: off normal incidence they are not", rel > 0.05)

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
