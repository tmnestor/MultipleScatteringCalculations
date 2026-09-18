#!/usr/bin/env python3
"""A single layer in a whole space, from the first-order system, against Kennett.

This is the one EXTERNAL arbiter in the whole first-order development.  Every
other comparison made so far sets two formulations against each other; this one
sets the first-order contrast operator against an independently validated
reflectivity code, and against a closed form written out from scratch below.

WHY A LAYER IS THE RIGHT TEST
-----------------------------
For a laterally invariant contrast the operator difference is simply

    DeltaA(z) = [ A_total(k) - A_background(k) ] * 1_{[0,h]}(z) ,

a piecewise-constant 6x6 matrix.  There is NO augmentation, no by-parts
transfer, no surface term and no differentiated indicator -- because A carries no
d3, a jump in the medium is harmless.  That is the structural advantage of the
first-order form, and it is exactly the property the cube development had to work
around.  So this tests A, Gamma and the Lippmann--Schwinger composition with
none of the cube machinery in the way.

It is also nonlinear in the contrast, so it tests the RESUMMATION rather than
just the Born term; and DeltaA carries k explicitly (A12 ~ nu1 k^2 / i omega), so
scanning slowness tests the k dependence that the density channel could not.

THE NORMALISATION TRAP
----------------------
Reflection coefficients differ between the displacement and flux bases by
diag(alpha sqrt(eta_P), i beta sqrt(eta_S)), which no per-mode ratio test can
see.  So nothing here is compared against Kennett until the convention is pinned
at normal incidence against ``normal_incidence_layer``, which is derived in this
file by continuity of displacement and traction across the two interfaces and
owes nothing to any other code.  R is defined throughout as

    R = [u3 of the up-going field at z = 0-] / [u3 of the incident field at z = 0-]

on both sides of every comparison, which leaves no sign convention free.

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
from scripts.gate_first_order_schwinger import (  # noqa: E402
    amat_paper_batch,
    balance_batch,
    downgoing,
)

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


# ---------------------------------------------------------------------------
# The independent closed form: normal incidence, by continuity
# ---------------------------------------------------------------------------


def normal_incidence_layer(
    rho1: float, a1: float, rho2: float, a2: float, h: float, omega: float
) -> complex:
    """Reflection of a normally incident P wave from a layer, by continuity.

    Derived here rather than quoted.  With u the vertical displacement and
    e^{-i omega t}, in each medium u = D e^{i k z} + U e^{-i k z} with k = omega/c,
    and the traction is tau = (rho c^2) du/dz = i rho c omega (D e^{ikz} - U
    e^{-ikz}).  Matching u and tau at z = 0 and z = h with no up-going wave in the
    lower half-space gives the two-interface result.  Written with the impedance
    Z = rho c, the standard form follows, and the point of writing it out is that
    the SIGN is then fixed by the derivation and not by a convention.

    Args:
        rho1: Density outside.
        a1: P velocity outside.
        rho2: Density in the layer.
        a2: P velocity in the layer.
        h: Layer thickness.
        omega: Angular frequency.

    Returns:
        U/D for the incident medium, at z = 0.
    """
    z1, z2 = rho1 * a1, rho2 * a2
    r12 = (z1 - z2) / (z1 + z2)  # u-reflection: u continuous, tau flips with Z
    r23 = (z2 - z1) / (z2 + z1)
    e2 = np.exp(2j * omega * h / a2)
    return complex((r12 + r23 * e2) / (1.0 + r12 * r23 * e2))


# ---------------------------------------------------------------------------
# The first-order route
# ---------------------------------------------------------------------------


# With k along x1 the system is block diagonal: {-tau_23, v2} is SH and does not
# couple to {-tau_13, -tau_33, v1, v3}.  Working in the P-SV block is not an
# approximation, and it removes the double degeneracy of q_S -- which is a real
# hazard, not a nicety: eig splits a degenerate eigenspace arbitrarily, so SV and
# SH come back mixed, and the SS coefficient was wrong by a factor of four at one
# slowness and right at three others.
PSV = [0, 2, 3, 5]


def amat_psv(ref: ReferenceMedium, omega: complex, k1: float) -> np.ndarray:
    """The 4x4 P-SV block of A at lateral wavenumber k1 along x1.

    Args:
        ref: Medium.
        omega: Angular frequency.
        k1: Lateral wavenumber.

    Returns:
        Shape (4, 4) complex.
    """
    a6 = amat_paper_batch(ref, omega, np.array([k1]), np.array([0.0]))[0]
    return a6[np.ix_(PSV, PSV)]


def spectral(ref: ReferenceMedium, omega: complex, k1: float) -> tuple:
    """Eigenvalues, right and left eigenvectors of the P-SV block, with up/down.

    Args:
        ref: Medium.
        omega: Angular frequency.
        k1: Lateral wavenumber, along x1.

    Returns:
        (ev, rv, lv, dn) with rv columns the right eigenvectors, lv rows the left.
    """
    a4 = amat_psv(ref, omega, k1)
    bal, s = balance_batch(a4[None, :, :])
    ev, rvb = np.linalg.eig(bal)
    ev, rvb = ev[0], rvb[0]
    rv = rvb * s[0][:, None]
    lv = np.linalg.inv(rv)
    return ev, rv, lv, downgoing(ev)


def mode_index(
    ev: np.ndarray, dn: np.ndarray, ref: ReferenceMedium, omega: complex, k1: float, want: str
) -> int:
    """Index of the down- or up-going P or SV mode in the P-SV block.

    P and SV are told apart by their vertical wavenumbers, which are known
    analytically.  Inside the P-SV block there is no degeneracy to resolve, so
    this is a two-way choice with no tie to break.

    Args:
        ev: Eigenvalues, shape (4,).
        dn: Downgoing mask.
        ref: Medium.
        omega: Angular frequency.
        k1: Lateral wavenumber.
        want: One of 'Pd', 'Pu', 'Sd', 'Su'.

    Returns:
        The index.

    Raises:
        RuntimeError: if the up/down split does not give exactly two of each.
    """
    qp = np.sqrt(complex(k1**2 - (omega / ref.alpha) ** 2))
    qs = np.sqrt(complex(k1**2 - (omega / ref.beta) ** 2))
    target = qp if want[0] == "P" else qs
    mask = dn if want[1] == "d" else ~dn
    if int(np.sum(mask)) != 2:
        msg = f"expected two {'down' if want[1] == 'd' else 'up'}-going modes, got {int(np.sum(mask))}"
        raise RuntimeError(msg)
    cand = [i for i in range(4) if mask[i]]
    return min(cand, key=lambda i: abs(abs(ev[i]) - abs(target)))


def layer_reflection_matrix(
    ref: ReferenceMedium,
    lay: ReferenceMedium,
    h: float,
    omega: complex,
    p: float,
    ncell: int,
    zero_k_contrast: bool = False,
) -> np.ndarray:
    """R for a layer in a whole space, from the first-order contrast operator.

    Solves the laterally invariant Lippmann--Schwinger equation

        q(z) = q_inc(z) + int_0^h Gamma(k; z, z') DeltaA q(z') dz' ,

    by piecewise-constant collocation.  The cell integrals are done in CLOSED
    FORM from the spectral representation of Gamma, split at z' = z so that no
    cell integral straddles the jump in the kernel -- the jump is the source
    delta and must not be smeared by quadrature.

    Args:
        ref: Background medium.
        lay: Layer medium.
        h: Layer thickness.
        omega: Angular frequency.
        p: Horizontal slowness, so that k1 = omega p.
        ncell: Number of cells across the layer.
        incident: 'P' or 'S'.

    Returns:
        U/D in the u3 convention at z = 0.
    """
    k1 = float(np.real(omega) * p)
    ev, rv, lv, dn = spectral(ref, omega, k1)
    kc = 0.0 if zero_k_contrast else k1
    da = amat_psv(lay, omega, kc) - amat_psv(ref, omega, kc)

    edge = np.linspace(0.0, h, ncell + 1)
    zc = 0.5 * (edge[:-1] + edge[1:])
    proj = np.stack([np.outer(rv[:, i], lv[i, :]) for i in range(4)])

    def kernel_block(z: float, lo: float, hi: float) -> np.ndarray:
        """int_lo^hi Gamma(z - z') dz', exactly, split at z' = z."""
        out = np.zeros((4, 4), dtype=complex)
        for sel, sgn in ((dn, 1.0), (~dn, -1.0)):
            # the down-going part carries z' < z, the up-going part z' > z, and
            # the split is done here rather than by quadrature: the jump across
            # z' = z IS the source delta and must not be smeared
            a, b = (lo, min(hi, z)) if sgn > 0 else (max(lo, z), hi)
            if b <= a:
                continue
            for i in range(4):
                if not sel[i]:
                    continue
                e = ev[i]
                val = (b - a) if abs(e) < 1e-300 else (np.exp(e * (z - a)) - np.exp(e * (z - b))) / e
                out += sgn * val * proj[i]
        return out

    big = np.zeros((4 * ncell, 4 * ncell), dtype=complex)
    for m in range(ncell):
        for n in range(ncell):
            big[4 * m : 4 * m + 4, 4 * n : 4 * n + 4] = kernel_block(zc[m], edge[n], edge[n + 1]) @ da
    lhs = np.eye(4 * ncell, dtype=complex) - big

    # Each mode is normalised by a FIXED component -- P by v3, SV by v1 -- so the
    # amplitude has a definite meaning and a definite sign, both of which survive
    # to normal incidence where the other component vanishes.
    # In the reduced basis (-tau_13, -tau_33, v1, v3), P is normalised by v3 and
    # SV by v1 -- each by the component that survives at normal incidence.
    ncomp = {"P": 3, "S": 2}
    out = np.zeros((2, 2), dtype=complex)
    ztop = -0.25 * h if h > 0 else -1.0
    for jc, jn in enumerate(("P", "S")):
        idx = mode_index(ev, dn, ref, omega, k1, f"{jn}d")
        vin = rv[:, idx] / rv[ncomp[jn], idx]
        rhs = np.concatenate([vin * np.exp(ev[idx] * z) for z in zc])
        qs = np.linalg.solve(lhs, rhs).reshape(ncell, 4)
        sc = np.zeros(4, dtype=complex)
        for n in range(ncell):
            sc += kernel_block(ztop, edge[n], edge[n + 1]) @ da @ qs[n]
        for ic, iname in enumerate(("P", "S")):
            iu = mode_index(ev, dn, ref, omega, k1, f"{iname}u")
            cu = (lv[iu, :] @ sc) * np.exp(-ev[iu] * ztop)
            out[ic, jc] = cu * rv[ncomp[iname], iu]
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
    """The PP entry of the reflection matrix, in the u3 convention.

    Args:
        ref: Background medium.
        lay: Layer medium.
        h: Layer thickness.
        omega: Angular frequency.
        p: Horizontal slowness.
        ncell: Number of cells across the layer.
        zero_k_contrast: Build DeltaA at k = 0 -- a control, not physics.

    Returns:
        R_PP.
    """
    return complex(layer_reflection_matrix(ref, lay, h, omega, p, ncell, zero_k_contrast)[0, 0])


def layer_reflection_kennett(
    ref: ReferenceMedium, lay: ReferenceMedium, h: float, omega: float, p: float
) -> np.ndarray:
    """RD_psv for the same layer, from the validated Kennett recursion.

    Args:
        ref: Background medium.
        lay: Layer medium.
        h: Layer thickness.
        omega: Angular frequency.
        p: Horizontal slowness.

    Returns:
        The PP entry of RD_psv.
    """
    # Kennett needs a finite top layer but does NOT reference RD to the top of
    # it: RD is unchanged for d = 50, 100, 200, 400, so it already sits at the
    # bottom of layer 1, which is the top of the target layer.  Dividing out a
    # two-way phase here -- the natural guess -- manufactures a spurious
    # slowness-dependent offset.  Measured, not assumed.
    stack = LayerStack(
        [
            IsotropicLayer(ref.alpha, ref.beta, ref.rho, 100.0),
            IsotropicLayer(lay.alpha, lay.beta, lay.rho, h),
            IsotropicLayer(ref.alpha, ref.beta, ref.rho, np.inf),
        ]
    )
    res = kennett_layers(stack, p, np.array([omega]))
    return np.asarray(res.RD_psv[0], dtype=complex)


def main() -> int:
    """Run the layer gate.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  A layer in a whole space: the first-order system against Kennett")
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
    print("--- 3: oblique incidence, the full 2x2, against Kennett -----------")
    print("    Kennett's RD is referenced at the top of the target layer -- it is")
    print("    unchanged for d = 50, 100, 200, 400 -- so no phase is applied.")
    print("    The two conventions differ by a fixed matrix; what is CHECKED is")
    print("    that it is fixed, since a convention cannot depend on angle.")
    ratios = []
    ps = (0.0, 5e-5, 1.0e-4, 1.5e-4)
    for p in ps:
        rmat = layer_reflection_matrix(ref, lay, h, omega, p, 120)
        ken = layer_reflection_kennett(ref, lay, h, omega, p)
        rat = np.where(np.abs(ken) > 1e-12, rmat / np.where(np.abs(ken) > 1e-12, ken, 1.0), np.nan)
        ratios.append(rat)
        print(f"    p={p:8.2e} (sin i={p * ref.alpha:.3f})")
        print(
            f"        first-order PP {rmat[0, 0].real:+.6f}{rmat[0, 0].imag:+.6f}i"
            f"   PS {rmat[0, 1].real:+.6f}{rmat[0, 1].imag:+.6f}i"
        )
        print(
            f"        kennett     PP {ken[0, 0].real:+.6f}{ken[0, 0].imag:+.6f}i"
            f"   PS {ken[0, 1].real:+.6f}{ken[0, 1].imag:+.6f}i"
        )
        print(
            f"        ratio       PP {rat[0, 0]:+.5f}   SS {rat[1, 1]:+.5f}"
            f"   PS {rat[0, 1]:+.5f}   SP {rat[1, 0]:+.5f}"
        )

    diag = np.array([[r[0, 0], r[1, 1]] for r in ratios])
    spread = float(np.max(np.abs(diag - diag[0])))
    print(f"    spread of the diagonal ratios across slowness = {spread:.3e}")
    report("the PP and SS conventions are constant across slowness", spread < 1e-3)
    off = np.abs(diag[0] + 1.0)
    print(f"    the constant is {diag[0][0]:+.6f} (PP), {diag[0][1]:+.6f} (SS)")
    report("that constant is exactly -1, the sign pinned by the closed form", float(np.max(off)) < 1e-3)

    # If the two conventions differ by a diagonal rescaling D = diag(d_P, d_S),
    # then ratio_PS = d_P/d_S and ratio_SP = d_S/d_P, so their PRODUCT is one
    # whatever D is.  That is a statement about the off-diagonals -- the
    # mode-converted coefficients -- and it holds without knowing D.
    prods = [r[0, 1] * r[1, 0] for r in ratios[1:]]
    worst = max(abs(x - 1.0) for x in prods)
    print(f"    PS x SP = {', '.join(f'{x.real:+.5f}' for x in prods)}")
    print(f"    worst departure from unity = {worst:.3e}")
    report("the mode-converted coefficients agree up to a diagonal rescaling", worst < 1e-3)
    print(f"    the measured d_P/d_S = {', '.join(f'{r[0, 1]:+.4f}' for r in ratios[1:])}")

    print("")
    print("--- 4: the k dependence of DeltaA is load-bearing -----------------")
    print("    Building DeltaA at k = 0 while leaving Gamma at the true k must")
    print("    change nothing at normal incidence and must break everything off it.")
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
