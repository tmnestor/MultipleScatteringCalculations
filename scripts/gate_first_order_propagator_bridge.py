#!/usr/bin/env python3
"""Is ``corrected_layered_6x6`` the Green's matrix of the derived system matrix A?

Why this gate exists
--------------------
``Mathematica/MatrixVectorWaveEquation.wl`` derived the operator matrix ``A`` of
the first-order system ``d3 q = A q + d`` from the stiffness tensor, and checked
it against two independent objects (Appendix J of the paper, and the thesis
``Akdef``).  ``Mathematica/FirstOrderContrastOperator.wl`` then derived the
contrast operator ``DeltaA`` and its augmentation.

Everything downstream needs the Green's matrix of that system.  The survey found
it already exists and is validated: ``GlobalMatrix.layered_greens`` builds it and
``cubic_scattering.layered_correction`` wraps it with the three corrections, in
the basis ``(u_z, u_x, u_y, T_zz, T_xz, T_yz)`` -- which is exactly the basis the
thesis writes ``A`` in.  Nothing new needs to be constructed.

But "the same basis" has been assumed, not measured.  This gate measures it.

The identity
------------
Away from the source plane the Green's matrix solves the HOMOGENEOUS system in
its receiver coordinate, so for two receiver depths in one uniform layer

    Gamma(z2, z') = expm( A * (z2 - z1) ) . Gamma(z1, z')

with no free parameters.  The identity is LOCAL: it holds however complicated the
medium between the source and z1 is, because the ODE knows only about the layer
the receiver sits in.  The gate therefore places the source ABOVE a fast slab and
the receivers BELOW it, so transmission, reflection, conversion and interbed
multiples are all present in Gamma and the identity must still hold exactly.

Raw versus corrected
--------------------
Both are measured, because which one is the physical state vector is the open
question.  ``correct_6x6`` applies a receiver-side operator W_r and a source-side
operator W_s.  W_s cancels out of the ratio, but W_r does not: if the RAW object
propagates by ``expm(A dz)`` then the corrected one propagates by
``W_r expm(A dz) W_r^-1``, and vice versa.  Only one of them can satisfy the
identity with the A written in the physical basis, so the gate names which.

This bears directly on the ungated 9-component source-convention question.

Run:  conda run -n seismic python scripts/gate_first_order_propagator_bridge.py
"""

import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.layered_correction import correct_6x6  # isort: skip

from GlobalMatrix.layered_greens import layered_greens_6x6  # isort: skip
from Kennett_Reflectivity.layer_model import LayerModel  # isort: skip

# Residuals are reported in units of the arithmetic floor (see bridge_residual),
# so the threshold is a small multiple of 1, not an absolute size.
TOL = 1.0e3
CTRL_MIN = 1.0e6

A_MED = (4.0, 2.22, 2.6)
B_MED = (6.5, 3.7, 3.3)

# Interface k lies at the bottom of layer k.  Layers 6-9 are medium A, so the
# three receiver interfaces below are separated by exactly one layer of A each.
SRC_IFACE = 2
RCV_IFACES = (6, 7, 8)
DZ = 1.0
MED_LAYER = 7  # the layer between receiver interfaces 6 and 7


def sandwich_model(q: float = 1000.0) -> LayerModel:
    """ocean | A A A | B B | A A A A, interface k at the bottom of layer k.

    Reused verbatim from ``gate_stratified_correction.py`` so that the bridge is
    measured on a model whose corrected 6x6 is already validated.

    Args:
        q: Quality factor; high, so the slab contrast really reflects.

    Returns:
        The layered model.
    """
    al = [1.5] + [A_MED[0]] * 3 + [B_MED[0]] * 2 + [A_MED[0]] * 4
    be = [0.0] + [A_MED[1]] * 3 + [B_MED[1]] * 2 + [A_MED[1]] * 4
    rh = [1.03] + [A_MED[2]] * 3 + [B_MED[2]] * 2 + [A_MED[2]] * 4
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([1.0] * 8), np.inf],
        Q_alpha=[q] * 10,
        Q_beta=[1e10, *([q] * 9)],
    )


def system_matrix(model: LayerModel, layer: int, omega: float, kx: float, ky: float) -> np.ndarray:
    """The 6x6 system matrix A(kx, ky) in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).

    Transcribed from the thesis eq. ``Akdef`` (GRepresentations.tex), which
    ``Mathematica/MatrixVectorWaveEquation.wl`` proved equal to the operator
    matrix of the paper's Appendix J under one constant similarity transform.
    The shorthand is local to that equation:

        gamma = lam/(lam+2mu),  a = 1/(lam+2mu),  b = 1/mu,
        zeta  = 4 mu (lam+mu)/(lam+2mu) = nu1,
        chi   = 2 mu lam/(lam+2mu)      = nu2,   with zeta - chi = 2 mu.

    The moduli are built from the model's COMPLEX velocities, so the comparison
    is exact rather than approximate at finite Q.

    Args:
        model: Stratified model.
        layer: Layer index supplying the medium.
        omega: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).

    Returns:
        Shape (6, 6) complex.
    """
    alpha_c = 1.0 / model.complex_slowness_p()[layer]
    beta_c = 1.0 / model.complex_slowness_s()[layer]
    rho = float(model.rho[layer])

    mu = rho * beta_c**2
    lam = rho * alpha_c**2 - 2.0 * mu
    kc = lam + 2.0 * mu

    gam = lam / kc
    a = 1.0 / kc
    b = 1.0 / mu
    zet = 4.0 * mu * (lam + mu) / kc
    chi = 2.0 * mu * lam / kc

    rw2 = rho * omega**2
    j = 1j
    return np.array(
        [
            [0.0, -j * gam * kx, -j * gam * ky, a, 0.0, 0.0],
            [-j * kx, 0.0, 0.0, 0.0, b, 0.0],
            [-j * ky, 0.0, 0.0, 0.0, 0.0, b],
            [-rw2, 0.0, 0.0, 0.0, -j * kx, -j * ky],
            [0.0, -rw2 + zet * kx**2 + mu * ky**2, kx * ky * (chi + mu), -j * kx * gam, 0.0, 0.0],
            [0.0, kx * ky * (chi + mu), -rw2 + zet * ky**2 + mu * kx**2, -j * ky * gam, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )


def greens(
    model: LayerModel, omega: float, kx: float, ky: float, rcv: int, *, corrected: bool
) -> np.ndarray:
    """Green's matrix at one receiver interface, raw or corrected.

    Args:
        model: Stratified model.
        omega: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        rcv: Receiver interface index.
        corrected: Apply the D1/D2/D3 wrapper corrections.

    Returns:
        Shape (6, 6) complex in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).
    """
    raw = layered_greens_6x6(
        model,
        omega,
        np.array([kx]),
        np.array([ky]),
        source_iface=SRC_IFACE,
        receiver_iface=rcv,
    )[0]
    if not corrected:
        return np.asarray(raw, dtype=np.complex128)
    s_s = model.complex_slowness_s()
    return np.asarray(
        correct_6x6(raw, omega, s_s[max(SRC_IFACE, 1)], s_s[max(rcv, 1)], kx, ky),
        dtype=np.complex128,
    )


def bridge_residual(
    model: LayerModel, omega: float, kx: float, ky: float, amat: np.ndarray, *, corrected: bool
) -> float:
    """Worst residual of Gamma(z_{n+1}) = expm(A dz) Gamma(z_n), in units of the
    arithmetic floor.

    A bare relative residual is not comparable across the table.  Beyond the
    critical slowness the propagator carries both a growing and a decaying
    evanescent branch, so ``expm(A dz)`` has dynamic range ~exp(|k| dz) -- 1.6e7
    at the widest angle here.  Forming ``prop @ lower`` then amplifies the
    rounding already present in ``lower`` by ``||prop||``, and no correct
    implementation can do better than

        floor = eps * ||prop|| * ||lower|| / ||upper|| .

    Reporting residual/floor makes every row comparable: O(1) means the identity
    holds to the last bit available, whatever the conditioning.  This is the
    absolute companion the project requires alongside any ratio test.

    Args:
        model: Stratified model.
        omega: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        amat: Candidate system matrix.
        corrected: Measure the corrected object rather than the raw one.

    Returns:
        Worst residual/floor over the consecutive receiver pairs.
    """
    eps = float(np.finfo(np.float64).eps)
    prop = expm(amat * DZ)
    pnorm = float(la.norm(prop))
    gam = [greens(model, omega, kx, ky, r, corrected=corrected) for r in RCV_IFACES]
    worst = 0.0
    for lower, upper in zip(gam[:-1], gam[1:], strict=True):
        denom = la.norm(upper)
        if denom == 0.0:
            return float("inf")
        floor = eps * pnorm * float(la.norm(lower)) / float(denom)
        worst = max(worst, float(la.norm(upper - prop @ lower) / denom) / floor)
    return worst


def main() -> int:
    """Run the bridge gate over several (kx, ky), plus a negative control.

    Returns:
        0 if the bridge holds in exactly one of the two bases, 1 otherwise.
    """
    model = sandwich_model()
    omega = 2.0 * np.pi * 5.0

    # propagating, non-axial; propagating, near-axial; evanescent for S
    cases = [(2.0, 1.3), (0.4, 0.05), (14.0, 9.0)]

    print("=" * 72)
    print("  Gamma(z2) = expm(A dz) Gamma(z1)   --- source above a fast slab,")
    print("  receivers below it, so all multiples are present in Gamma.")
    print(f"  omega = {omega:.4f} rad/s    dz = {DZ} km    medium A layer {MED_LAYER}")
    print("=" * 72)
    print("  residuals are in units of the arithmetic floor; O(1) = exact")
    print(f"{'kx':>7} {'ky':>7} {'|k|/omega':>10} {'raw':>13} {'corrected':>13}")

    raw_worst = 0.0
    cor_worst = 0.0
    for kx, ky in cases:
        amat = system_matrix(model, MED_LAYER, omega, kx, ky)
        r_raw = bridge_residual(model, omega, kx, ky, amat, corrected=False)
        r_cor = bridge_residual(model, omega, kx, ky, amat, corrected=True)
        raw_worst = max(raw_worst, r_raw)
        cor_worst = max(cor_worst, r_cor)
        p = np.hypot(kx, ky) / omega
        print(f"{kx:7.2f} {ky:7.2f} {p:10.4f} {r_raw:13.3e} {r_cor:13.3e}")

    print("-" * 72)
    print(f"{'worst':>26} {raw_worst:13.3e} {cor_worst:13.3e}")

    # Negative control: swap zeta and chi.  Both are in-plane stiffnesses that
    # remain after e_zz is eliminated and they differ by exactly 2 mu, so the
    # swap is a physically plausible transcription slip that no symmetry check
    # would catch.  It must break the bridge.
    kx, ky = cases[0]
    good = system_matrix(model, MED_LAYER, omega, kx, ky)
    bad = good.copy()
    alpha_c = 1.0 / model.complex_slowness_p()[MED_LAYER]
    beta_c = 1.0 / model.complex_slowness_s()[MED_LAYER]
    rho = float(model.rho[MED_LAYER])
    mu = rho * beta_c**2
    lam = rho * alpha_c**2 - 2.0 * mu
    kc = lam + 2.0 * mu
    zet, chi = 4.0 * mu * (lam + mu) / kc, 2.0 * mu * lam / kc
    rw2 = rho * omega**2
    bad[4, 1] = -rw2 + chi * kx**2 + mu * ky**2
    bad[5, 2] = -rw2 + chi * ky**2 + mu * kx**2
    bad[4, 2] = bad[5, 1] = kx * ky * (zet + mu)
    ctrl_raw = bridge_residual(model, omega, kx, ky, bad, corrected=False)
    ctrl_cor = bridge_residual(model, omega, kx, ky, bad, corrected=True)
    print(f"{'control (zeta<->chi)':>26} {ctrl_raw:13.3e} {ctrl_cor:13.3e}")

    print("=" * 72)
    raw_ok = raw_worst < TOL
    cor_ok = cor_worst < TOL
    ctrl_ok = min(ctrl_raw, ctrl_cor) > CTRL_MIN

    for label, ok in (
        ("raw layered_greens_6x6 propagates by expm(A dz)", raw_ok),
        ("corrected_layered_6x6 propagates by expm(A dz)", cor_ok),
        ("negative control (zeta<->chi swap) is rejected", ctrl_ok),
    ):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    if raw_ok == cor_ok:
        print("\n  INCONCLUSIVE: the two bases are not separated by this test.")
        return 1
    which = "RAW" if raw_ok else "CORRECTED"
    print(f"\n  The physical state vector of d3 q = A q is the {which} object.")
    return 0 if ctrl_ok else 1


if __name__ == "__main__":
    sys.exit(main())
