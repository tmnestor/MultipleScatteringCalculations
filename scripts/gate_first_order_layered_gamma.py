#!/usr/bin/env python3
"""Gamma for a LAYERED background, at arbitrary depth, gated against the whole space.

ANCHOR: Nestor (1996) Ch.2 (Akdef), (specA); Ch.5 GstratRep.tex (PstratDef).

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT BUILD
-------------------------------------------------------
The first-order contrast operator needs a background Green's matrix Gamma.  So
far that has been the whole-space one, built from the thesis spectral
representation.  The layered background is the payoff: there is no closed-form
whole-space G to fall back on, so the moment route is the only route.

The stratified 6x6 already exists, is validated, and is already bridged to this
A: ``GlobalMatrix.layered_greens`` builds it by a two-pass block-Riccati sweep,
``cubic_scattering.layered_correction`` wraps it with the three corrections, and
``gate_first_order_propagator_bridge`` measures that the corrected object
satisfies ``Gamma(z2) = expm(A dz) Gamma(z1)``.  Nothing here rebuilds any of it.

A first attempt DID rebuild it, as a naive propagator-matrix stack traversal.
It reduced correctly in the no-contrast limit, to 7.4e-15 -- and was unusable:
cond(Y_up) reached 9.0e15, the Thomson--Haskell instability, because a
propagator-matrix sweep forms the growing evanescent branch explicitly.  The
Riccati sweep never forms it.  That build was discarded rather than patched.

Two things genuinely were missing, and are supplied here:

  1. ARBITRARY DEPTH.  The Riccati object is indexed by INTERFACE, and the
     moment integral needs Gamma(z, z') at continuously varying z, z'.  The
     stable way to get it is not to propagate away from an interface -- that is
     the discarded route again -- but to CUT the model at the depths wanted,
     with zero contrast across the cut.  The sweep is unchanged and still never
     forms the growing branch.  Part 2 measures that the cut is harmless.

  2. THE SOURCE NORMALISATION.  The bridge tests a RATIO of two receiver depths,
     in which the source-side operator cancels; it therefore says nothing about
     how Gamma is normalised at the source.  Part 1 measures the jump directly
     and finds it is exactly I_6, so ``[d_z - A] Gamma = I_6 delta(z - z')``
     holds with no calibration constant anywhere.  That closes the 6x6 half of
     the source-convention question the bridge left open.

THE UNIFORM LIMIT, STATED SHARPLY
---------------------------------
The plan asked for "the uniform limit reduces to the whole space".  It cannot
reduce exactly: layer 0 of the stratified solver is acoustic, so a uniform
ELASTIC whole space is not reachable -- the ocean floor is always there and
always reflects.  What CAN be said is stronger and is parameter-free.  Whatever
the ocean floor sends back is, at any receiver, DOWNGOING: it left the source
upward, turned once, and is heading down.  So

    Gamma_layered - Gamma_wholespace   is purely downgoing,

with the split taken from the column ordering of (eigDef) and no numerical
classification.  Part 5 measures it at 1e-16, for receivers both below and
ABOVE the source.  Beyond the critical wavenumber the returned field is
evanescent over the whole round trip and the difference vanishes outright, so
there the layered Gamma IS the whole-space Gamma, to the arithmetic floor.
Part 6 puts a reflector BELOW instead, which sends an UPGOING field back, and
the projection fails by fourteen orders -- so Part 5 is not vacuous.

Units are km, km/s, g/cm^3, rad/km, rad/s throughout, as in the sibling repo.

Run:  conda run -n seismic python scripts/gate_first_order_layered_gamma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.layered_correction import correct_6x6  # noqa: E402
from GlobalMatrix.layered_greens import layered_greens_6x6  # noqa: E402
from Kennett_Reflectivity.layer_model import LayerModel  # noqa: E402
from scripts.gate_first_order_propagator_bridge import system_matrix  # noqa: E402
from scripts.gate_thesis_spectral import J6, dz_normalised, gamma_thesis  # noqa: E402

MED = (4.0, 2.22, 2.6)
SLAB = (6.5, 3.7, 3.3)
OCEAN = (1.5, 0.0, 1.03)
OCEAN_H = 3.0
Q = 600.0
OMEGA = 2.0 * np.pi * 2.0

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def stack(cuts: list[float], slab: tuple[int, int] | None = None) -> LayerModel:
    """Ocean over an elastic medium cut at the given depths below the floor.

    Interface k is the bottom of layer k, so interface 0 is the ocean floor and
    interface j is at depth ``OCEAN_H + cuts[j-1]``.  The last layer is a
    radiating half-space, which is what makes any upgoing field in Part 5 have
    to have come from the ocean floor.

    Args:
        cuts: Depths below the ocean floor, increasing, where interfaces sit.
        slab: Optional (first, last) elastic layer indices given SLAB material,
            used only as the negative control of Part 6.

    Returns:
        The layered model.
    """
    th = [OCEAN_H, *np.diff([0.0, *cuts]).tolist(), np.inf]
    n = len(th)
    al = [OCEAN[0]] + [MED[0]] * (n - 1)
    be = [OCEAN[1]] + [MED[1]] * (n - 1)
    rh = [OCEAN[2]] + [MED[2]] * (n - 1)
    if slab is not None:
        for j in range(slab[0], slab[1] + 1):
            al[j], be[j], rh[j] = SLAB
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=th,
        Q_alpha=[Q] * n,
        Q_beta=[1e10, *([Q] * (n - 1))],
    )


def gamma_layered(model: LayerModel, kx: float, ky: float, src: int, rcv: int) -> np.ndarray:
    """The corrected stratified Gamma at one (source, receiver) interface pair.

    This is the whole of the new object: the validated Riccati Green's matrix,
    wrapped by the validated corrections, read in the thesis basis.  Arbitrary
    depth comes from where the model is CUT, not from anything done here.

    Args:
        model: Stratified model.
        kx: Lateral wavenumber, x (rad/km).
        ky: Lateral wavenumber, y (rad/km).
        src: Source interface index.
        rcv: Receiver interface index.

    Returns:
        Shape (6, 6) complex in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).
    """
    raw = layered_greens_6x6(
        model, OMEGA, np.array([kx]), np.array([ky]), source_iface=src, receiver_iface=rcv
    )[0]
    s_s = model.complex_slowness_s()
    return np.asarray(
        correct_6x6(raw, OMEGA, s_s[max(src, 1)], s_s[max(rcv, 1)], kx, ky), dtype=np.complex128
    )


def medium_of(model: LayerModel, layer: int = 1) -> ReferenceMedium:
    """The reference medium of one layer, with its COMPLEX velocities.

    The comparison against the whole-space Gamma is only exact if both carry the
    same attenuation, so the complex slownesses are used rather than the nominal
    real velocities.  ``ReferenceMedium`` is a plain dataclass whose derived
    lam and mu are arithmetic, so complex arguments work; the annotation says
    float because the package's own use of it is lossless.

    Args:
        model: Stratified model.
        layer: Layer index.

    Returns:
        The medium.
    """
    return ReferenceMedium(
        1.0 / model.complex_slowness_p()[layer],  # type: ignore[arg-type]
        1.0 / model.complex_slowness_s()[layer],  # type: ignore[arg-type]
        float(model.rho[layer]),
    )


def updown(ref: ReferenceMedium, kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
    """The downgoing and upgoing spectral projectors of A.

    Built from the analytic eigenvectors with the split taken from the COLUMN
    ORDERING of (eigDef) -- the first three columns are downgoing -- so no
    numerical quantity's sign is consulted.

    Args:
        ref: Medium.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        (P_down, P_up), each shape (6, 6).
    """
    dzm, inv, _ = dz_normalised(ref, OMEGA, kx, ky)
    return dzm[:, :3] @ inv[:3, :], dzm[:, 3:] @ inv[3:, :]


def main() -> int:
    """Gate the layered Gamma.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  Gamma for a layered background -- Task 6")
    print("=" * 74)
    kb = OMEGA / MED[1]
    print(f"    omega = {OMEGA:.6f} rad/s,  S branch radius omega/beta = {kb:.4f} rad/km")

    print("")
    print("--- 1: the source jump is exactly I_6 -----------------------------")
    print("    The bridge tests a RATIO of receiver depths, so the source-side")
    print("    operator cancels out of it and the normalisation is untested.")
    print("    Here both receivers are propagated back ONTO the source plane")
    print("    with expm(A dz) and subtracted, which tests it directly.")
    print(f"    {'delta (km)':>12} {'| jump - I_6 |':>16} {'| jump |':>12}")
    worst_jump = 0.0
    for delta in (0.4, 0.2, 0.1, 0.05):
        mod = stack([2.0, 4.0 - delta, 4.0, 4.0 + delta, 6.0])
        amat = system_matrix(mod, 1, OMEGA, 0.7, 0.3)
        above = expm(amat * delta) @ gamma_layered(mod, 0.7, 0.3, 3, 2)
        below = expm(-amat * delta) @ gamma_layered(mod, 0.7, 0.3, 3, 4)
        err = float(np.max(np.abs(below - above - np.eye(6))))
        worst_jump = max(worst_jump, err)
        print(f"    {delta:12.3f} {err:16.3e} {np.max(np.abs(below - above)):12.3e}")
    report("[d_z - A] Gamma = I_6 delta(z-z'), with no calibration constant", worst_jump < 1e-11)

    print("")
    print("--- 2: cutting the model at a depth is harmless --------------------")
    print("    This is how arbitrary depth is reached.  A cut with zero contrast")
    print("    across it must not change Gamma between two FIXED physical")
    print("    depths -- and the Riccati sweep still never forms the growing")
    print("    evanescent branch, which is what the discarded route did.")
    worst_cut = 0.0
    for kx in (0.4, 2.0, 5.0, 9.0):
        coarse = gamma_layered(stack([4.0, 9.0]), kx, 0.0, 1, 2)
        extra = [4.0, 5.3, 6.1, 7.77, 8.4, 9.0]
        fine = gamma_layered(stack(extra), kx, 0.0, 1, len(extra))
        rel = float(np.max(np.abs(fine - coarse)) / np.max(np.abs(coarse)))
        worst_cut = max(worst_cut, rel)
        print(f"    kx = {kx:5.2f}   2 cuts vs 6 cuts, same two depths: {rel:10.3e}")
    report("zero-contrast cuts leave Gamma unchanged, so any depth is reachable", worst_cut < 1e-11)

    print("")
    print("--- 3: the bridge identity, now at ARBITRARY depths ----------------")
    print("    The existing bridge holds it at interfaces of a fixed stack.  The")
    print("    cuts here are at depths chosen to be on no lattice at all.")
    zs = [1.0, 2.34, 3.61, 5.07, 6.0]
    mod = stack(zs)
    worst_br = 0.0
    for kx, ky in ((0.4, 0.0), (1.3, 0.9), (3.0, -1.1), (7.0, 2.0)):
        amat = system_matrix(mod, 1, OMEGA, kx, ky)
        for lo in (2, 3):
            dz = zs[lo] - zs[lo - 1]
            lower = gamma_layered(mod, kx, ky, 1, lo + 1)
            upper = gamma_layered(mod, kx, ky, 1, lo)
            prop = expm(amat * dz)
            # Normalised by the amplification the propagator itself applies:
            # beyond critical it carries a growing branch, and no correct
            # arithmetic can do better than ||prop|| times the rounding in
            # ``upper``.  See the bridge gate for the same normalisation.
            floor = float(np.max(np.abs(prop)) * np.max(np.abs(upper)) * 2.2e-16)
            resid = float(np.max(np.abs(lower - prop @ upper)) / floor)
            worst_br = max(worst_br, resid)
            print(f"    k = ({kx:5.2f},{ky:5.2f})  dz = {dz:5.2f}   residual = {resid:9.2f} floors")
    report("Gamma(z2) = expm(A dz) Gamma(z1) at depths on no lattice", worst_br < 3e2)

    print("")
    print("--- 4: reciprocity of the layered Gamma ----------------------------")
    print("    Gamma(-k; z', z)^T = J6 Gamma(k; z, z') J6, which exchanges the")
    print("    source and receiver interfaces as well as reversing k.")
    mod = stack([2.0, 5.0, 9.0])
    worst_rec = 0.0
    for kx, ky in ((0.5, 0.0), (1.7, 1.1), (4.0, -2.0), (8.0, 3.0)):
        fwd = gamma_layered(mod, kx, ky, 1, 3)
        bwd = gamma_layered(mod, -kx, -ky, 3, 1)
        rel = float(np.max(np.abs(bwd.T - J6 @ fwd @ J6)) / np.max(np.abs(fwd)))
        worst_rec = max(worst_rec, rel)
        print(f"    k = ({kx:5.2f},{ky:5.2f})   {rel:10.3e}")
    report("the stratified Gamma is reciprocal in the thesis basis", worst_rec < 1e-10)

    print("")
    print("--- 5: the uniform limit, named exactly ----------------------------")
    print("    A uniform ELASTIC whole space is not reachable: layer 0 of the")
    print("    solver is acoustic, so the ocean floor is always there.  But what")
    print("    it returns is DOWNGOING at every receiver -- it went up, turned")
    print("    once, and is coming down -- and the half-space below radiates, so")
    print("    nothing else can contribute.  Hence the difference from the")
    print("    whole-space Gamma is purely downgoing, with no free parameter.")
    mod = stack([10.0, 14.0, 18.0])
    ref = medium_of(mod)
    print(f"    {'kx':>6} {'dz':>7} {'|D|/|Gamma|':>13} {'|P_up D|/|D|':>14}")
    worst_up, seen_big, worst_evan = 0.0, 0.0, 0.0
    for kx in (0.4, 1.2, 2.5, 4.0, 5.0):
        _, pup = updown(ref, kx, 0.0)
        for rcv, dz in ((2, 4.0), (3, 8.0), (0, -10.0)):
            diff = gamma_layered(mod, kx, 0.0, 1, rcv) - gamma_thesis(ref, OMEGA, kx, 0.0, dz)
            nd = float(np.max(np.abs(diff)))
            rel = nd / float(np.max(np.abs(gamma_thesis(ref, OMEGA, kx, 0.0, dz))))
            up = float(np.max(np.abs(pup @ diff))) / nd
            worst_up, seen_big = max(worst_up, up), max(seen_big, rel)
            print(f"    {kx:6.2f} {dz:7.1f} {rel:13.3e} {up:14.3e}")
    report("Gamma_layered - Gamma_wholespace is purely DOWNGOING", worst_up < 1e-12)
    report("and is not small, so the projection is not trivially satisfied", seen_big > 0.1)
    print("")
    print("    Beyond the critical wavenumber the returned field is evanescent")
    print("    over the whole round trip, so the difference vanishes outright")
    print("    and the layered Gamma IS the whole-space Gamma:")
    for kx in (6.0, 7.0, 9.0):
        for rcv, dz in ((2, 4.0), (3, 8.0)):
            tt = gamma_thesis(ref, OMEGA, kx, 0.0, dz)
            rel = float(np.max(np.abs(gamma_layered(mod, kx, 0.0, 1, rcv) - tt)))
            rel /= float(np.max(np.abs(tt)))
            worst_evan = max(worst_evan, rel)
            print(f"    {kx:6.2f} {dz:7.1f} {rel:13.3e}")
    report("at kx > omega/beta the layered Gamma reduces to the whole space", worst_evan < 1e-12)

    print("")
    print("--- 6: negative control -- a reflector BELOW instead ----------------")
    print("    Same geometry, but the contrast is now under the receiver, so it")
    print("    returns an UPGOING field.  Part 5's projection must fail.")
    modc = stack([10.0, 14.0, 18.0], slab=(4, 4))
    best_up = 0.0
    for kx in (0.4, 1.2, 2.5):
        _, pup = updown(ref, kx, 0.0)
        diff = gamma_layered(modc, kx, 0.0, 1, 2) - gamma_thesis(ref, OMEGA, kx, 0.0, 4.0)
        up = float(np.max(np.abs(pup @ diff)) / np.max(np.abs(diff)))
        best_up = max(best_up, up)
        print(f"    kx = {kx:5.2f}   | P_up D | / | D | = {up:10.3e}")
    report("a reflector below puts an upgoing part back, by 12+ orders", best_up > 1e-2)

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
