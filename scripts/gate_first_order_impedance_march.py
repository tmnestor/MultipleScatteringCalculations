#!/usr/bin/env python3
"""March the impedance through a slab, and score it against Kennett.

ANCHOR: Nestor (1996) Ch.2 (Akdef), (eigDef), (kzcDef); Ch.5 Algorithm 5.1.

WHAT THIS IS
------------
``gate_first_order_impedance`` built the impedance Y and showed it is far better
conditioned than the mode-basis coefficients V.  But it only ever solved the
ALGEBRAIC Riccati equation, in a homogeneous medium.  This gate does the thing
the whole construction exists for: it MARCHES

    Y' = A21 + A22 Y - Y A11 - Y A12 Y                                  (*)

through a slab and reads a reflection matrix off the result, then scores that
against objects this note does not own -- a closed form at normal incidence, and
the Kennett recursion at angle.

THE MARCH
---------
Start below the slab with the downgoing impedance of the background half-space:
that is the radiation condition, the continuum form of the sibling repository's
``Y[M+1] = 0``.  Integrate (*) upward to the top of the slab.  Then match to
up- and downgoing modes in the background above, using continuity of u and
tau_3 and nothing else:

    (T_u - Y U_u) a_u = -(T_d - Y U_d) a_d,
    R = -(T_u - Y U_u)^-1 (T_d - Y U_d).

Two integrators are carried, because each checks the other:

  * MOBIUS.  For a piecewise-constant medium the step is EXACT:
    Y_top = (M21 + M22 Y_bot)(M11 + M12 Y_bot)^-1 with M = expm(-A h).  One
    step is exact for a homogeneous slab.  It does form expm, which carries the
    growing evanescent branch, so it is expected to have a range.
  * RK4 on (*) directly.  Y stays bounded, so no growing exponential is ever
    formed, but the step is limited by the decay rate 2|k_z| of
    ``gate_first_order_riccati_blocks``.

They are independent, and Part 2 measures where each is trustworthy rather than
assuming either.

WHY THE GRADED SLAB IS THE POINT
--------------------------------
A homogeneous slab has a closed form and needs no march.  A slab whose contrast
VARIES with depth does not, and that is what a march buys.  Part 5 grades the
slab and scores against a Kennett stack cut at the same depths -- where the two
are solving the identical piecewise-constant problem, so they must agree to the
ARITHMETIC FLOOR at finite step count, not merely converge together.

Units are SI throughout -- m, m/s, kg/m^3, rad/m, rad/s -- matching
``gate_first_order_layer_vs_kennett``, whose arbiter and amplitude convention
are reused so the two routes can be compared directly.

Run:  conda run -n seismic python scripts/gate_first_order_impedance_march.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack, kennett_layers  # noqa: E402
from scripts.gate_first_order_impedance import ablocks, newton_are  # noqa: E402
from scripts.gate_first_order_layer_vs_kennett import (  # noqa: E402
    IDX,
    NORM,
    normal_incidence_layer,
)
from scripts.gate_thesis_spectral import dz_normalised  # noqa: E402

#: The thesis-to-Kennett amplitude convention, per channel.  P-SV differs by a
#: sign and SH does NOT -- confirmed independently by the collocation route of
#: ``gate_first_order_layer_vs_kennett``, which reports
#: (-1.000042, -1.000166, +1.000166) at 120 cells.  The sign split is a real
#: feature of the two conventions, not an error, and lumping the three together
#: is what made this gate's first run report a spurious 2.000.
CONV = np.array([-1.0, -1.0, +1.0])

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
LAY = ReferenceMedium(5500.0, 3300.0, 2750.0)
OMEGA = 60.0
H = 200.0

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: Description.
        ok: Whether it passed.
    """
    _PASS.append((label, bool(ok)))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def mode_columns(med: ReferenceMedium, omega: complex, kx: float) -> tuple[np.ndarray, np.ndarray]:
    """The down- and upgoing mode columns, in the displacement-component convention.

    Each column is divided by its own displacement component -- u_z for P, u_x
    for S, u_y for H -- which is the amplitude convention
    ``gate_first_order_layer_vs_kennett`` uses.  Reusing it is what lets the two
    routes be compared against Kennett on the same footing.

    Args:
        med: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber.

    Returns:
        (D_down, D_up), each shape (6, 3).
    """
    dzm, _, _ = dz_normalised(med, omega, kx, 0.0)
    out = []
    for suffix in ("d", "u"):
        cols = [dzm[:, IDX[f"{n}{suffix}"]] / dzm[NORM[n], IDX[f"{n}{suffix}"]] for n in "PSH"]
        out.append(np.stack(cols, axis=-1))
    return out[0], out[1]


def y_downgoing(med: ReferenceMedium, omega: complex, kx: float) -> np.ndarray:
    """The downgoing half-space impedance: D_z names the branch, Newton refines it.

    Args:
        med: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber.

    Returns:
        Shape (3, 3) complex.
    """
    dd, _ = mode_columns(med, omega, kx)
    y0 = dd[3:, :] @ np.linalg.inv(dd[:3, :])
    return newton_are(y0, ablocks(med, omega, kx, 0.0))[0]


def step_mobius(y: np.ndarray, blk: tuple, h: float) -> np.ndarray:
    """One EXACT step up through a uniform sublayer of thickness h.

    Args:
        y: Impedance at the bottom of the sublayer.
        blk: The four blocks of A there.
        h: Sublayer thickness, positive.

    Returns:
        The impedance at the top.
    """
    a11, a12, a21, a22 = blk
    amat = np.block([[a11, a12], [a21, a22]])
    m = expm(-amat * h)
    num = m[3:, :3] + m[3:, 3:] @ y
    den = m[:3, :3] + m[:3, 3:] @ y
    return np.asarray(num @ np.linalg.inv(den))


def step_rk4(y: np.ndarray, blk: tuple, h: float, nsub: int = 1) -> np.ndarray:
    """One step up through a uniform sublayer, by RK4 on (*) directly.

    Args:
        y: Impedance at the bottom.
        blk: The four blocks of A there.
        h: Sublayer thickness, positive.
        nsub: RK4 substeps within the sublayer.

    Returns:
        The impedance at the top.
    """
    a11, a12, a21, a22 = blk

    def f(yy: np.ndarray) -> np.ndarray:
        return a21 + a22 @ yy - yy @ a11 - yy @ a12 @ yy

    dz = -h / nsub
    for _ in range(nsub):
        k1 = f(y)
        k2 = f(y + 0.5 * dz * k1)
        k3 = f(y + 0.5 * dz * k2)
        k4 = f(y + dz * k3)
        y = y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


def march(
    medium_at: Callable[[float], ReferenceMedium],
    omega: complex,
    kx: float,
    thickness: float,
    nstep: int,
    *,
    method: str = "mobius",
    nsub: int = 1,
) -> np.ndarray:
    """March Y from the base of the slab to its top.

    The slab occupies 0 <= z <= thickness, z downward, with the background
    half-space below supplying the starting impedance.

    Args:
        medium_at: Medium as a function of depth within the slab.
        omega: Angular frequency.
        kx: Lateral wavenumber.
        thickness: Slab thickness.
        nstep: Sublayers.
        method: "mobius" or "rk4".
        nsub: RK4 substeps per sublayer.

    Returns:
        The impedance at the top of the slab, shape (3, 3).
    """
    y = y_downgoing(REF, omega, kx)
    edge = np.linspace(0.0, thickness, nstep + 1)
    for n in range(nstep - 1, -1, -1):
        zc = 0.5 * (edge[n] + edge[n + 1])
        blk = ablocks(medium_at(float(zc)), omega, kx, 0.0)
        hstep = float(edge[n + 1] - edge[n])
        y = step_mobius(y, blk, hstep) if method == "mobius" else step_rk4(y, blk, hstep, nsub)
    return y


def reflection(y: np.ndarray, omega: complex, kx: float) -> np.ndarray:
    """R from the marched impedance, by continuity of u and tau_3 alone.

    Args:
        y: Impedance at the top of the slab.
        omega: Angular frequency.
        kx: Lateral wavenumber.

    Returns:
        Shape (3, 3), indexed [out, in] over (P, S, H).
    """
    dd, du = mode_columns(REF, omega, kx)
    lhs = du[3:, :] - y @ du[:3, :]
    rhs = dd[3:, :] - y @ dd[:3, :]
    return np.asarray(-np.linalg.solve(lhs, rhs))


def r_march(lay: ReferenceMedium, p: float, nstep: int, **kw) -> np.ndarray:
    """R for a uniform slab of medium `lay`.

    Args:
        lay: Slab medium.
        p: Horizontal slowness.
        nstep: Sublayers.
        **kw: Passed to ``march``.

    Returns:
        Shape (3, 3).
    """
    kx = float(np.real(OMEGA) * p)
    return reflection(march(lambda _z: lay, OMEGA, kx, H, nstep, **kw), OMEGA, kx)


def graded(z: float) -> ReferenceMedium:
    """A slab whose contrast varies smoothly with depth -- no closed form exists.

    Args:
        z: Depth within the slab.

    Returns:
        The medium there.
    """
    s = 1.0 + 0.10 * np.sin(np.pi * z / H) + 0.04 * (z / H)
    return ReferenceMedium(REF.alpha * s, REF.beta * s, REF.rho * s)


def kennett_r(layers: list[tuple[ReferenceMedium, float]], p: float) -> tuple[np.ndarray, complex]:
    """RD from the Kennett recursion, for a stack embedded in the background.

    Args:
        layers: (medium, thickness) pairs for the slab.
        p: Horizontal slowness.

    Returns:
        (RD_psv as (2, 2), RD_sh).
    """
    stack = LayerStack(
        [
            IsotropicLayer(REF.alpha, REF.beta, REF.rho, 100.0),
            *[IsotropicLayer(m.alpha, m.beta, m.rho, t) for m, t in layers],
            IsotropicLayer(REF.alpha, REF.beta, REF.rho, np.inf),
        ]
    )
    res = kennett_layers(stack, p, np.array([float(np.real(OMEGA))]))
    return np.asarray(res.RD_psv[0], dtype=complex), complex(res.RD_sh[0])


def diag_ratio(rmat: np.ndarray, ken: np.ndarray, ksh: complex) -> np.ndarray:
    """The three diagonal convention ratios against Kennett.

    Args:
        rmat: R from the march.
        ken: RD_psv.
        ksh: RD_sh.

    Returns:
        Shape (3,) complex.
    """
    return np.array([rmat[0, 0] / ken[0, 0], rmat[1, 1] / ken[1, 1], rmat[2, 2] / ksh])


def main() -> int:
    """Run the march gate.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  Marching the impedance through a slab, scored against Kennett")
    print("=" * 74)
    print(f"    background {REF.alpha}/{REF.beta}/{REF.rho}, h = {H} m, omega = {OMEGA} rad/s")
    print(f"    omega/alpha = {OMEGA / REF.alpha:.5f}, omega/beta = {OMEGA / REF.beta:.5f} rad/m")

    print("")
    print("--- 1: normal incidence, against the closed form -------------------")
    print("    One Mobius step is EXACT for a uniform slab, so this is an")
    print("    absolute check with no discretisation in it at all.")
    worst_ni = 0.0
    for name, lay in (
        ("weak   (+2%)", ReferenceMedium(5100.0, 3060.0, 2550.0)),
        ("medium (+10%)", LAY),
        ("strong (+40%)", ReferenceMedium(7000.0, 4200.0, 3500.0)),
    ):
        exact = normal_incidence_layer(REF.rho, REF.alpha, lay.rho, lay.alpha, H, OMEGA)
        got = complex(r_march(lay, 0.0, 1)[0, 0])
        rel = abs(got - exact) / abs(exact)
        worst_ni = max(worst_ni, rel)
        print(f"    {name}:  closed form {exact.real:+.8f}{exact.imag:+.8f}i")
        print(f"                  march       {got.real:+.8f}{got.imag:+.8f}i   rel {rel:.3e}")
    report("ONE exact step reproduces the closed form for a uniform slab", worst_ni < 1e-12)

    print("")
    print("--- 2: the two integrators, and where each is trustworthy ----------")
    print("    Mobius is exact per uniform sublayer but forms expm(-A h), which")
    print("    carries the growing evanescent branch.  RK4 never forms it but is")
    print("    step-limited by the decay rate.  Neither is assumed.")
    exact = normal_incidence_layer(REF.rho, REF.alpha, LAY.rho, LAY.alpha, H, OMEGA)
    print(f"    {'steps':>7} {'Mobius':>12} {'RK4':>12} {'RK4 ratio':>11}")
    prev, rates, worst_mob = None, [], 0.0
    for ns in (2, 4, 8, 16, 32):
        em = abs(complex(r_march(LAY, 0.0, ns)[0, 0]) - exact) / abs(exact)
        er = abs(complex(r_march(LAY, 0.0, ns, method="rk4")[0, 0]) - exact) / abs(exact)
        worst_mob = max(worst_mob, em)
        rate = prev / er if prev else float("nan")
        if prev:
            rates.append(rate)
        prev = er
        print(f"    {ns:7d} {em:12.3e} {er:12.3e} {rate:11.2f}")
    print(f"    RK4 refinement ratios: {', '.join(f'{r:.1f}' for r in rates)}")
    report("Mobius stays exact at every step count, as it must", worst_mob < 1e-12)
    report("RK4 converges at fourth order, so (*) is being integrated right", min(rates) > 10.0)

    print("")
    print("--- 3: oblique incidence, against Kennett ---------------------------")
    print("    The thesis and Kennett amplitude conventions differ by a fixed")
    print("    matrix.  What is testable is that it is FIXED -- a convention")
    print("    cannot depend on angle -- and that it is the SAME constant the")
    print("    collocation route of the layer gate already found, which is a")
    print("    cross-check between two independent routes, not a free fit.")
    ratios = []
    for p in (0.0, 5e-5, 1.0e-4, 1.5e-4):
        rmat = r_march(LAY, p, 1)
        ken, ksh = kennett_r([(LAY, H)], p)
        rat = diag_ratio(rmat, ken, ksh)
        ratios.append(rat)
        print(
            f"    p={p:8.2e} (sin i={p * REF.alpha:.3f})   PP {rat[0]:+.6f}"
            f"   SS {rat[1]:+.6f}   HH {rat[2]:+.6f}"
        )
    arr = np.array(ratios)
    spread = float(np.max(np.abs(arr - arr[0])))
    print(f"    spread across slowness = {spread:.3e}")
    report("the conventions are constant across slowness", spread < 1e-6)
    near = float(np.max(np.abs(arr - CONV[None, :])))
    print(f"    the constants are (-1, -1, +1) to {near:.3e}.  The collocation")
    print("    route reaches the same three to 1.7e-4 at 120 cells, by a")
    print("    completely different path -- so this is a cross-check between two")
    print("    independent routes, not a fitted offset.  Note the SH sign does")
    print("    NOT flip where P-SV does; that split is real.")
    report("the conventions are (-1, -1, +1), agreeing with the collocation route", near < 1e-9)

    print("")
    print("--- 4: past critical, where the march has to carry evanescence ------")
    print(f"    P goes evanescent at p = 1/alpha = {1 / REF.alpha:.3e},")
    print(f"    S at p = 1/beta = {1 / REF.beta:.3e}.")
    worst_pc = 0.0
    for p in (1.8e-4, 2.2e-4, 3.0e-4):
        rmat = r_march(LAY, p, 1)
        ken, ksh = kennett_r([(LAY, H)], p)
        rat = diag_ratio(rmat, ken, ksh)
        dev = float(np.max(np.abs(rat - CONV)))
        worst_pc = max(worst_pc, dev)
        tag = "P evan" if p < 1 / REF.beta else "P,S evan"
        print(f"    p={p:8.2e} {tag:>9}   PP {rat[0]:+.6f}  SS {rat[1]:+.6f}  HH {rat[2]:+.6f}")
    report("the same constant holds past the P and the S critical slowness", worst_pc < 1e-6)

    print("")
    print("--- 5: THE POINT -- a GRADED slab, where no closed form exists ------")
    print("    A uniform slab never needed a march.  Here the contrast varies")
    print("    with depth, so the march is the only route -- and Kennett is cut")
    print("    at the SAME depths, making both solve the identical")
    print("    piecewise-constant problem.  They must therefore agree to the")
    print("    ARITHMETIC FLOOR at finite step count, not merely converge.")
    print(f"    {'steps':>7} {'p':>10} {'PP ratio':>22} {'max |ratio - conv|':>19}")
    worst_gr = 0.0
    for ns in (4, 16, 64):
        for p in (0.0, 1.0e-4):
            edge = np.linspace(0.0, H, ns + 1)
            mids = 0.5 * (edge[:-1] + edge[1:])
            layers = [(graded(float(z)), float(H / ns)) for z in mids]
            kx = float(OMEGA * p)
            rmat = reflection(march(graded, OMEGA, kx, H, ns), OMEGA, kx)
            ken, ksh = kennett_r(layers, p)
            rat = diag_ratio(rmat, ken, ksh)
            dev = float(np.max(np.abs(rat - CONV)))
            worst_gr = max(worst_gr, dev)
            print(f"    {ns:7d} {p:10.2e} {rat[0]:+.12f} {dev:17.3e}")
    report("the graded march agrees with Kennett to the floor at every mesh", worst_gr < 1e-9)

    print("")
    print("--- 6: negative control -- the grading must matter ------------------")
    print("    If the march were insensitive to where the contrast sits, Part 5")
    print("    would be passing for the wrong reason.  Reversing the profile")
    print("    keeps every layer and every thickness, and only reorders them.")
    kx = float(OMEGA * 1.0e-4)
    r_fwd = reflection(march(graded, OMEGA, kx, H, 32), OMEGA, kx)
    r_rev = reflection(march(lambda zz: graded(H - zz), OMEGA, kx, H, 32), OMEGA, kx)
    rel = float(np.max(np.abs(r_fwd - r_rev)) / np.max(np.abs(r_fwd)))
    print(f"    forward PP {complex(r_fwd[0, 0]):+.8f}")
    print(f"    reversed PP {complex(r_rev[0, 0]):+.8f}   relative change {rel:.3e}")
    report("reversing the depth profile changes R, so the march sees order", rel > 1e-3)

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
