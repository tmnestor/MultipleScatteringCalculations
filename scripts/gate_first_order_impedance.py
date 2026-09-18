#!/usr/bin/env python3
"""The impedance Riccati form, and whether it clears the mode basis's ceiling.

ANCHOR: Nestor (1996) Ch.2 (Akdef), (eigDef), (kzcDef), (specA).

THE QUESTION
------------
``gate_first_order_riccati_blocks`` found that the mode-basis Riccati
coefficients V = D_z^-1 dA D_z carry a ceiling.  Not from stiffness -- the stiff
part is diagonal and an integrating factor removes it exactly -- but from
NON-NORMALITY: eig(V)/2|k| is flat at 0.0656 over four decades while |V|/2|k|
grows as |k|^2, because k_zP and k_zS both tend to i|k| and the P and S columns
of D_z become parallel.  cond(D_z) ~ |k|^2.001, V_du and V_ud cancel over nine
orders, and the similarity eig(V) = eig(dA) fails by |k| = 600.

The conjecture was that the mode basis is simply the wrong basis where the modes
coalesce, and that marching an IMPEDANCE in the physical variables would avoid
the problem because D_z never appears.  This gate tests that.

WHAT THE SURVEY FOUND FIRST
---------------------------
No matrix impedance exists anywhere: not in this package, not in the sibling
repository, not in the thesis.  What does exist is a SCALAR SH impedance used as
an eigenvector normalisation convention, and two Riccati sweeps -- both of which
march a reflection matrix in MODE AMPLITUDES, not an impedance.  So the sibling
repository is not already doing this, and an earlier note in the write-up saying
it was has been corrected.

THE EQUATION
------------
Split q = (u, tau_3) and A into blocks.  For a field with tau_3 = Y u,

    u'      = A11 u + A12 tau_3 = (A11 + A12 Y) u,
    tau_3'  = A21 u + A22 tau_3 = (A21 + A22 Y) u,
    tau_3'  = Y' u + Y u',

whence the impedance Riccati equation

    Y' = A21 + A22 Y - Y A11 - Y A12 Y.                              (*)

Its coefficients are the blocks of A itself -- polynomial in k, no eigenvectors
anywhere.  In a homogeneous medium Y' = 0 and (*) becomes the ALGEBRAIC Riccati
equation, whose stabilising solution is the downgoing half-space impedance.

THE ARBITER
-----------
Self-consistency is not enough, so the test is against an independently known
analytic quantity.  A purely downgoing field obeys u' = (A11 + A12 Y) u, so

    eig(A11 + A12 Y) = { i k_zP, i k_zS, i k_zS }

with k_zc from (kzcDef) in closed form.  Part 5 scores Y on that, at wavenumbers
where the mode basis has already failed.

Units are km, km/s, g/cm^3, rad/km, rad/s.

Run:  conda run -n seismic python scripts/gate_first_order_impedance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_sylvester

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from scripts.gate_thesis_spectral import (  # noqa: E402
    amat_thesis,
    dz_balanced,
    kz_c,
    rowscale,
)

BG = ReferenceMedium(4.0, 2.22, 2.6)
QFAC = 600.0
OMEGA = 2.0 * np.pi * 2.0 * (1.0 + 0.5j / QFAC)

#: A contrast, used only by Part 2 to put V's failure beside Y's on one table.
PER = ReferenceMedium(
    float(np.sqrt((BG.lam + 2.0 + 2.0 * (BG.mu + 1.0)) / (BG.rho + 0.1))),
    float(np.sqrt((BG.mu + 1.0) / (BG.rho + 0.1))),
    BG.rho + 0.1,
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


def ablocks(
    med: ReferenceMedium, omega: complex, kx: float, ky: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four 3x3 blocks of (Akdef), in the thesis basis.

    No transcription: the blocks are read off the already-validated matrix.

    Args:
        med: Medium.
        omega: Angular frequency.
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.

    Returns:
        (A11, A12, A21, A22).
    """
    a = amat_thesis(med, omega, kx, ky)
    return a[:3, :3], a[:3, 3:], a[3:, :3], a[3:, 3:]


def are_residual(y: np.ndarray, blk: tuple, *, relative: bool = True) -> float:
    """Residual of A21 + A22 Y - Y A11 - Y A12 Y = 0.

    Args:
        y: Candidate impedance, shape (3, 3).
        blk: The four blocks of A.
        relative: Normalise by the largest term rather than returning it raw.

    Returns:
        The residual.
    """
    a11, a12, a21, a22 = blk
    terms = (a21, a22 @ y, -y @ a11, -y @ a12 @ y)
    res = float(np.max(np.abs(sum(terms))))
    if not relative:
        return res
    return res / max(float(max(np.max(np.abs(t)) for t in terms)), 1e-300)


def y_from_dz(kx: float, ky: float, omega: complex = OMEGA) -> np.ndarray:
    """The downgoing impedance read off D_z: Y = T_d U_d^-1.

    This is the route that inherits D_z's conditioning, and is used as a
    starting point and as the thing to beat -- not as the answer.

    Args:
        kx: Lateral wavenumber, x.
        ky: Lateral wavenumber, y.
        omega: Angular frequency.

    Returns:
        Shape (3, 3) complex.
    """
    scl = rowscale(BG, omega, kx, ky)
    dzb, _, _, _ = dz_balanced(BG, omega, kx, ky)
    dzu = dzb * (1.0 / scl)[:, None]  # undo the row scaling: back to (u, tau_3)
    return np.asarray(dzu[3:, :3] @ np.linalg.inv(dzu[:3, :3]))


def newton_are(y0: np.ndarray, blk: tuple, iters: int = 40) -> tuple[np.ndarray, float]:
    """Refine Y by Newton's method on the algebraic Riccati equation.

    Uses ONLY the blocks of A.  The Newton correction E solves the Sylvester
    equation (A22 - Y A12) E - E (A11 + A12 Y) = -R(Y), and the iteration is
    the standard one for the stabilising solution -- so if the equation is well
    conditioned, this converges to the arithmetic floor whatever D_z is doing.

    Args:
        y0: Starting impedance.
        blk: The four blocks of A.
        iters: Maximum iterations.

    Returns:
        (Y, residual).
    """
    a11, a12, a21, a22 = blk
    y = np.array(y0, dtype=np.complex128, copy=True)
    best, best_r = y, are_residual(y, blk)
    for _ in range(iters):
        res = a21 + a22 @ y - y @ a11 - y @ a12 @ y
        try:
            e = solve_sylvester(a22 - y @ a12, -(a11 + a12 @ y), -res)
        except Exception:  # noqa: BLE001 - a singular Sylvester ends the refinement
            break
        y = y + e
        r = are_residual(y, blk)
        if r < best_r:
            best, best_r = np.array(y, copy=True), r
        if r < 1e-15:
            break
    return best, best_r


def y_refined(k: float, omega: complex = OMEGA) -> np.ndarray:
    """Y at one wavenumber: D_z picks the branch, Newton supplies the precision.

    The algebraic Riccati equation has more than one solution -- the downgoing
    and upgoing impedances among them -- and Newton converges to whichever root
    it is started near.  Continuation in |k| was tried first and FLIPPED BRANCH
    on crossing omega/alpha, returning the upgoing root: the arbiter of Part 5
    came back at exactly 2.000, which is |-i k_z - i k_z| / |k_z|, the signature
    of a sign flip rather than of inaccuracy.

    The division of labour that works is the natural one.  D_z is analytic, so
    it names the right branch however imprecise it has become; Newton then
    refines using the blocks of A alone, which is where the precision is.
    Part 3 verifies the branch intrinsically afterwards, without D_z.

    Args:
        k: Lateral wavenumber.
        omega: Angular frequency.

    Returns:
        Shape (3, 3) complex.
    """
    return newton_are(y_from_dz(k, 0.0, omega), ablocks(BG, omega, k, 0.0))[0]


def downgoing_branch(y: np.ndarray, blk: tuple) -> bool:
    """Is Y the DOWNGOING root, tested without reference to D_z?

    A downgoing field obeys u' = (A11 + A12 Y) u with e^{+i k_z z} and
    Im(k_z) >= 0, so every eigenvalue must have non-positive real part.  With
    attenuation the inequality is strict, which makes this a clean test.

    Args:
        y: Candidate impedance.
        blk: The four blocks of A.

    Returns:
        Whether every mode decays downward.
    """
    a11, a12, _, _ = blk
    return bool(np.all(np.real(np.linalg.eigvals(a11 + a12 @ y)) < 0.0))


def main() -> int:
    """Gate the impedance form.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    print("=" * 74)
    print("  The impedance Riccati form -- does it clear the mode basis ceiling?")
    print("=" * 74)
    print(f"    omega/alpha = {abs(OMEGA / BG.alpha):.4f}, omega/beta = {abs(OMEGA / BG.beta):.4f} rad/km")
    sweep = [2.0, 8.0, 20.0, 60.0, 200.0, 600.0, 2000.0, 6000.0]

    print("")
    print("--- 1: the equation is right -- Y from D_z satisfies it -------------")
    print("    In a homogeneous medium Y' = 0, so the downgoing impedance read")
    print("    off D_z must satisfy A21 + A22 Y - Y A11 - Y A12 Y = 0 exactly.")
    print("    Checked where D_z is still trustworthy, which validates (*).")
    worst_eq = 0.0
    for kx, ky in ((0.0, 0.0), (2.0, 0.0), (1.3, 0.9), (20.0, 0.0), (11.0, -7.0)):
        blk = ablocks(BG, OMEGA, kx, ky)
        r = are_residual(y_from_dz(kx, ky), blk)
        worst_eq = max(worst_eq, r)
        print(f"    k = ({kx:6.1f},{ky:6.1f})   residual = {r:10.3e}")
    report("the impedance Riccati equation is satisfied by the D_z impedance", worst_eq < 1e-12)

    print("")
    print("--- 2: WHY the impedance is different -- it needs only half of D_z ---")
    print("    V = D_z^-1 dA D_z needs the FULL inverse, which mixes up- and")
    print("    downgoing and is where the nine-order cancellation lives.")
    print("    Y = T_d U_d^-1 needs only the three DOWNGOING columns.  So the")
    print("    two degrade quite differently: Y linearly in cond(U_d), V by")
    print("    cancellation.  Both are measured here, on one line each.")
    print(
        f"    {'|k|':>8} {'cond(D_z)':>11} {'cond(U_d)':>11} {'Y: ARE resid':>14} {'V: eig(V)-eig(dA)':>19}"
    )
    worst_lin, v_fails = 0.0, False
    for k in sweep:
        scl = rowscale(BG, OMEGA, k, 0.0)
        dzb, invb, _, _ = dz_balanced(BG, OMEGA, k, 0.0)
        dzu = dzb * (1.0 / scl)[:, None]
        cu = float(np.linalg.cond(dzu[:3, :3]))
        r = are_residual(y_from_dz(k, 0.0), ablocks(BG, OMEGA, k, 0.0))
        da = amat_thesis(PER, OMEGA, k, 0.0) - amat_thesis(BG, OMEGA, k, 0.0)
        vm = invb @ (scl[:, None] * da * (1.0 / scl)[None, :]) @ dzb
        ea = np.sort_complex(np.linalg.eigvals(da))
        sim = float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(vm)) - ea)))
        sim /= float(np.max(np.abs(ea)))
        worst_lin = max(worst_lin, r / (cu * 2.3e-16))
        if k >= 600.0:
            v_fails = v_fails or sim > 1e-2
        print(f"    {k:8.1f} {np.linalg.cond(dzb):11.3e} {cu:11.3e} {r:14.3e} {sim:19.3e}")
    report("Y degrades only LINEARLY in cond(U_d), not by cancellation", worst_lin < 50.0)
    report("V, needing the full inverse, fails outright over the same range", v_fails)

    print("")
    print("--- 3: Newton on the ARE, using only the blocks of A ----------------")
    print("    D_z names the branch -- it is analytic, so it does that correctly")
    print("    however imprecise it has become -- and Newton then supplies the")
    print("    precision from the blocks of A alone.  Continuation in |k| was")
    print("    tried instead and FLIPPED BRANCH at omega/alpha, returning the")
    print("    upgoing root; see the note in ``y_refined``.")
    ys = {k: y_refined(k) for k in sweep}
    print(
        f"    {'|k|':>8} {'ARE residual':>14} {'|Y|':>12} {'|Y|/(mu|k|)':>14} "
        f"{'cond(Y)':>10} {'downgoing':>10}"
    )
    worst_new, all_down = 0.0, True
    for k in sweep:
        y, blk = ys[k], ablocks(BG, OMEGA, k, 0.0)
        r = are_residual(y, blk)
        down = downgoing_branch(y, blk)
        worst_new, all_down = max(worst_new, r), all_down and down
        print(
            f"    {k:8.1f} {r:14.3e} {np.max(np.abs(y)):12.4e} "
            f"{np.max(np.abs(y)) / abs(BG.mu * k):14.4f} {np.linalg.cond(y):10.4f} "
            f"{str(down):>10}"
        )
    report("Newton holds the ARE to the floor at every wavenumber", worst_new < 1e-13)
    report("and lands on the DOWNGOING root, tested without reference to D_z", all_down)

    print("")
    print("--- 4: Y is bounded and physical -----------------------------------")
    print("    |Y| must grow like mu|k| -- the elastostatic half-space stiffness")
    print("    -- and cond(Y) must stay O(1).  That is the whole difference from")
    print("    V, whose norm grew as |k|^2 relative to its own spectrum.")
    print("    The limit must also be omega-INDEPENDENT, since it is static.")
    om2 = 2.0 * np.pi * 8.0 * (1.0 + 0.5j / QFAC)
    ys2 = {k: y_refined(k, om2) for k in sweep}
    print(f"    {'|k|':>8} {'Y/(mu|k|) at 2 Hz':>20} {'at 8 Hz':>14} {'difference':>13}")
    worst_om, worst_cond = 0.0, 0.0
    for k in sweep:
        n1 = ys[k] / (BG.mu * k)
        n2 = ys2[k] / (BG.mu * k)
        d = float(np.max(np.abs(n1 - n2)) / np.max(np.abs(n1)))
        if k >= 600.0:
            worst_om = max(worst_om, d)
        worst_cond = max(worst_cond, float(np.linalg.cond(ys[k])))
        print(f"    {k:8.1f} {float(np.max(np.abs(n1))):20.6f} {float(np.max(np.abs(n2))):14.6f} {d:13.3e}")
    report("cond(Y) stays O(1) across four decades -- Y is well conditioned", worst_cond < 20.0)
    report("the static limit of Y/(mu|k|) is omega-independent, as it must be", worst_om < 1e-3)

    print("")
    print("--- 5: THE ARBITER -- Y reproduces k_z, in closed form ---------------")
    print("    A downgoing field obeys u' = (A11 + A12 Y) u, so the eigenvalues")
    print("    of A11 + A12 Y must be i k_zP, i k_zS, i k_zS with k_zc from")
    print("    (kzcDef).  That is an independent analytic quantity, not a")
    print("    self-consistency check, and it is where the two routes are")
    print("    scored against each other.")
    print(f"    {'|k|':>8} {'from Y (Newton)':>18} {'from Y (D_z)':>16} {'verdict':>12}")
    worst_arb, mode_broke = 0.0, False
    for k in sweep:
        a11, a12, _, _ = ablocks(BG, OMEGA, k, 0.0)
        want = np.sort_complex(
            np.array([1j * kz_c(c, OMEGA, k, 0.0) for c in (BG.alpha, BG.beta, BG.beta)])
        )
        sc = float(np.max(np.abs(want)))
        got_n = np.sort_complex(np.linalg.eigvals(a11 + a12 @ ys[k]))
        got_d = np.sort_complex(np.linalg.eigvals(a11 + a12 @ y_from_dz(k, 0.0)))
        e_n = float(np.max(np.abs(got_n - want))) / sc
        e_d = float(np.max(np.abs(got_d - want))) / sc
        worst_arb = max(worst_arb, e_n)
        if k >= 2000.0:
            mode_broke = mode_broke or e_d > 1e3 * max(e_n, 1e-16)
        print(
            f"    {k:8.1f} {e_n:18.3e} {e_d:16.3e} {'Newton wins' if e_d > 10 * e_n else 'comparable':>12}"
        )
    print("    The impedance route is not exact either -- it drifts as about")
    print("    |k|^1.6, from 2e-13 at |k| = 60 to 3e-10 at |k| = 6000.  What it")
    print("    does not do is FAIL: the mode basis returns 2.000 by |k| = 600,")
    print("    which is |-i k_z - i k_z|/|k_z| and so a lost branch, not a lost")
    print("    digit.  Four orders better, and degrading gracefully.")
    report("the impedance holds k_z to 1e-9 out to |k| = 6000, thirty times further", worst_arb < 1e-9)
    report("and beats the D_z route by orders where D_z has coalesced", mode_broke)

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
