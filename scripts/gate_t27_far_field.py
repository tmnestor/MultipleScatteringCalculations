"""GATE: the 27-mode coupling reduces to the point propagator in the far field.

WHY THIS GATE EXISTS AND WHY IT IS THE ONE THAT MATTERS. Every other check on
this coupling compares the construction against ITSELF -- reciprocity, the
centre-of-mass reduction, the nesting of the tiers, the convergence of the
contact quadrature. The 9-mode normalisation took three attempts, and the first
two passed all of those (1e-16, 6e-14, 5e-13) while the strain sector was wrong
by factors of 12 and 24. Only the far-field limit compares against an object the
construction does not contain.

For the quadratic tier no 27x27 point propagator exists to compare with, so the
independent object is built here: with u = s - s' (s in the receiver cell, s' in
the source cell),

    Gamma^{ab}(R) = Int C_ab(u) G_{d(a),d(b)}(R + u) du,

and expanding G(R+u), every coefficient is a moment of the two scalar monomials
over the cube. Keeping the lowest surviving moment on each side gives

    Gamma^{ab} -> m_a m_b * [ (-1)^{o_b} / (o_a! o_b!) ]
                          * (Mhat_a (x) Mhat_b) : d^{o_a+o_b} G ,

the (-1)^{o_b} because the source coordinate enters with a minus sign and the
factorials from the multinomial. Everything after `m_a m_b` is built from the
point propagator's own derivative tensors, so the measured ratio
`Gamma_raw / structure` IS the normalisation -- measured, not assumed.

GATES
  [A] the raw 27x27 against the full second-order expansion, with the RATE
  [B] the block scale against m_a m_b, for all 16 sub-tier pairs
  [C] that scale as a LIMIT: the residual must vanish as (d/R)^2

Sub-tiers, not tiers: `s_p^2` and `s_p s_q` sit in one tier but radiate into
different channels, so a table written by tier would average them together and
hide it.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.galerkin_propagator import (
    basis_terms,
    far_field_moment,
    galerkin_block_9x9,
    gram_diagonal,
)
from cubic_scattering.kupradze_derivatives import scalar_derivative_tensors
from cubic_scattering.resonance_tmatrix import elastodynamic_greens_deriv

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA, D = 60.0, 1.0
A = 0.5 * D
NM = 27

# A GENERIC direction. Along a coordinate axis whole families of components of
# G, dG and ddG vanish identically by symmetry, so a check written on those
# entries measures zero against zero and says nothing. (2,6,3)/7 has no zero
# component and no two equal, so nothing is degenerate.
UNIT = np.array([2.0, 6.0, 3.0]) / 7.0

TIERS = [("const", range(0, 3)), ("linear", range(3, 9)), ("quad", range(9, 27))]
SUB = [
    ("const", [0, 1, 2]),
    ("linear", list(range(3, 9))),
    ("quad s_p^2", [9 + 6 * k + m for k in range(3) for m in range(3)]),
    ("quad s_ps_q", [9 + 6 * k + m for k in range(3) for m in range(3, 6)]),
]


def moment(e: tuple[int, ...], a: float) -> float:
    """Int_{[-a,a]^3} s^e dV -- factorises, and an odd exponent kills it."""
    out = 1.0
    for n in e:
        if n % 2:
            return 0.0
        out *= 2.0 * a ** (n + 1) / (n + 1)
    return out


def bumped(e: tuple[int, int, int], *axes: int) -> tuple[int, int, int]:
    v = list(e)
    for ax in axes:
        v[ax] += 1
    return (v[0], v[1], v[2])


def leading_moment(e: tuple[int, int, int], a: float) -> tuple[int, float, NDArray]:
    """(order, scale, unit tensor) of the lowest surviving moment of s^e."""
    p = moment(e, a)
    if p:
        return 0, p, np.array(1.0)
    q = np.array([moment(bumped(e, k), a) for k in range(3)])
    if np.any(q):
        s = float(np.abs(q).max())
        return 1, s, q / s
    r = np.array([[moment(bumped(e, k, m), a) for m in range(3)] for k in range(3)])
    s = float(np.abs(r).max())
    return 2, s, r / s


def greens_derivatives(r_vec: NDArray, nmax: int) -> list[NDArray]:
    """[G, dG, ddG, ...] to order `nmax`.

    The quadrupole x quadrupole block needs FOURTH derivatives, which
    `elastodynamic_greens_deriv` stops short of. Kupradze supplies any order,
    because differentiating

        G_ij = (1/rho w^2) [ kS^2 g_S delta_ij + d_i d_j (g_S - g_P) ]

    only raises the order of the two SCALARS -- the operator's coefficients are
    constants. Leaving those blocks out was not an option: they would have come
    back as nan, and nan is silently dropped by `max()`.
    """
    kp, ks = OMEGA / REF.alpha, OMEGA / REF.beta
    d_p = scalar_derivative_tensors(r_vec, kp, order=nmax + 2)
    d_s = scalar_derivative_tensors(r_vec, ks, order=nmax + 2)
    pref = 1.0 / (REF.rho * OMEGA**2)
    delta = np.eye(3)
    out = []
    for n in range(nmax + 1):
        term = np.multiply.outer(delta, d_s[n]) if n else delta * d_s[0]
        out.append(pref * (ks**2 * term + (d_s[n + 2] - d_p[n + 2])))
    return out


def expansion(r_vec: NDArray, a: float) -> NDArray:
    """The coupling expanded to second order in the cell size.

    Int C_ab(u) u_p u_q du and its lower moments all reduce to cube monomial
    moments of the two trial functions, so this shares no code with the
    quadrature it is testing.
    """
    g, gd, gdd = elastodynamic_greens_deriv(r_vec, OMEGA, REF)
    out = np.zeros((NM, NM), dtype=complex)
    terms = [basis_terms(al) for al in range(NM)]
    for al in range(NM):
        for be in range(NM):
            acc = 0.0 + 0.0j
            for ca, ea, da in terms[al]:
                pa = moment(ea, a)
                qa = np.array([moment(bumped(ea, p), a) for p in range(3)])
                ra = np.array([[moment(bumped(ea, p, q), a) for q in range(3)] for p in range(3)])
                for cb, eb, db in terms[be]:
                    pb = moment(eb, a)
                    qb = np.array([moment(bumped(eb, p), a) for p in range(3)])
                    rb = np.array([[moment(bumped(eb, p, q), a) for q in range(3)] for p in range(3)])
                    m2 = ra * pb - np.outer(qa, qb) - np.outer(qb, qa) + pa * rb
                    acc += (
                        ca
                        * cb
                        * (
                            pa * pb * g[da, db]
                            + np.dot(qa * pb - pa * qb, gd[da, db, :])
                            + 0.5 * np.tensordot(m2, gdd[da, db, :, :], axes=2)
                        )
                    )
            out[al, be] = acc
    return out


def point_structure(r_vec: NDArray, a: float) -> tuple[NDArray, NDArray]:
    """The point-propagator block each mode pair reduces to, and its scale."""
    dg = greens_derivatives(r_vec, 4)
    struct = np.zeros((NM, NM), dtype=complex)
    scale = np.zeros((NM, NM))
    terms = [basis_terms(al) for al in range(NM)]
    fact = [1.0, 1.0, 2.0]
    for al in range(NM):
        for be in range(NM):
            acc = 0.0 + 0.0j
            sc_ab = 0.0
            for ca, ea, da in terms[al]:
                oa, ma, ta = leading_moment(ea, a)
                for cb, eb, db in terms[be]:
                    ob, mb, tb = leading_moment(eb, a)
                    coef = ca * cb * (-1.0) ** ob / (fact[oa] * fact[ob])
                    n = oa + ob
                    deriv = dg[n][da, db]
                    comb = np.multiply.outer(ta, tb)
                    contracted = comb * deriv if n == 0 else np.tensordot(comb, deriv, axes=n)
                    acc += coef * complex(contracted)
                    sc_ab = ma * mb
            struct[al, be] = acc
            scale[al, be] = sc_ab
    return struct, scale


def block_scale_error(r_vec: NDArray) -> tuple[float, float]:
    """Worst |measured/(m_a m_b) - 1| over the 16 sub-tier pairs, and the spread.

    BLOCK MAXIMA, not entrywise ratios: many entries vanish identically by
    parity, so an entrywise relative error is 0/0 noise.
    """
    gam = galerkin_block_9x9(r_vec, D, OMEGA, REF, n_quad=8, n_modes=NM)
    struct, scale = point_structure(r_vec, A)
    worst = worst_spread = 0.0
    for (_, ia), (_, ib) in itertools.product(SUB, SUB):
        sl = np.ix_(ia, ib)
        big = np.abs(struct[sl]) > 0.05 * np.abs(struct[sl]).max()
        ratio = (gam[sl][big] / struct[sl][big]).real
        meas = float(np.median(ratio))
        rel = abs(meas / float(np.median(scale[sl][big])) - 1.0)
        # A nan must FAIL, not be skipped -- max() silently drops it.
        worst = float("inf") if not np.isfinite(rel) else max(worst, rel)
        worst_spread = max(worst_spread, float(ratio.std() / abs(meas)))
    return worst, worst_spread


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print("=" * 74)

    # Pin the two Green's routines together before trusting orders 3-4.
    r0 = 40.0 * D * UNIT
    guard = max(
        np.abs(k - x).max() / np.abs(x).max()
        for k, x in zip(greens_derivatives(r0, 2), elastodynamic_greens_deriv(r0, OMEGA, REF), strict=True)
    )
    print(f"  guard  Kupradze vs elastodynamic_greens_deriv, orders 0-2 : {guard:.2e}")
    if guard > 1e-12:
        print("  FAIL -- the two Green's routines disagree; orders 3-4 untrustworthy")
        return 1

    print()
    print("  [A] raw 27x27 vs the analytic second-order expansion (block maxima)")
    print("      " + "".join(f"{na[:5]}x{nb[:5]:>6}" for na, _ in TIERS for nb, _ in TIERS))
    a_rows = []
    for n in (10, 20, 40):
        r = float(n) * D * UNIT
        gam = galerkin_block_9x9(r, D, OMEGA, REF, n_quad=8, n_modes=NM)
        want = expansion(r, A)
        err = {}
        for (na, ia), (nb, ib) in itertools.product(TIERS, TIERS):
            sl = np.ix_(list(ia), list(ib))
            scale = np.abs(want[sl]).max()
            err[(na, nb)] = float(np.abs(gam[sl] - want[sl]).max() / scale)
        a_rows.append(err)
        print(f"  {n:3d} " + "".join(f"{err[k]:12.2e}" for k in err))
    rate = min(a_rows[0][k] / a_rows[-1][k] for k in a_rows[0])
    print(f"      slowest error reduction over R = 10d -> 40d : {rate:.1f}x  (expect >= 8)")

    print()
    print("  [B] block scale vs m_a m_b, all 16 sub-tier pairs, at R = 40d")
    worst, spread = block_scale_error(r0)
    print(f"      worst |measured / (m_a m_b) - 1| : {worst:.3e}")
    print(f"      worst spread within a block      : {spread:.3e}  (a block-CONSTANT ratio)")

    print()
    print("  [C] that scale as a LIMIT -- the residual must vanish, not just be small")
    tail = [(n, block_scale_error(float(n) * D * UNIT)[0]) for n in (20, 40, 80)]
    for n, w in tail:
        print(f"      R = {n:3d} d   {w:.3e}")
    ratios = [tail[i][1] / tail[i + 1][1] for i in range(len(tail) - 1)]
    print(f"      ratios {', '.join(f'{x:.2f}' for x in ratios)}   (expect ~4 = (d/R)^2)")

    print()
    print("  the scales this pins, as returned by the module")
    g27, m27 = gram_diagonal(D, 27), far_field_moment(D, 27)
    for al, lab in (
        (0, "constant"),
        (3, "linear axial"),
        (6, "linear shear"),
        (9, "quad s_p^2"),
        (12, "quad s_p s_q"),
    ):
        print(
            f"      {lab:>14}   Gram {g27[al]:12.6g}   moment {m27[al]:12.6g}"
            f"   ratio {g27[al] / m27[al]:7.4f}"
        )
    print("      The two scales already differ in the T9 tier, on the SHEAR modes,")
    print("      by exactly the factor of two of the engineering convention. In the")
    print("      quadratic tier they differ again but for a different reason: the")
    print("      s_p^2 modes carry a MONOPOLE, so their self-energy (V d^4/80) and")
    print("      their radiating moment (V d^2/12) are unrelated quantities. The T9")
    print("      rule does not carry across unchanged.")

    ok = rate >= 8.0 and worst < 2e-3 and spread < 5e-3 and all(x > 3.0 for x in ratios)
    print()
    print("=" * 74)
    if ok:
        print("  PASS -- the 27-mode coupling reduces to the point propagator, every")
        print("  block scaling as the product of the two modes' lowest surviving")
        print("  moments. The normalisation is pinned by an independent object.")
        return 0
    print("  FAIL -- the 27-mode normalisation is NOT established.")
    print("  Do not widen the solver or wire T27 in until this passes.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
