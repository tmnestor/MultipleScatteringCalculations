"""The 9x9 BUBNOV-GALERKIN inter-cell propagator.

═══ ⚠ NOT WIRED IN. THE HYPOTHESIS THIS WAS BUILT TO TEST IS REFUTED. ═══════════
This module was built to test whether the residual comes from pairing a Galerkin
`T_0` with a MOMENT `G_0`. Measured end to end, with the normalisation correct,
the Galerkin contact coupling is WORSE than the moment operator:

    scheme                                   residual floor
    moment contact correction                8.6e-4
    Galerkin, correct normalisation          5.1e-3
    Galerkin, WRONG (row-only) normalisation 8.6e-5   <- see below

⚠ THE MIDDLE ROW IS THE RESULT; THE BOTTOM ROW IS A CAUTIONARY TALE. An earlier
row-only normalisation scaled the contact strain coupling down by 12x and looked
like a 6x improvement. It was an accidental FUDGE FACTOR, arrived at by error
while trying to eliminate one -- and it passed reciprocity (1e-16), the
centre-of-mass reduction (6e-14) and the G-block check (5e-13), because every one
of those compares the construction against itself. Only the FAR-FIELD LIMIT,
which compares against an independent object, caught it.

WHY THE HYPOTHESIS PROBABLY FAILS. `T_0` is Galerkin-DERIVED, but as USED it is
`T = V . Delta c*` -- a local constitutive multiplication on `(u, eps)`, which is
a moment convention. The reduction to four effective contrasts discards the
Galerkin projection structure, so the moment propagator is arguably its correct
partner after all.

WHY THIS FILE IS KEPT. It is correct and gated, and its machinery generalises
directly to the 27-mode route that the evidence now favours: the autocorrelation
kernels `C_ab`, the apex-pyramid rule for the contact singularity, and above all
the far-field check that must be run FIRST on any future coupling.
════════════════════════════════════════════════════════════════════════════════

ORIGINAL MOTIVATION, kept because the observation stands even though the
inference from it did not. The lattice solver pairs a Galerkin-derived T-matrix
with a MOMENT propagator, and the two are not the same operator. `T_0` is the first tier
of the Bubnov-Galerkin hierarchy -- trial functions are 3 constant plus 6 linear
(`r_m e_k`, symmetric gradient) over the cell -- whereas `inter_voxel_propagator`
supplies cell-AVERAGED field and derivatives, `<<G>>`, `<<dG>>`, `<<ddG>>`. Those
agree only when the field is exactly linear across a cell. A touching neighbour's
near field has O(1) curvature across the cell AT EVERY SCALE, so the mismatch
does not vanish under refinement -- which is exactly the measured behaviour: the
refinement ladder saturates, converging to a WRONG limit rather than diverging.

A consistent Galerkin discretisation converges under mesh refinement at fixed
polynomial degree. Saturation is therefore the signature of INCONSISTENCY, not of
insufficient basis order, and the cure is to put the propagator on the same basis
as the T-matrix rather than to add modes.

THE OBJECT.

    Gamma^{ab}(R) = Int_Vi Int_Vj  phi_a(r - R_i) . G(r - r') . phi_b(r' - R_j)

By the centre-of-mass reduction (BubnovGalerkinCubicScatter.tex, eq. Gprop-reduced)
this collapses from six dimensions to three:

    Gamma^{ab}(R) = Int_{[-2a,2a]^3}  G_{d(a),d(b)}(u + R)  C_{ab}(u)  du,

    C_{ab}(u) = Int_{B(u)} p_a(xi + u/2) p_b(xi - u/2) dxi,

with B(u) the centred box of half-lengths b_k = a - |u_k|/2. For the 9-mode basis
the polynomials are degree <= 1, so every C_ab is elementary and is written in
closed form below -- no quadrature enters the autocorrelation.

THE SINGULARITY IS MILDER HERE THAN IN THE MOMENT FORM, and this is the practical
point. Integration by parts has moved the derivatives onto the TRIAL FUNCTIONS,
so the kernel is G itself (~1/rho) rather than dd G (~1/rho^3). The face-contact
tensor-product Gauss bias that this project has recorded as a trap applies to the
1/w^3 kernel; at 1/rho, with the apex-pyramid radial rule used here, the
singularity is integrated exactly in the radial variable.

Coordinates: z = axis 0 (down), x = axis 1, y = axis 2 -- the project's
seismological ordering. Voigt: (zz, xx, yy, xy, zy, zx) with ENGINEERING shear,
matching `resonance_tmatrix.VOIGT_PAIRS` and the 9-component state
(u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx).
"""

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .resonance_tmatrix import VOIGT_PAIRS, elastodynamic_greens

# A basis function is a sum of terms (coefficient, monomial exponents, direction).
# Monomial exponents are per-axis powers of the LOCAL coordinate s.
BasisTerm = tuple[float, tuple[int, int, int], int]


def basis_terms(alpha: int) -> list[BasisTerm]:
    """The 9-mode Bubnov-Galerkin trial function, as (coeff, exponents, dir) terms.

    alpha 0..2   constant displacement e_alpha                      (degree 0)
    alpha 3..8   linear displacement whose symmetric gradient is the
                 unit Voigt strain vector, ENGINEERING convention    (degree 1)

    The off-diagonal (shear) modes are genuinely two-term: the field
    0.5 (s_q e_p + s_p e_q) has symmetric gradient with 2 eps_pq = 1, which is
    what the engineering convention requires. Writing them as a single
    (monomial, direction) pair -- as the general derivation's notation suggests --
    would silently halve the shear sector.
    """
    if alpha < 0 or alpha > 26:
        msg = (
            f"basis index must be in 0..26, got {alpha}.\n"
            "  Where: cubic_scattering/galerkin_propagator.py, basis_terms(alpha)\n"
            "  Valid: 0-2 constant displacement, 3-8 linear (Voigt strain),\n"
            "         9-26 quadratic (6 monomials x 3 directions)\n"
            "  Fix:   indices 0-8 are the T9 tier; 9-26 extend it to T27"
        )
        raise ValueError(msg)
    if alpha < 3:
        return [(1.0, (0, 0, 0), alpha)]
    if alpha < 9:
        p, q = VOIGT_PAIRS[alpha - 3]
        if p == q:
            e = [0, 0, 0]
            e[p] = 1
            return [(1.0, (e[0], e[1], e[2]), p)]
        ep = [0, 0, 0]
        ep[q] = 1
        eq = [0, 0, 0]
        eq[p] = 1
        return [(0.5, (ep[0], ep[1], ep[2]), p), (0.5, (eq[0], eq[1], eq[2]), q)]
    # T27 quadratic tier. Ordering follows `tmatrix_assembly`: six monomials
    # (r1^2, r2^2, r3^2, r2r3, r1r3, r1r2) for each of the three directions.
    # That file's strain ordering was checked against VOIGT_PAIRS and agrees, so
    # indices 0-8 here ARE its first nine and the tiers stack cleanly.
    idx = alpha - 9
    direction, mono = divmod(idx, 6)
    quad_exponents: list[tuple[int, int, int]] = [
        (2, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ]
    return [(1.0, quad_exponents[mono], direction)]


def autocorrelation(ea: tuple[int, int, int], eb: tuple[int, int, int], u: NDArray, a: float) -> NDArray:
    """C(u) for one monomial pair, in closed form. Vectorised over u.

    C(u) = Int_{B(u)} prod_k (xi_k + u_k/2)^{ea_k} (xi_k - u_k/2)^{eb_k} dxi

    The box B(u) is centred, so it factorises axis by axis and each axis integral
    is elementary for degree <= 1. With b = a - |u|/2 on that axis:

        (0,0):  2b
        (1,0):  Int (xi + u/2) dxi  =  b u        [Int xi = 0 by symmetry]
        (0,1):  Int (xi - u/2) dxi  = -b u
        (1,1):  Int (xi^2 - u^2/4)  =  2b^3/3 - b u^2/2

    Checked against the three worked examples in the derivation: the (0,0)^3 case
    is the 3-D tent prod (2a - |u_k|); displacement x axial strain gives
    -u_k b_k prod_{j!=k} (2 b_j); axial x axial gives
    (2 b_k^3/3 - u_k^2 b_k/2) prod_{j!=k} (2 b_j).
    """
    b = a - 0.5 * np.abs(u)  # (..., 3)
    out = np.ones(u.shape[:-1], dtype=float)
    for k in range(3):
        bk, uk = b[..., k], u[..., k]
        key = (ea[k], eb[k])
        if key == (0, 0):
            fac = 2.0 * bk
        elif key == (1, 0):
            fac = bk * uk
        elif key == (0, 1):
            fac = -bk * uk
        elif key == (1, 1):
            fac = 2.0 * bk**3 / 3.0 - bk * uk**2 / 2.0
        # ---- degree-2 cases, needed only by the T27 quadratic tier ----------
        elif key in {(2, 0), (0, 2)}:
            # Int (xi +- u/2)^2 = 2b^3/3 + b u^2/2 -- EVEN in u, unlike (1,1).
            fac = 2.0 * bk**3 / 3.0 + bk * uk**2 / 2.0
        elif key == (2, 1):
            fac = bk**3 * uk / 3.0 - bk * uk**3 / 4.0
        elif key == (1, 2):
            fac = -(bk**3) * uk / 3.0 + bk * uk**3 / 4.0
        elif key == (2, 2):
            fac = 2.0 * bk**5 / 5.0 - bk**3 * uk**2 / 3.0 + bk * uk**4 / 8.0
        else:
            msg = (
                f"autocorrelation: unsupported monomial pair {key} on axis {k}.\n"
                "  Where: cubic_scattering/galerkin_propagator.py, autocorrelation\n"
                "  Valid: per-axis exponents 0, 1 or 2 (T9 uses 0-1, T27 adds 2)\n"
                "  Fix:   extend the closed forms above for higher degree (T57)"
            )
            raise ValueError(msg)
        out = out * fac
    return out


def _pyramid_nodes(lo: NDArray, hi: NDArray, apex: NDArray, n: int) -> tuple[NDArray, NDArray]:
    """Quadrature for a box, split into pyramids with apex at a singular point.

    Any point of a pyramid with apex at the origin and base on the plane
    w_d = c is w = t * base_point, t in [0,1], with volume element
    dw = t^2 |c| dt dA. A kernel behaving as 1/|w| then enters as
    t^2 / t = t -- smooth -- so the 1/rho singularity is integrated exactly in
    the radial variable rather than fought with more points.

    Faces containing the apex give degenerate (zero-volume) pyramids and are
    skipped, which is what makes a boundary singularity cheaper than an interior
    one.

    Returns (points, weights) in ABSOLUTE coordinates.
    """
    gx, gw = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (gx + 1.0)
    tw = 0.5 * gw

    pts: list[NDArray] = []
    wts: list[NDArray] = []
    for d in range(3):
        for c in (lo[d], hi[d]):
            cd = c - apex[d]
            if abs(cd) < 1e-14:
                continue  # apex lies in this face: degenerate pyramid
            o1, o2 = [k for k in range(3) if k != d]
            # Base rectangle on the plane w_d = cd, relative to the apex.
            a1 = 0.5 * (hi[o1] - lo[o1]) * gx + 0.5 * (hi[o1] + lo[o1]) - apex[o1]
            w1 = 0.5 * (hi[o1] - lo[o1]) * gw
            a2 = 0.5 * (hi[o2] - lo[o2]) * gx + 0.5 * (hi[o2] + lo[o2]) - apex[o2]
            w2 = 0.5 * (hi[o2] - lo[o2]) * gw

            T, A1, A2 = np.meshgrid(t, a1, a2, indexing="ij")
            TW, W1, W2 = np.meshgrid(tw, w1, w2, indexing="ij")
            w = np.empty(T.shape + (3,))
            w[..., d] = T * cd
            w[..., o1] = T * A1
            w[..., o2] = T * A2
            pts.append((w + apex).reshape(-1, 3))
            wts.append((TW * W1 * W2 * T**2 * abs(cd)).reshape(-1))
    return np.concatenate(pts), np.concatenate(wts)


def _tensor_nodes(lo: NDArray, hi: NDArray, n: int) -> tuple[NDArray, NDArray]:
    """Plain tensor-product Gauss on a box, for octants with no singularity."""
    gx, gw = np.polynomial.legendre.leggauss(n)
    ax = [0.5 * (hi[k] - lo[k]) * gx + 0.5 * (hi[k] + lo[k]) for k in range(3)]
    aw = [0.5 * (hi[k] - lo[k]) * gw for k in range(3)]
    p = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    w = np.prod(np.stack(np.meshgrid(*aw, indexing="ij"), axis=-1), axis=-1).ravel()
    return p, w


def _integration_nodes(a: float, r_vec: NDArray, n: int) -> tuple[NDArray, NDArray]:
    """Nodes for Int_{[-2a,2a]^3} G(u+R) C(u) du.

    THE DOMAIN IS SPLIT INTO THE EIGHT OCTANTS OF u, and this is not an
    optimisation. C_ab(u) is only PIECEWISE polynomial: b_k = a - |u_k|/2 puts a
    kink on each plane u_k = 0. A rule that straddles those kinks integrates a
    piecewise-smooth function with a smooth-function rule, and converges at a
    crawl -- measured, before this split was added: 8.7e-3, 2.6e-3, 1.6e-3,
    6.7e-4 over n_quad = 8..24, still 1.7e-3 short of an independently converged
    6-D reference.

    Within an octant C is a single polynomial. The singular point u = -R is
    handled by the apex-pyramid rule in whichever octants touch it, and the rest
    take plain tensor Gauss.
    """
    sing = -np.asarray(r_vec, dtype=float)
    pts: list[NDArray] = []
    wts: list[NDArray] = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                sgn = np.array([sx, sy, sz], dtype=float)
                lo = np.where(sgn < 0, -2.0 * a, 0.0)
                hi = np.where(sgn < 0, 0.0, 2.0 * a)
                # Does the singular point touch this octant's closure?
                touches = bool(np.all(sing >= lo - 1e-12) and np.all(sing <= hi + 1e-12))
                if touches:
                    p, w = _pyramid_nodes(lo, hi, np.clip(sing, lo, hi), n)
                else:
                    p, w = _tensor_nodes(lo, hi, n)
                pts.append(p)
                wts.append(w)
    return np.concatenate(pts), np.concatenate(wts)


def galerkin_block_9x9(
    r_vec: NDArray,
    d: float,
    omega: float,
    ref: ReferenceMedium,
    n_quad: int = 10,
    n_modes: int = 9,
) -> NDArray:
    """The 9x9 Bubnov-Galerkin coupling between two cells of side d at offset r_vec.

    ⚠ REAL omega only, because `elastodynamic_greens` is typed and written for a
    real frequency. Attenuative media -- which this project does care about, and
    which the Ewald lattice sum already supports -- would need that routine
    widened to complex first. Annotating `complex` here without doing so would
    only move the failure somewhere less obvious.

    Args:
        r_vec: Cell-centre separation (z, x, y), metres.
        d: Cell side (= 2a).
        omega: Angular frequency, REAL.
        ref: Background medium.
        n_quad: Gauss points per dimension of the reduced 3-D integral.

    Returns:
        (n_modes, n_modes) complex. Row = receiver trial function, column =
        source. n_modes is 9 for the T9 tier, 27 with the quadratic tier.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    a = 0.5 * d
    u, w = _integration_nodes(a, r_vec, n_quad)

    # Green's tensor at u + r_vec for every node, once.
    g = np.empty((u.shape[0], 3, 3), dtype=complex)
    for i, uu in enumerate(u):
        g[i] = elastodynamic_greens(uu + r_vec, omega, ref)

    # The autocorrelation depends only on the MONOMIAL PAIR, and the distinct
    # pairs are far fewer than the mode pairs -- 16 at T9, 100 at T27, against
    # 729 mode pairs each with several terms. Cache the weighted kernel per pair,
    # gathered from the basis actually in use rather than hard-coded, so the
    # quadratic tier needs no second list to keep in step.
    terms = [basis_terms(al) for al in range(n_modes)]
    exps = sorted({e for t in terms for _, e, _ in t})
    wc: dict[tuple[tuple[int, int, int], tuple[int, int, int]], NDArray] = {}
    for ea in exps:
        for eb in exps:
            wc[(ea, eb)] = w * autocorrelation(ea, eb, u, a)

    out = np.zeros((n_modes, n_modes), dtype=complex)
    for alpha in range(n_modes):
        for beta in range(n_modes):
            acc = 0.0 + 0.0j
            for ca, ea, da in terms[alpha]:
                for cb, eb, db in terms[beta]:
                    acc += ca * cb * np.dot(wc[(ea, eb)], g[:, da, db])
            out[alpha, beta] = acc
    return out


def gram_diagonal(d: float) -> NDArray:
    """The Gram matrix of the 9 trial functions over a cell of side d, diagonal.

    M_ab = Int_V phi_a . phi_b. The basis is orthogonal, so only the diagonal
    survives:
        constant modes        Int e_a . e_a          = V
        axial strain (p == q) Int s_p^2              = V d^2 / 12
        shear  (p != q)       Int |0.5(s_q e_p + s_p e_q)|^2
                              = 0.25 (Int s_q^2 + Int s_p^2) = V d^2 / 24

    The shear entry is HALF the axial one, which is the engineering-convention
    factor showing up in the mass rather than in the trial function.
    """
    v = d**3
    return v * np.array([1.0, 1.0, 1.0] + [d**2 / 12.0] * 3 + [d**2 / 24.0] * 3)


def galerkin_propagator_9x9(
    r_vec: NDArray,
    d: float,
    omega: float,
    ref: ReferenceMedium,
    n_quad: int = 10,
) -> NDArray:
    """Galerkin coupling NORMALISED for the solver's convention: M^-1 Gamma / V.

    WHY THE NORMALISATION IS WHAT IT IS. The Galerkin system is
    `M c = M c0 + Gamma q`, whereas the solver iterates `psi = psi0 + G0 T0 psi`
    with no Gram matrix. Two facts close the gap:

      * the trial functions are normalised so their coefficients ARE the
        9-component state -- the linear modes vanish at the cell centre, so
        c[0:3] is the centre displacement, and each linear mode carries unit
        Voigt strain -- hence psi = c, with no conversion;
      * `T0 = V . Delta c*` returns a cell TOTAL (force, moment), while the
        Galerkin source coefficient q is a DENSITY. That is the factor of V.

    So the object the solver needs is `M^-1 Gamma / V`.

    THE NORMALISATION IS ASYMMETRIC, and the asymmetry is the ENGINEERING SHEAR
    CONVENTION rather than an accident:

        rows (field side)   scaled by 1 / M          , M_shear = V d^2 / 24
        cols (source side)  scaled by 1 / M_src      , M_src   = V d^2 / 12 for
                                                       ALL SIX strain modes

    The field-side shear component is `2 eps` and carries the extra factor of
    two; the source-side shear is a stress and does not. That is the same
    asymmetry the moment propagator records as `H = W C^T`,
    `W = diag(1,1,1,2,2,2)`.

    ⚠ THIS TOOK THREE ATTEMPTS AND TWO OF THEM PASSED THE OBVIOUS CHECKS. The G
    block reduces to `<<G>>` to 5e-13 under ANY of them, because the constant
    mode's mass is V on both sides -- so that check, and reciprocity, and the
    centre-of-mass reduction, all pass while the strain sector is wrong. They
    compare the construction against itself.

    Only the FAR-FIELD LIMIT compares it against an INDEPENDENT object: at large
    R two cells look like points, so the coupling must reduce to the point
    propagator. Measured `point / Gamma_raw` at R = 12d gave block maxima
    1.0008, 12.035, 23.91, 289.5 -- that is rows 24 and columns 12 on the strain
    sector, which is what the scaling above encodes.

    The far-field limit is also what makes a short-ranged CORRECTION legitimate
    at all: `[Galerkin - point]` must vanish with R, or every extra shell adds
    spurious contribution. Under the first (row-only) attempt it tended to a
    non-zero constant, and the measured symptom was the conversion getting
    monotonically WORSE as its reach was extended.

    WHY THIS MATTERS BEYOND TIDINESS. A short-ranged CORRECTION is only
    legitimate if `[Galerkin - point]` vanishes at large separation. Under
    row-only scaling it tended to a non-zero constant, so every extra shell added
    spurious contribution -- measured, as monotone worsening with conversion
    reach. Under `M^-1 Gamma M^-1` the difference decays and the correction is
    well posed.

    Both limits are analytic: `Gamma_GG = V^2 <<G>>` gives `<<G>>`, and
    `Gamma_SS -> -(V d^2/12)^2 d_p d_q G` gives `-d_p d_q G`. Both are the point
    propagator's own blocks.
    """
    gam = galerkin_block_9x9(r_vec, d, omega, ref, n_quad)
    m_row = gram_diagonal(d)
    v = d**3
    # Source side: no engineering factor of two on the shear modes.
    m_col = v * np.array([1.0, 1.0, 1.0] + [d**2 / 12.0] * 6)
    return gam / m_row[:, None] / m_col[None, :]


def galerkin_plane_wave_state(k_vec: NDArray, pol: NDArray, centre: NDArray, d: float) -> NDArray:
    """A plane wave PROJECTED onto the 9 trial functions, not sampled at a point.

    The solver's incident field is `pol * exp(i k . r_centre)` with the strain
    read off analytically -- a POINT VALUE at the cell centre. That is a third
    convention alongside a Galerkin `T_0` and a Galerkin `G_0`, and finishing the
    conversion means projecting it too:

        c_a = (1 / M_aa) Int_V phi_a(s) . u0(centre + s) ds.

    Everything factorises because the cell is a box and the wave is a product:

        I0(k) = Int_{-d/2}^{d/2} e^{i k s} ds        = 2 sin(k d / 2) / k
        I1(k) = Int_{-d/2}^{d/2} s e^{i k s} ds      = -i dI0/dk

    THE LONG-WAVELENGTH LIMIT IS THE CHECK, and it is exact rather than
    approximate: I0 -> d and I1 -> i k d^3 / 12, so the constant modes tend to
    `pol` and the axial modes to `i k_p pol_p = eps_pp`, the shear modes to
    `i(k_q pol_p + k_p pol_q) = 2 eps_pq`. So this reduces to the existing
    point-sampled state as k d -> 0, and differs from it at O((k d)^2).
    """
    k_vec = np.asarray(k_vec, dtype=complex)
    pol = np.asarray(pol, dtype=complex)
    a = 0.5 * d

    def i0(k: complex) -> complex:
        # 2 sin(k a)/k, with the removable singularity at k = 0 handled.
        return complex(d) if abs(k) < 1e-12 else complex(2.0 * np.sin(k * a) / k)

    def i1(k: complex) -> complex:
        # Int s e^{iks} ds = -i d/dk [2 sin(k a)/k]
        if abs(k) < 1e-12:
            return 1j * 0.0
        return complex(-1j * (2.0 * a * np.cos(k * a) / k - 2.0 * np.sin(k * a) / k**2))

    i0s = [i0(k_vec[j]) for j in range(3)]
    i1s = [i1(k_vec[j]) for j in range(3)]
    phase = np.exp(1j * float(np.real(np.dot(k_vec, centre))))

    gram = gram_diagonal(d)

    out = np.zeros(9, dtype=complex)
    for alpha in range(9):
        acc = 0.0 + 0.0j
        for c, e, dirn in basis_terms(alpha):
            term = c * pol[dirn]
            for j in range(3):
                term *= i1s[j] if e[j] == 1 else i0s[j]
            acc += term
        out[alpha] = acc * phase / gram[alpha]  # gram already carries the V
    return out


__all__ = [
    "autocorrelation",
    "basis_terms",
    "galerkin_block_9x9",
    "galerkin_plane_wave_state",
    "galerkin_propagator_9x9",
    "gram_diagonal",
]
