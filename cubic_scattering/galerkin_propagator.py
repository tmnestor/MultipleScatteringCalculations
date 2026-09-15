"""The 9x9 BUBNOV-GALERKIN inter-cell propagator.

WHY THIS EXISTS. The lattice solver currently pairs a Galerkin T-matrix with a
MOMENT propagator, and the two are not the same operator. `T_0` is the first tier
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
    if alpha < 0 or alpha > 8:
        msg = (
            f"basis index must be in 0..8, got {alpha}.\n"
            "  Where: cubic_scattering/galerkin_propagator.py, basis_terms(alpha)\n"
            "  Valid: 0-2 constant displacement, 3-8 linear (Voigt strain) modes\n"
            "  Fix:   the 9-component state is (u_z,u_x,u_y, e_zz,e_xx,e_yy,"
            " 2e_xy,2e_zy,2e_zx)"
        )
        raise ValueError(msg)
    if alpha < 3:
        return [(1.0, (0, 0, 0), alpha)]
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
        pa, pb = ea[k], eb[k]
        if pa == 0 and pb == 0:
            fac = 2.0 * bk
        elif pa == 1 and pb == 0:
            fac = bk * uk
        elif pa == 0 and pb == 1:
            fac = -bk * uk
        else:
            fac = 2.0 * bk**3 / 3.0 - bk * uk**2 / 2.0
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
        (9, 9) complex. Row = receiver trial function, column = source.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    a = 0.5 * d
    u, w = _integration_nodes(a, r_vec, n_quad)

    # Green's tensor at u + r_vec for every node, once.
    g = np.empty((u.shape[0], 3, 3), dtype=complex)
    for i, uu in enumerate(u):
        g[i] = elastodynamic_greens(uu + r_vec, omega, ref)

    out = np.zeros((9, 9), dtype=complex)
    for alpha in range(9):
        for beta in range(9):
            acc = 0.0 + 0.0j
            for ca, ea, da in basis_terms(alpha):
                for cb, eb, db in basis_terms(beta):
                    c_u = autocorrelation(ea, eb, u, a)
                    acc += ca * cb * np.sum(w * c_u * g[:, da, db])
            out[alpha, beta] = acc
    return out


__all__ = ["autocorrelation", "basis_terms", "galerkin_block_9x9"]
