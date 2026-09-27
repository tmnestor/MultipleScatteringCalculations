"""The SAME-PLANE (dz = 0) lattice sum, averaged over the source cell.

WHY THIS FILE EXISTS, AND WHY THE FORM FACTOR IS NOT THE ANSWER HERE. For
dz != 0 the Bloch kernel is a plane-wave sum, so averaging a cube-shaped source
cell is exactly a multiplication by the form factor sinc(k_x h) sinc(k_y h)
sinc(k_z h) -- one power, the SINGLE average a collocation scheme needs.
`sweep_kernels.vertical_kernel_9x9(..., cell_half_width=h)` does that.

At dz = 0 it cannot. The route there is EWALD, and Ewald deliberately moves part
of the sum back into REAL space precisely because the plain reciprocal sum
diverges at equal depth. A form factor is the Fourier transform of the cell
indicator, so it multiplies plane waves; it has no meaning against a spatial
summation. And the plain reciprocal sum cannot be substituted: with the SINGLE
form factor its terms decay as 1/|G|^2 against ~|G| growth of the strain block
and ~|G| states per shell -- log-divergent. (The DOUBLE form factor gains a
further 1/|G|^2 and does converge, which is why `inter_voxel_propagator`'s
sinc^2 route exists and never met this wall.)

THE ROUTE TAKEN INSTEAD: AN EXACT ANALYTIC TAIL. The midpoint error of a cell
average is

    <g> - g = (d^2/24) grad^2 g + O(d^4),

and for the Helmholtz kernel grad^2 g_c = -k_c^2 g_c EXACTLY away from the
origin. Derivatives commute with grad^2, so the same holds for every derivative
tensor. Hence, beyond a near shell where the average is done directly,

    sum_{|R| > R0} <D>  =  (1 - k_c^2 d^2 / 24) * (D_full - D_near) + O(d^4),

and D_full is exactly what the existing Ewald sum returns. So the tail needs no
Ewald surgery at all -- it reuses `origin_scalar_tensors`, which is already
called once per mode (k_P and k_S), which is also the granularity the tail needs
because each mode carries its own k^2.

⚠ WHY A TAIL IS NEEDED AT ALL, rather than just summing further. The correction
<D> - D has an O(1/R) tail and, past k r ~ 1, the sum becomes CONDITIONALLY
convergent and summation-shape dependent (`scripts/investigate_correction_tail_shape.py`).
Enlarging the box cannot converge it: the O(1/R) tail needs R ~ 80 while the
shape term switches on near R ~ 1/(k_S d) ~ 50. The tail term above removes the
slowly-convergent part in closed form, which is the only way out of that.

⚠ WHAT THE O(d^4) MEANS. The tail is exact through d^2. The residual is the
next term of the Euler-Maclaurin-style cell expansion, which involves the cube's
non-isotropic fourth moments and decays two further powers in R. R0 is therefore
a CONVERGENCE parameter with a measurable effect, and `R0-independence` is the
sharp test of the construction -- if the tail were wrong, the answer would drift
with R0.

AT k_par = 0 NO LATTICE SUM IS NEEDED AT ALL: THE TILING IDENTITY. Cubes tile the
plane, so the source-cell-averaged sum over ALL cells, the cube's own included, is
the field at the mid-plane of a uniform plate of thickness d -- a 1-D closed form.
Removing the self cell (which the T-matrix carries) leaves the exact kernel,
``exact_same_plane_9x9 = P0 - self/d^3``, which ``averaged_same_plane_9x9`` returns
at k_par = 0. Measured against the exact analytic value
(``Mathematica/ContinuumLimit_AveragedSums.wl``), the Ewald + near-shell + tail route
below carried a STATIC strain-from-moment error of 9.4e-4 at its defaults (r0_cells
2, n_gauss 6), falling to 3.6e-4 at n_gauss 20 and 2.6e-5 at (8, 20): both the
near-shell quadrature at face contact and the O(d^4) tail truncation contribute.
The route remains in use at k_par != 0, where the form factor's zeros shift and the
identity does not apply directly.

Conventions inherited: (z, x, y) ordering, time e^{-i w t}, outgoing h^(1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import ReferenceMedium
from .kupradze_derivatives import MAX_ORDER, scalar_derivative_tensors
from .lattice_kupradze import origin_scalar_tensors


def _check_r0_cells(r0_cells: int) -> None:
    """Reject a near shell that cannot contain the nearest neighbours."""
    if r0_cells < 1:
        msg = (
            f"r0_cells must be >= 1, got {r0_cells}.\n"
            "  Where: cubic_scattering/cell_averaged_lattice.py,\n"
            "         averaged_origin_scalar_tensors() / averaged_same_plane_9x9()\n"
            "  Valid: a near-shell Chebyshev radius in cells, >= 1. The shell\n"
            "         must contain at least the nearest neighbours, where the\n"
            "         d^2 tail is least accurate.\n"
            "  Fix:   pass r0_cells=2 (the default) or larger"
        )
        raise ValueError(msg)


def _cell_nodes(h: float, n_gauss: int) -> tuple[NDArray, NDArray]:
    """Gauss nodes and weights for the SINGLE cell average on [-h, h]."""
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    return h * x, 0.5 * w


def averaged_origin_scalar_tensors(
    kappa: complex,
    eta: float,
    n_real: int,
    n_recip: int,
    a_l: float,
    k_par: NDArray,
    *,
    r0_cells: int = 2,
    n_gauss: int = 6,
    order: int = MAX_ORDER,
) -> list[NDArray]:
    """Same-plane Bloch sum over R != 0, averaged over a cubic SOURCE cell.

    Near shell direct, far field by the analytic d^2 tail::

        <D> = sum_{0 < |R| <= R0} <D_R>  +  (1 - kappa^2 d^2/24)(D_full - D_near)

    Args:
        kappa: Scalar wavenumber for ONE mode (k_P or k_S) -- the tail carries
            kappa^2, so the two modes must not be mixed before this point.
        eta: Ewald splitting parameter.
        n_real: Real-space half-width for the Ewald sum.
        n_recip: Reciprocal half-width for the Ewald sum.
        a_l: Lattice pitch, equal to the cube side.
        k_par: In-plane Bloch vector, shape (2,).
        r0_cells: Near-shell Chebyshev radius R0, in cells. A CONVERGENCE
            parameter, not a constant: raise it until the answer stops moving.
        n_gauss: Gauss points per axis for the near-shell cell average.
        order: Highest derivative order.

    Returns:
        A list whose n-th entry has shape (3,)*n, complex.
    """
    _check_r0_cells(r0_cells)

    h = 0.5 * a_l
    nodes, wts = _cell_nodes(h, n_gauss)

    near_avg: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    near_plain: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]

    for i in range(-r0_cells, r0_cells + 1):
        for j in range(-r0_cells, r0_cells + 1):
            if i == 0 and j == 0:
                continue
            # Separation from the field point at the origin to lattice site R,
            # in (z, x, y). Matches origin_scalar_tensors' sign convention.
            s_vec = np.array([0.0, -a_l * i, -a_l * j])
            phase = np.exp(1j * a_l * (k_par[0] * i + k_par[1] * j))

            for n, t in enumerate(scalar_derivative_tensors(s_vec, kappa, order)):
                near_plain[n] = near_plain[n] + phase * t

            # The cell average of this one term, by product Gauss. Regular:
            # |s| >= a_l and |u| <= (sqrt3/2) a_l, so |s - u| >= 0.134 a_l.
            for iz, uz in enumerate(nodes):
                for ix, ux in enumerate(nodes):
                    for iy, uy in enumerate(nodes):
                        wgt = wts[iz] * wts[ix] * wts[iy] * phase
                        shifted = s_vec - np.array([uz, ux, uy])
                        for n, t in enumerate(scalar_derivative_tensors(shifted, kappa, order)):
                            near_avg[n] = near_avg[n] + wgt * t

    full = origin_scalar_tensors(kappa, eta, n_real, n_recip, a_l, k_par, order)

    # The analytic tail. grad^2 g = -kappa^2 g exactly away from the origin, so
    # the cell average of the FAR field is a pure scalar multiple of it.
    tail_factor = 1.0 - (kappa**2) * (a_l**2) / 24.0
    return [near_avg[n] + tail_factor * (full[n] - near_plain[n]) for n in range(order + 1)]


def averaged_same_plane_9x9(
    d: float,
    omega: float,
    ref: ReferenceMedium,
    k_par: NDArray,
    *,
    eta: float | None = None,
    cutoff: int = 4,
    r0_cells: int = 2,
    n_gauss: int = 6,
) -> NDArray:
    """The dz = 0 Bloch block with the source cell averaged, as a 9x9.

    At k_par = 0 this is the exact closed form ``exact_same_plane_9x9`` (the
    tiling identity; see the module docstring) and no lattice sum is formed.
    Elsewhere the two modes are averaged SEPARATELY and only then assembled,
    because the tail factor carries kappa^2 and P and S do not share it.
    """
    _check_r0_cells(r0_cells)
    if not np.any(np.asarray(k_par)):
        return exact_same_plane_9x9(d, omega, ref)

    from .kupradze_derivatives import greens_from_scalars
    from .resonance_tmatrix import _voigt_contract

    eta_val = float(np.sqrt(np.pi) / d) if eta is None else float(eta)
    # Passed explicitly rather than through a **kwargs dict: mypy cannot keep
    # per-key types through `dict[str, object]`, and this call is exactly where
    # a mode mix-up would be invisible.
    d_p = averaged_origin_scalar_tensors(
        omega / ref.alpha,
        eta_val,
        cutoff,
        cutoff,
        d,
        k_par,
        r0_cells=r0_cells,
        n_gauss=n_gauss,
    )
    d_s = averaged_origin_scalar_tensors(
        omega / ref.beta,
        eta_val,
        cutoff,
        cutoff,
        d,
        k_par,
        r0_cells=r0_cells,
        n_gauss=n_gauss,
    )

    g, gd, gdd = greens_from_scalars(d_p, d_s, omega, ref)
    c, h_blk, s = _voigt_contract(gd, gdd)
    out = np.zeros((9, 9), dtype=complex)
    out[:3, :3] = g
    out[:3, 3:] = c
    out[3:, :3] = h_blk
    out[3:, 3:] = s
    return out


# (z, x, y) index pairs of the strain rows / moment columns: e_zz e_xx e_yy 2e_xy 2e_zy 2e_zx
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_ENG = (1, 1, 1, 2, 2, 2)


def plate_average_9x9(d: float, omega: complex, ref: ReferenceMedium) -> NDArray:
    """The plane Green's tensor at k_par = 0, averaged over a source slab of thickness d, per cell.

    ``P0 = (1/d^2) (1/d) int_{-d/2}^{d/2} g_plate(-z') dz'``: the field at the mid-plane of a uniform
    plate of thickness d, per unit cell area.  At k_par = 0 only the vertical wavenumber acts, so
    each entry is ``c0 delta(z) + A / (M k_z^2 - rho w^2)`` in the 1-D transform, and averages to
    ``c0/d + A (e^{i k h} - 1)/(d M k^2)``.  Nonzero entries: u <- f on the diagonal, and the three
    normal strain-from-moment entries, whose delta weights -1/M_P and -1/(2 mu) are the plate's
    static depolarisation.

    Args:
        d: Lattice pitch (cube side).
        omega: Angular frequency.
        ref: Background medium (real or attenuative).

    Returns:
        Shape (9, 9) complex.
    """
    h = 0.5 * d
    m_p, m_s = ref.rho * ref.alpha**2, ref.rho * ref.beta**2
    k_p, k_s = omega / ref.alpha, omega / ref.beta
    rw2 = ref.rho * omega**2

    def avg(m: complex, k: complex) -> complex:
        return complex((np.exp(1j * k * h) - 1.0) / (d * m * k**2))

    out = np.zeros((9, 9), dtype=complex)
    out[0, 0] = avg(m_p, k_p)
    out[1, 1] = out[2, 2] = avg(m_s, k_s)
    out[3, 3] = -1.0 / (m_p * d) - (rw2 / m_p) * avg(m_p, k_p)  # e_zz <- M_zz
    shear = -1.0 / (2.0 * m_s * d) - (rw2 / (2.0 * m_s)) * avg(m_s, k_s)
    out[7, 7] = out[8, 8] = shear  # 2e_zy <- M_zy, 2e_zx <- M_zx
    return out / d**2


def cube_self_9x9(d: float, omega: complex, ref: ReferenceMedium) -> NDArray:
    """The cube's own cell integral int_cube Gamma(-y) dy, as a 9x9 -- the term the T-matrix carries.

    Built from the T-matrix's own integrals, so the kernel and the single site agree on what the
    self cell is: ``Gamma0 = int_cube G_11`` for u <- f, and ``I_ijkl = int_cube d_k d_l G_ij``
    (``A^c, B^c, C^c``, static Eshelby plus radiation) for strain <- moment.  The odd blocks vanish
    by the cube's symmetry.

    Args:
        d: Cube side.
        omega: Angular frequency.
        ref: Background medium.

    Returns:
        Shape (9, 9) complex.
    """
    from .effective_contrasts import _compute_ABC_polynomial, _compute_Gamma0_analytical

    a = 0.5 * d
    g0 = _compute_Gamma0_analytical(omega, a, ref.alpha, ref.beta, ref.rho)
    ac, bc, cc = _compute_ABC_polynomial(omega, a, ref.alpha, ref.beta, ref.rho)

    def i_t(i: int, j: int, k: int, l: int) -> complex:  # noqa: E741 -- I_ijkl = int d_k d_l G_ij
        return complex(
            ac * (i == j) * (k == l)
            + bc * ((i == k) * (j == l) + (i == l) * (j == k))
            + cc * (i == j == k == l)
        )

    out = np.zeros((9, 9), dtype=complex)
    out[:3, :3] = g0 * np.eye(3)
    for v, (ra, rb) in enumerate(_VOIGT):
        for w, (ca, ce) in enumerate(_VOIGT):
            # row (ra, rb) engineering, column (ca, ce) symmetrised; int d_a d_e G_bc = I_{b c a e}
            out[3 + v, 3 + w] = (
                _ENG[v]
                * 0.25
                * (i_t(rb, ca, ra, ce) + i_t(rb, ce, ra, ca) + i_t(ra, ca, rb, ce) + i_t(ra, ce, rb, ca))
            )
    return out


def exact_same_plane_9x9(d: float, omega: complex, ref: ReferenceMedium) -> NDArray:
    """The same-plane cell-averaged lattice sum at k_par = 0, EXACTLY: ``P0 - self / d^3``.

    Cubes tile the plane, so the sum over ALL cells of the source-cell-averaged Green's tensor, the
    cube's own included, is the field at the mid-plane of a uniform plate of thickness d
    (``plate_average_9x9``).  Removing the self cell, which lives in the T-matrix, leaves the sum
    the Foldy-Lax kernel needs -- with no lattice sum, Ewald split, near shell or tail.  Proved and
    checked analytically in ``Mathematica/ContinuumLimit_AveragedSums.wl`` (static, all 21
    components of the plate identity to 4e-9); the Ewald + near-shell + O(d^2)-tail route it
    replaces here carried a static strain-from-moment error of 9.4e-4 at its defaults.

    Args:
        d: Lattice pitch (cube side).
        omega: Angular frequency.
        ref: Background medium.

    Returns:
        Shape (9, 9) complex.
    """
    return plate_average_9x9(d, omega, ref) - cube_self_9x9(d, omega, ref) / d**3
