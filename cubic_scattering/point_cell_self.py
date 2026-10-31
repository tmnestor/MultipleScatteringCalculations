"""The point (collocation) cell's self term and its exact single-site T-matrix, at any frequency.

WHAT THE POINT CELL NEEDS
-------------------------
The classical point-matching cell takes the nine-component state e = (u, strain) as uniform in the cube and
imposes the equation at the centre. Its own field there is S_self D e, with

    S_self = integral over the cube of P(y) dy,

P the 9x9 point propagator and D = diag(omega^2 drho I_3, dC_V) the contrast. So e = psi + S_self D e and

    T = d^3 D (I - S_self D)^{-1},

the uniform-strain closure. Statically S_self is the cube's depolarisation (Chiu 1977, Nozaki 2001). At
finite frequency it is new, and this module evaluates it exactly.

WHY THE FACES
-------------
The strain-from-moment block of P is a fourth derivative of D = g_S - g_P (and a second of g_S), which
decays as r^-3 at the centre: not absolutely integrable, and its value depends on the excluded region. The
integral is a distribution, and pairing with the cube's indicator moves one derivative onto the faces,

    integral_V d_{i1} d_{i2..in} f dV = sum over faces of n_{i1} integral_face d_{i2..in} f dA,

valid when d^{n-1} f is locally integrable (here at worst r^-2). The faces lie a distance h from the
centre, so the face integrands are smooth and a Gauss rule on each face converges exponentially. The
delta-function content of the inclusion problem, which a volume quadrature misses, is carried by the
divergence theorem itself. The undifferentiated g_S, weakly singular, is written as the divergence of the
radial field F(r) r_hat / r^2 with F(r) = integral_0^r f(s) s^2 ds, which moves it onto the faces too.

Nothing here is a series in k h, so the self term is exact at every frequency, not to a fixed order. The
same face route at a non-zero offset reproduces the receiver-averaged blocks of ``cell_averaged_pair``,
which is how it is checked.

Conventions inherited from ``kupradze_derivatives`` and ``resonance_tmatrix``: g = exp(i kappa r)/(4 pi r),
(z, x, y) ordering, time e^{-i omega t}, the 9x9 [[G, C], [H, S]] of ``_voigt_contract``.
"""

import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from .cell_averaged_pair import _assemble
from .effective_contrasts import CubeTMatrixResult, MaterialContrast, ReferenceMedium, compute_cube_tmatrix
from .kupradze_derivatives import (
    MAX_ORDER,
    difference_derivative_tensors,
    scalar_derivative_tensors,
)
from .voigt_tmatrix import effective_stiffness_voigt

#: Gauss points per axis on each quarter of a face. The integrand peaks at the face centre, a distance h
#: from the cube's centre, so on a quarter face its nearest complex singularity lies about one panel width
#: off the interval; 16 points give about 1e-13 (``test_point_cell_self``).
N_FACE = 16

#: Gauss points for F(r) = r^2 integral_0^1 t exp(i kappa r t) dt / (4 pi), smooth on [0, 1].
N_RADIAL = 24


def _face_nodes(centre: NDArray, h: float, n: int) -> list[tuple[int, float, NDArray, NDArray]]:
    """(axis, sign, points, weights) for the six faces, each split into four panels."""
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * h
    # two panels [-h, 0] and [0, h] per face axis
    u = np.concatenate([half * x - half, half * x + half])
    wu = np.concatenate([half * w, half * w])
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ww = np.outer(wu, wu).ravel()
    faces = []
    for axis in range(3):
        others = [b for b in range(3) if b != axis]
        for sign in (-1.0, 1.0):
            pts = np.empty((uu.size, 3))
            pts[:, axis] = centre[axis] + sign * h
            pts[:, others[0]] = centre[others[0]] + uu.ravel()
            pts[:, others[1]] = centre[others[1]] + vv.ravel()
            faces.append((axis, sign, pts, ww))
    return faces


def cube_integral_tensors(
    centre: NDArray,
    h: float,
    tensors_at: Callable[[NDArray], list[NDArray]],
    kappa: complex | None,
    order: int = MAX_ORDER,
    n_face: int = N_FACE,
) -> list[NDArray]:
    """Integrals over the cube of the derivative tensors d_{i1..in} f, n = 0..order, taken as distributions.

    Args:
        centre: The cube's centre relative to the source, shape (3,); the origin for the self term.
        h: The cube's half-width.
        tensors_at: The derivative tensors of f at a point, orders 0..order - 1 at least (as
            ``kupradze_derivatives.scalar_derivative_tensors``).
        kappa: The wavenumber when f = exp(i kappa r)/(4 pi r), whose order-0 integral is moved onto the
            faces by the radial field; None when f is regular at the origin (D = g_S - g_P), whose order-0
            integral is then not needed and returned as NaN.
        order: The highest derivative order.
        n_face: Gauss points per axis on each quarter face.

    Returns:
        A list whose n-th entry has shape (3,)*n, complex: integral_V d^n f dV (not the average).
    """
    c = np.asarray(centre, dtype=float)
    out: list[NDArray] = [np.zeros((3,) * n, dtype=complex) for n in range(order + 1)]
    out[0] = np.array(np.nan + 0j) if kappa is None else np.array(0j)
    t, wt = np.polynomial.legendre.leggauss(N_RADIAL)
    t, wt = 0.5 * (t + 1.0), 0.5 * wt
    for axis, sign, pts, wts in _face_nodes(c, h, n_face):
        for p, w in zip(pts, wts, strict=True):
            tens = tensors_at(p)
            for n in range(1, order + 1):
                # n_{i1} times the order n-1 tensor: only the face's own axis in the first index
                out[n][axis] = out[n][axis] + sign * w * tens[n - 1]
            if kappa is not None:
                r = float(np.linalg.norm(p))
                f_r = r**2 * np.sum(wt * t * np.exp(1j * kappa * r * t)) / (4.0 * np.pi)
                out[0] = out[0] + w * f_r * sign * p[axis] / r**3
    return out


def self_block_9x9(omega: complex, ref: ReferenceMedium, d: float, n_face: int = N_FACE) -> NDArray:
    """S_self: the 9x9 point propagator integrated over the cube of side ``d`` about the source, exactly.

    Args:
        omega: Angular frequency (may be complex).
        ref: Background medium.
        d: Cell SIDE.
        n_face: Gauss points per axis on each quarter face.

    Returns:
        Shape (9, 9), complex, in the layout of ``cell_averaged_pair.averaged_pair_block_9x9`` times d^3.
    """
    return offset_block_9x9(np.zeros(3), omega, ref, d, n_face=n_face)


def offset_block_9x9(
    s_vec: NDArray, omega: complex, ref: ReferenceMedium, d: float, n_face: int = N_FACE
) -> NDArray:
    """The 9x9 point propagator integrated over the cube of side ``d`` centred at ``s_vec`` from the source.

    At a non-zero lattice offset it is d^3 times ``averaged_pair_block_9x9``; at the origin it is S_self.
    """
    k_p, k_s = omega / ref.alpha, omega / ref.beta
    h = 0.5 * d
    d_diff = cube_integral_tensors(
        s_vec,
        h,
        lambda r: difference_derivative_tensors(r, k_p, k_s, MAX_ORDER - 1),
        None,
        n_face=n_face,
    )
    d_s = cube_integral_tensors(s_vec, h, lambda r: scalar_derivative_tensors(r, k_s, 1), k_s, 2, n_face)
    return _assemble(d_diff, d_s, omega, ref)


#: Terms of the series in k h of the self term, and the largest |k_S| h it is used for. Its terms grow to
#: about e^(sqrt(3) k h) before they cancel, which costs digits; against the face rule, measured
#: (31 October 2026):
#:   |k_S| h       <= 1     2        3        4        5        6        8
#:   worst error   9e-15    2.3e-14  7e-14    2.0e-13  7.5e-13  4.1e-12  9.5e-8
#: Beyond the reach the face rule is used directly (0.6 s rather than 0.6 ms).
SERIES_TERMS = 48
SERIES_REACH = 4.0


@lru_cache(maxsize=1)
def _unit_cube_moments() -> list[list[NDArray]]:
    """J[t][n] = integral over the cube [-1, 1]^3 of the derivative tensor d^n r^(t-1), n = 0..4, as
    distributions, for t = 0..SERIES_TERMS - 1. Frequency, cell size and medium free: computed once.

    The same face rule as ``cube_integral_tensors``, vectorised over the face points: for n >= 1 the face
    integral of n_{i1} d^(n-1) r^m with m = t - 1, the derivatives from the radial ladder
    f_k = (1/r d/dr)^k r^m = m (m - 2) ... (m - 2k + 2) r^(m - 2k); for n = 0 the radial field
    r^(m+3)/(m+3) x/r^3, whose divergence is r^m.
    """
    faces = _face_nodes(np.zeros(3), 1.0, N_FACE)
    x = np.concatenate([f[2] for f in faces])  # (P, 3)
    w = np.concatenate([f[3] for f in faces])
    nrm = np.concatenate(
        [np.tile(sign * np.eye(3)[axis], (len(f[3]), 1)) for f in faces for axis, sign in [f[:2]]]
    )
    r = np.linalg.norm(x, axis=1)
    eye = np.eye(3)
    out: list[list[NDArray]] = []
    for t in range(SERIES_TERMS):
        m = t - 1
        lad = []
        for k in range(4):
            c = 1.0
            for q in range(k):
                c *= m - 2 * q
            lad.append(c * r ** (m - 2 * k))
        f0, f1, f2, f3 = lad
        j0 = np.sum(w * r ** (m + 3) / (m + 3) * np.einsum("pi,pi->p", nrm, x) / r**3)
        d1 = f1[:, None] * x
        d2 = f1[:, None, None] * eye + f2[:, None, None] * np.einsum("pi,pj->pij", x, x)
        sym = (
            np.einsum("jk,pl->pjkl", eye, x)
            + np.einsum("jl,pk->pjkl", eye, x)
            + np.einsum("kl,pj->pjkl", eye, x)
        )
        d3 = f2[:, None, None, None] * sym + f3[:, None, None, None] * np.einsum("pj,pk,pl->pjkl", x, x, x)
        out.append(
            [
                np.array(j0),
                np.einsum("p,pi,p->i", w, nrm, f0),
                np.einsum("p,pi,pj->ij", w, nrm, d1),
                np.einsum("p,pi,pjk->ijk", w, nrm, d2),
                np.einsum("p,pi,pjkl->ijkl", w, nrm, d3),
            ]
        )
    return out


def self_block_9x9_series(omega: complex, ref: ReferenceMedium, d: float) -> NDArray:
    """S_self as the series in k h with the coefficients of ``_unit_cube_moments``: a short sum a frequency.

    g_c = (1/4 pi) sum_t (i k_c)^t r^(t-1) / t! and D = g_S - g_P, whose t = 0 term cancels; the integral
    over the cube of half-width h of d^n r^(t-1) is h^(t+2-n) J[t][n].
    """
    h = 0.5 * d
    k_p, k_s = omega / ref.alpha, omega / ref.beta
    moments = _unit_cube_moments()
    d_s = [np.zeros((3,) * n, dtype=complex) for n in range(3)]
    d_diff = [np.zeros((3,) * n, dtype=complex) for n in range(MAX_ORDER + 1)]
    for t, jt in enumerate(moments):
        inv = 1.0 / (4.0 * np.pi * math.factorial(t))
        c_s = (1j * k_s) ** t * inv
        c_d = (1j**t) * (k_s**t - k_p**t) * inv
        for n in range(MAX_ORDER + 1):
            term = h ** (t + 2 - n) * jt[n]
            if n < 3:
                d_s[n] = d_s[n] + c_s * term
            d_diff[n] = d_diff[n] + c_d * term
    return _assemble(d_diff, d_s, omega, ref)


def contrast_9x9(omega: complex, contrast: MaterialContrast) -> NDArray:
    """D = diag(omega^2 drho I_3, dC_V), the Born limit of ``_sub_cell_tmatrix_9x9`` divided by d^3."""
    out = np.zeros((9, 9), dtype=complex)
    out[:3, :3] = omega**2 * contrast.Drho * np.eye(3)
    out[3:, 3:] = effective_stiffness_voigt(contrast.Dlambda, contrast.Dmu, contrast.Dmu)
    return out


def exact_point_tmatrix_9x9(
    omega: complex, d: float, contrast: MaterialContrast, s_self: NDArray
) -> NDArray:
    """The uniform-strain closure T = d^3 D (I - S_self D)^{-1} with the exact self term.

    Args:
        omega: Angular frequency.
        d: Cell SIDE.
        contrast: The cell's contrast.
        s_self: ``self_block_9x9(omega, ref, d)``, which does not depend on the contrast.

    Returns:
        Shape (9, 9), complex, in the layout of ``_sub_cell_tmatrix_9x9``.
    """
    dm = contrast_9x9(omega, contrast)
    return d**3 * dm @ np.linalg.inv(np.eye(9) - s_self @ dm)


# --------------------------------------------------------------------------------------------------------
# The single site of a cube inside a lattice, as numerics.yml chooses it
# --------------------------------------------------------------------------------------------------------

NUMERICS_YAML = Path(__file__).with_name("numerics.yml")
SINGLE_SITES = ("exact", "package")
_SITE_EXAMPLE = "         single_site:\n           lattice: exact      # one of: exact, package"


@lru_cache(maxsize=1)
def lattice_single_site() -> str:
    """``single_site.lattice`` from ``cubic_scattering/numerics.yml``: a lattice cell's single site.

    Read once per process; there is no default.

    Raises:
        ValueError: with what is wrong, where to fix it, a valid example, and how to recover.
    """

    def error(problem: str) -> ValueError:
        return ValueError(
            f"single_site.lattice {problem}\n"
            f"  Where: {NUMERICS_YAML}, key single_site.lattice\n"
            f"  Valid:\n{_SITE_EXAMPLE}\n"
            "  Fix:   add the block above at the top level of that file; exact is the uniform-strain\n"
            "         closure, package the cube T-matrix with far-field form-factor corrections."
        )

    if not NUMERICS_YAML.is_file():
        raise error("cannot be read: the numerics configuration file is missing.")
    block = (yaml.safe_load(NUMERICS_YAML.read_text()) or {}).get("single_site")
    if not isinstance(block, dict) or "lattice" not in block:
        raise error("is missing.")
    site = block["lattice"]
    if site not in SINGLE_SITES:
        raise error(f"is {site!r}; it must be one of {SINGLE_SITES}.")
    return str(site)


#: S_self by (omega, d, alpha, beta, rho), the speeds kept complex for an attenuating medium; bounded so a
#: long sweep cannot grow it without limit (each entry is a 9x9 complex array, about 1.3 kB).
_SELF_BLOCKS: dict[tuple[complex, float, complex, complex, complex], NDArray] = {}
_SELF_BLOCKS_MAX = 100_000


def _cached_self_block(omega: complex, d: float, ref: ReferenceMedium) -> NDArray:
    """S_self for one frequency, cell size and medium; it does not depend on the contrast. Read-only. The
    series in k h within SERIES_REACH, the face rule beyond."""
    key = (complex(omega), float(d), complex(ref.alpha), complex(ref.beta), complex(ref.rho))
    block = _SELF_BLOCKS.get(key)
    if block is None:
        if abs(omega / ref.beta) * 0.5 * d <= SERIES_REACH:
            block = self_block_9x9_series(omega, ref, d)
        else:
            block = self_block_9x9(omega, ref, d)
        block.setflags(write=False)
        if len(_SELF_BLOCKS) < _SELF_BLOCKS_MAX:
            _SELF_BLOCKS[key] = block
    return block


def exact_cube_tmatrix(
    omega: complex, a: float, ref: ReferenceMedium, contrast: MaterialContrast
) -> CubeTMatrixResult:
    """The uniform-strain closure with the exact self term, as a CubeTMatrixResult's effective contrasts.

    The closure is cubic-symmetric, so its 9x9 T-matrix is fixed by four effective contrasts: T_uu =
    V omega^2 drho* I_3 and T_ee = V dC*_V(dlambda*, dmu*_diag, dmu*_off), the form
    ``resonance_tmatrix._sub_cell_tmatrix_9x9`` builds from them. The intermediate quantities of
    ``compute_cube_tmatrix`` (Gamma0, Ac, ..., the amplification factors) belong to its own derivation and
    are NaN here, so that code reading them fails visibly instead of mixing the two single sites.

    Args:
        omega: Angular frequency (may be complex).
        a: Cube HALF-width.
        ref: Background medium.
        contrast: The cube's contrast.

    Raises:
        ValueError: if the closure is not of the cubic form (a defect of the self term, not of the input).
    """
    d = 2.0 * a
    # the medium's speeds may be complex (an attenuating background): keyed as given, never cast to real
    s_self = _cached_self_block(omega, d, ref)  # the speeds may be complex (an attenuating medium)
    dm = contrast_9x9(omega, contrast)
    # (I - S D)^-1 is block diagonal by parity: the density and stiffness channels close separately
    drho_star = complex(contrast.Drho / (1.0 - s_self[0, 0] * dm[0, 0])) if contrast.Drho else 0j
    dc_star = dm[3:, 3:] @ np.linalg.inv(np.eye(6) - s_self[3:, 3:] @ dm[3:, 3:])
    dlam, dmu_diag, dmu_off = dc_star[0, 1], 0.5 * (dc_star[0, 0] - dc_star[0, 1]), 0.5 * dc_star[3, 3]
    rebuilt = effective_stiffness_voigt(dlam, dmu_diag, dmu_off)
    scale = max(float(np.max(np.abs(dc_star))), 1e-300)
    if float(np.max(np.abs(rebuilt - dc_star))) > 1e-9 * scale:
        raise ValueError(
            "the exact closure's stiffness block is not of the cubic form "
            "dC*(dlambda*, dmu*_diag, dmu*_off): "
            f"misfit {float(np.max(np.abs(rebuilt - dc_star))) / scale:.2e} of its largest entry.\n"
            "  Where: cubic_scattering/point_cell_self.py, exact_cube_tmatrix\n"
            "  Valid: a misfit at round-off, as test_point_cell_self checks\n"
            "  Fix:   check self_block_9x9 (the face rule N_FACE) for this omega and cell size."
        )
    nan = complex(np.nan, np.nan)
    return CubeTMatrixResult(
        Gamma0=nan,
        Ac=nan,
        Bc=nan,
        Cc=nan,
        T1c=nan,
        T2c=nan,
        T3c=nan,
        amp_u=nan,
        amp_theta=nan,
        amp_e_off=nan,
        amp_e_diag=nan,
        Drho_star=drho_star,
        Dlambda_star=complex(dlam),
        Dmu_star_off=complex(dmu_off),
        Dmu_star_diag=complex(dmu_diag),
    )


def lattice_cube_tmatrix(
    omega: complex,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    k_hat: NDArray | None = None,
) -> CubeTMatrixResult:
    """The single site of a cube inside a lattice, the one ``single_site.lattice`` in numerics.yml names.

    ``k_hat`` is passed to ``compute_cube_tmatrix`` (its directional form factor) when that is the site.

    Every solver that couples cells takes its cell T-matrix from here; a lone cube takes
    ``compute_cube_tmatrix`` directly (``tests/test_single_site_guard.py`` keeps the two apart).
    """
    if lattice_single_site() == "exact":
        return exact_cube_tmatrix(omega, a, ref, contrast)  # isotropic: k_hat has no part in it
    return compute_cube_tmatrix(omega, a, ref, contrast, k_hat=k_hat)
