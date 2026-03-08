"""Sphere vs cubic T-matrix validation module.

Provides:
  1. Rayleigh sphere T-matrix (analytical, isotropic C=0)
  2. Sphere decomposition via Foldy-Lax (voxelized sphere)
  3. Elastic Mie theory (partial wave expansion)
  4. Far-field scattering amplitudes

The sphere has isotropic symmetry (no cubic anisotropy), so comparison
with the cubic T-matrix infrastructure provides an independent
validation of the multiple scattering formalism.

References:
    Shekhar et al. (2023) - Small sphere self-consistent solution
    Pao & Mow (1973) - Elastic Mie scattering
    Korneev & Johnson (1993) - Elastic scattering from a sphere
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate
from scipy.special import hankel1, jv, lpmv

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    _compute_T123,
    compute_cube_tmatrix,
)
from .resonance_tmatrix import (
    _build_incident_field_coupled,
    _propagator_block_9x9,
    _sub_cell_tmatrix_9x9,
    sub_cell_centres,
)

# =====================================================================
# Result dataclasses
# =====================================================================


@dataclass
class SphereTMatrixResult:
    """Result from Rayleigh sphere T-matrix computation.

    Attributes:
        Gamma0: Volume integral of Green's tensor trace.
        A: Second-derivative integral A (isotropic).
        B: Second-derivative integral B (isotropic).
        T1: T-matrix coupling coefficient T1.
        T2: T-matrix coupling coefficient T2.
        amp_u: Displacement amplification factor.
        amp_theta: Dilatation amplification factor.
        amp_dev: Deviatoric strain amplification factor.
        Drho_star: Effective density contrast.
        Dlambda_star: Effective bulk modulus contrast.
        Dmu_star: Effective shear modulus contrast (single value, isotropic).
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrast.
    """

    Gamma0: complex
    A: complex
    B: complex
    T1: complex
    T2: complex
    amp_u: complex
    amp_theta: complex
    amp_dev: complex
    Drho_star: complex
    Dlambda_star: complex
    Dmu_star: complex
    omega: float
    radius: float
    ref: ReferenceMedium
    contrast: MaterialContrast


@dataclass
class SphereDecompositionResult:
    """Result from Foldy-Lax sphere decomposition.

    Attributes:
        T3x3: 3x3 effective displacement T-matrix.
        T_comp_9x9: Full 9x9 composite T-matrix.
        centres: Sub-cell centre coordinates, shape (N, 3).
        n_sub: Number of sub-cells per edge of bounding cube.
        n_cells: Number of cells inside sphere.
        a_sub: Sub-cell half-width (m).
        condition_number: Condition number of Foldy-Lax matrix.
        psi_exc: Exciting field solution, shape (9*N, 9).
        omega: Angular frequency.
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrast.
    """

    T3x3: NDArray[np.complexfloating]
    T_comp_9x9: NDArray[np.complexfloating]
    centres: NDArray[np.floating]
    n_sub: int
    n_cells: int
    a_sub: float
    condition_number: float
    psi_exc: NDArray[np.complexfloating]
    omega: float
    radius: float
    ref: ReferenceMedium
    contrast: MaterialContrast


@dataclass
class MieResult:
    """Result from elastic Mie theory computation.

    Attributes:
        a_n: P-wave scattering coefficients, shape (n_max+1,).
            a_n[n] for angular order n=0,...,n_max.
        b_n: SV-wave scattering coefficients, shape (n_max+1,).
            b_n[0] = 0 (no SV monopole).
        c_n: SH scattering coefficients, shape (n_max+1,).
            c_n[0] = 0 (no SH monopole).
        n_max: Maximum angular order.
        omega: Angular frequency.
        radius: Sphere radius.
        ref: Background medium.
        contrast: Material contrast.
        ka_P: Dimensionless P-wave frequency.
        ka_S: Dimensionless S-wave frequency.
    """

    a_n: NDArray[np.complexfloating]
    b_n: NDArray[np.complexfloating]
    c_n: NDArray[np.complexfloating]
    n_max: int
    omega: float
    radius: float
    ref: ReferenceMedium
    contrast: MaterialContrast
    ka_P: float
    ka_S: float


# =====================================================================
# Phase 1: Rayleigh Sphere T-Matrix
# =====================================================================


def _sphere_radial_fg(
    r: float,
    omega: float,
    alpha: float,
    beta: float,
    rho: float,
) -> tuple[complex, complex]:
    """Radial Green's tensor functions f(r) and g(r).

    The Green's tensor decomposes as G_{ij}(x) = f(r) delta_{ij} + g(r) n_i n_j
    where n_i = x_i/r.

    Args:
        r: Radial distance (m), must be > 0.
        omega: Angular frequency (rad/s).
        alpha: P-wave velocity (m/s).
        beta: S-wave velocity (m/s).
        rho: Density (kg/m^3).

    Returns:
        (f, g) complex radial functions.
    """
    c_nf = (1.0 / (8.0 * np.pi * rho * beta**2)) * (1.0 - beta**2 / alpha**2)

    exp_p = np.exp(1j * omega * r / alpha)
    exp_s = np.exp(1j * omega * r / beta)

    f = -c_nf / r + exp_s / (4.0 * np.pi * rho * beta**2 * r)
    g = (
        3.0 * c_nf / r
        + exp_p / (4.0 * np.pi * rho * alpha**2 * r)
        - exp_s / (4.0 * np.pi * rho * beta**2 * r)
    )

    return f, g


def _sphere_Gamma0(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
) -> complex:
    """Volume integral of Green's tensor diagonal over sphere.

    Gamma0 = integral_0^a r^2 [4*pi*f(r) + (4*pi/3)*g(r)] dr

    where the angular integrals have been evaluated analytically using
    spherical symmetry.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.

    Returns:
        Gamma0 (complex).
    """
    alpha, beta, rho = ref.alpha, ref.beta, ref.rho

    def integrand_real(r: float) -> float:
        f, g = _sphere_radial_fg(r, omega, alpha, beta, rho)
        val = r**2 * (4.0 * np.pi * f + (4.0 * np.pi / 3.0) * g)
        return float(np.real(val))

    def integrand_imag(r: float) -> float:
        f, g = _sphere_radial_fg(r, omega, alpha, beta, rho)
        val = r**2 * (4.0 * np.pi * f + (4.0 * np.pi / 3.0) * g)
        return float(np.imag(val))

    re, _ = integrate.quad(integrand_real, 0, radius, limit=100)
    im, _ = integrate.quad(integrand_imag, 0, radius, limit=100)

    return complex(re, im)


def _sphere_AB(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
) -> tuple[complex, complex]:
    """Green's tensor second-derivative integral A, B for a sphere.

    For an isotropic sphere:
        I_{ijkl} = A δ_{ij}δ_{kl} + B (δ_{ik}δ_{jl} + δ_{il}δ_{jk})

    (No C term since sphere has full rotational symmetry.)

    The volume integral ∫_V ∂²G_{ij}/∂x_k∂x_l dV has two contributions:

    1. **Eshelby (static, singular)**: The Green's tensor 1/r singularity
       produces a delta function under second differentiation. This gives
       the exact Eshelby S-tensor for a sphere and is independent of ω:

           A_E = -(9 - 10ν) / (30μ(1 - ν))
           B_E = 1 / (30μ(1 - ν))

       where ν is the background Poisson ratio.

    2. **Dynamic (smooth, O(ω²))**: The oscillating exponentials
       exp(iωr/α), exp(iωr/β) give frequency-dependent corrections
       that vanish as ω → 0.

    In the Rayleigh regime (ka ≪ 1), the dynamic correction is O(ka²)
    relative to the Eshelby part and can be safely neglected.

    The Eshelby result is **exact for all contrast levels** — the
    amplification factors it produces match the exact Mie solution
    in the ka → 0 limit regardless of material contrast magnitude.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.

    Returns:
        (A, B) complex, dominated by the Eshelby contribution.
    """
    # Analytical Eshelby S-tensor for a sphere in an isotropic matrix.
    # Derived from S_A = (5ν-1)/(15(1-ν)), S_B = (4-5ν)/(15(1-ν))
    # and the relation S^E = -S^code : C^0 which inverts to give:
    #   A = -(9-10ν)/(30μ(1-ν))
    #   B = 1/(30μ(1-ν))
    nu = ref.lam / (2.0 * (ref.lam + ref.mu))
    A_eshelby = -(9.0 - 10.0 * nu) / (30.0 * ref.mu * (1.0 - nu))
    B_eshelby = 1.0 / (30.0 * ref.mu * (1.0 - nu))

    # Dynamic correction: integrate the smooth (non-singular) part of
    # the Green's tensor second derivative.  This is the difference
    # between the full dynamic integrand and its static 1/r limit.
    # At ka << 1 this correction is O(ka²) and negligible for the
    # effective contrast extraction.
    ka_S = omega * radius / ref.beta
    if ka_S < 0.3:
        # Deep Rayleigh: dynamic correction is negligible
        return complex(A_eshelby), complex(B_eshelby)

    # For larger ka: numerically integrate the dynamic correction
    # by subtracting the static 1/r integrand (which integrates to zero
    # by cancellation) from the full dynamic integrand.  This avoids
    # the catastrophic cancellation that plagues the naive integration.
    alpha, beta, rho = ref.alpha, ref.beta, ref.rho

    # Static radial functions: f_s = F/r, g_s = G/r
    F_static = (ref.lam + 3.0 * ref.mu) / (
        8.0 * np.pi * ref.mu * (ref.lam + 2.0 * ref.mu)
    )
    G_static = (ref.lam + ref.mu) / (8.0 * np.pi * ref.mu * (ref.lam + 2.0 * ref.mu))

    def _fg_dynamic(r: float) -> tuple[complex, complex]:
        """Return (f_dyn, g_dyn) = f(r) - F/r, g(r) - G/r."""
        f, g = _sphere_radial_fg(r, omega, alpha, beta, rho)
        return f - F_static / r, g - G_static / r

    dr = 1e-8 * radius

    def _h_dyn_deriv(r: float) -> tuple[complex, complex]:
        """(h_dyn', h_dyn'') for h_dyn = 3f_dyn + g_dyn."""

        def hd(rr: float) -> complex:
            fd, gd = _fg_dynamic(rr)
            return 3.0 * fd + gd

        hp = (hd(r + dr) - hd(r - dr)) / (2.0 * dr)
        hpp = (hd(r + dr) - 2.0 * hd(r) + hd(r - dr)) / dr**2
        return hp, hpp

    def _q_dyn_and_deriv(r: float) -> tuple[complex, complex]:
        """(q_dyn, q_dyn') for the dynamic correction to q = f'+g'+2g/r."""
        fd_p, gd_p = _fg_dynamic(r + dr)
        fd_m, gd_m = _fg_dynamic(r - dr)
        fd_c, gd_c = _fg_dynamic(r)
        fp = (fd_p - fd_m) / (2.0 * dr)
        gp = (gd_p - gd_m) / (2.0 * dr)
        q = fp + gp + 2.0 * gd_c / r
        fd_pp, gd_pp = _fg_dynamic(r + 2.0 * dr)
        fd_mm, gd_mm = _fg_dynamic(r - 2.0 * dr)
        fp_p = (fd_pp - fd_c) / (2.0 * dr)
        gp_p = (gd_pp - gd_c) / (2.0 * dr)
        q_plus = fp_p + gp_p + 2.0 * gd_p / (r + dr)
        fp_m = (fd_c - fd_mm) / (2.0 * dr)
        gp_m = (gd_c - gd_mm) / (2.0 * dr)
        q_minus = fp_m + gp_m + 2.0 * gd_m / (r - dr)
        qp = (q_plus - q_minus) / (2.0 * dr)
        return q, qp

    def g1_re(r: float) -> float:
        hp, hpp = _h_dyn_deriv(r)
        return float(np.real(r**2 * ((4 * np.pi / 3) * hpp + (8 * np.pi / 3) * hp / r)))

    def g1_im(r: float) -> float:
        hp, hpp = _h_dyn_deriv(r)
        return float(np.imag(r**2 * ((4 * np.pi / 3) * hpp + (8 * np.pi / 3) * hp / r)))

    def g2_re(r: float) -> float:
        q, qp = _q_dyn_and_deriv(r)
        return float(np.real(r**2 * ((4 * np.pi / 3) * qp + (8 * np.pi / 3) * q / r)))

    def g2_im(r: float) -> float:
        q, qp = _q_dyn_and_deriv(r)
        return float(np.imag(r**2 * ((4 * np.pi / 3) * qp + (8 * np.pi / 3) * q / r)))

    r_min = 1e-6 * radius
    Gamma1_re, _ = integrate.quad(g1_re, r_min, radius, limit=200)
    Gamma1_im, _ = integrate.quad(g1_im, r_min, radius, limit=200)
    Gamma1 = complex(Gamma1_re, Gamma1_im)
    Gamma2_re, _ = integrate.quad(g2_re, r_min, radius, limit=200)
    Gamma2_im, _ = integrate.quad(g2_im, r_min, radius, limit=200)
    Gamma2 = complex(Gamma2_re, Gamma2_im)

    A_dyn = (4.0 * Gamma1 - 2.0 * Gamma2) / 10.0
    B_dyn = (3.0 * Gamma2 - Gamma1) / 10.0

    return complex(A_eshelby + A_dyn), complex(B_eshelby + B_dyn)


def compute_sphere_tmatrix(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
) -> SphereTMatrixResult:
    """Compute the Rayleigh T-matrix for an isotropic sphere.

    The sphere has isotropic symmetry (C=0, no cubic anisotropy).
    Uses _compute_T123 with C=0 to get T1, T2 (T3=0).

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts.

    Returns:
        SphereTMatrixResult with all computed quantities.
    """
    Gamma0 = _sphere_Gamma0(omega, radius, ref)
    A, B = _sphere_AB(omega, radius, ref)

    # Isotropic sphere: C = 0
    T1, T2, T3 = _compute_T123(A, B, 0.0, contrast.Dlambda, contrast.Dmu)

    # Amplification factors (isotropic: amp_e_off = amp_e_diag = amp_dev)
    amp_u = 1.0 / (1.0 - omega**2 * contrast.Drho * Gamma0)
    amp_theta = 1.0 / (1.0 - 3.0 * T1 - 2.0 * T2)
    amp_dev = 1.0 / (1.0 - 2.0 * T2)

    # Effective contrasts (isotropic: single Dmu_star)
    Drho_star = contrast.Drho * amp_u
    Dmu_star = contrast.Dmu * amp_dev
    Dlambda_star = (
        contrast.Dlambda + 2.0 / 3.0 * contrast.Dmu
    ) * amp_theta - contrast.Dmu / 3.0 * (amp_dev + amp_dev)

    return SphereTMatrixResult(
        Gamma0=Gamma0,
        A=A,
        B=B,
        T1=T1,
        T2=T2,
        amp_u=amp_u,
        amp_theta=amp_theta,
        amp_dev=amp_dev,
        Drho_star=Drho_star,
        Dlambda_star=Dlambda_star,
        Dmu_star=Dmu_star,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
    )


# =====================================================================
# Phase 2: Sphere Decomposition (Foldy-Lax)
# =====================================================================


def sphere_sub_cell_centres(
    radius: float,
    n_sub: int,
) -> tuple[NDArray[np.floating], float]:
    """Generate sub-cell centres inside a sphere.

    Creates a cubic grid using sub_cell_centres(radius, n_sub) and
    filters to keep only sub-cubes whose centres lie inside the sphere.

    Args:
        radius: Sphere radius (m).
        n_sub: Number of sub-cells per edge of the bounding cube.

    Returns:
        (centres, a_sub) where centres has shape (N, 3) and
        a_sub = radius / n_sub.
    """
    a_sub = radius / n_sub
    all_centres = sub_cell_centres(radius, n_sub)
    distances = np.linalg.norm(all_centres, axis=1)
    mask = distances <= radius
    return all_centres[mask], a_sub


def compute_sphere_foldy_lax(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
) -> SphereDecompositionResult:
    """Compute sphere T-matrix via Foldy-Lax decomposition.

    Voxelizes the sphere into small cubes and solves the 9N x 9N
    multiple scattering problem.

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts.
        n_sub: Number of sub-cells per edge of bounding cube.
        k_hat: Unit incident direction (default z-hat).
        wave_type: 'S' or 'P'.

    Returns:
        SphereDecompositionResult with composite T-matrix.
    """
    centres, a_sub = sphere_sub_cell_centres(radius, n_sub)
    N = len(centres)

    # Sub-cell Rayleigh T-matrix (same for all cells)
    rayleigh_sub = compute_cube_tmatrix(omega, a_sub, ref, contrast)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, omega, a_sub)

    # Build 9N x 9N propagator (off-diagonal only)
    P_tilde = np.zeros((9 * N, 9 * N), dtype=complex)
    for m in range(N):
        for n in range(N):
            if m != n:
                r_vec = centres[m] - centres[n]
                P_tilde[9 * m : 9 * m + 9, 9 * n : 9 * n + 9] = _propagator_block_9x9(
                    r_vec, omega, ref
                )

    # Block-diagonal T_tilde
    T_block = np.kron(np.eye(N, dtype=complex), T_loc)

    # Foldy-Lax matrix: A = I - P_tilde @ T_tilde
    A_mat = np.eye(9 * N, dtype=complex) - P_tilde @ T_block
    cond_num = float(np.linalg.cond(A_mat))

    # Incident field
    psi_inc = _build_incident_field_coupled(
        centres, omega, ref, k_hat=k_hat, wave_type=wave_type
    )

    # Solve
    psi_exc = np.linalg.solve(A_mat, psi_inc)

    # Composite T-matrix
    T_comp = np.zeros((9, 9), dtype=complex)
    for n in range(N):
        T_comp += T_loc @ psi_exc[9 * n : 9 * n + 9, :]

    T3x3 = T_comp[:3, :3].copy()

    return SphereDecompositionResult(
        T3x3=T3x3,
        T_comp_9x9=T_comp,
        centres=centres,
        n_sub=n_sub,
        n_cells=N,
        a_sub=a_sub,
        condition_number=cond_num,
        psi_exc=psi_exc,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
    )


# =====================================================================
# Phase 3: Elastic Mie Theory (Partial Waves)
# =====================================================================


def _spherical_jn_complex(n: int, z: complex) -> complex:
    """Spherical Bessel function j_n(z) for complex z.

    Uses j_n(z) = sqrt(pi/(2z)) * J_{n+0.5}(z).
    """
    if abs(z) < 1e-30:
        return 1.0 if n == 0 else 0.0
    return complex(np.sqrt(np.pi / (2.0 * z)) * jv(n + 0.5, z))


def _spherical_yn_complex(n: int, z: complex) -> complex:
    """Spherical Bessel function y_n(z) for complex z.

    Uses y_n(z) = sqrt(pi/(2z)) * Y_{n+0.5}(z).
    Y_{n+0.5}(z) = (-1)^{n+1} * J_{-(n+0.5)}(z) for half-integer order.
    """
    if abs(z) < 1e-30:
        return -np.inf
    yn = complex(np.sqrt(np.pi / (2.0 * z)) * ((-1) ** (n + 1)) * jv(-(n + 0.5), z))
    return yn


def _spherical_h1_complex(n: int, z: complex) -> complex:
    """Spherical Hankel function h_n^(1)(z) for complex z.

    Uses h_n^(1)(z) = sqrt(pi/(2z)) * H_{n+0.5}^(1)(z).
    """
    if abs(z) < 1e-30:
        return complex(np.inf)
    return complex(np.sqrt(np.pi / (2.0 * z)) * hankel1(n + 0.5, z))


def _spherical_jn_deriv(n: int, z: complex) -> complex:
    """Derivative d/dz [z * j_n(z)] / z using recurrence.

    [z * j_n(z)]' = z * j_{n-1}(z) - n * j_n(z)
    So the derivative of j_n w.r.t. z is:
        j_n'(z) = j_{n-1}(z) - (n+1)/z * j_n(z)  (wrong sign convention)

    Actually: j_n'(z) = (n/z) j_n(z) - j_{n+1}(z)
    """
    if abs(z) < 1e-30:
        return 1.0 / 3.0 if n == 1 else 0.0
    return complex(
        (n / z) * _spherical_jn_complex(n, z) - _spherical_jn_complex(n + 1, z)
    )


def _spherical_h1_deriv(n: int, z: complex) -> complex:
    """Derivative h_n^(1)'(z) using recurrence.

    h_n^(1)'(z) = (n/z) h_n^(1)(z) - h_n^(1)_{n+1}(z)
    """
    return complex(
        (n / z) * _spherical_h1_complex(n, z) - _spherical_h1_complex(n + 1, z)
    )


def _mie_pwave_fields(
    n: int,
    k: complex,
    r: float,
    lam: float,
    mu: float,
    z_type: str = "j",
) -> tuple[complex, complex, complex, complex]:
    """Displacement and stress from P-wave potential phi = z_n(kr) P_n.

    Uses the Helmholtz decomposition u = grad(phi):
        u_r = k z_n'(kr) P_n
        u_theta = z_n(kr)/r dP_n/dtheta

    The P_n and dP_n factors are applied externally. Returns the radial
    coefficient functions only.

    Args:
        n: Angular order (>= 1).
        k: P-wavenumber.
        r: Radial distance (m).
        lam: Lame lambda.
        mu: Shear modulus.
        z_type: 'j' for regular (interior), 'h1' for scattered (outgoing).

    Returns:
        (ur_coeff, ut_coeff, srr_coeff, srt_coeff) where the displacement
        and stress are:
            u_r = ur_coeff * P_n
            u_theta = ut_coeff * dP_n/dtheta
            sigma_rr = srr_coeff * P_n
            sigma_rtheta = srt_coeff * dP_n/dtheta
    """
    z = k * r
    if z_type == "j":
        zn = _spherical_jn_complex(n, z)
        zn_p = _spherical_jn_deriv(n, z)
    else:
        zn = _spherical_h1_complex(n, z)
        zn_p = _spherical_h1_deriv(n, z)

    # u_r = k z_n'(kr)
    ur = k * zn_p

    # u_theta = z_n(kr) / r
    ut = zn / r

    # sigma_rr = -(lam+2mu) k^2 z_n - 4mu k z_n'/r + 2mu n(n+1) z_n/r^2
    # Derived from: sigma_rr = lam(-k^2 z_n) + 2mu k^2 z_n''
    # with z_n'' eliminated via Bessel equation
    srr = (
        -(lam + 2.0 * mu) * k**2 * zn
        - 4.0 * mu * k * zn_p / r
        + 2.0 * mu * n * (n + 1) * zn / r**2
    )

    # sigma_rtheta = 2mu [k z_n' - z_n/r] / r
    srt = 2.0 * mu * (k * zn_p - zn / r) / r

    return ur, ut, srr, srt


def _mie_swave_fields(
    n: int,
    k: complex,
    r: float,
    mu: float,
    z_type: str = "j",
) -> tuple[complex, complex, complex, complex]:
    """Displacement and stress from SV-wave potential psi = z_n(kr) P_n.

    Uses u = curl(curl(r_hat * r * psi)):
        u_r = n(n+1) z_n(kr)/r P_n
        u_theta = [z_n(kr) + kr z_n'(kr)] / r dP_n/dtheta

    The P_n, dP_n, and n(n+1) factors are applied externally for u_r
    but NOT for u_theta. Returns the radial coefficient functions.

    Note: The n(n+1) factor for u_r is included HERE (not externally),
    so the matrix column directly multiplies the coefficient B_n.

    Args:
        n: Angular order (>= 1).
        k: S-wavenumber.
        r: Radial distance (m).
        mu: Shear modulus.
        z_type: 'j' for regular (interior), 'h1' for scattered (outgoing).

    Returns:
        (ur_coeff, ut_coeff, srr_coeff, srt_coeff) where:
            u_r = ur_coeff * P_n
            u_theta = ut_coeff * dP_n/dtheta
            sigma_rr = srr_coeff * P_n
            sigma_rtheta = srt_coeff * dP_n/dtheta
    """
    z = k * r
    if z_type == "j":
        zn = _spherical_jn_complex(n, z)
        zn_p = _spherical_jn_deriv(n, z)
    else:
        zn = _spherical_h1_complex(n, z)
        zn_p = _spherical_h1_deriv(n, z)

    nn1 = n * (n + 1)

    # u_r = n(n+1) z_n / r
    ur = nn1 * zn / r

    # u_theta = [z_n + kr z_n'] / r = D1 / r
    ut = (zn + z * zn_p) / r

    # sigma_rr = 2mu n(n+1) [k z_n'/r - z_n/r^2]
    # (div(u^S) = 0, so sigma_rr = 2mu du_r/dr)
    srr = 2.0 * mu * nn1 * (k * zn_p / r - zn / r**2)

    # sigma_rtheta = mu [(2n(n+1) - 2 - k^2 r^2) z_n - 2kr z_n'] / r^2
    srt = mu * ((2 * nn1 - 2 - z**2) * zn - 2.0 * z * zn_p) / r**2

    return ur, ut, srr, srt


def _mie_matrix_psv(
    n: int,
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
) -> NDArray[np.complexfloating]:
    """Build 4x4 boundary condition matrix for P-SV Mie scattering.

    Uses Helmholtz decomposition: u = grad(phi) + curl(curl(r*psi)).
    Rows: continuity of [u_r, u_theta, sigma_rr, sigma_r_theta] at r=a.
    Columns: [A_n (scattered P), B_n (scattered S),
              C_n (interior P), D_n (interior S)]

    Convention: scattered + incident = interior
    => scattered - interior = -incident

    Args:
        n: Angular order.
        omega: Angular frequency.
        radius: Sphere radius.
        ref: Background medium.
        contrast: Material contrast.

    Returns:
        4x4 complex boundary matrix.
    """
    a = radius
    lam_out = ref.lam
    mu_out = ref.mu

    lam_in = lam_out + contrast.Dlambda
    mu_in = mu_out + contrast.Dmu
    rho_in = ref.rho + contrast.Drho

    alpha_in = np.sqrt((lam_in + 2.0 * mu_in) / rho_in)
    beta_in = np.sqrt(mu_in / rho_in)

    kP_out = omega / ref.alpha
    kS_out = omega / ref.beta
    kP_in = omega / alpha_in
    kS_in = omega / beta_in

    # Scattered P (outgoing h1, exterior)
    ur_Ps, ut_Ps, srr_Ps, srt_Ps = _mie_pwave_fields(
        n, kP_out, a, lam_out, mu_out, "h1"
    )
    # Scattered S (outgoing h1, exterior)
    ur_Ss, ut_Ss, srr_Ss, srt_Ss = _mie_swave_fields(n, kS_out, a, mu_out, "h1")
    # Interior P (regular j)
    ur_Pi, ut_Pi, srr_Pi, srt_Pi = _mie_pwave_fields(n, kP_in, a, lam_in, mu_in, "j")
    # Interior S (regular j)
    ur_Si, ut_Si, srr_Si, srt_Si = _mie_swave_fields(n, kS_in, a, mu_in, "j")

    M = np.zeros((4, 4), dtype=complex)

    # Row 0: u_r continuity
    M[0, 0] = ur_Ps
    M[0, 1] = ur_Ss
    M[0, 2] = -ur_Pi
    M[0, 3] = -ur_Si

    # Row 1: u_theta continuity
    M[1, 0] = ut_Ps
    M[1, 1] = ut_Ss
    M[1, 2] = -ut_Pi
    M[1, 3] = -ut_Si

    # Row 2: sigma_rr continuity
    M[2, 0] = srr_Ps
    M[2, 1] = srr_Ss
    M[2, 2] = -srr_Pi
    M[2, 3] = -srr_Si

    # Row 3: sigma_r_theta continuity
    M[3, 0] = srt_Ps
    M[3, 1] = srt_Ss
    M[3, 2] = -srt_Pi
    M[3, 3] = -srt_Si

    return M


def _mie_incident_psv(
    n: int,
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    incident_type: str = "P",
) -> NDArray[np.complexfloating]:
    """Incident field RHS for P-SV Mie scattering.

    For a unit-amplitude P-wave u = z_hat exp(ik_P z), the scalar
    potential is phi = exp(ik_P z)/(ik_P), giving partial wave
    coefficient (2n+1) i^n / (ik_P) for order n.

    Args:
        n: Angular order.
        omega: Angular frequency.
        radius: Sphere radius.
        ref: Background medium.
        incident_type: 'P' or 'S'.

    Returns:
        RHS vector of length 4 (negative of incident field at boundary).
    """
    a = radius
    rhs = np.zeros(4, dtype=complex)

    if incident_type == "P":
        k_inc = omega / ref.alpha
        # Potential coefficient: (2n+1) i^n / (ik_P)
        coeff = (2 * n + 1) * (1j) ** n / (1j * k_inc)
        ur, ut, srr, srt = _mie_pwave_fields(n, k_inc, a, ref.lam, ref.mu, "j")
        rhs[0] = -coeff * ur
        rhs[1] = -coeff * ut
        rhs[2] = -coeff * srr
        rhs[3] = -coeff * srt
    else:
        k_inc = omega / ref.beta
        # SV-wave potential coefficient: (2n+1) i^n / (ik_S)
        coeff = (2 * n + 1) * (1j) ** n / (1j * k_inc)
        ur, ut, srr, srt = _mie_swave_fields(n, k_inc, a, ref.mu, "j")
        rhs[0] = -coeff * ur
        rhs[1] = -coeff * ut
        rhs[2] = -coeff * srr
        rhs[3] = -coeff * srt

    return rhs


def _mie_matrix_sh(
    n: int,
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
) -> NDArray[np.complexfloating]:
    """Build 2x2 boundary condition matrix for SH Mie scattering.

    SH modes decouple from P-SV and only involve the S-wavenumber.

    Args:
        n: Angular order.
        omega: Angular frequency.
        radius: Sphere radius.
        ref: Background medium.
        contrast: Material contrast.

    Returns:
        2x2 complex boundary matrix.
    """
    a = radius
    mu_out = ref.mu
    rho_in = ref.rho + contrast.Drho
    mu_in = mu_out + contrast.Dmu

    kS_out = omega / ref.beta
    beta_in = np.sqrt(mu_in / rho_in)
    kS_in = omega / beta_in

    z_out = kS_out * a
    z_in = kS_in * a

    # Scattered (h1) and interior (j) spherical Bessel values
    h_out = _spherical_h1_complex(n, z_out)
    hp_out = _spherical_h1_deriv(n, z_out)
    j_in = _spherical_jn_complex(n, z_in)
    jp_in = _spherical_jn_deriv(n, z_in)

    M = np.zeros((2, 2), dtype=complex)
    # u_phi continuity: h_n(kS_out * a) * c_n = j_n(kS_in * a) * d_n
    M[0, 0] = h_out
    M[0, 1] = -j_in

    # tau_r_phi continuity: mu_out * kS_out * h'_n = mu_in * kS_in * j'_n
    M[1, 0] = mu_out * kS_out * hp_out
    M[1, 1] = -mu_in * kS_in * jp_in

    return M


def compute_elastic_mie(
    omega: float,
    radius: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_max: int | None = None,
) -> MieResult:
    """Compute elastic Mie scattering coefficients.

    Full elastic Mie solution via partial wave expansion.
    Follows Pao & Mow (1973) / Korneev & Johnson (1993).

    Args:
        omega: Angular frequency (rad/s).
        radius: Sphere radius (m).
        ref: Background medium.
        contrast: Material contrasts.
        n_max: Maximum angular order. If None, auto-selected via
            Wiscombe criterion.

    Returns:
        MieResult with scattering coefficients per order.
    """
    ka_P = omega * radius / ref.alpha
    ka_S = omega * radius / ref.beta

    if n_max is None:
        n_max = max(2, int(np.ceil(ka_S + 4.0 * ka_S ** (1.0 / 3.0) + 2)))

    # Arrays indexed from 0: a_n[n] for order n=0,...,n_max
    a_n = np.zeros(n_max + 1, dtype=complex)
    b_n = np.zeros(n_max + 1, dtype=complex)
    c_n = np.zeros(n_max + 1, dtype=complex)

    # n=0 monopole: purely P-wave, 2x2 system
    a = radius
    lam_out = ref.lam
    mu_out = ref.mu
    lam_in = lam_out + contrast.Dlambda
    mu_in = mu_out + contrast.Dmu
    rho_in = ref.rho + contrast.Drho
    alpha_in = np.sqrt((lam_in + 2.0 * mu_in) / rho_in)
    kP_out = omega / ref.alpha
    kP_in = omega / alpha_in

    ur_s, _, srr_s, _ = _mie_pwave_fields(0, kP_out, a, lam_out, mu_out, "h1")
    ur_i, _, srr_i, _ = _mie_pwave_fields(0, kP_in, a, lam_in, mu_in, "j")
    ur_inc, _, srr_inc, _ = _mie_pwave_fields(0, kP_out, a, lam_out, mu_out, "j")

    M0 = np.array([[ur_s, -ur_i], [srr_s, -srr_i]], dtype=complex)
    coeff_0 = 1.0 / (1j * kP_out)  # (2*0+1)*i^0 / (ik_P)
    rhs_0 = np.array([-coeff_0 * ur_inc, -coeff_0 * srr_inc], dtype=complex)
    try:
        sol_0 = np.linalg.solve(M0, rhs_0)
        a_n[0] = sol_0[0]
    except np.linalg.LinAlgError:
        pass

    # n>=1: P-SV 4x4 system + SH 2x2 system
    for n in range(1, n_max + 1):
        M_psv = _mie_matrix_psv(n, omega, radius, ref, contrast)
        rhs_psv = _mie_incident_psv(n, omega, radius, ref, incident_type="P")

        # Sign convention: the standard Mie partial wave expansion
        # exp(ikz) = sum (2n+1) i^n j_n P_n(cos theta) uses a polar axis
        # convention that produces backward-peaked scattering in our
        # coordinate system (axis 0 = propagation direction, z-down).
        # The factor (-1)^n corrects the angular pattern so that the
        # scattering is forward-peaked, consistent with the Rayleigh
        # and Foldy-Lax implementations.
        sign = (-1.0) ** n

        try:
            sol_psv = np.linalg.solve(M_psv, rhs_psv)
            a_n[n] = sign * sol_psv[0]  # scattered P coefficient
            b_n[n] = sign * sol_psv[1]  # scattered S coefficient
        except np.linalg.LinAlgError:
            pass

        # SH modes: 2x2 system
        M_sh = _mie_matrix_sh(n, omega, radius, ref, contrast)
        z_inc = omega / ref.beta * radius
        j_inc = _spherical_jn_complex(n, z_inc)
        coeff_sh = (2 * n + 1) * (1j) ** n / (1j * omega / ref.beta)
        rhs_sh = np.array(
            [
                -coeff_sh * j_inc,
                -coeff_sh * ref.mu * omega / ref.beta * _spherical_jn_deriv(n, z_inc),
            ],
            dtype=complex,
        )

        try:
            sol_sh = np.linalg.solve(M_sh, rhs_sh)
            c_n[n] = sign * sol_sh[0]  # scattered SH coefficient
        except np.linalg.LinAlgError:
            pass

    return MieResult(
        a_n=a_n,
        b_n=b_n,
        c_n=c_n,
        n_max=n_max,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
        ka_P=ka_P,
        ka_S=ka_S,
    )


@dataclass
class MieEffectiveContrasts:
    """Effective material contrasts extracted from Mie scattering coefficients.

    Provides an independent route to the effective contrasts that does NOT
    use the Eshelby/Green's-tensor volume integral. Instead, it extracts
    Δλ*, Δμ*, Δρ* by matching the Mie far-field angular pattern to the
    Rayleigh scattering formula:

        f_P(θ) ∝ k²Δλ* + ω²Δρ* cos θ + 2k²Δμ* cos²θ

    The three Legendre projections (P₀, P₁, P₂) give bulk modulus (Δκ*),
    density (Δρ*), and shear modulus (Δμ*) independently.

    Agreement with Rayleigh route:
        Δρ*: exact at all contrasts (volume integral is equivalent to
              boundary matching for the dipole/translational mode)
        Δκ*, Δμ*: exact in the Born limit; O(ε) discrepancy at finite
              contrast because the Eshelby volume-average amplification
              differs from the exact Mie surface-matching amplification.
    """

    Dlambda_star: complex
    Dmu_star: complex
    Dkappa_star: complex
    Drho_star: complex
    Drho_star_S: complex  # independent extraction from S-wave at broadside


def mie_extract_effective_contrasts(mie_result: MieResult) -> MieEffectiveContrasts:
    """Extract effective material contrasts from Mie partial wave coefficients.

    Uses direct Legendre projection of the Mie partial wave coefficients
    rather than evaluating the far-field at discrete angles.  This avoids
    contamination from higher-order (n ≥ 3) partial waves that pollute the
    three-angle inversion at finite contrast.

    The Mie P-wave far-field decomposes as:

        f_P(θ) = Σ_n a_n (-i)^n P_n(cos θ)

    Matching against the Rayleigh scattering formula (rewritten in Legendre
    polynomials using cos²θ = (2P₂ + 1)/3):

        f_P(θ) = −C_P {k² Δκ* P₀ + ω² Δρ* P₁ + (4k²Δμ*/3) P₂}

    where C_P = V/(4πρα²), V = (4/3)πa³, gives one-to-one matching of
    the a³ (leading Rayleigh-order) scattering coefficient per Legendre
    mode:

        n = 0  (monopole):   a₀ = −C_P k² Δκ*        → bulk modulus
        n = 1  (dipole):  −i a₁ = −C_P ω² Δρ*        → density
        n = 2  (quadrupole): −a₂ = −C_P (4k²Δμ*/3)   → shear modulus

    The S-wave dipole coefficient b₁ gives an independent density estimate:

        −i b₁ = −C_S ω² Δρ*_S

    These projections are exact to each partial wave order and do not mix
    higher multipoles into the extracted contrasts.

    Args:
        mie_result: Output of compute_elastic_mie.

    Returns:
        MieEffectiveContrasts with independently extracted contrasts.
    """
    ref = mie_result.ref
    omega = mie_result.omega
    radius = mie_result.radius
    V = (4.0 / 3.0) * np.pi * radius**3
    kP = omega / ref.alpha

    a_n = mie_result.a_n
    b_n = mie_result.b_n

    # Prefactors: C_P = V/(4πρα²), C_S = V/(4πρβ²)
    C_P = V / (4.0 * np.pi * ref.rho * ref.alpha**2)
    C_S = V / (4.0 * np.pi * ref.rho * ref.beta**2)

    # Direct extraction from partial wave coefficients (Legendre projection):
    #   n=0: a₀ = -C_P k² Δκ*
    #   n=1: -i a₁ = -C_P ω² Δρ*   →  Δρ* = i a₁ / (C_P ω²)
    #   n=2: -a₂ = -C_P (4k²Δμ*/3) →  Δμ* = 3 a₂ / (4 C_P k²)
    Dkappa_star = -a_n[0] / (C_P * kP**2)
    Drho_star = 1j * a_n[1] / (C_P * omega**2)
    Dmu_star = 3.0 * a_n[2] / (4.0 * C_P * kP**2)
    Dlambda_star = Dkappa_star - 2.0 * Dmu_star / 3.0

    # S-wave dipole coefficient gives independent density:
    #   -i b₁ = -C_S ω² Δρ*_S  →  Δρ*_S = i b₁ / (C_S ω²)
    Drho_star_S = 1j * b_n[1] / (C_S * omega**2)

    return MieEffectiveContrasts(
        Dlambda_star=complex(Dlambda_star),
        Dmu_star=complex(Dmu_star),
        Dkappa_star=complex(Dkappa_star),
        Drho_star=complex(Drho_star),
        Drho_star_S=complex(Drho_star_S),
    )


# =====================================================================
# Phase 4: Far-Field Scattering Amplitudes
# =====================================================================


def _dPn_dtheta(n: int, theta: float) -> float:
    """Derivative dP_n(cos theta)/dtheta via finite differences."""
    dt = 1e-8
    Pn_p = float(lpmv(0, n, np.cos(theta + dt)))
    Pn_m = float(lpmv(0, n, np.cos(theta - dt)))
    return (Pn_p - Pn_m) / (2.0 * dt)


def mie_scattered_displacement(
    mie_result: MieResult,
    r_points: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Evaluate Mie scattered displacement at exterior points.

    Uses the Helmholtz decomposition with correct 1/r decay:
        P-wave: u_r = A_n kP h_n'(kP r) P_n,
                u_theta = A_n h_n(kP r)/r dP_n
        S-wave: u_r = B_n n(n+1) h_n(kS r)/r P_n,
                u_theta = B_n [h_n + kS r h_n']/r dP_n

    The incident wave is a P-wave along the z-axis (index 0).
    Converted to Cartesian coordinates (z=0, x=1, y=2).

    Args:
        mie_result: Output of compute_elastic_mie.
        r_points: Observation points, shape (M, 3). Must be outside sphere.

    Returns:
        Scattered displacement, shape (M, 3), complex.
    """
    ref = mie_result.ref
    omega = mie_result.omega
    n_max = mie_result.n_max
    kP = omega / ref.alpha
    kS = omega / ref.beta

    r_points = np.asarray(r_points, dtype=float)
    M_pts = r_points.shape[0]
    u_scat = np.zeros((M_pts, 3), dtype=complex)

    for idx in range(M_pts):
        pos = r_points[idx]
        r = float(np.linalg.norm(pos))
        if r < 1e-14:
            continue

        cos_theta = pos[0] / r
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        if sin_theta > 1e-12:
            cos_phi = pos[1] / (r * sin_theta)
            sin_phi = pos[2] / (r * sin_theta)
        else:
            cos_phi = 1.0
            sin_phi = 0.0

        u_r = 0.0j
        u_theta = 0.0j

        for n in range(0, n_max + 1):
            an = mie_result.a_n[n]
            bn = mie_result.b_n[n]

            Pn = float(lpmv(0, n, cos_theta))
            dPn = _dPn_dtheta(n, theta) if (n > 0 and abs(sin_theta) > 1e-10) else 0.0

            # Scattered P-wave fields at r (outgoing h1)
            ur_P, ut_P, _, _ = _mie_pwave_fields(n, kP, r, ref.lam, ref.mu, "h1")

            u_r += an * ur_P * Pn
            u_theta += an * ut_P * dPn

            # S-wave only for n >= 1
            if n >= 1:
                ur_S, ut_S, _, _ = _mie_swave_fields(n, kS, r, ref.mu, "h1")
                u_r += bn * ur_S * Pn
                u_theta += bn * ut_S * dPn

        # Convert (u_r, u_theta) to Cartesian (z, x, y)
        # r_hat = (cos_theta, sin_theta cos_phi, sin_theta sin_phi)
        # theta_hat = (-sin_theta, cos_theta cos_phi, cos_theta sin_phi)
        u_scat[idx, 0] = u_r * cos_theta - u_theta * sin_theta
        u_scat[idx, 1] = u_r * sin_theta * cos_phi + u_theta * cos_theta * cos_phi
        u_scat[idx, 2] = u_r * sin_theta * sin_phi + u_theta * cos_theta * sin_phi

    return u_scat


def mie_far_field(
    mie_result: MieResult,
    theta_arr: NDArray[np.floating],
    incident_type: str = "P",
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Far-field scattering amplitudes from Mie solution.

    Evaluates the Mie scattered displacement at large r and extracts
    the far-field pattern coefficients f_P, f_S defined by:
        u_P ~ f_P(theta) * r_hat * exp(ik_P r) / r
        u_S ~ f_S(theta) * theta_hat * exp(ik_S r) / r

    Args:
        mie_result: Output of compute_elastic_mie.
        theta_arr: Scattering angles (radians), shape (M,).
        incident_type: 'P' for P-wave incidence.

    Returns:
        (f_P, f_S) far-field amplitudes, each shape (M,).
    """
    theta_arr = np.asarray(theta_arr, dtype=float)
    ref = mie_result.ref
    omega = mie_result.omega
    n_max = mie_result.n_max
    kP = omega / ref.alpha
    kS = omega / ref.beta

    r_eval = 1.0e6 * mie_result.radius
    f_P = np.zeros_like(theta_arr, dtype=complex)
    f_S = np.zeros_like(theta_arr, dtype=complex)

    for i, theta in enumerate(theta_arr):
        cos_t = np.cos(theta)
        u_r = 0.0j
        u_theta = 0.0j

        for n in range(0, n_max + 1):
            an = mie_result.a_n[n]
            bn = mie_result.b_n[n]

            Pn = float(lpmv(0, n, cos_t))
            dPn = (
                _dPn_dtheta(n, theta) if (n > 0 and abs(np.sin(theta)) > 1e-10) else 0.0
            )

            ur_P, ut_P, _, _ = _mie_pwave_fields(n, kP, r_eval, ref.lam, ref.mu, "h1")
            u_r += an * ur_P * Pn
            u_theta += an * ut_P * dPn

            if n >= 1:
                ur_S, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, ref.mu, "h1")
                u_r += bn * ur_S * Pn
                u_theta += bn * ut_S * dPn

        # Extract far-field amplitude: u ~ f * exp(ikr)/r
        f_P[i] = u_r * r_eval * np.exp(-1j * kP * r_eval)
        f_S[i] = u_theta * r_eval * np.exp(-1j * kS * r_eval)

    return f_P, f_S


def sphere_rayleigh_far_field(
    sphere_result: SphereTMatrixResult,
    r_obs: NDArray[np.floating],
    k_hat: NDArray[np.floating],
    pol: NDArray[np.floating],
    wave_type: str = "P",
) -> NDArray[np.complexfloating]:
    """Scattered field from Rayleigh sphere at an observation point.

    u^scat(x) = V * {H_{ijk}(x) * Dc_{jklm} * eps_{lm}
                      + omega^2 * G_{ij}(x) * Drho * u_j}

    where V = (4/3) pi a^3 and eps, u are the self-consistent
    interior values.

    Args:
        sphere_result: Output of compute_sphere_tmatrix.
        r_obs: Observation point position (m), shape (3,).
        k_hat: Unit incident propagation direction, shape (3,).
        pol: Incident polarisation vector, shape (3,).
        wave_type: 'P' or 'S' for incident wave type.

    Returns:
        Scattered displacement at r_obs, shape (3,).
    """
    omega = sphere_result.omega
    radius = sphere_result.radius
    ref = sphere_result.ref
    contrast = sphere_result.contrast
    V_sphere = (4.0 / 3.0) * np.pi * radius**3

    k_hat = np.asarray(k_hat, dtype=float)
    k_hat /= np.linalg.norm(k_hat)
    pol = np.asarray(pol, dtype=float)
    r_obs = np.asarray(r_obs, dtype=float)

    # Incident field at sphere centre (origin)
    u_inc = pol.copy()

    # Incident strain at sphere centre
    if wave_type == "P":
        k_mag = omega / ref.alpha
    else:
        k_mag = omega / ref.beta
    # eps_{lm}^(0) = (i k / 2) (k_hat_l * pol_m + k_hat_m * pol_l)
    eps_inc = 0.5j * k_mag * (np.outer(k_hat, pol) + np.outer(pol, k_hat))

    # Self-consistent interior fields
    u_int = sphere_result.amp_u * u_inc
    theta_int = sphere_result.amp_theta * np.trace(eps_inc)
    eps_dev_inc = eps_inc - np.trace(eps_inc) / 3.0 * np.eye(3)
    eps_dev_int = sphere_result.amp_dev * eps_dev_inc
    eps_int = eps_dev_int + theta_int / 3.0 * np.eye(3)

    # Stress perturbation: Dc_{jklm} eps_{lm}
    Dc_eps = (
        contrast.Dlambda * np.trace(eps_int) * np.eye(3) + 2.0 * contrast.Dmu * eps_int
    )

    # Green's tensor and its derivative at observation point
    from .resonance_tmatrix import elastodynamic_greens_deriv as _egd

    G, Gd, _ = _egd(r_obs, omega, ref)

    # H_{ijk} = (1/2)(G_{ij,k} + G_{ik,j})  — strain Green's tensor
    # Actually H_{ijk} = G_{ij,k} (derivative w.r.t. source, not field)
    # For source at origin, observation at r_obs: G_{ij,k} is the derivative
    # w.r.t. x'_k, which equals -G_{ij,k}^{(x)} for translation invariance.
    # But the Lippmann-Schwinger uses G_{ij,k}^{source derivative}
    # = -Gd[i,j,k] (negative of field-point derivative)

    # u^scat_i = V * {-Gd[i,j,k] * Dc_eps[j,k] + omega^2 * G[i,j] * Drho_star * u_int[j]}
    # Wait: the effective contrasts already include amplification
    # The correct formula using effective contrasts:
    # u^scat_i = V * {-Gd[i,j,k] * [Dlambda_star * theta * delta_jk + 2*Dmu_star * eps_jk]
    #               + omega^2 * G[i,j] * Drho_star * u_inc_j}

    Dc_star_eps = (
        complex(sphere_result.Dlambda_star) * np.trace(eps_inc) * np.eye(3)
        + 2.0 * complex(sphere_result.Dmu_star) * eps_inc
    )

    u_scat = np.zeros(3, dtype=complex)
    for i in range(3):
        # Stiffness term: H_{ijk} Dc*_{jk...} uses source-point derivative
        stiff_term = 0.0j
        for j in range(3):
            for k in range(3):
                stiff_term += -Gd[i, j, k] * Dc_star_eps[j, k]

        # Density term
        dens_term = omega**2 * complex(sphere_result.Drho_star) * (G[i, :] @ u_inc)

        u_scat[i] = V_sphere * (stiff_term + dens_term)

    return u_scat


def _voigt_to_tensor(voigt_6: NDArray) -> NDArray:
    """Convert Voigt stress vector to 3x3 symmetric tensor.

    The Voigt convention stores doubled off-diagonal components:
    [sigma_zz, sigma_xx, sigma_yy, 2*sigma_xy, 2*sigma_zy, 2*sigma_zx]

    Returns the actual tensor sigma_{ij}.
    """
    T = np.zeros((3, 3), dtype=complex)
    T[0, 0] = voigt_6[0]  # zz
    T[1, 1] = voigt_6[1]  # xx
    T[2, 2] = voigt_6[2]  # yy
    T[1, 2] = T[2, 1] = voigt_6[3] / 2.0  # xy
    T[0, 2] = T[2, 0] = voigt_6[4] / 2.0  # zy
    T[0, 1] = T[1, 0] = voigt_6[5] / 2.0  # zx
    return T


def foldy_lax_far_field(
    decomp_result: SphereDecompositionResult,
    r_hat_arr: NDArray[np.floating],
    r_distance: float,
    k_hat: NDArray[np.floating],
    pol: NDArray[np.floating],
    wave_type: str = "P",
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Far-field scattering amplitude from Foldy-Lax decomposition.

    Sums contributions from each sub-cell using the far-field
    asymptotic Green's tensor, including both force monopole and
    stress dipole contributions.

    The far-field Green's tensor source-derivative gives:
        G_{ij,k'}^P ~ -ik_P r_hat_k G_P^far r_hat_i r_hat_j
        G_{ij,k'}^S ~ -ik_S r_hat_k G_S^far (delta_ij - r_hat_i r_hat_j)

    So the total scattered displacement from cell m is:
        u_P = G_P^far r_hat [(r_hat . F) - ik_P (r_hat . sigma . r_hat)]
        u_S = G_S^far [(F - ik_S sigma . r_hat)_perp]

    where F = source[:3] (force) and sigma = tensor(source[3:]) (stress dipole).

    Args:
        decomp_result: Output of compute_sphere_foldy_lax.
        r_hat_arr: Observation directions, shape (M, 3).
        r_distance: Distance from origin to observation points (m).
        k_hat: Unit incident propagation direction, shape (3,).
        pol: Incident polarisation vector, shape (3,).
        wave_type: 'P' or 'S'.

    Returns:
        (u_P, u_S) P-wave and S-wave far-field displacements,
        each of shape (M, 3).
    """
    omega = decomp_result.omega
    ref = decomp_result.ref
    centres = decomp_result.centres
    N = decomp_result.n_cells
    a_sub = decomp_result.a_sub

    k_hat = np.asarray(k_hat, dtype=float)
    k_hat /= np.linalg.norm(k_hat)
    pol = np.asarray(pol, dtype=float)
    r_hat_arr = np.asarray(r_hat_arr, dtype=float)

    kP = omega / ref.alpha
    kS = omega / ref.beta
    r = r_distance

    if wave_type == "P":
        k_mag = kP
    else:
        k_mag = kS

    M = r_hat_arr.shape[0]
    u_P = np.zeros((M, 3), dtype=complex)
    u_S = np.zeros((M, 3), dtype=complex)

    # Get the sub-cell Rayleigh T-matrix
    rayleigh_sub = compute_cube_tmatrix(omega, a_sub, ref, decomp_result.contrast)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, omega, a_sub)

    psi_exc = decomp_result.psi_exc

    # Precompute incident vector (same for all cells)
    eps_inc_voigt = _plane_wave_strain_voigt(k_hat, pol, k_mag)
    inc_vec = np.zeros(9, dtype=complex)
    inc_vec[:3] = pol
    inc_vec[3:] = eps_inc_voigt

    for obs_idx in range(M):
        r_hat = r_hat_arr[obs_idx]

        for n in range(N):
            # Scattered source at cell n
            psi_n = psi_exc[9 * n : 9 * n + 9, :] @ inc_vec
            source = T_loc @ psi_n  # 9-vector: [force_3, stress_dipole_6]

            force = source[:3]
            sigma = _voigt_to_tensor(source[3:])  # 3x3 stress dipole tensor

            # Phase from cell position to observation point
            phase_out_P = np.exp(1j * kP * (r - np.dot(r_hat, centres[n])))
            phase_out_S = np.exp(1j * kS * (r - np.dot(r_hat, centres[n])))

            G_far_P = phase_out_P / (4.0 * np.pi * ref.rho * ref.alpha**2 * r)
            G_far_S = phase_out_S / (4.0 * np.pi * ref.rho * ref.beta**2 * r)

            # Stress dipole contractions with r_hat
            sigma_r = sigma @ r_hat  # sigma . r_hat (vector)
            sigma_rr = np.dot(r_hat, sigma_r)  # r_hat . sigma . r_hat (scalar)

            # P-wave: u_P = G_P r_hat [(r_hat.F) - ik_P (r_hat.sigma.r_hat)]
            Q_P = np.dot(r_hat, force) - 1j * kP * sigma_rr
            u_P[obs_idx] += G_far_P * Q_P * r_hat

            # S-wave: u_S = G_S [(F - ik_S sigma.r_hat)_perp]
            Q_S = force - 1j * kS * sigma_r
            Q_S_perp = Q_S - np.dot(r_hat, Q_S) * r_hat
            u_S[obs_idx] += G_far_S * Q_S_perp

    return u_P, u_S


def _plane_wave_strain_voigt(
    k_hat: NDArray[np.floating],
    pol: NDArray[np.floating],
    k_mag: float,
) -> NDArray[np.complexfloating]:
    """Compute Voigt strain vector for a plane wave.

    eps_{lm} = (ik/2)(k_hat_l * pol_m + k_hat_m * pol_l)

    Voigt ordering: (zz, xx, yy, xy, zy, zx) with indices (0,0), (1,1), (2,2),
    (1,2), (0,2), (0,1).

    Args:
        k_hat: Unit propagation direction, shape (3,).
        pol: Polarisation vector, shape (3,).
        k_mag: Wavenumber magnitude.

    Returns:
        Voigt strain vector, shape (6,).
    """
    eps = 0.5j * k_mag * (np.outer(k_hat, pol) + np.outer(pol, k_hat))

    # Voigt ordering: (0,0), (1,1), (2,2), (1,2), (0,2), (0,1)
    voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    eps_voigt = np.zeros(6, dtype=complex)
    for idx, (p, q) in enumerate(voigt_pairs):
        if p == q:
            eps_voigt[idx] = eps[p, q]
        else:
            eps_voigt[idx] = 2.0 * eps[p, q]  # engineering strain

    return eps_voigt
