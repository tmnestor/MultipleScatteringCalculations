"""
voigt_tmatrix.py
Express the cubic T-matrix as a 6×6 matrix in the
displacement-traction basis (uz, ux, uy, tzz, txz, tyz).

Coordinate system: z (down), x (right), y (out of page) — right-handed.
The mapping from generic (1,2,3) Cartesian to (z,x,y) is: 1→z, 2→x, 3→y.

VOIGT ORDERING
--------------
The T-matrix tensor T_{mnlp} in Voigt notation uses the ordering:
  (zz, xx, yy, xy, zy, zx) → indices (1,2,3,4,5,6)

With 1→z, 2→x, 3→y:
  Voigt 1 = (1,1) = zz
  Voigt 2 = (2,2) = xx
  Voigt 3 = (3,3) = yy
  Voigt 4 = (2,3) = xy
  Voigt 5 = (1,3) = zy
  Voigt 6 = (1,2) = zx

The 6×6 Voigt T-matrix (Eqs 39-41):
  T_V = [[D, 0], [0, S]]
where D = T1 J + (2T2+T3) I3,  S = 2T2 I3.

DISPLACEMENT-TRACTION BASIS
----------------------------
The user's basis is (uz, ux, uy, tzz, txz, tyz).

Tractions on a z=const surface:
  tzz = σ_zz,  txz = σ_zx,  tyz = σ_zy

For a plane wave with horizontal wavenumber (kx, ky):
  εzz = (tzz − λ ikx ux − λ iky uy) / (λ + 2μ)
  εxx = ikx ux
  εyy = iky uy
  2εzx = txz / μ
  2εzy = tyz / μ
  2εxy = iky ux + ikx uy

This gives a 6×6 matrix S that maps (uz,ux,uy,tzz,txz,tyz) → Voigt strain.
Note that uz appears ONLY through the traction (tzz absorbs the ikz uz part).

SCATTERED FIELD
---------------
The scattered displacement-traction from a single cube scatterer with
volume V = (2a)³ and effective contrasts (Δρ*, Δλ*, Δμ*_diag, Δμ*_off):

  u^scat_i = V [H_{ijk} Δσ*_{jk} + ω² G_{ij} Δρ* u^0_j]

where Δσ* is the self-consistent effective stress perturbation and
G, H are the Green's tensor and its derivative at the observation point.

The 6×6 T-matrix in the (u,t) basis is the composition:
  T_6x6(kx, ky) = R(kx, ky) @ T_eff @ S(kx, ky)
where S extracts strains from (u,t), T_eff applies the effective contrasts,
and R converts back to (u,t).

For the LOCAL SITE T-matrix (self-interaction at the scatterer),
we provide the Voigt T-matrix and the strain extraction separately,
since the full 6×6 depends on the Green's function (handled by the
propagation framework).
"""

import numpy as np

from .effective_contrasts import CubeTMatrixResult, ReferenceMedium

# ================================================================
# Voigt T-matrix in (z, x, y) coordinate system
# ================================================================


def voigt_tmatrix_6x6(T1c: complex, T2c: complex, T3c: complex) -> np.ndarray:
    """
    Build the 6×6 Voigt T-matrix for the cubic scatterer.

    T_V = [[D, 0], [0, S]]

    with D = T1 J + (2T2+T3) I3,  S = 2T2 I3.

    Voigt ordering: (εzz, εxx, εyy, 2εxy, 2εzy, 2εzx).

    The T-matrix maps Voigt strain → Voigt effective stress perturbation:
      Δσ*_V = T_V @ ε^0_V

    Parameters
    ----------
    T1c, T2c, T3c : complex
        Cubic T-matrix coupling coefficients.

    Returns
    -------
    T_V : ndarray, shape (6, 6), complex
        Voigt T-matrix.
    """
    T_V = np.zeros((6, 6), dtype=complex)

    # D block (3×3, diagonal strains)
    diag_val = T1c + 2.0 * T2c + T3c
    off_val = T1c
    T_V[0, 0] = diag_val
    T_V[1, 1] = diag_val
    T_V[2, 2] = diag_val
    T_V[0, 1] = off_val
    T_V[0, 2] = off_val
    T_V[1, 0] = off_val
    T_V[1, 2] = off_val
    T_V[2, 0] = off_val
    T_V[2, 1] = off_val

    # S block (3×3, off-diagonal shear)
    shear_val = 2.0 * T2c
    T_V[3, 3] = shear_val
    T_V[4, 4] = shear_val
    T_V[5, 5] = shear_val

    return T_V


def voigt_tmatrix_from_result(result: CubeTMatrixResult) -> np.ndarray:
    """Build 6×6 Voigt T-matrix from a CubeTMatrixResult."""
    return voigt_tmatrix_6x6(result.T1c, result.T2c, result.T3c)


# ================================================================
# Effective stiffness contrast in Voigt form
# ================================================================


def effective_stiffness_voigt(
    Dlambda_star: complex, Dmu_star_diag: complex, Dmu_star_off: complex
) -> np.ndarray:
    """
    Build the 6×6 effective stiffness contrast matrix Δc* in Voigt form.

    For the cube, Δc* has cubic (not isotropic) symmetry:

      Δc*_V = [[D*, 0], [0, S*]]

    where:
      D*_{ij} = Δλ* + 2Δμ*_diag  (if i=j)
              = Δλ*               (if i≠j)
      S*_{ij} = 2Δμ*_off δ_{ij}

    This maps Voigt strain → Voigt stress perturbation.
    """
    Dc = np.zeros((6, 6), dtype=complex)

    # D* block
    diag_val = Dlambda_star + 2.0 * Dmu_star_diag
    off_val = Dlambda_star
    Dc[0, 0] = diag_val
    Dc[1, 1] = diag_val
    Dc[2, 2] = diag_val
    Dc[0, 1] = off_val
    Dc[0, 2] = off_val
    Dc[1, 0] = off_val
    Dc[1, 2] = off_val
    Dc[2, 0] = off_val
    Dc[2, 1] = off_val

    # S* block
    shear_val = 2.0 * Dmu_star_off
    Dc[3, 3] = shear_val
    Dc[4, 4] = shear_val
    Dc[5, 5] = shear_val

    return Dc


# ================================================================
# Strain extraction from displacement-traction vector
# ================================================================


def strain_from_displacement_traction(
    kx: float, ky: float, ref: ReferenceMedium
) -> np.ndarray:
    """
    6×6 matrix S that maps (uz, ux, uy, tzz, txz, tyz) → Voigt strain.

    For a plane wave with horizontal wavenumbers (kx, ky), the six
    independent strains are determined by the six displacement-traction
    components without needing kz (the traction absorbs kz-dependence):

      εzz  = (tzz − λ ikx ux − λ iky uy) / (λ+2μ)
      εxx  = ikx ux
      εyy  = iky uy
      2εxy = iky ux + ikx uy
      2εzy = tyz / μ
      2εzx = txz / μ

    Voigt ordering: (εzz, εxx, εyy, 2εxy, 2εzy, 2εzx)
    Input ordering: (uz, ux, uy, tzz, txz, tyz)

    Parameters
    ----------
    kx : float
        Horizontal wavenumber in x-direction.
    ky : float
        Horizontal wavenumber in y-direction (cylindrical harmonic order
        for 2.5D).
    ref : ReferenceMedium
        Background medium (provides λ, μ).

    Returns
    -------
    S : ndarray, shape (6, 6), complex
        Strain extraction matrix.
    """
    lam = ref.lam
    mu = ref.mu
    lam_2mu = lam + 2.0 * mu

    S = np.zeros((6, 6), dtype=complex)

    # Voigt row 0: εzz = (tzz − λ ikx ux − λ iky uy) / (λ+2μ)
    # uz  ux              uy              tzz         txz  tyz
    S[0, 1] = -lam * 1j * kx / lam_2mu  # ux coeff
    S[0, 2] = -lam * 1j * ky / lam_2mu  # uy coeff
    S[0, 3] = 1.0 / lam_2mu  # tzz coeff

    # Voigt row 1: εxx = ikx ux
    S[1, 1] = 1j * kx

    # Voigt row 2: εyy = iky uy
    S[2, 2] = 1j * ky

    # Voigt row 3: 2εxy = iky ux + ikx uy
    S[3, 1] = 1j * ky
    S[3, 2] = 1j * kx

    # Voigt row 4: 2εzy = tyz / μ
    S[4, 5] = 1.0 / mu

    # Voigt row 5: 2εzx = txz / μ
    S[5, 4] = 1.0 / mu

    return S


def traction_from_strain(kx: float, ky: float, ref: ReferenceMedium) -> np.ndarray:
    """
    Build 3×6 matrix R mapping Voigt stress → traction components.

    Given Voigt stress (σzz, σxx, σyy, σxy, σzy, σzx), extract
    the traction on z=const: (tzz, txz, tyz) = (σzz, σzx, σzy).

    Returns matrix P such that (tzz, txz, tyz)^T = P @ σ_Voigt.
    """
    P = np.zeros((3, 6), dtype=complex)
    P[0, 0] = 1.0  # tzz = σzz (Voigt index 0)
    P[1, 5] = 1.0  # txz = σzx (Voigt index 5)
    P[2, 4] = 1.0  # tyz = σzy (Voigt index 4)
    return P


# ================================================================
# Full 6×6 T-matrix in displacement-traction basis
# ================================================================


def tmatrix_displacement_traction(
    result: CubeTMatrixResult,
    omega: float,
    a: float,
    kx: float,
    ky: float,
    ref: ReferenceMedium,
) -> np.ndarray:
    """
    Build the full 6×6 T-matrix in the (uz, ux, uy, tzz, txz, tyz) basis.

    This maps the incident displacement-traction vector at the scatterer
    site to the scattered displacement-traction vector.

    The scattered field from Eq (27) of TMatrix_Derivation.pdf:
      u^scat_i = V [H_{ijk} Δσ*_{jk} + ω² G_{ij} Δρ* u^0_j]

    For a plane wave with wavenumber (kx, kz), this becomes a 6×6
    linear mapping on (u, t).

    The T-matrix has the structure:
      T_6x6 = V × [ density_block + stiffness_block ]

    where:
      density_block: Δρ* ω² G_{ij} maps u^0 → u^scat
      stiffness_block: Δc*_{jklm} maps ε^0 → Δσ* → u^scat

    NOTE: This function provides the LOCAL SITE contribution only.
    The full scattered field requires the Green's function G and H
    evaluated at the appropriate source-receiver separation, which
    is handled by the propagation framework (FFTProp).

    For the self-consistent local site T-matrix, G and H are the
    volume integrals Γ₀ and (A, B, C) already computed in the
    CubeTMatrixResult. The 6×6 in the (u,t) basis is then:

      [u^scat]   [Δρ* ω² Γ₀ I₃    0₃] [u^0]     [Δc*_V @ S]
      [      ] = [                    ] [   ] + V [          ]
      [t^scat]   [       ?        ?  ] [t^0]     [    ?     ]

    This is complex because the scattered traction depends on the
    observation point (through gradients of G). For the local site,
    we provide the Voigt effective stiffness Δc* and the strain
    extraction matrix S separately.

    Parameters
    ----------
    result : CubeTMatrixResult
        Pre-computed T-matrix parameters.
    omega : float
        Angular frequency.
    a : float
        Cube half-width.
    kx, ky : float
        Horizontal wavenumbers.
    ref : ReferenceMedium
        Background medium.

    Returns
    -------
    T_6x6 : ndarray, shape (6, 6), complex
        T-matrix in displacement-traction basis.
    """
    V_cube = (2.0 * a) ** 3  # cube volume

    # Strain extraction: (u,t) → Voigt strain
    S = strain_from_displacement_traction(kx, ky, ref)

    # Effective stiffness contrast Voigt matrix
    Dc_star = effective_stiffness_voigt(
        result.Dlambda_star, result.Dmu_star_diag, result.Dmu_star_off
    )

    # Stiffness scattering: scattered Voigt stress = Δc* @ ε^0
    # = Δc* @ S @ (u,t)^0
    stress_from_ut = Dc_star @ S  # (6,6) Voigt stress from (u,t)

    # Convert Voigt stress back to traction on z=const surface
    P_trac = traction_from_strain(kx, ky, ref)

    # For converting scattered stress to scattered (u,t), we need:
    # - Scattered displacements: involve Green's function G (not available here)
    # - Scattered tractions: involve derivatives of G

    # At the LOCAL SITE (self-interaction), the relationship simplifies:
    # The self-consistent amplification already accounts for the local
    # Green's function integrals. The scattered displacement is:
    #   u^scat = V × [ω² Δρ* Γ₀ u^0 + strain_Green_integral @ Δσ*]
    #
    # For the self-consistent formulation, the TOTAL field is:
    #   u = A_u u^0  (displacement amplification)
    # so the scattered displacement is:
    #   u^scat = (A_u - 1) u^0
    #
    # Similarly for strain components via the four amplification factors.

    # Build the amplified strain from Eq (46):
    # ε* = (A_θ θ^0/3) δ + A_e^diag e^0,diag + A_e^off e^0,off
    # This maps incident Voigt strain → total internal Voigt strain.

    # Amplification matrix in Voigt form (6×6)
    A_mat = _amplification_voigt(result.amp_theta, result.amp_e_diag, result.amp_e_off)

    # Scattered strain = (A - I) @ ε^0 in Voigt form
    scat_strain_from_incident = A_mat - np.eye(6)

    # Scattered Voigt stress = Δc_bare @ ε*_scat = Δc_bare @ (A-I) @ ε^0
    # where Δc_bare is the bare (unamplified) stiffness contrast
    # But actually, Δσ* = Δc @ ε* = Δc @ A @ ε^0

    # For the local site T-matrix T_0, we define:
    # T_0 maps the total field at the site to the scattered field.
    # In the self-consistent picture:
    #   u^scat_i = V [H_{ijk}(x_obs, x_s) Δσ*_{jk} + ω² G_{ij}(x_obs,x_s) Δρ* u^0_j]
    # where Δσ*_{jk} = Δc*_{jklm} ε^0_{lm} (effective stiffness × incident strain)

    # The observation point x_obs is NOT the scatterer site for inter-site coupling.
    # For the local self-interaction, G and H are the volume integrals.

    # For NOW, we return a practical 6×6 matrix that computes:
    # Row 0-2 (scattered displacement): V ω² Δρ* Γ₀ δ_{ij} u^0_j
    #   (density scattering contribution only — stiffness part needs H)
    # Row 3-5 (scattered traction): V × P @ Δc* @ S @ (u,t)^0

    T_6x6 = np.zeros((6, 6), dtype=complex)

    # Density scattering: u^scat = V ω² Δρ* Γ₀ u^0
    density_factor = V_cube * omega**2 * result.Drho_star * result.Gamma0
    T_6x6[0, 0] = density_factor  # uz → uz
    T_6x6[1, 1] = density_factor  # ux → ux
    T_6x6[2, 2] = density_factor  # uy → uy

    # Stiffness scattering for tractions: t^scat = V × P @ Δc* @ S @ (u,t)^0
    T_6x6[3:6, :] = V_cube * P_trac @ stress_from_ut

    return T_6x6


# ================================================================
# Amplification matrix in Voigt form
# ================================================================


def _amplification_voigt(
    amp_theta: complex, amp_e_diag: complex, amp_e_off: complex
) -> np.ndarray:
    """
    6×6 amplification matrix in Voigt strain space.

    Decomposes incident strain into three eigenspaces and amplifies:
      ε* = A_θ (θ^0/3) δ + A_e^diag e^0,diag + A_e^off e^0,off

    where θ^0 = ε^0_{ii} (trace = dilatation) and
    e^0 = ε^0 - (θ^0/3)δ (deviatoric strain).
    """
    A = np.zeros((6, 6), dtype=complex)

    # Dilatation projector P_θ: projects onto (1,1,1,0,0,0)/3
    # in Voigt notation, θ = ε1+ε2+ε3
    # P_θ @ ε_V = (θ/3, θ/3, θ/3, 0, 0, 0)
    P_theta = np.zeros((6, 6))
    P_theta[:3, :3] = 1.0 / 3.0

    # Diagonal deviatoric projector: D block - P_θ
    P_diag_dev = np.zeros((6, 6))
    P_diag_dev[:3, :3] = np.eye(3) - 1.0 / 3.0

    # Off-diagonal projector
    P_off = np.zeros((6, 6))
    P_off[3, 3] = 1.0
    P_off[4, 4] = 1.0
    P_off[5, 5] = 1.0

    A = amp_theta * P_theta + amp_e_diag * P_diag_dev + amp_e_off * P_off

    return A


# ================================================================
# Convenience: scattered field components
# ================================================================


def scattered_stress_voigt(
    result: CubeTMatrixResult, incident_strain_voigt: np.ndarray
) -> np.ndarray:
    """
    Compute scattered Voigt stress from incident Voigt strain.

    Δσ*_V = Δc*_V @ ε^0_V

    where Δc* is the effective stiffness contrast matrix.

    Parameters
    ----------
    result : CubeTMatrixResult
        T-matrix result.
    incident_strain_voigt : ndarray, shape (6,)
        Incident strain in Voigt notation.

    Returns
    -------
    stress_voigt : ndarray, shape (6,)
        Scattered stress perturbation in Voigt notation.
    """
    Dc = effective_stiffness_voigt(
        result.Dlambda_star, result.Dmu_star_diag, result.Dmu_star_off
    )
    return Dc @ incident_strain_voigt
