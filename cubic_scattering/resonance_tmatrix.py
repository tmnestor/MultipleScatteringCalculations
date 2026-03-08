"""
Resonance-regime T-matrix via internal Foldy-Lax subdivision.

Full-wave T-matrix for cubic elastic scatterers in the resonance regime
(ka ~ O(1)).

Rayleigh regime (ka << 1)
    ``effective_contrasts.py`` handles this via Taylor-expanded Green's
    tensor with analytic self-consistent amplification factors.  The
    Eshelby singularity is avoided by polynomial expansion --- valid only
    when the field is effectively uniform across the scatterer.

Resonance regime (ka ~ O(1))
    Field variation across the cube interior is no longer negligible.
    This module handles the resonance regime by:

    1. Subdividing the cube of half-width *a* into n³ sub-cells, each
       of half-width ``a_sub = a / n``.  For n large enough,
       ``ka_sub << 1``, so each sub-cell IS in the Rayleigh regime.
    2. Computing the single-site Rayleigh T-matrix for each sub-cell via
       the existing ``compute_cube_tmatrix`` (re-uses all validated code).
    3. Building and solving the internal Foldy-Lax system for the n³
       sub-cells, using the full elastodynamic Green's tensor (near- +
       intermediate- + far-field) for inter-sub-cell coupling.
    4. Extracting the effective composite 3×3 T-matrix that maps an
       incident displacement at the cube centre to the total scattered
       displacement amplitude.

    The composite T-matrix reduces to the Rayleigh result as n → 1
    (verified by ``validate_rayleigh_limit``) and captures internal
    resonances as ka increases.

Green's tensor
--------------
Kupradze solution (Ben-Menahem & Singh 1981, eq. 2.138)::

    G_ij = (1/4πρω²) { k_S² δ_ij h_S + ∂_i ∂_j (h_S − h_P) }

where ``h_X = exp(i k_X r) / r``.  Expanding the double derivative
explicitly::

    G_ij = (1/4πρω²) [φ(r) δ_ij + ψ(r) γ_i γ_j]

    φ(r) = k_S² e^{ik_S r}/r − (1 − ik_S r) e^{ik_S r}/r³
           + (1 − ik_P r) e^{ik_P r}/r³

    ψ(r) = 3(1 − ik_S r) e^{ik_S r}/r³ − 3(1 − ik_P r) e^{ik_P r}/r³
           + k_P² e^{ik_P r}/r − k_S² e^{ik_S r}/r

As r → 0 the 1/r³ terms cancel between P and S (they are equal and
opposite), leaving an integrable 1/r singularity --- consistent with
Kelvin's static solution.  No principal-value treatment of G itself is
required; the singular self-interaction is absorbed into the sub-cell
Rayleigh T-matrix (computed by ``effective_contrasts.py``).

Sub-cell T-matrix (coupled 9×9)
-------------------------------
Each sub-cell carries a 9×9 block-diagonal local T-matrix coupling
displacement (3 DOF) and Voigt strain (6 DOF)::

    T_loc = [[ω²Δρ* V_sub I₃,    0₃ₓ₆     ],
             [0₆ₓ₃,              V_sub Δc*_V]]

This captures both density scattering (Δρ) and stiffness scattering
(Δλ, Δμ) simultaneously.

Foldy-Lax system (9N×9N)
-------------------------
::

    (I − P̃·T̃) · Ψ_exc = Ψ⁰

    P̃[9m:9m+9, 9n:9n+9] = [[G, C], [H, S]](x_m − x_n)   (m ≠ n)
    T̃ = diag(T_loc, …, T_loc)             (block-diagonal, 9N × 9N)

where P̃ is the 9×9 propagator with displacement–displacement (G),
displacement–strain (C, H), and strain–strain (S) blocks from the
elastodynamic Green's tensor and its derivatives.

Composite T-matrix
------------------
::

    T_comp[9×9] = Σ_n T_loc · Ψ_exc[9n:9n+9, :]
    T3x3 = T_comp[:3, :3]

The 9×9 composite T-matrix maps incident (displacement, strain) to
scattered (force monopole, stress dipole).  The 3×3 displacement block
T3x3 is the effective displacement T-matrix for the full cube.

Interface compatibility
-----------------------
The module is designed as a drop-in complement to
``effective_contrasts.py``::

    from cubic_scattering import ReferenceMedium, MaterialContrast
    from cubic_scattering.resonance_tmatrix import (
        ResonanceTmatrixResult,
        compute_resonance_tmatrix,
        suggest_n_subcells,
        voigt_tmatrix_from_resonance_result,
    )

References
----------
Ben-Menahem, A. & Singh, S.J. (1981). *Seismic Waves and Sources*.
    Springer.
Gubernatis, J.E., Domany, E. & Krumhansl, J.A. (1977). J. Appl. Phys.,
    48, 2804.
Wu, R.-S. & Ben-Menahem, A. (1985). Geophys. J. R. Astr. Soc., 81, 609.
Waterman, P.C. (1969). Phys. Rev. D, 3, 825. (T-matrix / null-field
    method)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import (
    CubeTMatrixResult,
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from .voigt_tmatrix import (
    effective_stiffness_voigt,
    strain_from_displacement_traction,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Complex3x3 = NDArray[np.complexfloating]  # shape (3, 3)
Complex6x6 = NDArray[np.complexfloating]  # shape (6, 6)

# Voigt index → pair of Cartesian indices (z=0, x=1, y=2)
VOIGT_PAIRS: list[tuple[int, int]] = [
    (0, 0),  # zz
    (1, 1),  # xx
    (2, 2),  # yy
    (1, 2),  # xy
    (0, 2),  # zy
    (0, 1),  # zx
]

# ---------------------------------------------------------------------------
# Validity threshold (ka_sub)
# ---------------------------------------------------------------------------
_KA_SUB_WARN: float = 0.50  # warn if sub-cells approach resonance
_KA_SUB_MAX: float = 0.30  # recommended upper bound for Rayleigh validity


# ===========================================================================
# Section 1 — Full elastodynamic Green's tensor
# ===========================================================================


def elastodynamic_greens(
    r_vec: NDArray[np.floating],
    omega: float,
    ref: ReferenceMedium,
) -> Complex3x3:
    """Full elastodynamic (Stokes) Green's tensor G_ij(r, ω).

    Kupradze solution for a homogeneous isotropic full space::

        G_ij = (1/4πρω²) [φ(r) δ_ij + ψ(r) γ_i γ_j]

    Args:
        r_vec: Displacement vector ``x_field − x_source`` (m),
            shape (3,).
        omega: Angular frequency (rad/s).
        ref: Background medium (alpha, beta, rho).

    Returns:
        Displacement Green's tensor G, shape (3, 3), complex (m/N).
        Returns ``np.zeros((3, 3), complex)`` at r = 0; the caller is
        responsible for the self-interaction (handled via the sub-cell
        Rayleigh T-matrix).

    Notes:
        The 1/r³ near-field singularity from P and S contributions
        cancels identically (the terms are equal in magnitude and
        opposite in sign as r → 0), leaving an integrable 1/r
        singularity --- no PV treatment of G itself is required.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0e-14:
        return np.zeros((3, 3), dtype=complex)

    kP: float = omega / ref.alpha
    kS: float = omega / ref.beta
    gamma: NDArray = r_vec / r

    expP = np.exp(1j * kP * r)
    expS = np.exp(1j * kS * r)

    # Near-field factors (1 − ikr) e^{ikr}/r³
    nfP = (1.0 - 1j * kP * r) * expP / r**3
    nfS = (1.0 - 1j * kS * r) * expS / r**3

    phi: complex = kS**2 * expS / r - nfS + nfP
    psi: complex = 3.0 * nfS - 3.0 * nfP + kP**2 * expP / r - kS**2 * expS / r

    prefactor = 1.0 / (4.0 * np.pi * ref.rho * omega**2)
    return prefactor * (phi * np.eye(3) + psi * np.outer(gamma, gamma))


def _radial_functions(
    r: float, kP: float, kS: float
) -> tuple[complex, complex, complex, complex, complex, complex]:
    """Compute radial scalar functions φ, ψ and their first/second derivatives.

    Both φ and ψ are sums of terms ``c · e^{ik r} · r^p`` for p in {-1,-2,-3}.
    Derivatives use::

        d/dr [e^{ikr} r^p]   = e^{ikr} r^{p-1} (ikr + p)
        d²/dr² [e^{ikr} r^p] = e^{ikr} r^{p-2} [(ikr)² + 2p(ikr) + p(p-1)]

    Args:
        r: Distance (must be > 0).
        kP: P-wave wavenumber.
        kS: S-wave wavenumber.

    Returns:
        (φ, ψ, φ′, ψ′, φ″, ψ″).
    """
    expP = np.exp(1j * kP * r)
    expS = np.exp(1j * kS * r)
    exps = {kP: expP, kS: expS}

    # Terms: (coefficient, wavenumber, power_of_r)
    phi_terms: list[tuple[complex, float, int]] = [
        (kS**2, kS, -1),
        (1j * kS, kS, -2),
        (-1.0, kS, -3),
        (-1j * kP, kP, -2),
        (1.0, kP, -3),
    ]
    psi_terms: list[tuple[complex, float, int]] = [
        (-(kS**2), kS, -1),
        (-3j * kS, kS, -2),
        (3.0, kS, -3),
        (kP**2, kP, -1),
        (3j * kP, kP, -2),
        (-3.0, kP, -3),
    ]

    phi = phi_p = phi_pp = 0j
    psi = psi_p = psi_pp = 0j

    for c, k, p in phi_terms:
        e = exps[k]
        ikr = 1j * k * r
        phi += c * e * r**p
        phi_p += c * e * r ** (p - 1) * (ikr + p)
        phi_pp += c * e * r ** (p - 2) * (ikr**2 + 2 * p * ikr + p * (p - 1))

    for c, k, p in psi_terms:
        e = exps[k]
        ikr = 1j * k * r
        psi += c * e * r**p
        psi_p += c * e * r ** (p - 1) * (ikr + p)
        psi_pp += c * e * r ** (p - 2) * (ikr**2 + 2 * p * ikr + p * (p - 1))

    return phi, psi, phi_p, psi_p, phi_pp, psi_pp


def elastodynamic_greens_deriv(
    r_vec: NDArray[np.floating],
    omega: float,
    ref: ReferenceMedium,
) -> tuple[Complex3x3, NDArray, NDArray]:
    """Green's tensor G, first derivative Gd, and second derivative Gdd.

    Returns:
        (G, Gd, Gdd) with shapes (3,3), (3,3,3), (3,3,3,3).
        All zero at r=0 (self-interaction handled by local T-matrix).

    The formulae follow from the chain rule on
    ``G_{ij} = P [φ δ_{ij} + ψ γ_i γ_j]``::

        G_{ij,k}  → 3 tensor structures (φ′, ψ′−2ψ/r, ψ/r)
        G_{ij,kl} → 7 tensor structures (φ′/r, φ″−φ′/r, ψ′/r−2ψ/r²,
                                          ψ/r², ψ″−5ψ′/r+8ψ/r²)
    """
    r_vec = np.asarray(r_vec, dtype=float)
    r = float(np.linalg.norm(r_vec))
    z3 = np.zeros((3, 3), dtype=complex)
    z33 = np.zeros((3, 3, 3), dtype=complex)
    z333 = np.zeros((3, 3, 3, 3), dtype=complex)
    if r < 1.0e-14:
        return z3, z33, z333

    kP: float = omega / ref.alpha
    kS: float = omega / ref.beta
    g = r_vec / r  # γ_i = unit direction vector

    P = 1.0 / (4.0 * np.pi * ref.rho * omega**2)
    phi, psi, phi_p, psi_p, phi_pp, psi_pp = _radial_functions(r, kP, kS)

    # --- G_{ij} ---
    G = P * (phi * np.eye(3) + psi * np.outer(g, g))

    # --- G_{ij,k} ---
    c1: complex = phi_p
    c2: complex = psi_p - 2.0 * psi / r
    c3: complex = psi / r
    delta = np.eye(3)
    Gd = P * (
        c1 * np.einsum("k,ij->ijk", g, delta)
        + c2 * np.einsum("i,j,k->ijk", g, g, g)
        + c3 * (np.einsum("ik,j->ijk", delta, g) + np.einsum("jk,i->ijk", delta, g))
    )

    # --- G_{ij,kl} --- 7 tensor structures
    t1: complex = phi_p / r  # δ_{ij}δ_{kl}
    t2: complex = phi_pp - phi_p / r  # δ_{ij}γ_kγ_l
    t3: complex = psi_p / r - 2.0 * psi / r**2  # γ_iγ_jδ_{kl}, and 4 mixed terms
    t4: complex = psi / r**2  # δ_{ik}δ_{jl}+δ_{jk}δ_{il}
    t7: complex = psi_pp - 5.0 * psi_p / r + 8.0 * psi / r**2  # γ_iγ_jγ_kγ_l

    Gdd = P * (
        t1 * np.einsum("ij,kl->ijkl", delta, delta)
        + t2 * np.einsum("ij,k,l->ijkl", delta, g, g)
        + t3 * np.einsum("i,j,kl->ijkl", g, g, delta)
        + t4
        * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("jk,il->ijkl", delta, delta)
        )
        + t3
        * (
            np.einsum("il,j,k->ijkl", delta, g, g)
            + np.einsum("jl,i,k->ijkl", delta, g, g)
        )
        + t3
        * (
            np.einsum("ik,j,l->ijkl", delta, g, g)
            + np.einsum("jk,i,l->ijkl", delta, g, g)
        )
        + t7 * np.einsum("i,j,k,l->ijkl", g, g, g, g)
    )

    return G, Gd, Gdd


def _voigt_contract(Gd: NDArray, Gdd: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Contract Green's tensor derivatives to Voigt propagator blocks.

    Args:
        Gd: First derivative G_{ij,k}, shape (3,3,3).
        Gdd: Second derivative G_{ij,kl}, shape (3,3,3,3).

    Returns:
        (C, H, S) with shapes (3,6), (6,3), (6,6).

        - C[i,α]: stress source α → displacement i.
        - H[α,j]: force j → Voigt strain α.
        - S[α,β]: stress source β → Voigt strain α.
    """
    C = np.zeros((3, 6), dtype=complex)
    H = np.zeros((6, 3), dtype=complex)
    S = np.zeros((6, 6), dtype=complex)

    for alpha, (p, q) in enumerate(VOIGT_PAIRS):
        # C[i,α]: G_{ip,p} for diagonal, G_{ip,q}+G_{iq,p} for off-diag
        for i in range(3):
            if p == q:
                C[i, alpha] = Gd[i, p, p]
            else:
                C[i, alpha] = Gd[i, p, q] + Gd[i, q, p]

        # H[α,j]: G_{pj,p} for diagonal, G_{pj,q}+G_{qj,p} for off-diag
        for j in range(3):
            if p == q:
                H[alpha, j] = Gd[p, j, p]
            else:
                H[alpha, j] = Gd[p, j, q] + Gd[q, j, p]

        # S[α,β]: double Voigt contraction of G_{ij,kl}
        for beta, (m, n) in enumerate(VOIGT_PAIRS):
            if p == q and m == n:
                S[alpha, beta] = Gdd[p, m, m, p]
            elif p == q and m != n:
                S[alpha, beta] = Gdd[p, m, n, p] + Gdd[p, n, m, p]
            elif p != q and m == n:
                S[alpha, beta] = Gdd[p, m, m, q] + Gdd[q, m, m, p]
            else:
                S[alpha, beta] = (
                    Gdd[p, m, n, q]
                    + Gdd[p, n, m, q]
                    + Gdd[q, m, n, p]
                    + Gdd[q, n, m, p]
                )

    return C, H, S


def _propagator_block_9x9(
    r_vec: NDArray[np.floating],
    omega: float,
    ref: ReferenceMedium,
) -> NDArray:
    """9x9 inter-sub-cell propagator [[G, C], [H, S]].

    Returns zeros at r=0 (self-interaction is in T_loc).
    """
    G, Gd, Gdd = elastodynamic_greens_deriv(r_vec, omega, ref)
    C, H, S = _voigt_contract(Gd, Gdd)
    P = np.zeros((9, 9), dtype=complex)
    P[:3, :3] = G
    P[:3, 3:] = C
    P[3:, :3] = H
    P[3:, 3:] = S
    return P


# ===========================================================================
# Section 2 — Sub-cell geometry
# ===========================================================================


def sub_cell_centres(a: float, n: int) -> NDArray[np.floating]:
    """Centres of the n³ sub-cells for a cube of half-width *a*.

    The cube occupies [−a, a]³.  Sub-cells have half-width
    ``a_sub = a / n`` and are arranged on a regular Cartesian grid.

    Args:
        a: Cube half-width (m).
        n: Number of sub-cells per edge (total n³ sub-cells).

    Returns:
        Sub-cell centre coordinates, shape (n³, 3), float64 (m).
    """
    a_sub = a / n
    # n uniformly-spaced centres from (−a + a_sub) to (a − a_sub)
    coords = np.linspace(-a + a_sub, a - a_sub, n)
    ix, iy, iz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])


# ===========================================================================
# Section 3 — Sub-cell T-matrix (coupled 9×9)
# ===========================================================================


def _sub_cell_tmatrix_9x9(
    rayleigh: CubeTMatrixResult,
    omega: float,
    a_sub: float,
) -> NDArray:
    """9x9 local T-matrix for a sub-cell (displacement + Voigt strain).

    Block-diagonal::

        T = [[ω²Δρ* V_sub I₃,    0₃ₓ₆     ],
             [0₆ₓ₃,              V_sub Δc*_V]]

    Args:
        rayleigh: Rayleigh T-matrix result for the sub-cell.
        omega: Angular frequency (rad/s).
        a_sub: Sub-cell half-width (m).

    Returns:
        Local T-matrix, shape (9, 9), complex.
    """
    T = np.zeros((9, 9), dtype=complex)
    V_sub = (2.0 * a_sub) ** 3
    T[:3, :3] = omega**2 * complex(rayleigh.Drho_star) * V_sub * np.eye(3)
    T[3:, 3:] = V_sub * effective_stiffness_voigt(
        rayleigh.Dlambda_star, rayleigh.Dmu_star_diag, rayleigh.Dmu_star_off
    )
    return T


# ===========================================================================
# Section 4 — Utility helpers
# ===========================================================================


def suggest_n_subcells(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    ka_threshold: float = _KA_SUB_MAX,
) -> int:
    """Minimum sub-cells per edge to keep each sub-cell Rayleigh-valid.

    Returns the smallest integer n such that
    ``k_S · (a / n) ≤ ka_threshold``.

    Args:
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        ref: Background medium.
        ka_threshold: Target ka_sub upper bound (default 0.30).

    Returns:
        n (always ≥ 1).

    Examples:
        >>> ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2700.0)
        >>> suggest_n_subcells(omega=2*np.pi*50, a=1.0, ref=ref)
        1
        >>> suggest_n_subcells(omega=2*np.pi*50, a=10.0, ref=ref)
        4
    """
    ka_cube = (omega / ref.beta) * a
    return max(1, int(np.ceil(ka_cube / ka_threshold)))


def _build_incident_field_coupled(
    centres: NDArray,
    omega: float,
    ref: ReferenceMedium,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
) -> NDArray:
    """Build the (9N, 9) incident-field matrix for the coupled system.

    9 independent incident patterns:

    - **Columns 0--2** (displacement inputs): uniform displacement
      ``u^inc(x_m) = ê_p``, ``ε^inc_V = 0``.
    - **Columns 3--8** (strain inputs): linear displacement variation
      ``u^inc(x_m) = ε⁰_tensor · (x_m − x_centre)`` plus uniform
      Voigt strain ``ε^inc_V = ê_α``.

    Phase factor ``exp(i k · x_m)`` is applied to all patterns.

    Args:
        centres: Sub-cell centre coordinates, shape (N, 3).
        omega: Angular frequency (rad/s).
        ref: Background medium.
        k_hat: Unit propagation direction (default ẑ).
        wave_type: ``'S'`` or ``'P'``.

    Returns:
        Incident-field matrix, shape (9N, 9), complex.
    """
    N = len(centres)
    if k_hat is None:
        k_hat = np.array([0.0, 0.0, 1.0])
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat /= np.linalg.norm(k_hat)

    k_mag = omega / (ref.beta if wave_type == "S" else ref.alpha)
    phases = np.exp(1j * k_mag * (centres @ k_hat))  # (N,)
    x_centre = np.mean(centres, axis=0)

    U0 = np.zeros((9 * N, 9), dtype=complex)
    for m in range(N):
        phase = phases[m]
        # Columns 0-2: uniform displacement, zero strain
        U0[9 * m : 9 * m + 3, :3] = np.eye(3) * phase

        # Columns 3-8: strain inputs
        dx = centres[m] - x_centre
        for alpha, (p, q) in enumerate(VOIGT_PAIRS):
            # Strain tensor from unit Voigt strain ê_α
            eps_tensor = np.zeros((3, 3))
            if p == q:
                eps_tensor[p, p] = 1.0
            else:
                eps_tensor[p, q] = 0.5
                eps_tensor[q, p] = 0.5

            # u^inc = ε⁰_tensor · (x_m − x_centre)
            U0[9 * m : 9 * m + 3, 3 + alpha] = (eps_tensor @ dx) * phase
            # Voigt strain = unit vector ê_α
            U0[9 * m + 3 + alpha, 3 + alpha] = phase

    return U0


# ===========================================================================
# Section 5 — Result dataclass
# ===========================================================================


@dataclass
class ResonanceTmatrixResult:
    """Result from ``compute_resonance_tmatrix``.

    Attributes:
        T3x3: Effective displacement T-matrix for the full cube,
            shape (3, 3), complex.  ``T3x3[:, p]`` = total scattered
            amplitude for incident polarisation e_p.
        ka_cube: Dimensionless frequency k_S · a (S-wave).
        ka_sub: Dimensionless frequency k_S · (a/n_sub) for each
            sub-cell.  Should be < 0.3 for Rayleigh validity.
        n_sub: Number of sub-cells per edge (n_sub³ total).
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        ref: Background medium used.
        contrast: Material contrast used.
        condition_number: Condition number of the Foldy-Lax matrix.
            Values > 1e10 indicate near-resonance or solver instability.
        rayleigh_result: Full-cube Rayleigh T-matrix (n_sub=1 reference),
            used for Rayleigh-limit validation.
        n_iterations_converged: If a Neumann-series solve was used
            (``neumann_order > 0``), the order at which the series
            converged.  ``None`` for direct solve.
    """

    T3x3: Complex3x3
    ka_cube: float
    ka_sub: float
    n_sub: int
    omega: float
    a: float
    ref: ReferenceMedium
    contrast: MaterialContrast
    condition_number: float
    rayleigh_result: CubeTMatrixResult
    T_comp_9x9: NDArray
    force_monopole: Complex3x3
    stress_dipole_voigt: NDArray
    n_iterations_converged: int | None = None


# ===========================================================================
# Section 6 — Main computation
# ===========================================================================


def compute_resonance_tmatrix(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int | None = None,
    k_hat: NDArray | None = None,
    wave_type: str = "S",
    neumann_order: int = 0,
    neumann_tol: float = 1.0e-8,
) -> ResonanceTmatrixResult:
    """Full-wave T-matrix for a cubic scatterer (resonance regime).

    Subdivides the cube into ``n_sub³`` sub-cells, solves the coupled
    9N×9N internal Foldy-Lax system (displacement + Voigt strain), and
    returns the composite T-matrix for the full cube.

    Args:
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        ref: Background medium.
        contrast: Material contrast.
        n_sub: Sub-cells per edge.  If ``None``, auto-selected via
            ``suggest_n_subcells`` to keep ``ka_sub < 0.3``.
        k_hat: Unit incident propagation direction, shape (3,).
            Default: ẑ.
        wave_type: ``'S'`` or ``'P'`` --- determines incident-field
            phase (default ``'S'``).
        neumann_order: If > 0, solve via truncated Neumann series to
            this order instead of a direct (LU) solve.  Order 0 =
            direct solve only (default).
        neumann_tol: Convergence tolerance for the Neumann series
            (relative norm of last correction).

    Returns:
        ``ResonanceTmatrixResult`` with the composite 3×3 and 9×9
        T-matrices.

    Notes:
        Incident-field phase variation (``exp(i k · x_m)``) is included
        for all sub-cells, which is essential when ``ka_cube ~ O(1)``.
        In the Rayleigh limit (``ka_cube << 1``) this reduces to a
        uniform incident field.

    Examples:
        >>> ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2700.0)
        >>> c = MaterialContrast(Dlambda=1e9, Dmu=0.5e9, Drho=200.0)
        >>> n = suggest_n_subcells(omega=2*np.pi*50, a=5.0, ref=ref)
        >>> result = compute_resonance_tmatrix(
        ...     2*np.pi*50, a=5.0, ref=ref, contrast=c, n_sub=n,
        ... )
        >>> print(f"ka = {result.ka_cube:.3f}, n = {result.n_sub}")
    """
    # --- Auto-select n_sub ---
    if n_sub is None:
        n_sub = suggest_n_subcells(omega, a, ref)

    a_sub = a / n_sub
    kS = omega / ref.beta
    ka_cube = kS * a
    ka_sub = kS * a_sub

    if ka_sub > _KA_SUB_WARN:
        warnings.warn(
            f"ka_sub = {ka_sub:.3f} > {_KA_SUB_WARN}. "
            f"Sub-cells are outside the Rayleigh regime. "
            f"Use n_sub >= {suggest_n_subcells(omega, a, ref)} "
            f"(currently n_sub = {n_sub}).",
            UserWarning,
            stacklevel=2,
        )

    # --- Sub-cell Rayleigh T-matrix (same for all cells) ---
    rayleigh_sub = compute_cube_tmatrix(omega, a_sub, ref, contrast)

    # --- Full-cube Rayleigh result (n_sub=1 reference for validation) ---
    rayleigh_full = compute_cube_tmatrix(omega, a, ref, contrast)

    # --- Sub-cell centres ---
    centres = sub_cell_centres(a, n_sub)  # (N, 3)
    N = centres.shape[0]  # n_sub³

    return _solve_coupled(
        omega,
        a,
        a_sub,
        ref,
        contrast,
        n_sub,
        centres,
        N,
        rayleigh_sub,
        rayleigh_full,
        ka_cube,
        ka_sub,
        k_hat,
        wave_type,
        neumann_order,
        neumann_tol,
    )


def _solve_coupled(
    omega: float,
    a: float,
    a_sub: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    centres: NDArray,
    N: int,
    rayleigh_sub: CubeTMatrixResult,
    rayleigh_full: CubeTMatrixResult,
    ka_cube: float,
    ka_sub: float,
    k_hat: NDArray | None,
    wave_type: str,
    neumann_order: int,
    neumann_tol: float,
) -> ResonanceTmatrixResult:
    """Coupled (Δρ, Δλ, Δμ) 9N×9N Foldy-Lax solve."""
    # --- 9×9 local T-matrix ---
    T_loc = _sub_cell_tmatrix_9x9(rayleigh_sub, omega, a_sub)

    # --- 9N×9N propagator (off-diagonal only) ---
    P_tilde = np.zeros((9 * N, 9 * N), dtype=complex)
    for m in range(N):
        for n in range(N):
            if m != n:
                P_tilde[9 * m : 9 * m + 9, 9 * n : 9 * n + 9] = _propagator_block_9x9(
                    centres[m] - centres[n], omega, ref
                )

    # --- Block-diagonal T̃ = I_N ⊗ T_loc ---
    T_block = np.kron(np.eye(N, dtype=complex), T_loc)  # (9N, 9N)

    # --- Foldy-Lax matrix A = I − P̃·T̃ ---
    A = np.eye(9 * N, dtype=complex) - P_tilde @ T_block
    cond_num = float(np.linalg.cond(A))
    if cond_num > 1.0e10:
        warnings.warn(
            f"Coupled Foldy-Lax condition number = {cond_num:.2e}. "
            "Near-resonance or strong-coupling instability.",
            UserWarning,
            stacklevel=2,
        )

    # --- 9N×9 incident field ---
    psi_inc = _build_incident_field_coupled(
        centres, omega, ref, k_hat=k_hat, wave_type=wave_type
    )

    # --- Solve for exciting field ---
    n_iters_converged: int | None = None
    if neumann_order == 0:
        psi_exc = np.linalg.solve(A, psi_inc)
    else:
        B_mat = P_tilde @ T_block
        psi_exc = psi_inc.copy()
        term = psi_inc.copy()
        for k in range(1, neumann_order + 1):
            term = B_mat @ term
            psi_exc = psi_exc + term
            rel_norm = np.linalg.norm(term) / (np.linalg.norm(psi_exc) + 1.0e-300)
            if rel_norm < neumann_tol:
                n_iters_converged = k
                break
        else:
            warnings.warn(
                f"Coupled Neumann series did not converge in "
                f"{neumann_order} orders (ka_cube = {ka_cube:.3f}).",
                UserWarning,
                stacklevel=2,
            )

    # --- 9×9 composite T-matrix ---
    T_comp = np.zeros((9, 9), dtype=complex)
    for n in range(N):
        T_comp += T_loc @ psi_exc[9 * n : 9 * n + 9, :]

    T3x3 = T_comp[:3, :3].copy()

    return ResonanceTmatrixResult(
        T3x3=T3x3,
        ka_cube=ka_cube,
        ka_sub=ka_sub,
        n_sub=n_sub,
        omega=omega,
        a=a,
        ref=ref,
        contrast=contrast,
        condition_number=cond_num,
        rayleigh_result=rayleigh_full,
        n_iterations_converged=n_iters_converged,
        T_comp_9x9=T_comp,
        force_monopole=T_comp[:3, :3].copy(),
        stress_dipole_voigt=T_comp[3:, 3:].copy(),
    )


# ===========================================================================
# Section 7 — Output: Voigt 6×6 T-matrix
# ===========================================================================


def _volume_integral_voigt(Ac: complex, Bc: complex, Cc: complex) -> NDArray:
    """6×6 symmetrized volume-integral tensor from A, B, C values.

    Builds S_V_{αβ} by Voigt contraction of::

        I_{ijkl} = A δ_{ij}δ_{kl} + B(δ_{ik}δ_{jl} + δ_{il}δ_{jk}) + C E_{ijkl}

    The result has the same block-diagonal structure as the Voigt T-matrix:
    D block (3×3): diag = A+2B+C, off-diag = B.
    S block (3×3): diag = 2(A+B), off-diag = 0.
    """
    S_V = np.zeros((6, 6), dtype=complex)
    # D block
    diag_D = Ac + 2.0 * Bc + Cc
    for i in range(3):
        S_V[i, i] = diag_D
        for j in range(3):
            if i != j:
                S_V[i, j] = Bc
    # S block
    shear = 2.0 * (Ac + Bc)
    S_V[3, 3] = shear
    S_V[4, 4] = shear
    S_V[5, 5] = shear
    return S_V


def _traction_from_voigt_strain(ref: ReferenceMedium) -> NDArray:
    """3×6 background-stiffness traction extraction on z=const.

    Maps Voigt strain (εzz, εxx, εyy, 2εxy, 2εzy, 2εzx) to
    traction (tzz, txz, tyz) via the background stiffness::

        tzz = (λ+2μ) εzz + λ εxx + λ εyy
        txz = μ · 2εzx
        tyz = μ · 2εzy
    """
    lam = ref.lam
    mu = ref.mu
    C_trac = np.zeros((3, 6), dtype=complex)
    C_trac[0, 0] = lam + 2.0 * mu  # tzz from εzz
    C_trac[0, 1] = lam  # tzz from εxx
    C_trac[0, 2] = lam  # tzz from εyy
    C_trac[1, 5] = mu  # txz from 2εzx
    C_trac[2, 4] = mu  # tyz from 2εzy
    return C_trac


def voigt_tmatrix_from_resonance_result(
    result: ResonanceTmatrixResult,
    kx: float = 0.0,
    ky: float = 0.0,
) -> Complex6x6:
    """6×6 Voigt T-matrix in the (uz, ux, uy, tzz, txz, tyz) basis.

    Constructs the **near-field exact** 6×6 via the volume-integrated
    Green's tensor::

        T_6x6 = G_self @ T_comp_9x9 @ input_conv

    where:

    - ``input_conv`` (9×6): extracts (u, ε_V) from (u, t).
    - ``T_comp_9x9`` (9×9): coupled composite T-matrix.
    - ``G_self`` (6×9): converts (F, Δσ*_V) to (u, t)^scat via Γ₀ and
      the symmetrized volume integral (A, B, C).

    Args:
        result: Output of ``compute_resonance_tmatrix``.
        kx: Horizontal wavenumber in x-direction (for strain extraction).
        ky: Horizontal wavenumber in y-direction.

    Returns:
        Voigt T-matrix, shape (6, 6), complex.
    """
    return _voigt_tmatrix_near_field(result, kx, ky)


def _voigt_tmatrix_near_field(
    result: ResonanceTmatrixResult,
    kx: float,
    ky: float,
) -> Complex6x6:
    """Near-field exact 6×6 via volume-integrated Green's tensor."""
    ref = result.ref

    # Input conversion: (u,t) → (u, ε_V)
    S = strain_from_displacement_traction(kx, ky, ref)  # 6×6
    input_conv = np.zeros((9, 6), dtype=complex)
    input_conv[:3, :3] = np.eye(3)  # u passes through
    input_conv[3:, :] = S  # ε_V from (u,t)

    # Output conversion: (F, Δσ*_V) → (u,t)^scat
    rayleigh = result.rayleigh_result
    Gamma0 = rayleigh.Gamma0
    S_V = _volume_integral_voigt(rayleigh.Ac, rayleigh.Bc, rayleigh.Cc)
    C_trac = _traction_from_voigt_strain(ref)

    G_self = np.zeros((6, 9), dtype=complex)
    G_self[:3, :3] = Gamma0 * np.eye(3)  # force → displacement (Γ₀)
    G_self[3:, 3:] = C_trac @ S_V  # stress → traction

    return G_self @ result.T_comp_9x9 @ input_conv


# ===========================================================================
# Section 8 — Validation helpers
# ===========================================================================


def validate_rayleigh_limit(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    rtol: float = 0.05,
) -> dict:
    """Verify that the resonance T-matrix reduces to Rayleigh at n=1.

    At n=1 with no inter-cell coupling, the displacement block of the
    coupled 9×9 T-matrix must match the Rayleigh density-only T-matrix
    ``ω² Δρ* V I₃``.

    Args:
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        ref: Background medium.
        contrast: Material contrast.
        rtol: Relative tolerance for agreement (default 5%).

    Returns:
        Dict with keys ``'ka_cube'``, ``'T3x3_resonance'``,
        ``'T3x3_rayleigh'``, ``'relative_error'``, ``'passed'``.
    """
    result = compute_resonance_tmatrix(omega, a, ref, contrast, n_sub=1)
    T_res = result.T3x3

    # Rayleigh reference: density-only T-matrix = ω² Δρ* V I₃
    rayleigh = compute_cube_tmatrix(omega, a, ref, contrast)
    V = (2.0 * a) ** 3
    t_eff = V * omega**2 * complex(rayleigh.Drho_star)
    T_ray = t_eff * np.eye(3, dtype=complex)

    norm_ray = float(np.linalg.norm(T_ray))
    denom = max(norm_ray, 1.0e-300)
    rel_err = float(np.linalg.norm(T_res - T_ray)) / denom

    return {
        "ka_cube": result.ka_cube,
        "T3x3_resonance": T_res,
        "T3x3_rayleigh": T_ray,
        "relative_error": rel_err,
        "passed": rel_err < rtol,
    }


def scattering_order_decomposition(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_sub: int,
    max_order: int = 10,
    tol: float = 1.0e-8,
) -> dict:
    """Decompose the composite T-matrix by Neumann scattering order.

    Returns the partial T-matrix for each order k = 0 (direct), 1
    (single intralayer multiple), 2 (double), … and the fully summed
    result from a direct solve.  Divergence of the Neumann series
    (strong contrast) is clearly visible as growing ``‖T_k‖``.

    Args:
        omega: Angular frequency (rad/s).
        a: Cube half-width (m).
        ref: Background medium.
        contrast: Material contrast.
        n_sub: Sub-cells per edge.
        max_order: Maximum Neumann order to compute.
        tol: Convergence tolerance.

    Returns:
        Dict with keys ``'orders'``, ``'T3x3_by_order'``,
        ``'T3x3_partial'``, ``'T3x3_full'``, ``'norm_by_order'``,
        ``'converged_order'``, ``'ka_cube'``.
    """
    # Direct solve (ground truth)
    full_result = compute_resonance_tmatrix(
        omega, a, ref, contrast, n_sub=n_sub, neumann_order=0
    )
    T_full = full_result.T3x3
    norm_full = float(np.linalg.norm(T_full))

    # Neumann orders
    orders: list[int] = []
    T_by_order: list[Complex3x3] = []
    T_partial: list[Complex3x3] = []
    norms: list[float] = []
    converged_at: int | None = None

    T_by_order_cumulative: Complex3x3 | None = None
    for k in range(0, max_order + 1):
        res_k = compute_resonance_tmatrix(
            omega,
            a,
            ref,
            contrast,
            n_sub=n_sub,
            neumann_order=k,
            neumann_tol=0.0,  # fixed order, no early stop
        )
        T_k = res_k.T3x3

        # Extract k-th order contribution
        if k == 0:
            T_k_contribution = T_k.copy()
        else:
            assert T_by_order_cumulative is not None
            T_k_contribution = T_k - T_by_order_cumulative

        orders.append(k)
        T_by_order.append(T_k_contribution)
        T_partial.append(T_k.copy())
        norm_k = float(np.linalg.norm(T_k_contribution))
        norms.append(norm_k)

        if converged_at is None and norm_k < tol * max(norm_full, 1.0e-300):
            converged_at = k

        T_by_order_cumulative = T_k.copy()

    return {
        "orders": orders,
        "T3x3_by_order": T_by_order,
        "T3x3_partial": T_partial,
        "T3x3_full": T_full,
        "norm_by_order": norms,
        "converged_order": converged_at,
        "ka_cube": full_result.ka_cube,
    }
