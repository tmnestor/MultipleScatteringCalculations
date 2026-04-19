"""
incident_field.py
Project plane-wave incident fields onto the Galerkin basis (27 or 57 components).

For u_inc(r) = pol * exp(i k·r), the overlap with basis function φ_α
over cube [-a,a]³ factorizes into 1D integrals:

  ∫_{-a}^{a} x^m exp(ikx x) dx

These integrals are computed analytically for m = 0, 1, 2, 3.

Basis ordering (0-indexed, matching tmatrix_assembly.py):
  0-2:   constant displacement: e_1, e_2, e_3
  3-5:   axial strain: r_1 e_1, r_2 e_2, r_3 e_3
  6-8:   shear strain
  9-26:  quadratic modes (6 monomials × 3 directions)
  27-56: cubic modes (10 monomials × 3 directions)  [T₅₇ only]
"""

import numpy as np

from .effective_contrasts import ReferenceMedium

# ================================================================
# 1D Fourier-monomial integrals
# ================================================================


def _monomial_fourier_1d(m: int, k: float, a: float) -> complex:
    """Compute ∫_{-a}^{a} x^m exp(ikx) dx analytically.

    Parameters
    ----------
    m : Monomial power (0, 1, 2, or 3).
    k : Wavenumber component along this axis.
    a : Half-width of cube.

    Returns
    -------
    Complex integral value.
    """
    ka = k * a
    abs_ka = abs(ka)

    if m == 0:
        if abs_ka < 1e-8:
            # Taylor: 2a(1 - (ka)²/6 + (ka)⁴/120)
            ka2 = ka * ka
            return 2.0 * a * (1.0 - ka2 / 6.0 + ka2 * ka2 / 120.0)
        return 2.0 * a * np.sinc(ka / np.pi)  # np.sinc(x) = sin(πx)/(πx)

    if m == 1:
        if abs_ka < 1e-6:
            # Taylor: 2ia³k/3 (1 - (ka)²/10 + (ka)⁴/280)
            ka2 = ka * ka
            return 2j * a**3 * k / 3.0 * (1.0 - ka2 / 10.0 + ka2 * ka2 / 280.0)
        # Exact: 2i[sin(ka)/k² - a cos(ka)/k] = (2i/k²)[sin(ka) - ka cos(ka)]
        return 2j / k**2 * (np.sin(ka) - ka * np.cos(ka))

    if m == 2:
        if abs_ka < 1e-5:
            # Taylor: 2a³/3 (1 - (ka)²/5 + (ka)⁴/70 - ...)
            # From expansion of integral = 2a³/3 - 2a⁵k²/15 + ...
            ka2 = ka * ka
            return 2.0 * a**3 / 3.0 * (1.0 - 3.0 * ka2 / 10.0 + ka2 * ka2 / 56.0)
        # Exact: (2/k³)[2ka cos(ka) + (k²a² - 2) sin(ka)]
        # = 2a/k² [(k²a² - 2) sin(ka)/(ka) + 2cos(ka)]
        return 2.0 / k**3 * (2.0 * ka * np.cos(ka) + (ka**2 - 2.0) * np.sin(ka))

    if m == 3:
        if abs_ka < 1e-2:
            # Taylor: ∫ x³ e^{ikx} dx = 2ia⁵k/5 · [1 - 5(ka)²/42 + (ka)⁴/252 - ...]
            # From ∫₀ᵃ x³ sin(kx) dx with sin series: term n = (-1)ⁿ k^{2n+1} a^{2n+5}/((2n+1)!(2n+5))
            ka2 = ka * ka
            return 2j * a**5 * k / 5.0 * (1.0 - 5.0 * ka2 / 42.0 + ka2 * ka2 / 252.0)
        # Exact: (2i/k⁴)[(3k²a² - 6)sin(ka) - (k³a³ - 6ka)cos(ka)]
        # Derived by integration by parts:
        #   ∫ x³ e^{ikx} dx = [x³/(ik)]e^{ikx} - (3/ik)∫ x² e^{ikx} dx
        # After evaluating at ±a and simplifying:
        return (
            2j
            / k**4
            * ((3.0 * ka**2 - 6.0) * np.sin(ka) - (ka**3 - 6.0 * ka) * np.cos(ka))
        )

    msg = f"Monomial power m={m} not supported (only 0, 1, 2, 3)"
    raise ValueError(msg)


# ================================================================
# Full cube overlap integrals
# ================================================================

# Monomial exponents for quadratic monomials (0-indexed):
# 0: r1², 1: r2², 2: r3², 3: r2r3, 4: r1r3, 5: r1r2
_QUAD_EXPONENTS = [
    (2, 0, 0),  # r1²
    (0, 2, 0),  # r2²
    (0, 0, 2),  # r3²
    (0, 1, 1),  # r2 r3
    (1, 0, 1),  # r1 r3
    (1, 1, 0),  # r1 r2
]


# Monomial exponents for cubic monomials (0-indexed):
# 0: r1³, 1: r2³, 2: r3³,
# 3: r1²r2, 4: r1²r3, 5: r2²r1, 6: r2²r3, 7: r3²r1, 8: r3²r2,
# 9: r1r2r3
_CUBIC_EXPONENTS = [
    (3, 0, 0),  # r1³
    (0, 3, 0),  # r2³
    (0, 0, 3),  # r3³
    (2, 1, 0),  # r1²r2
    (2, 0, 1),  # r1²r3
    (1, 2, 0),  # r2²r1
    (0, 2, 1),  # r2²r3
    (1, 0, 2),  # r3²r1
    (0, 1, 2),  # r3²r2
    (1, 1, 1),  # r1r2r3
]


def cube_overlap_integrals(k_vec: np.ndarray, pol: np.ndarray, a: float) -> np.ndarray:
    """Compute the 27-component projection of a plane wave onto the T27 basis.

    For u_inc(r) = pol * exp(ik·r), computes:
      c_α = ∫_{cube} φ_α(r) · u_inc(r) d³r

    Parameters
    ----------
    k_vec : Wave vector (3,).
    pol : Polarization vector (3,).
    a : Cube half-width.

    Returns
    -------
    c : ndarray, shape (27,), complex — projection coefficients.
    """
    k1, k2, k3 = k_vec

    # Precompute all needed 1D integrals: I_m(k_j, a) for m=0,1,2 and j=1,2,3
    I = np.zeros((3, 3), dtype=complex)  # I[m, axis]
    for axis, kj in enumerate([k1, k2, k3]):
        for m in range(3):
            I[m, axis] = _monomial_fourier_1d(m, kj, a)

    # Shorthand: product of three 1D integrals
    def prod3(m1, m2, m3):
        return I[m1, 0] * I[m2, 1] * I[m3, 2]

    c = np.zeros(27, dtype=complex)

    # ── Constant displacement (0-2): φ = e_k, integral = pol_k * I000
    I000 = prod3(0, 0, 0)
    for k in range(3):
        c[k] = pol[k] * I000

    # ── Axial strain (3-5): φ_k = r_k e_k
    # integral = pol_k * I[1,k] * product of I[0] for other axes
    for k in range(3):
        axes = [0, 0, 0]
        axes[k] = 1
        c[3 + k] = pol[k] * prod3(*axes)

    # ── Shear strain (6-8):
    # 6: (r_3 e_2 + r_2 e_3)/2 → (pol_2 * I[0,0]*I[0,1]*I[1,2] + pol_3 * I[0,0]*I[1,1]*I[0,2])/2
    # 7: (r_3 e_1 + r_1 e_3)/2 → (pol_1 * I[0,0]*I[0,1]*I[1,2] + pol_3 * I[1,0]*I[0,1]*I[0,2])/2
    # 8: (r_2 e_1 + r_1 e_2)/2 → (pol_1 * I[0,0]*I[1,1]*I[0,2] + pol_2 * I[1,0]*I[0,1]*I[0,2])/2

    # Pair indices for shear: 6→{1,2}, 7→{0,2}, 8→{0,1}
    shear_pairs = [(1, 2), (0, 2), (0, 1)]
    for s, (p, q) in enumerate(shear_pairs):
        # Term 1: pol_q * (r_p-monomial)
        m1 = [0, 0, 0]
        m1[p] = 1
        # Term 2: pol_p * (r_q-monomial)
        m2 = [0, 0, 0]
        m2[q] = 1
        c[6 + s] = 0.5 * (pol[q] * prod3(*m1) + pol[p] * prod3(*m2))

    # ── Quadratic block (9-26): q_m * e_k
    # 6 monomials × 3 directions
    for direction in range(3):
        for mono_idx, (e1, e2, e3) in enumerate(_QUAD_EXPONENTS):
            idx = 9 + direction * 6 + mono_idx
            c[idx] = pol[direction] * prod3(e1, e2, e3)

    return c


def cube_overlap_integrals_57(
    k_vec: np.ndarray, pol: np.ndarray, a: float
) -> np.ndarray:
    """Compute the 57-component projection of a plane wave onto the T57 basis.

    Extends cube_overlap_integrals() with 30 cubic modes (indices 27-56):
      c_m(r) · e_k for 10 cubic monomials × 3 directions.

    Parameters
    ----------
    k_vec : Wave vector (3,).
    pol : Polarization vector (3,).
    a : Cube half-width.

    Returns
    -------
    c : ndarray, shape (57,), complex — projection coefficients.
    """
    k1, k2, k3 = k_vec

    # Precompute all needed 1D integrals: I_m(k_j, a) for m=0,1,2,3 and j=1,2,3
    I = np.zeros((4, 3), dtype=complex)  # I[m, axis]
    for axis, kj in enumerate([k1, k2, k3]):
        for m in range(4):
            I[m, axis] = _monomial_fourier_1d(m, kj, a)

    def prod3(m1, m2, m3):
        return I[m1, 0] * I[m2, 1] * I[m3, 2]

    c = np.zeros(57, dtype=complex)

    # ── First 27 components: identical to T27 ──
    # Constant displacement (0-2)
    I000 = prod3(0, 0, 0)
    for k in range(3):
        c[k] = pol[k] * I000

    # Axial strain (3-5)
    for k in range(3):
        axes = [0, 0, 0]
        axes[k] = 1
        c[3 + k] = pol[k] * prod3(*axes)

    # Shear strain (6-8)
    shear_pairs = [(1, 2), (0, 2), (0, 1)]
    for s, (p, q) in enumerate(shear_pairs):
        m1 = [0, 0, 0]
        m1[p] = 1
        m2 = [0, 0, 0]
        m2[q] = 1
        c[6 + s] = 0.5 * (pol[q] * prod3(*m1) + pol[p] * prod3(*m2))

    # Quadratic block (9-26): 6 monomials × 3 directions
    for direction in range(3):
        for mono_idx, (e1, e2, e3) in enumerate(_QUAD_EXPONENTS):
            idx = 9 + direction * 6 + mono_idx
            c[idx] = pol[direction] * prod3(e1, e2, e3)

    # ── Cubic block (27-56): 10 monomials × 3 directions ──
    for direction in range(3):
        for mono_idx, (e1, e2, e3) in enumerate(_CUBIC_EXPONENTS):
            idx = 27 + direction * 10 + mono_idx
            c[idx] = pol[direction] * prod3(e1, e2, e3)

    return c


# ================================================================
# Plane-wave generation
# ================================================================


def plane_wave_PSV_SH(
    k_hat: np.ndarray, omega: float, ref: ReferenceMedium
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate P, SV, SH incident plane waves for a given propagation direction.

    Parameters
    ----------
    k_hat : Unit propagation direction (3,).
    omega : Angular frequency.
    ref : Background medium.

    Returns
    -------
    waves : List of (k_vec, pol, label) tuples.
        k_vec is the full wave vector, pol is the polarization unit vector.
    """
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)

    kP = omega / ref.alpha
    kS = omega / ref.beta

    waves = []

    # P-wave: pol parallel to k_hat
    waves.append((kP * k_hat, k_hat.copy(), "P"))

    # SV and SH: need two vectors perpendicular to k_hat
    # Use a reference vector to define the scattering plane
    if abs(k_hat[0]) < 0.9:
        ref_vec = np.array([1.0, 0.0, 0.0])
    else:
        ref_vec = np.array([0.0, 1.0, 0.0])

    # SV: in scattering plane (k_hat, ref_vec), perpendicular to k_hat
    sv = ref_vec - np.dot(ref_vec, k_hat) * k_hat
    sv = sv / np.linalg.norm(sv)
    waves.append((kS * k_hat, sv, "SV"))

    # SH: perpendicular to both k_hat and SV
    sh = np.cross(k_hat, sv)
    sh = sh / np.linalg.norm(sh)
    waves.append((kS * k_hat, sh, "SH"))

    return waves
