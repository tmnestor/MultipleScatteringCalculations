"""
compute_gerade_blocks.py
Compute per-irrep mass and body bilinear values for T₅₇ gerade blocks.

Strategy:
1. Build 57×57 mass matrix (exact polynomial integration)
2. Build 36×36 gerade body bilinear (A and B channels)
   - A-channel: from K1at (analytical from Mp masters)
   - B-channel: from K3 (analytical from MpB masters)
3. Project through Usym₅₇ to per-irrep blocks
4. Extract per-irrep blocks and print Python-ready values

Run:  conda run -n seismic python -m cubic_scattering.compute_gerade_blocks
"""

from __future__ import annotations

import time
from math import comb, factorial

import numpy as np
from scipy import integrate

from cubic_scattering.tmatrix_assembly import _build_usym_57

# ================================================================
# Basis definition (matching tmatrix_assembly.py ordering)
# ================================================================

_QUAD_EXP = [
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

_CUBIC_EXP = [
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (0, 2, 1),
    (1, 0, 2),
    (0, 1, 2),
    (1, 1, 1),
]

# Gerade indices in the 57-element basis
GERADE_INDICES = list(range(3, 9)) + list(range(27, 57))  # 6 strain + 30 cubic = 36


def _build_basis_components():
    """Return list of 57 basis functions, each as [(comp, exp, coeff), ...]."""
    basis = []
    # 0-2: constant displacement e_k
    for k in range(3):
        basis.append([(k, (0, 0, 0), 1.0)])
    # 3-5: axial strain r_k e_k
    for k in range(3):
        exp = [0, 0, 0]
        exp[k] = 1
        basis.append([(k, tuple(exp), 1.0)])
    # 6-8: shear strain (symmetrized)
    shear_pairs = [(1, 2), (0, 2), (0, 1)]
    for a, b in shear_pairs:
        exp_a = [0, 0, 0]
        exp_a[a] = 1
        exp_b = [0, 0, 0]
        exp_b[b] = 1
        basis.append([(b, tuple(exp_a), 0.5), (a, tuple(exp_b), 0.5)])
    # 9-26: quadratic (6 mono × 3 dir)
    for d in range(3):
        for exp in _QUAD_EXP:
            basis.append([(d, exp, 1.0)])
    # 27-56: cubic (10 mono × 3 dir)
    for d in range(3):
        for exp in _CUBIC_EXP:
            basis.append([(d, exp, 1.0)])
    assert len(basis) == 57
    return basis


# ================================================================
# Mass matrix (exact polynomial integration on [-1,1]³)
# ================================================================


def _int_mono(n: int) -> float:
    """∫_{-1}^{1} x^n dx = 2/(n+1) if n even, 0 if odd."""
    return 0.0 if n % 2 == 1 else 2.0 / (n + 1)


def _int_cube_mono(e1: int, e2: int, e3: int) -> float:
    """∫_{[-1,1]³} r1^e1 r2^e2 r3^e3 dV."""
    return _int_mono(e1) * _int_mono(e2) * _int_mono(e3)


def compute_mass_matrix(basis, indices: list[int]) -> np.ndarray:
    """Compute mass sub-matrix M[i,j] = ∫ φ_i · φ_j dV for given indices."""
    n = len(indices)
    M = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii, n):
            val = 0.0
            for comp_i, exp_i, c_i in basis[indices[ii]]:
                for comp_j, exp_j, c_j in basis[indices[jj]]:
                    if comp_i != comp_j:
                        continue
                    esum = (
                        exp_i[0] + exp_j[0],
                        exp_i[1] + exp_j[1],
                        exp_i[2] + exp_j[2],
                    )
                    val += c_i * c_j * _int_cube_mono(*esum)
            M[ii, jj] = val
            M[jj, ii] = val
    return M


# ================================================================
# Mp master integrals (1/ρ kernel, body channel A)
# ================================================================

_MP_VALUES: dict[tuple[int, int, int], float] = {
    # Mp[p,q,r] = ∫_{[0,1]³} x^p y^q z^r / √(x²+y²+z²) dxdydz
    # All values verified by scipy NIntegrate (14-digit precision)
    (0, 0, 0): 1.19003868198977658e00,
    (0, 0, 1): 5.15593558809205987e-01,
    (0, 0, 2): 3.20197318818350951e-01,
    (0, 1, 1): 2.33296903209784939e-01,
    (0, 0, 3): 2.30382186007204504e-01,
    (0, 1, 2): 1.47410679312947518e-01,
    (1, 1, 1): 1.07859634642850338e-01,
    (0, 0, 4): 1.79393565881997064e-01,
    (0, 1, 3): 1.06988102001234972e-01,
    (0, 2, 2): 9.38691686242359574e-02,
    (1, 1, 2): 6.88368569699362165e-02,
    (0, 0, 5): 1.46696864954606515e-01,
    (0, 1, 4): 8.37322945328095036e-02,
    (0, 2, 3): 6.84131188125464557e-02,
    (1, 1, 3): 5.02357025631917711e-02,
    (1, 2, 2): 4.41485631257232877e-02,
    (0, 0, 6): 1.24002234752386586e-01,
    (0, 1, 5): 6.86946607168422568e-02,
    (0, 2, 4): 5.36775866759084466e-02,
    (0, 3, 3): 4.99764397776433153e-02,
    (1, 1, 4): 3.94492940163620742e-02,
    (1, 2, 3): 3.23087903128156662e-02,
    (2, 2, 2): 2.84193216744485916e-02,
}

_mp_cache: dict[tuple[int, int, int], float] = dict(_MP_VALUES)


def _mp(p: int, q: int, r: int) -> float:
    """Mp[p,q,r] = ∫_{[0,1]³} x^p y^q z^r / √(x²+y²+z²) dxdydz, canonicalized."""
    s = sorted([p, q, r])
    key: tuple[int, int, int] = (s[0], s[1], s[2])
    if key in _mp_cache:
        return _mp_cache[key]
    # NIntegrate fallback
    p0, q0, r0 = key

    def integrand(z, y, x):
        return x**p0 * y**q0 * z**r0 / np.sqrt(x**2 + y**2 + z**2)

    val, err = integrate.tplquad(
        integrand,
        0,
        1,
        0,
        1,
        0,
        1,
        epsabs=1e-14,
        epsrel=1e-14,
    )
    _mp_cache[key] = val
    print(f"    Mp{key} = {val:.15e}  (NIntegrate, err={err:.1e})")
    return val


def _k1at(p: int, q: int, r: int) -> float:
    """K1at[p,q,r] = Σ_{a,b,c∈{0,1}} (-1)^{a+b+c} Mp[p+a, q+b, r+c]."""
    val = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                val += (-1) ** (a + b + c) * _mp(p + a, q + b, r + c)
    return val


# ================================================================
# MpB master integrals (1/ρ³ kernel, body channel B)
# ================================================================

_MPB_VALUES: dict[tuple[int, int, int], float] = {
    # Degree 2-10 from CubeT57MpBValues.wl
    (0, 0, 2): 0.39667956066325892,
    (0, 1, 1): 0.27108279714579107,
    (0, 0, 3): 0.23782799629186467,
    (0, 1, 2): 0.13888278125866329,
    (1, 1, 1): 0.09637631717731280,
    (0, 0, 4): 0.16723283512853512,
    (0, 1, 3): 0.08988302886904195,
    (0, 2, 2): 0.07648224184490793,
    (1, 1, 2): 0.05353084547170106,
    (0, 0, 5): 0.12816962270230052,
    (0, 1, 4): 0.06552126038831467,
    (0, 2, 3): 0.05110628165245203,
    (1, 1, 3): 0.03595321154761678,
    (1, 2, 2): 0.03078313727218084,
    (0, 0, 6): 0.10360870808346750,
    (0, 1, 5): 0.05124163045441187,
    (0, 2, 4): 0.03789242889926478,
    (0, 3, 3): 0.03470029964771736,
    (1, 1, 4): 0.02674451317172476,
    (1, 2, 3): 0.02104617189910573,
    (2, 2, 2): 0.01808431082571208,
    (0, 0, 7): 0.08682206840112112,
    (0, 1, 6): 0.04195069511351963,
    (0, 2, 5): 0.02993739827674269,
    (0, 3, 4): 0.02596345208288675,
    (1, 1, 5): 0.02117675251468316,
    (1, 2, 4): 0.01581814733640314,
    (1, 3, 3): 0.01452947502425430,
    (2, 2, 3): 0.01251226845291700,
    (0, 0, 8): 0.07465652194018819,
    (0, 1, 7): 0.03545718675052557,
    (0, 2, 6): 0.02467285640609919,
    (0, 3, 5): 0.02062985475582070,
    (0, 4, 4): 0.01953162304499306,
    (1, 1, 6): 0.01748041234372651,
    (1, 2, 5): 0.01260761921049598,
    (1, 3, 4): 0.01098444083631779,
    (2, 2, 4): 0.00947310722481620,
    (2, 3, 3): 0.00871673026600191,
    (0, 0, 9): 0.06544988299025351,
    (0, 1, 8): 0.03067740392139402,
    (0, 2, 7): 0.02095062127467145,
    (0, 3, 6): 0.01706724493140461,
    (0, 4, 5): 0.01557338244619345,
    (1, 1, 7): 0.01486070276243374,
    (1, 2, 6): 0.01045366278604737,
    (1, 3, 5): 0.00878854532715960,
    (1, 4, 4): 0.00833457569968207,
    (2, 2, 5): 0.00758694598715418,
    (2, 3, 4): 0.00662800247963338,
    (3, 3, 3): 0.00610552820300301,
    (0, 0, 10): 0.05824710501223141,
    (0, 1, 9): 0.02701870822319017,
    (0, 2, 8): 0.01818790451270370,
    (0, 3, 7): 0.01453186717850457,
    (0, 4, 6): 0.01291475657634694,
    (0, 5, 5): 0.01244571434430761,
    (1, 1, 8): 0.01291269107249291,
    (1, 2, 7): 0.00891561032002002,
    (1, 3, 6): 0.00730642223593760,
    (1, 4, 5): 0.00668453467404709,
    (2, 2, 6): 0.00631205055182747,
    (2, 3, 5): 0.00532371373273068,
    (2, 4, 4): 0.00505366555912197,
    (3, 3, 4): 0.00465871355762868,
}

_mpb_cache: dict[tuple[int, int, int], float] = dict(_MPB_VALUES)


def _mpb(p: int, q: int, r: int) -> float:
    """MpB[p,q,r] = ∫_{[0,1]³} x^p y^q z^r / (x²+y²+z²)^{3/2} dxdydz, canonicalized."""
    s = sorted([p, q, r])
    key: tuple[int, int, int] = (s[0], s[1], s[2])
    if key in _mpb_cache:
        return _mpb_cache[key]
    p0, q0, r0 = key

    # NIntegrate fallback (singularity at origin is integrable for p+q+r ≥ 2)
    def integrand(z, y, x):
        rr2 = x**2 + y**2 + z**2
        if rr2 < 1e-60:
            return 0.0
        return x**p0 * y**q0 * z**r0 / rr2**1.5

    val, err = integrate.tplquad(
        integrand,
        0,
        1,
        0,
        1,
        0,
        1,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    _mpb_cache[key] = val
    print(f"    MpB{key} = {val:.15e}  (NIntegrate, err={err:.1e})")
    return val


# ================================================================
# K3 kernels via MpB (NOT direct NIntegrate!)
# ================================================================
#
# K3diag[p,q,r] = ∫_{[0,1]³} (1-u1)(1-u2)(1-u3) u1^{p+2} u2^q u3^r / |u|³ du
#              = Σ_{a,b,c∈{0,1}} (-1)^{a+b+c} MpB[p+2+a, q+b, r+c]
#
# K3off[p,q,r] = ∫_{[0,1]³} (1-u1)(1-u2)(1-u3) u1^{p+1} u2^{q+1} u3^r / |u|³ du
#             = Σ_{a,b,c∈{0,1}} (-1)^{a+b+c} MpB[p+1+a, q+1+b, r+c]

_k3diag_cache: dict[tuple[int, int, int], float] = {}
_k3off_cache: dict[tuple[int, int, int], float] = {}


def _k3diag(p: int, q: int, r: int) -> float:
    """K3diag via tent-expanded MpB."""
    key = (p, q, r)
    if key in _k3diag_cache:
        return _k3diag_cache[key]
    val = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                val += (-1) ** (a + b + c) * _mpb(p + 2 + a, q + b, r + c)
    _k3diag_cache[key] = val
    return val


def _k3off(p: int, q: int, r: int) -> float:
    """K3off via tent-expanded MpB."""
    key = (p, q, r)
    if key in _k3off_cache:
        return _k3off_cache[key]
    val = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                val += (-1) ** (a + b + c) * _mpb(p + 1 + a, q + 1 + b, r + c)
    _k3off_cache[key] = val
    return val


def _k3kernel(exps: tuple[int, int, int], comp_i: int, comp_j: int) -> float:
    """K3 kernel for component pair (comp_i, comp_j) with monomial exponents.

    The B-channel Green's function has kernel u_{comp_i} u_{comp_j} / |u|³.
    Combined with tent factor and monomial u^exps, the integral factorizes.
    """
    p, q, r = exps
    if comp_i == comp_j:
        # Diagonal: u_a² / |u|³ → K3diag with extra u_a²
        if comp_i == 0:
            return _k3diag(p, q, r)
        if comp_i == 1:
            return _k3diag(q, p, r)  # swap axes 0↔1
        return _k3diag(r, q, p)  # swap axes 0↔2
    # Off-diagonal: u_a u_b / |u|³ → K3off
    a, b = sorted([comp_i, comp_j])
    if (a, b) == (0, 1):
        return _k3off(p, q, r)
    if (a, b) == (0, 2):
        return _k3off(p, r, q)  # u1 u3 → swap axes 1↔2
    return _k3off(q, r, p)  # (1,2): u2 u3 → swap axes 0↔2, then 0↔1


# ================================================================
# Body bilinear computation via (ξ,s) substitution
# ================================================================


def _expand_1d_product(a: int, b: int) -> dict[tuple[int, int], float]:
    """Expand (ξ + s/2)^a · (ξ - s/2)^b as {(xi_pow, s_pow): coeff}."""
    result: dict[tuple[int, int], float] = {}
    for i in range(a + 1):
        for j in range(b + 1):
            xi_pow = (a - i) + (b - j)
            s_pow = i + j
            coeff = comb(a, i) * comb(b, j) * (0.5) ** i * (-0.5) ** j
            key = (xi_pow, s_pow)
            result[key] = result.get(key, 0.0) + coeff
    return result


def _xi_integrate_residual(poly_1d: dict[tuple[int, int], float]) -> dict[int, float]:
    """Integrate over ξ, substitute s=2u, divide out one (1-u) factor.

    ξ-integral: ∫_{-(1-u)}^{1-u} ξ^m dξ = 2(1-u)^{m+1}/(m+1) for even m.
    After s=2u: s^n → (2u)^n.
    Divide by (1-u): (1-u)^{m+1} → (1-u)^m.
    Expand (1-u)^m into powers of u.
    """
    result: dict[int, float] = {}
    for (xi_pow, s_pow), coeff in poly_1d.items():
        if abs(coeff) < 1e-30 or xi_pow % 2 == 1:
            continue
        xi_factor = 2.0 / (xi_pow + 1)
        s_factor = 2.0**s_pow
        tent_power = xi_pow  # after dividing out one (1-u)
        for k in range(tent_power + 1):
            u_pow = s_pow + k
            c = coeff * xi_factor * s_factor * comb(tent_power, k) * (-1) ** k
            result[u_pow] = result.get(u_pow, 0.0) + c
    return result


def _compute_axis_residuals(
    exp_i: tuple[int, int, int], exp_j: tuple[int, int, int]
) -> tuple[list[dict[int, float]], list[dict[int, float]]]:
    """Compute per-axis residuals for both exponent orderings.

    Returns (normal, swapped) where:
      normal[k] = residual from (exp_i[k], exp_j[k])
      swapped[k] = residual from (exp_j[k], exp_i[k])
    """
    normal = []
    swapped = []
    for k in range(3):
        normal.append(_xi_integrate_residual(_expand_1d_product(exp_i[k], exp_j[k])))
        if exp_i[k] == exp_j[k]:
            swapped.append(normal[-1])  # symmetric → same
        else:
            swapped.append(
                _xi_integrate_residual(_expand_1d_product(exp_j[k], exp_i[k]))
            )
    return normal, swapped


def _symmetrize_axis(
    normal: dict[int, float], swapped: dict[int, float], kernel_parity: int
) -> dict[int, float]:
    """Combine normal and swapped residuals for one axis.

    For even kernel parity: Q = normal + swapped  (symmetric)
    For odd kernel parity:  Q = normal - swapped  (anti-symmetric)
    """
    sign = 1.0 if kernel_parity % 2 == 0 else -1.0
    result: dict[int, float] = {}
    all_keys = set(normal.keys()) | set(swapped.keys())
    for m in all_keys:
        val = normal.get(m, 0.0) + sign * swapped.get(m, 0.0)
        if abs(val) > 1e-30:
            result[m] = val
    return result


def _form_3d_product(
    q0: dict[int, float], q1: dict[int, float], q2: dict[int, float]
) -> dict[tuple[int, int, int], float]:
    """Form 3D residual product from per-axis symmetrized residuals."""
    result: dict[tuple[int, int, int], float] = {}
    for p1, c1 in q0.items():
        if abs(c1) < 1e-30:
            continue
        for p2, c2 in q1.items():
            if abs(c2) < 1e-30:
                continue
            for p3, c3 in q2.items():
                c = c1 * c2 * c3
                if abs(c) < 1e-30:
                    continue
                key = (p1, p2, p3)
                result[key] = result.get(key, 0.0) + c
    return result


def _body_bilinear_entry(basis, i: int, j: int) -> tuple[float, float]:
    """Compute body bilinear entry (A_part, B_part) for basis functions i, j.

    Uses octant-symmetrized residuals to correctly handle asymmetric exponents.
    For symmetric exponents (exp_i[k] == exp_j[k]), Q_k = 2*P_k and the
    prefactor 4 × 2³ = 32 matches the standard formula.
    For asymmetric exponents, the symmetrization ensures correct cancellation.

    Total entry = 4 * (A_elas * A_part + B_elas * B_part)
    """
    a_sum = 0.0
    b_sum = 0.0

    for comp_i, exp_i, c_i in basis[i]:
        for comp_j, exp_j, c_j in basis[j]:
            p, q = comp_i, comp_j

            normal, swapped = _compute_axis_residuals(exp_i, exp_j)

            # A-channel: diagonal kernel only (p == q), isotropic 1/|u|
            if p == q:
                # Kernel parity (0,0,0) → all axes symmetrized
                q0 = _symmetrize_axis(normal[0], swapped[0], 0)
                q1 = _symmetrize_axis(normal[1], swapped[1], 0)
                q2 = _symmetrize_axis(normal[2], swapped[2], 0)
                residual_A = _form_3d_product(q0, q1, q2)
                for exps, coeff in residual_A.items():
                    if abs(coeff) < 1e-30:
                        continue
                    a_sum += c_i * c_j * coeff * _k1at(*exps)

            # B-channel: u_p * u_q / |u|³ kernel
            # Kernel parity: odd in axes p and q (mod 2)
            kp_B = [0, 0, 0]
            kp_B[p] = (kp_B[p] + 1) % 2
            kp_B[q] = (kp_B[q] + 1) % 2
            q0 = _symmetrize_axis(normal[0], swapped[0], kp_B[0])
            q1 = _symmetrize_axis(normal[1], swapped[1], kp_B[1])
            q2 = _symmetrize_axis(normal[2], swapped[2], kp_B[2])
            residual_B = _form_3d_product(q0, q1, q2)
            for exps, coeff in residual_B.items():
                if abs(coeff) < 1e-30:
                    continue
                b_sum += c_i * c_j * coeff * _k3kernel(exps, p, q)

    return a_sum, b_sum


def compute_body_bilinear(basis, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute body bilinear matrices (A and B channels) for given indices."""
    n = len(indices)
    BA = np.zeros((n, n))
    BB = np.zeros((n, n))

    total = n * (n + 1) // 2
    count = 0
    for ii in range(n):
        for jj in range(ii, n):
            count += 1
            if count % 50 == 0 or count == total:
                print(f"  body bilinear: {count}/{total} entries...", flush=True)
            a_part, b_part = _body_bilinear_entry(basis, indices[ii], indices[jj])
            BA[ii, jj] = 4.0 * a_part
            BA[jj, ii] = BA[ii, jj]
            BB[ii, jj] = 4.0 * b_part
            BB[jj, ii] = BB[ii, jj]

    return BA, BB


# ================================================================
# Polynomial master integrals for smooth (dynamic) body bilinear
# ================================================================
#
# The smooth Green's tensor G^s has kernel |u|^{2n} (polynomial)
# instead of 1/|u| (singular). The master integrals are exact
# rationals via the trinomial expansion of (x²+y²+z²)^n.


def _mp_poly(p: int, q: int, r: int, n: int) -> float:
    """MpPoly = ∫_{[0,1]³} x^p y^q z^r (x²+y²+z²)^n dx dy dz.

    Exact rational via trinomial expansion. O(n²) terms.
    """
    total = 0.0
    for aa in range(n + 1):
        for bb in range(n - aa + 1):
            cc = n - aa - bb
            coeff = factorial(n) / (factorial(aa) * factorial(bb) * factorial(cc))
            total += coeff / ((p + 2 * aa + 1) * (q + 2 * bb + 1) * (r + 2 * cc + 1))
    return total


def _k1at_poly(p: int, q: int, r: int, n: int) -> float:
    """Tent-convolved polynomial: Σ (-1)^{a+b+c} MpPoly(p+a,q+b,r+c,n)."""
    val = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                val += (-1) ** (a + b + c) * _mp_poly(p + a, q + b, r + c, n)
    return val


def _k3kernel_poly(
    exps: tuple[int, int, int], comp_i: int, comp_j: int, n: int
) -> float:
    """Smooth B-channel: K1atPoly with shifted exponents for u_iu_j factor.

    The B-channel smooth kernel is u_i u_j |u|^{2n}. This shifts the
    monomial exponents by +2 in one axis (diagonal) or +1 in two axes
    (off-diagonal), then uses the polynomial K1at.
    """
    p, q, r = exps
    if comp_i == comp_j:
        # Diagonal: u_a² |u|^{2n} → shift exponent by +2 in axis comp_i
        if comp_i == 0:
            return _k1at_poly(p + 2, q, r, n)
        if comp_i == 1:
            return _k1at_poly(p, q + 2, r, n)
        return _k1at_poly(p, q, r + 2, n)
    # Off-diagonal: u_a u_b |u|^{2n} → shift by +1 in both axes
    shifts = [0, 0, 0]
    shifts[comp_i] += 1
    shifts[comp_j] += 1
    return _k1at_poly(p + shifts[0], q + shifts[1], r + shifts[2], n)


def _body_bilinear_entry_smooth(basis, i: int, j: int, n: int) -> tuple[float, float]:
    """Compute smooth body bilinear entry at Taylor order n.

    Same loop structure as _body_bilinear_entry but with polynomial
    |u|^{2n} kernel instead of 1/|u| (A-channel) and u_iu_j|u|^{2n}
    instead of u_iu_j/|u|³ (B-channel).

    Returns (a_part, b_part) for A and B channels.
    """
    a_sum = 0.0
    b_sum = 0.0

    for comp_i, exp_i, c_i in basis[i]:
        for comp_j, exp_j, c_j in basis[j]:
            p, q = comp_i, comp_j

            normal, swapped = _compute_axis_residuals(exp_i, exp_j)

            # A-channel: diagonal kernel only (p == q)
            if p == q:
                q0 = _symmetrize_axis(normal[0], swapped[0], 0)
                q1 = _symmetrize_axis(normal[1], swapped[1], 0)
                q2 = _symmetrize_axis(normal[2], swapped[2], 0)
                residual_A = _form_3d_product(q0, q1, q2)
                for exps, coeff in residual_A.items():
                    if abs(coeff) < 1e-30:
                        continue
                    a_sum += c_i * c_j * coeff * _k1at_poly(*exps, n)

            # B-channel: u_p * u_q * |u|^{2n} kernel
            kp_B = [0, 0, 0]
            kp_B[p] = (kp_B[p] + 1) % 2
            kp_B[q] = (kp_B[q] + 1) % 2
            q0 = _symmetrize_axis(normal[0], swapped[0], kp_B[0])
            q1 = _symmetrize_axis(normal[1], swapped[1], kp_B[1])
            q2 = _symmetrize_axis(normal[2], swapped[2], kp_B[2])
            residual_B = _form_3d_product(q0, q1, q2)
            for exps, coeff in residual_B.items():
                if abs(coeff) < 1e-30:
                    continue
                b_sum += c_i * c_j * coeff * _k3kernel_poly(exps, p, q, n)

    return a_sum, b_sum


def compute_smooth_body_bilinear(
    basis, indices: list[int], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 36×36 smooth body bilinear at Taylor order n.

    Returns (BA_n, BB_n) with prefactors 8·4^n and 32·4^n built in.

    The smooth A-channel kernel replaces a₀/(2|u|) with φ_n·a^{2n}·4^n·|u|^{2n},
    giving prefactor 8·4^n (vs 4·a₀ for static).
    The smooth B-channel kernel replaces b₀·u_iu_j/(2|u|³) with
    ψ_n·a^{2n+2}·4·4^n·u_iu_j|u|^{2n}, giving prefactor 32·4^n.
    """
    m = len(indices)
    BA_n = np.zeros((m, m))
    BB_n = np.zeros((m, m))

    prefactor_A = 8.0 * 4.0**n
    prefactor_B = 32.0 * 4.0**n

    for ii in range(m):
        for jj in range(ii, m):
            a_part, b_part = _body_bilinear_entry_smooth(
                basis, indices[ii], indices[jj], n
            )
            BA_n[ii, jj] = prefactor_A * a_part
            BA_n[jj, ii] = BA_n[ii, jj]
            BB_n[ii, jj] = prefactor_B * b_part
            BB_n[jj, ii] = BB_n[ii, jj]

    return BA_n, BB_n


# ================================================================
# Per-irrep block extraction from projected gerade matrix
# ================================================================


def _extract_irrep_block(M_proj_gerade: np.ndarray, irrep: str) -> np.ndarray:
    """Extract per-irrep block from the 36×36 projected gerade matrix.

    Gerade column ranges within Usym (starting from col 21):
      A1g: 0-2   (d=1, 3 cols)
      A2g: 3     (d=1, 1 col)
      Eg:  4-11  (d=2, 8 cols → 4 seeds)
      T1g: 12-20 (d=3, 9 cols → 3 seeds)
      T2g: 21-35 (d=3, 15 cols → 5 seeds)

    For d=1: block = direct sub-matrix
    For d=2: block at stride-2 positions (first copy of each seed)
    For d=3: block at stride-3 positions (first copy of each seed)
    """
    if irrep == "A1g":
        idx = [0, 1, 2]
    elif irrep == "A2g":
        idx = [3]
    elif irrep == "Eg":
        # 4 seeds × 2 copies, take first copy: positions 4,6,8,10
        idx = [4, 6, 8, 10]
    elif irrep == "T1g":
        # 3 seeds × 3 copies, take first copy: positions 12,15,18
        idx = [12, 15, 18]
    elif irrep == "T2g":
        # 5 seeds × 3 copies, take first copy: positions 21,24,27,30,33
        idx = [21, 24, 27, 30, 33]
    else:
        msg = f"Unknown gerade irrep: {irrep}"
        raise ValueError(msg)

    return M_proj_gerade[np.ix_(idx, idx)]


# ================================================================
# Stiffness computation (volume part only)
# ================================================================


def _compute_stiffness_volume(
    basis, gerade_indices: list[int], BA_ger: np.ndarray, BB_ger: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute volume stiffness: Bstiff_vol[α,β] from div(Δc:ε(φ_β)) convolved with G.

    For φ_β with degree-d monomial, ε has degree d-1, σ has degree d-1,
    div(σ) has degree d-2. The body force is then convolved with G using
    the body bilinear form.

    The result decomposes into 4 channels: A·Δλ, A·Δμ, B·Δλ, B·Δμ.

    For strain modes (degree 1): div(σ) = 0 (constant strain has zero body force).
    For cubic modes (degree 3): div(σ) = degree-1 polynomial (linear).
    """
    n = len(gerade_indices)
    idx_map = {g: ii for ii, g in enumerate(gerade_indices)}

    S_Alam = np.zeros((n, n))
    S_Amu = np.zeros((n, n))
    S_Blam = np.zeros((n, n))
    S_Bmu = np.zeros((n, n))

    for jj, beta in enumerate(gerade_indices):
        # Compute strain of basis function β
        # For each term (comp, exp, coeff), the strain ε_{ab} = (∂u_a/∂x_b + ∂u_b/∂x_a)/2
        # The stress σ_{ij} = Δλ tr(ε) δ_{ij} + 2Δμ ε_{ij}
        # The body force f_i = ∂σ_{ij}/∂x_j

        # Collect derivatives to build the body force
        # force[direction] = list of (exponents, dlam_coeff, dmu_coeff)
        force: dict[int, dict[tuple[int, ...], list[float]]] = {0: {}, 1: {}, 2: {}}

        # First compute strain components
        strain: dict[tuple[int, int], list[tuple[tuple[int, ...], float]]] = {}
        for comp, exp, coeff in basis[beta]:
            for j in range(3):
                if exp[j] == 0:
                    continue
                new_exp = list(exp)
                new_exp[j] -= 1
                deriv_coeff = coeff * exp[j]
                # Contributes to ε[comp,j] and ε[j,comp] with factor 1/2
                for pair in [(comp, j), (j, comp)]:
                    if pair not in strain:
                        strain[pair] = []
                    strain[pair].append((tuple(new_exp), deriv_coeff / 2.0))

        # Compute trace of strain
        trace_poly: dict[tuple[int, ...], float] = {}
        for k in range(3):
            if (k, k) in strain:
                for exp, c in strain[(k, k)]:
                    trace_poly[exp] = trace_poly.get(exp, 0.0) + c

        # Build body force f_i = Δλ ∂tr(ε)/∂x_i + 2Δμ ∂ε_{ij}/∂x_j
        for i in range(3):
            # Δλ part: ∂tr(ε)/∂x_i
            for exp, c in trace_poly.items():
                if exp[i] == 0:
                    continue
                new_exp = list(exp)
                new_exp[i] -= 1
                new_exp_t = tuple(new_exp)
                dc = c * exp[i]
                if new_exp_t not in force[i]:
                    force[i][new_exp_t] = [0.0, 0.0]
                force[i][new_exp_t][0] += dc

            # 2Δμ part: Σ_j ∂ε_{ij}/∂x_j
            for j in range(3):
                if (i, j) not in strain:
                    continue
                for exp, c in strain[(i, j)]:
                    if exp[j] == 0:
                        continue
                    new_exp = list(exp)
                    new_exp[j] -= 1
                    new_exp_t = tuple(new_exp)
                    dc = 2.0 * c * exp[j]
                    if new_exp_t not in force[i]:
                        force[i][new_exp_t] = [0.0, 0.0]
                    force[i][new_exp_t][1] += dc

        # Now express the force as a linear combination of basis functions.
        # The force components are low-degree polynomials. We need to find
        # which basis function matches each (direction, monomial).
        for direction in range(3):
            for exp_t, (dlam_c, dmu_c) in force[direction].items():
                if abs(dlam_c) < 1e-30 and abs(dmu_c) < 1e-30:
                    continue
                # Find matching gerade basis function
                for _kk, gamma in enumerate(gerade_indices):
                    matched = False
                    for comp, bexp, bcoeff in basis[gamma]:
                        if comp == direction and bexp == exp_t and abs(bcoeff) > 1e-15:
                            gg = idx_map[gamma]
                            scale = 1.0 / bcoeff
                            for ii2 in range(n):
                                S_Alam[ii2, jj] += dlam_c * scale * BA_ger[ii2, gg]
                                S_Amu[ii2, jj] += dmu_c * scale * BA_ger[ii2, gg]
                                S_Blam[ii2, jj] += dlam_c * scale * BB_ger[ii2, gg]
                                S_Bmu[ii2, jj] += dmu_c * scale * BB_ger[ii2, gg]
                            matched = True
                            break
                    if matched:
                        break

    return S_Alam, S_Amu, S_Blam, S_Bmu


# ================================================================
# Surface stiffness helpers
# ================================================================

_k1at_surf_cache: dict[tuple[int, int, int], float] = {}


def _k1at_surf(p: int, q: int, r: int) -> float:
    """Partial-tent K1at: tent in first 2 axes only (for surface integral)."""
    key = (p, q, r)
    if key in _k1at_surf_cache:
        return _k1at_surf_cache[key]
    val = 0.0
    for a in range(2):
        for b in range(2):
            val += (-1) ** (a + b) * _mp(p + a, q + b, r)
    _k1at_surf_cache[key] = val
    return val


_k3_surf_cache: dict[tuple[int, int, int], float] = {}


def _k3_surf(p: int, q: int, r: int) -> float:
    """Partial-tent MpB kernel: tent in first 2 axes only (for surface integral)."""
    key = (p, q, r)
    if key in _k3_surf_cache:
        return _k3_surf_cache[key]
    val = 0.0
    for a in range(2):
        for b in range(2):
            val += (-1) ** (a + b) * _mpb(p + a, q + b, r)
    _k3_surf_cache[key] = val
    return val


def _normal_axis_poly(exp_n: int, face_sign: int) -> dict[int, float]:
    """Polynomial in v from r_n = face_sign*(1-2v).

    Returns {v_power: coefficient}.
    """
    result: dict[int, float] = {}
    fs_power = float(face_sign**exp_n)
    for m in range(exp_n + 1):
        coeff = fs_power * comb(exp_n, m) * float((-2) ** m)
        if abs(coeff) > 1e-30:
            result[m] = result.get(m, 0.0) + coeff
    return result


def _compute_face_traction(
    basis_terms: list, face_axis: int, face_sign: int
) -> list[tuple[int, tuple[int, int], float, float]]:
    """Compute traction of a basis function on one cube face.

    Traction: t_i = face_sign * [Dlam * tr(eps) * delta_{i,face_axis} + 2*Dmu * eps_{i,face_axis}]
    evaluated at r_{face_axis} = face_sign.

    Returns list of (trac_direction, free_exp_2d, dlam_coeff, dmu_coeff).
    """
    free_axes = [k for k in range(3) if k != face_axis]
    f1, f2 = free_axes

    result: dict[tuple[int, tuple[int, int]], list[float]] = {}

    for comp, exp, c in basis_terms:
        for j in range(3):
            if exp[j] == 0:
                continue
            new_exp = list(exp)
            new_exp[j] -= 1

            # Evaluate monomial on face: r_{face_axis} = face_sign
            face_factor = float(face_sign ** new_exp[face_axis])
            free_exp = (new_exp[f1], new_exp[f2])
            deriv = c * exp[j]

            # Dlam: trace contribution (only diagonal strain j == comp)
            if j == comp:
                key = (face_axis, free_exp)
                if key not in result:
                    result[key] = [0.0, 0.0]
                result[key][0] += face_sign * deriv * face_factor

            # Dmu: eps_{trac_dir, face_axis} contributions
            # From du_comp/dr_{face_axis} → eps_{comp, face_axis}
            if j == face_axis:
                key = (comp, free_exp)
                if key not in result:
                    result[key] = [0.0, 0.0]
                result[key][1] += face_sign * deriv * face_factor

            # From du_{face_axis}/dr_j → eps_{j, face_axis}
            if comp == face_axis:
                key = (j, free_exp)
                if key not in result:
                    result[key] = [0.0, 0.0]
                result[key][1] += face_sign * deriv * face_factor

    return [
        (d, fe, dl, dm)
        for (d, fe), (dl, dm) in result.items()
        if abs(dl) > 1e-30 or abs(dm) > 1e-30
    ]


def _surface_bilinear_1face(
    alpha_terms: list,
    traction_terms: list[tuple[int, tuple[int, int], float, float]],
    face_axis: int,
    face_sign: int,
) -> tuple[float, float, float, float]:
    """Surface bilinear for one face using (sigma,xi,v) substitution.

    For free axes: (sigma,xi) substitution, same as body bilinear.
    For normal axis: v substitution with r_n = face_sign*(1-2v).

    Prefactor derivation:
      Jacobian = 2 (from dv) × 1 (from sigma,xi per pair) = 2
      A-channel: 2/(2*rho) = 1/rho, fold xi to [0,1]^2: ×4 → prefactor 4
      B-channel: 2 × x_i*x_j/(8*rho^3), fold xi: ×4 → prefactor 4

    Returns (a_lam, a_mu, b_lam, b_mu).
    """
    free_axes = [k for k in range(3) if k != face_axis]
    f1, f2 = free_axes

    a_lam = 0.0
    a_mu = 0.0
    b_lam = 0.0
    b_mu = 0.0

    for comp_a, exp_a, c_a in alpha_terms:
        v_poly = _normal_axis_poly(exp_a[face_axis], face_sign)

        for trac_dir, trac_free_exp, dlam, dmu in traction_terms:
            # Free axis residuals via (sigma,xi) substitution
            a1, b1 = exp_a[f1], trac_free_exp[0]
            a2, b2 = exp_a[f2], trac_free_exp[1]

            poly1 = _expand_1d_product(a1, b1)
            poly2 = _expand_1d_product(a2, b2)
            norm1 = _xi_integrate_residual(poly1)
            norm2 = _xi_integrate_residual(poly2)
            swap1 = (
                norm1
                if a1 == b1
                else _xi_integrate_residual(_expand_1d_product(b1, a1))
            )
            swap2 = (
                norm2
                if a2 == b2
                else _xi_integrate_residual(_expand_1d_product(b2, a2))
            )

            # A-channel: diagonal (comp_a == trac_dir), kernel 1/rho (even)
            if comp_a == trac_dir:
                q1 = _symmetrize_axis(norm1, swap1, 0)
                q2 = _symmetrize_axis(norm2, swap2, 0)
                res_3d = _form_3d_product(q1, q2, v_poly)
                for exps, coeff in res_3d.items():
                    if abs(coeff) < 1e-30:
                        continue
                    master = _k1at_surf(*exps)
                    val = c_a * coeff * master
                    a_lam += dlam * val
                    a_mu += dmu * val

            # B-channel: x_tilde_{comp_a} * x_tilde_{trac_dir} / rho^3
            # x_tilde mapping: free_k -> u_k (odd), face_axis -> -face_sign*v (even in u)
            kp = [0, 0]
            extra_p, extra_q, extra_r = 0, 0, 0
            x_sign = 1.0
            for d in [comp_a, trac_dir]:
                if d == f1:
                    kp[0] = (kp[0] + 1) % 2
                    extra_p += 1
                elif d == f2:
                    kp[1] = (kp[1] + 1) % 2
                    extra_q += 1
                else:  # face_axis
                    extra_r += 1
                    x_sign *= -face_sign

            q1 = _symmetrize_axis(norm1, swap1, kp[0])
            q2 = _symmetrize_axis(norm2, swap2, kp[1])
            res_3d = _form_3d_product(q1, q2, v_poly)
            for exps, coeff in res_3d.items():
                if abs(coeff) < 1e-30:
                    continue
                p, q, r = exps
                master = _k3_surf(p + extra_p, q + extra_q, r + extra_r)
                val = c_a * x_sign * coeff * master
                b_lam += dlam * val
                b_mu += dmu * val

    return a_lam, a_mu, b_lam, b_mu


# ================================================================
# Surface stiffness (traction integral using MpB)
# ================================================================


def _compute_stiffness_surface(
    basis, gerade_indices: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute surface stiffness from traction jump at cube faces.

    For each face, the traction t_i = n_j Dc_{ijkl} eps_{kl}(phi_beta)
    is convolved with G and inner-producted with phi_alpha.

    Uses (sigma,xi,v) substitution:
      Free axes: same (sigma,xi) machinery as body bilinear
      Normal axis: r_n = face_sign*(1-2v), v in [0,1]
      Partial tent: tent in xi_1, xi_2 only (not v)

    Prefactor: 4 (same derivation as body bilinear).
    """
    n = len(gerade_indices)
    S_Alam = np.zeros((n, n))
    S_Amu = np.zeros((n, n))
    S_Blam = np.zeros((n, n))
    S_Bmu = np.zeros((n, n))

    total_pairs = n * n * 6
    count = 0

    for jj, beta in enumerate(gerade_indices):
        for face_axis in range(3):
            for face_sign in [1, -1]:
                traction_terms = _compute_face_traction(
                    basis[beta], face_axis, face_sign
                )
                if not traction_terms:
                    count += n
                    continue

                for ii, alpha in enumerate(gerade_indices):
                    count += 1
                    if count % 500 == 0:
                        print(
                            f"  surface stiffness: {count}/{total_pairs} ...",
                            flush=True,
                        )
                    a_lam, a_mu, b_lam, b_mu = _surface_bilinear_1face(
                        basis[alpha], traction_terms, face_axis, face_sign
                    )
                    S_Alam[ii, jj] += 4.0 * a_lam
                    S_Amu[ii, jj] += 4.0 * a_mu
                    S_Blam[ii, jj] += 4.0 * b_lam
                    S_Bmu[ii, jj] += 4.0 * b_mu

    return S_Alam, S_Amu, S_Blam, S_Bmu


# ================================================================
# Printing utilities
# ================================================================


def _print_matrix(name: str, M: np.ndarray, threshold: float = 1e-12):
    """Print matrix in Python-ready format."""
    n = M.shape[0]
    print(f"{name} = np.array([")
    for i in range(n):
        row = ", ".join(
            f"{M[i, j]:.14e}" if abs(M[i, j]) > threshold else "0.0" for j in range(n)
        )
        print(f"    [{row}],")
    print("])")


def _print_scalar(name: str, val: float):
    """Print scalar in Python-ready format."""
    print(f"{name} = {val:.14e}")


# ================================================================
# Main computation
# ================================================================


def main():
    t0 = time.time()
    print("=" * 60)
    print("T₅₇ Gerade Block Computation")
    print("=" * 60)

    basis = _build_basis_components()
    print(f"\nGerade modes: {len(GERADE_INDICES)} (6 strain + 30 cubic)")

    # ── Step 1: Mass matrix ──
    print("\n── Step 1: Mass matrix (exact) ──")
    M_ger = compute_mass_matrix(basis, GERADE_INDICES)
    print(f"  Symmetric: {np.allclose(M_ger, M_ger.T)}")
    eigvals_M = np.linalg.eigvalsh(M_ger)
    print(f"  Eigenvalue range: [{eigvals_M.min():.8f}, {eigvals_M.max():.8f}]")
    print(f"  Positive definite: {eigvals_M.min() > 0}")

    # ── Step 2: Body bilinear ──
    print("\n── Step 2: Body bilinear (36×36 gerade) ──")
    print("  Computing A-channel (from Mp masters)...")
    print("  Computing B-channel (from MpB masters)...")
    BA_ger, BB_ger = compute_body_bilinear(basis, GERADE_INDICES)
    print(f"  BA symmetric: {np.allclose(BA_ger, BA_ger.T, atol=1e-8)}")
    print(f"  BB symmetric: {np.allclose(BB_ger, BB_ger.T, atol=1e-8)}")
    dt = time.time() - t0
    print(f"  Elapsed: {dt:.1f}s")

    # ── Step 3: Stiffness (volume part) ──
    print("\n── Step 3: Stiffness volume part ──")
    SVol_Alam, SVol_Amu, SVol_Blam, SVol_Bmu = _compute_stiffness_volume(
        basis, GERADE_INDICES, BA_ger, BB_ger
    )
    print(f"  S_Alam norm: {np.linalg.norm(SVol_Alam):.6f}")
    print(f"  S_Amu norm: {np.linalg.norm(SVol_Amu):.6f}")

    # ── Step 3b: Stiffness (surface part) ──
    print("\n── Step 3b: Stiffness surface part ──")
    t_surf = time.time()
    SSurf_Alam, SSurf_Amu, SSurf_Blam, SSurf_Bmu = _compute_stiffness_surface(
        basis, GERADE_INDICES
    )
    print(f"  SSurf_Alam norm: {np.linalg.norm(SSurf_Alam):.6f}")
    print(f"  SSurf_Amu norm: {np.linalg.norm(SSurf_Amu):.6f}")
    print(f"  SSurf_Blam norm: {np.linalg.norm(SSurf_Blam):.6f}")
    print(f"  SSurf_Bmu norm: {np.linalg.norm(SSurf_Bmu):.6f}")
    print(f"  Surface elapsed: {time.time() - t_surf:.1f}s")

    # Combine volume + surface
    STot_Alam = SVol_Alam + SSurf_Alam
    STot_Amu = SVol_Amu + SSurf_Amu
    STot_Blam = SVol_Blam + SSurf_Blam
    STot_Bmu = SVol_Bmu + SSurf_Bmu

    # ── Step 4: Project through Usym₅₇ ──
    print("\n── Step 4: Usym₅₇ projection ──")
    Usym = _build_usym_57()
    orth_err = np.max(np.abs(Usym.T @ Usym - np.eye(57)))
    print(f"  Usym orthogonality error: {orth_err:.2e}")

    # Extract gerade block of Usym: Usym_g is 36×36
    g_idx = np.array(GERADE_INDICES)
    Usym_g = Usym[np.ix_(g_idx, np.arange(21, 57))]  # rows=gerade, cols=gerade irreps

    # Project matrices
    M_proj = Usym_g.T @ M_ger @ Usym_g
    BA_proj = Usym_g.T @ BA_ger @ Usym_g
    BB_proj = Usym_g.T @ BB_ger @ Usym_g
    STot_Alam_proj = Usym_g.T @ STot_Alam @ Usym_g
    STot_Amu_proj = Usym_g.T @ STot_Amu @ Usym_g
    STot_Blam_proj = Usym_g.T @ STot_Blam @ Usym_g
    STot_Bmu_proj = Usym_g.T @ STot_Bmu @ Usym_g

    # Check block-diagonal structure
    print("\n  Block-diagonal structure check (off-block max):")
    for name, mat in [("M", M_proj), ("BA", BA_proj), ("BB", BB_proj)]:
        # The blocks are at: A1g 0:3, A2g 3:4, Eg 4:12, T1g 12:21, T2g 21:36
        boundaries = [0, 3, 4, 12, 21, 36]
        max_offblock = 0.0
        for bi in range(len(boundaries) - 1):
            for bj in range(bi + 1, len(boundaries) - 1):
                block = mat[
                    boundaries[bi] : boundaries[bi + 1],
                    boundaries[bj] : boundaries[bj + 1],
                ]
                max_offblock = max(max_offblock, np.max(np.abs(block)))
        print(f"    {name}: {max_offblock:.2e}")

    # ── Step 5: Extract per-irrep blocks ──
    print("\n── Step 5: Per-irrep blocks ──")

    irreps = ["A1g", "A2g", "Eg", "T1g", "T2g"]
    for irrep in irreps:
        M_block = _extract_irrep_block(M_proj, irrep)
        BA_block = _extract_irrep_block(BA_proj, irrep)
        BB_block = _extract_irrep_block(BB_proj, irrep)

        dim = M_block.shape[0]
        print(f"\n  {irrep} ({dim}×{dim}):")
        print(f"    M diag: {np.diag(M_block)}")
        print(f"    BA diag: {np.diag(BA_block)}")
        print(f"    BB diag: {np.diag(BB_block)}")

        # Verify Kronecker structure for d>1 irreps
        if irrep == "Eg":
            idx_c0 = [4, 6, 8, 10]
            idx_c1 = [5, 7, 9, 11]
            M_c1 = M_proj[np.ix_(idx_c1, idx_c1)]
            print(f"    Eg copy consistency: {np.max(np.abs(M_block - M_c1)):.2e}")
        elif irrep in ("T1g", "T2g"):
            if irrep == "T1g":
                base = 12
                n_seeds = 3
            else:
                base = 21
                n_seeds = 5
            for copy in (1, 2):
                idx_copy = [base + 3 * s + copy for s in range(n_seeds)]
                M_copy = M_proj[np.ix_(idx_copy, idx_copy)]
                diff = np.max(np.abs(M_block - M_copy))
                print(f"    {irrep} copy {copy} consistency: {diff:.2e}")

    # ── Step 5b: Cross-validate strain stiffness ──
    print("\n── Step 5b: Strain stiffness cross-validation ──")
    # Strain modes have zero volume force, so total stiffness = surface only.
    # The [0,0] entry of each gerade irrep should match:
    #   A1g: 24*Dlam + 16*Dmu  → S_Alam[0,0]=24, S_Amu[0,0]=16
    #   Eg:  16*Dmu            → S_Alam[0,0]=0, S_Amu[0,0]=16
    #   T2g: 8*Dmu             → S_Alam[0,0]=0, S_Amu[0,0]=8
    for irrep, ref_Alam, ref_Amu in [
        ("A1g", 24.0, 16.0),
        ("Eg", 0.0, 16.0),
        ("T2g", 0.0, 8.0),
    ]:
        SA = _extract_irrep_block(STot_Alam_proj, irrep)
        SM = _extract_irrep_block(STot_Amu_proj, irrep)
        print(
            f"  {irrep}: S_Alam[0,0]={SA[0, 0]:.6f} (ref {ref_Alam:.1f})"
            f"  S_Amu[0,0]={SM[0, 0]:.6f} (ref {ref_Amu:.1f})"
        )

    # ── Step 6: Print Python-ready values ──
    print("\n" + "=" * 60)
    print("PYTHON-READY VALUES")
    print("(paste into effective_contrasts.py _build_galerkin_irrep_blocks_57)")
    print("=" * 60)

    for irrep in irreps:
        M_block = _extract_irrep_block(M_proj, irrep)
        BA_block = _extract_irrep_block(BA_proj, irrep)
        BB_block = _extract_irrep_block(BB_proj, irrep)
        SA_block = _extract_irrep_block(STot_Alam_proj, irrep)
        SB_block = _extract_irrep_block(STot_Amu_proj, irrep)
        SC_block = _extract_irrep_block(STot_Blam_proj, irrep)
        SD_block = _extract_irrep_block(STot_Bmu_proj, irrep)

        dim = M_block.shape[0]
        print(f"\n# ── {irrep} ({dim}×{dim}) ──")

        if dim == 1:
            _print_scalar(f"M_{irrep}", M_block[0, 0])
            _print_scalar(f"Bbody_A_{irrep}", BA_block[0, 0])
            _print_scalar(f"Bbody_B_{irrep}", BB_block[0, 0])
            _print_scalar(f"S_Alam_{irrep}", SA_block[0, 0])
            _print_scalar(f"S_Amu_{irrep}", SB_block[0, 0])
            _print_scalar(f"S_Blam_{irrep}", SC_block[0, 0])
            _print_scalar(f"S_Bmu_{irrep}", SD_block[0, 0])
        else:
            _print_matrix(f"M_{irrep}", M_block)
            _print_matrix(f"Bbody_A_{irrep}", BA_block)
            _print_matrix(f"Bbody_B_{irrep}", BB_block)
            _print_matrix(f"S_Alam_{irrep}", SA_block)
            _print_matrix(f"S_Amu_{irrep}", SB_block)
            _print_matrix(f"S_Blam_{irrep}", SC_block)
            _print_matrix(f"S_Bmu_{irrep}", SD_block)

    # ── Cross-validation ──
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION vs T₂₇")
    print("=" * 60)

    # Corrected strain-only body eigenvalues (validated by 6D Monte Carlo):
    strain_ev_A_ref = 1.755317619825370
    strain_ev_B_A1g_ref = -1.611924358631187
    strain_ev_B_Eg_ref = 0.043992031944107
    strain_ev_B_T2g_ref = 0.579676728845940

    for irrep in ["A1g", "Eg", "T2g"]:
        M_block = _extract_irrep_block(M_proj, irrep)
        BA_block = _extract_irrep_block(BA_proj, irrep)
        BB_block = _extract_irrep_block(BB_proj, irrep)

        # The [0,0] entry should be close to T₂₇ strain value
        # (but not exact due to cubic mode mixing)
        ratio_A = BA_block[0, 0] / M_block[0, 0] if abs(M_block[0, 0]) > 1e-15 else 0.0
        ratio_B = BB_block[0, 0] / M_block[0, 0] if abs(M_block[0, 0]) > 1e-15 else 0.0
        print(f"\n  {irrep}:")
        print(f"    BA[0,0]/M[0,0] = {ratio_A:.10f}  (T₂₇: {strain_ev_A_ref:.10f})")
        print(f"    BB[0,0]/M[0,0] = {ratio_B:.10f}", end="")
        if irrep == "A1g":
            print(f"  (T₂₇: {strain_ev_B_A1g_ref:.10f})")
        elif irrep == "Eg":
            print(f"  (T₂₇: {strain_ev_B_Eg_ref:.10f})")
        else:
            print(f"  (T₂₇: {strain_ev_B_T2g_ref:.10f})")

    dt_total = time.time() - t0
    print(f"\nTotal elapsed: {dt_total:.1f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
