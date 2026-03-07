"""
tmatrix_cube.py
Analytical T-Matrix for a Small CUBIC Scatterer
in an Isotropic Elastic Background Medium.

Semi-analytical approach: Taylor-expand the Green's tensor about the
centre of the cube, then integrate each monomial over [-a,a]^3 exactly.

The cube moments factor into products of 1D integrals:
    int_{-a}^{a} x^p dx = 2a/(p+1)  if p even, 0 if p odd

Key difference from sphere: the fourth-rank cube moment
    int_cube x_i x_j x_k x_l d^3x
has CUBIC (not isotropic) symmetry, introducing a third independent
tensor structure beyond delta_ij delta_kl and delta_ik delta_jl + ...

Usage:
    python tmatrix_cube.py
"""

import numpy as np
import sympy as sp
from sympy import (
    I,
    KroneckerDelta,
    Rational,
    diff,
    exp,
    latex,
    limit,
    pi,
    pprint,
    series,
    simplify,
    symbols,
)

sp.init_printing(use_unicode=True)

# ================================================================
# Parameters
# ================================================================
omega = symbols(r"\omega", positive=True)
alpha = symbols(r"\alpha", positive=True)
beta = symbols(r"\beta", positive=True)
rho = symbols(r"\rho", positive=True)
a = symbols("a", positive=True)
r = symbols("r", positive=True)

Dlambda = symbols(r"\Delta\lambda")
Dmu = symbols(r"\Delta\mu")
Drho = symbols(r"\Delta\rho")

# Spatial coordinates
x1, x2, x3 = symbols("x_1 x_2 x_3", real=True)
xx = [x1, x2, x3]

d = KroneckerDelta

print("=" * 70)
print("  Analytical T-Matrix for CUBIC Elastic Scatterer (SymPy)")
print("  Semi-analytical: Taylor expansion + cube moments")
print("=" * 70)

# ================================================================
# SECTION 1: Taylor expansion of radial functions
# ================================================================
# f(r) and g(r) expanded about r=0 to order N in r
# r^2 = x1^2 + x2^2 + x3^2

print("\n" + "-" * 70)
print("  Section 1: Taylor expansion of f(r), g(r)")
print("-" * 70)

# Near-field constant
C_nf = (1 - beta**2 / alpha**2) / (8 * pi * rho * beta**2)

# The Green's tensor components (from Appendix A)
# X(r) = exp(i*omega*r/alpha) / (4*pi*rho*alpha^2)
# V(r) = exp(i*omega*r/beta) / (4*pi*rho*beta^2)
# f(r) = V(r)/r - C/r = [V(r) - C]/r
# g(r) = 3C/r + X(r)/r - V(r)/r = [3C + X(r) - V(r)]/r
#
# Key: f(r) and g(r) have 1/r singularities, but G_{ij} is finite at r=0.
# We need to Taylor-expand G_{ij} directly, not f and g separately.
#
# Actually, let's expand the full expressions:
# f(r) = [exp(i*omega*r/beta) - (1 - beta^2/alpha^2)/2] / (4*pi*rho*beta^2*r)
# This has a 1/r singularity that cancels when combined properly.
#
# Better approach: expand f(r)*r and g(r)*r as Taylor series in r,
# then G_{ij} = [f(r)*r * delta_ij + g(r)*r * n_i*n_j] / r
#            = [F(r) * delta_ij] / r + [G_tilde(r) * x_i*x_j] / r^3
# where F = f*r, G_tilde = g*r
#
# Even better: write G_{ij} = f(r)*delta_{ij} + g(r)*x_i*x_j/r^2
# and note that f(r) and g(r)/r^2 need to be expanded.
#
# Cleanest approach: define
#   phi(r) = f(r)        — coefficient of delta_ij
#   psi(r) = g(r)/r^2    — coefficient of x_i x_j
# Then G_{ij} = phi(r) * delta_ij + psi(r) * x_i * x_j
# with r^2 = x1^2 + x2^2 + x3^2
#
# phi and psi are what we need to Taylor-expand about r=0.

# Let's compute phi(r) = f(r) and psi(r) = g(r)/r^2

# f(r) = V(r)/r - C/r where V(r) = exp(i*omega*r/beta)/(4*pi*rho*beta^2)
# g(r) = 3C/r + X(r)/r - V(r)/r

# phi(r) = f(r) = [exp(i*w*r/b) / (4*pi*rho*b^2) - C] / r
# psi(r) = g(r)/r^2 = [3C + exp(i*w*r/a)/(4*pi*rho*a^2) - exp(i*w*r/b)/(4*pi*rho*b^2)] / r^3

# These look singular but aren't. Let's use SymPy series expansion.

# Define intermediate
w = omega  # shorthand

# Full expressions for f*r and g*r (remove 1/r singularity)
fr_times_r = exp(I * w * r / beta) / (4 * pi * rho * beta**2) - C_nf
gr_times_r = (
    3 * C_nf
    + exp(I * w * r / alpha) / (4 * pi * rho * alpha**2)
    - exp(I * w * r / beta) / (4 * pi * rho * beta**2)
)

# phi(r) = (f*r)/r = f(r) — need series of f(r)
# psi(r) = g(r)/r^2 = (g*r)/(r^3) — need series of g*r / r^3

# Expand f*r and g*r as Taylor series in r about 0
N_ORDER = 8  # Keep terms up to r^N_ORDER in f*r (so r^(N-1) in f)

fr_r_series = series(fr_times_r, r, 0, n=N_ORDER + 1).removeO()
gr_r_series = series(gr_times_r, r, 0, n=N_ORDER + 1).removeO()

print(f"\nf(r)*r expanded: {len(str(fr_r_series))} chars (suppressed for brevity)")
print(f"g(r)*r expanded: {len(str(gr_r_series))} chars (suppressed for brevity)")

# Now phi(r) = f(r) = (f*r)/r — extract Taylor coefficients
# f*r = c0 + c1*r + c2*r^2 + ...
# BUT c0 should be nonzero (giving f ~ c0/r which would be singular)
# unless the constant term cancels. Let's check.

fr_r_poly = sp.Poly(fr_r_series, r)
gr_r_poly = sp.Poly(gr_r_series, r)

print("\nf*r coefficients (by power of r):")
for power, coeff in sorted(fr_r_poly.as_dict().items()):
    print(f"  r^{power[0]}: {simplify(coeff)}")

print("\ng*r coefficients (by power of r):")
for power, coeff in sorted(gr_r_poly.as_dict().items()):
    print(f"  r^{power[0]}: {simplify(coeff)}")

# ================================================================
# SECTION 2: G_{ij} as polynomial in x_i
# ================================================================
# G_{ij}(x) = f(r) * delta_{ij} + g(r)/r^2 * x_i * x_j
#
# where r^2 = x1^2 + x2^2 + x3^2
#
# We write f(r) = sum_n f_n * r^{2n}  (only even powers by parity of exp)
# and g(r)/r^2 = sum_n g_n * r^{2n}
# substituting r^2 = x1^2 + x2^2 + x3^2 throughout.

print("\n" + "-" * 70)
print("  Section 2: G_{ij} as Cartesian polynomial")
print("-" * 70)

# Extract Taylor coefficients of phi(r) = f(r) and psi(r) = g(r)/r^2
# From f*r: f(r) = sum_{k>=1} c_k * r^{k-1} where c_k is coeff of r^k in f*r
# From g*r: g(r)/r^2 = sum_{k>=1} c_k * r^{k-3} where c_k is coeff of r^k in g*r

# phi(r) = f(r): we need the even powers of r (since G_{ij} must be smooth)
# f*r has only odd powers of r (r^1, r^3, r^5, ...) so f(r) has even powers (r^0, r^2, r^4, ...)
# g*r has only odd powers of r as well, so g(r)/r^2 = (g*r)/r^3 has even powers (r^0, r^2, ...)

# Let's directly compute the Taylor expansion of f(r) and psi(r) = g(r)/r^2
# using series in r^2

# More robust approach: compute f(r) series directly
f_expr = fr_times_r / r
g_over_r2_expr = gr_times_r / r**3

# SymPy can expand these — the 1/r singularities cancel
f_series = series(f_expr, r, 0, n=N_ORDER).removeO()
psi_series = series(g_over_r2_expr, r, 0, n=N_ORDER).removeO()

f_series = simplify(f_series)
psi_series = simplify(psi_series)

print("\nphi(r) and psi(r) series computed successfully.")


# ================================================================
# SECTION 3: Cube moments
# ================================================================
# int_{-a}^{a} int_{-a}^{a} int_{-a}^{a} x1^p1 x2^p2 x3^p3 dx1 dx2 dx3
# = prod_i [2*a^(p_i+1)/(p_i+1)] if all p_i even, else 0
#
# For G_{ij} we need moments of the form:
#   int_cube (x1^2+x2^2+x3^2)^n dx = sum of multinomial moments
#   int_cube (x1^2+x2^2+x3^2)^n * x_i * x_j dx

print("\n" + "-" * 70)
print("  Section 3: Cube moments")
print("-" * 70)


def cube_moment(*exponents):
    """Compute int_{-a}^{a}^3 x1^p1 * x2^p2 * x3^p3 d^3x analytically."""
    result = sp.Integer(1)
    for p in exponents:
        if p % 2 != 0:  # odd exponent -> zero by symmetry
            return sp.Integer(0)
        result *= 2 * a ** (p + 1) / (p + 1)
    return result


# Verify basic moments
V_cube = cube_moment(0, 0, 0)  # = (2a)^3 = 8a^3
print(f"\nVolume of cube: {V_cube} = {simplify(V_cube)}")
assert V_cube == 8 * a**3

# Second moments: int x_i^2 d^3x = (2a)^3 * a^2/3 for each i
m2 = cube_moment(2, 0, 0)
print(f"int x1^2 d^3x = {m2}")

# Fourth moments — THIS is where cubic anisotropy enters
m4_iiii = cube_moment(4, 0, 0)  # int x1^4
m4_iijj = cube_moment(2, 2, 0)  # int x1^2 x2^2
print(f"int x1^4 d^3x = {m4_iiii}")
print(f"int x1^2 x2^2 d^3x = {m4_iijj}")

# For the sphere: int x_i^4 d^3x = (4pi/15) * int_0^a r^6 dr / a^0 ...
# Let me compare: the isotropic part would give
# int x_i x_j x_k x_l = C1 * (d_ij d_kl + d_ik d_jl + d_il d_jk)
# Contracting i=j=k=l=1: C1 * 3 = m4_iiii -> C1 = m4_iiii/3
# Contracting i=j=1, k=l=2: C1 = m4_iijj
# Check: m4_iiii/3 vs m4_iijj
print("\nCubic anisotropy check:")
print(f"  m4_iiii / 3 = {simplify(m4_iiii / 3)}")
print(f"  m4_iijj     = {simplify(m4_iijj)}")
print(f"  Ratio = {simplify(m4_iiii / (3 * m4_iijj))}")
print("  (Should be 1 for sphere, != 1 for cube)")

# The fourth moment tensor for the cube is:
# int_cube x_i x_j x_k x_l d^3x = P * (d_ij d_kl + d_ik d_jl + d_il d_jk) + Q * E_ijkl
# where E_ijkl = sum_m d_im d_jm d_km d_lm is the cubic anisotropy tensor
# (equals 1 when all indices equal, 0 otherwise)
#
# From the moments:
#   i=j=k=l: 3P + Q = m4_iiii = 8a^7/5
#   i=j!=k=l: P = m4_iijj = 8a^7/9
# So P = 8a^7/9, Q = 8a^7/5 - 3*8a^7/9 = 8a^7(1/5 - 1/3) = 8a^7*(-2/15) = -16a^7/15

P_coeff = m4_iijj
Q_coeff = simplify(m4_iiii - 3 * m4_iijj)

print("\nFourth moment decomposition:")
print(f"  P (isotropic part) = {P_coeff}")
print(f"  Q (cubic anisotropy) = {Q_coeff}")
print("  int x_i x_j x_k x_l = P*(d_ij d_kl + ...) + Q*E_ijkl")


# ================================================================
# SECTION 4: Volume integrals of G_{ij} over the cube
# ================================================================

print("\n" + "-" * 70)
print("  Section 4: Volume integral of G_{ij} over the cube")
print("-" * 70)

# G_{ij} = phi(r) * delta_{ij} + psi(r) * x_i * x_j
# where phi and psi are power series in r^2 = x1^2 + x2^2 + x3^2
#
# int_cube G_{ij} d^3x = delta_{ij} * int_cube phi(r) d^3x
#                       + int_cube psi(r) * x_i * x_j d^3x
#
# Each integrand is a polynomial in x1, x2, x3 after substituting
# r^2 = x1^2 + x2^2 + x3^2.

# Extract phi coefficients: phi(r) = sum_n phi_n * r^{2n}
# Extract psi coefficients: psi(r) = sum_n psi_n * r^{2n}
#
# Use SymPy's coeff() method to extract coefficients safely,
# since the series may contain 1/r terms that cancel in the full G_{ij}.

phi_coeffs = {}  # phi_n = coefficient of r^{2n} in phi(r)
psi_coeffs = {}

# Extract from the Taylor series of f*r (which is a polynomial in r)
# f(r) = (f*r)/r, so coeff of r^{2n} in f(r) = coeff of r^{2n+1} in f*r
for n in range(N_ORDER // 2 + 1):
    c = fr_r_series.coeff(r, 2 * n + 1)
    if c != 0:
        phi_coeffs[n] = simplify(c)

# psi(r) = g(r)/r^2 = (g*r)/r^3, so coeff of r^{2n} in psi = coeff of r^{2n+3} in g*r
for n in range(N_ORDER // 2 + 1):
    c = gr_r_series.coeff(r, 2 * n + 3)
    if c != 0:
        psi_coeffs[n] = simplify(c)

print("\nphi(r) = f(r) coefficients [phi_n for r^{2n}]:")
for n in sorted(phi_coeffs.keys()):
    print(f"  phi_{n} (r^{2 * n}): {phi_coeffs[n]}")

print("\npsi(r) = g(r)/r^2 coefficients [psi_n for r^{2n}]:")
for n in sorted(psi_coeffs.keys()):
    print(f"  psi_{n} (r^{2 * n}): {psi_coeffs[n]}")


# ================================================================
# Compute int_cube r^{2n} d^3x and int_cube r^{2n} x_i x_j d^3x
# ================================================================
# r^{2n} = (x1^2 + x2^2 + x3^2)^n
# We expand using multinomial theorem and integrate term by term.


def integrate_r2n_over_cube(n):
    """Compute int_cube (x1^2+x2^2+x3^2)^n d^3x."""
    if n == 0:
        return V_cube
    # Expand (x1^2 + x2^2 + x3^2)^n using multinomial
    rsq_expr = (x1**2 + x2**2 + x3**2) ** n
    expanded = sp.expand(rsq_expr)
    # Integrate each monomial over cube
    result = sp.Integer(0)
    poly = sp.Poly(expanded, x1, x2, x3)
    for powers, coeff in poly.as_dict().items():
        result += coeff * cube_moment(*powers)
    return simplify(result)


def integrate_r2n_xixj_over_cube(n, i, j):
    """Compute int_cube (x1^2+x2^2+x3^2)^n * x_i * x_j d^3x.
    i, j in {0, 1, 2} indexing x1, x2, x3."""
    rsq_expr = (x1**2 + x2**2 + x3**2) ** n * xx[i] * xx[j]
    expanded = sp.expand(rsq_expr)
    result = sp.Integer(0)
    poly = sp.Poly(expanded, x1, x2, x3)
    for powers, coeff in poly.as_dict().items():
        result += coeff * cube_moment(*powers)
    return simplify(result)


# Test
I0 = integrate_r2n_over_cube(0)
I1 = integrate_r2n_over_cube(1)
print(f"\nint_cube 1 d^3x = {I0}")
print(f"int_cube r^2 d^3x = {I1}")
print(f"int_cube r^2 x1^2 d^3x = {integrate_r2n_xixj_over_cube(1, 0, 0)}")
print(f"int_cube r^2 x1 x2 d^3x = {integrate_r2n_xixj_over_cube(1, 0, 1)}")

# ================================================================
# Assemble Gamma_0 (cube)
# ================================================================
# int_cube G_{ij} d^3x = delta_{ij} * sum_n phi_n * int_cube r^{2n} d^3x
#                       + sum_n psi_n * int_cube r^{2n} x_i x_j d^3x
#
# The first sum is proportional to delta_{ij} (isotropic).
# The second sum has the structure:
#   int_cube r^{2n} x_i x_j d^3x
# For n=0 this is just int x_i x_j = (2a)^3 * a^2/3 * delta_ij (isotropic!)
# For n>=1, these acquire cubic corrections but the leading structure is still delta_{ij}
#
# Actually: int_cube r^{2n} x_i x_j d^3x — by cubic symmetry, this IS proportional
# to delta_{ij} for all n! (The cubic anisotropy tensor E_ijkl only appears in rank-4 moments.)

print("\n" + "-" * 70)
print("  Section 4a: Gamma_0 for the cube")
print("-" * 70)

# Compute Gamma_0: int_cube G_{ij} d^3x = Gamma0_cube * delta_{ij}
# (since int_cube r^{2n} x_i x_j is proportional to delta_ij)

# Verify isotropy of second moment
test_11 = integrate_r2n_xixj_over_cube(1, 0, 0)
test_22 = integrate_r2n_xixj_over_cube(1, 1, 1)
test_12 = integrate_r2n_xixj_over_cube(1, 0, 1)
assert test_11 == test_22, f"Cubic moment not diagonal: {test_11} vs {test_22}"
assert test_12 == 0, f"Off-diagonal not zero: {test_12}"
print("Confirmed: int_cube r^{2n} x_i x_j d^3x is proportional to delta_ij ✓")

# Build Gamma_0
Gamma0_cube = sp.Integer(0)
max_n = max(max(phi_coeffs.keys()), max(psi_coeffs.keys()))

for n in sorted(phi_coeffs.keys()):
    I_n = integrate_r2n_over_cube(n)
    Gamma0_cube += phi_coeffs[n] * I_n
    print(f"  phi_{n} * int r^{2 * n} = {simplify(phi_coeffs[n])} * {I_n}")

for n in sorted(psi_coeffs.keys()):
    # Contribution: psi_n * int_cube r^{2n} x_i^2 d^3x (diagonal part, divide by delta_ii factor)
    I_n_diag = integrate_r2n_xixj_over_cube(n, 0, 0)  # same for all i
    Gamma0_cube += psi_coeffs[n] * I_n_diag
    print(f"  psi_{n} * int r^{2 * n} x1^2 = {simplify(psi_coeffs[n])} * {I_n_diag}")

Gamma0_cube = simplify(Gamma0_cube)
print("\nGamma_0 (cube) =")
pprint(Gamma0_cube)

# Compare with sphere Gamma_0 at same volume
# Sphere volume = 4/3 pi a_s^3 = (2a)^3 = 8a^3 -> a_s = a*(6/pi)^(1/3)
a_s = a * (6 / pi) ** Rational(1, 3)
print("\nEqual-volume sphere radius: a_s = a*(6/pi)^(1/3)")


# ================================================================
# SECTION 5: Second derivative integral — A, B, and C_cubic
# ================================================================
# For the cube:
# int_cube G_{ij,kl} d^3x = A_c * d_ij d_kl + B_c * (d_ik d_jl + d_il d_jk) + C_c * E_ijkl
#
# where E_ijkl = sum_m d_im d_jm d_km d_lm (cubic anisotropy tensor)
#
# We need THREE contractions now.

print("\n" + "-" * 70)
print("  Section 5: Second-derivative integral (cube)")
print("-" * 70)

# Strategy: compute int_cube G_{ij,kl} d^3x numerically for specific index
# combinations, or use the Taylor expansion approach.
#
# G_{ij,kl} = d^2/dx_k dx_l [phi(r) delta_ij + psi(r) x_i x_j]
#
# Let's compute this symbolically using SymPy differentiation on the
# truncated polynomial.

# Build G_{ij} as explicit polynomial in x1, x2, x3
r_sq = x1**2 + x2**2 + x3**2

# phi(r) as polynomial in x1, x2, x3
phi_poly = sum(phi_coeffs[n] * r_sq**n for n in phi_coeffs)
psi_poly = sum(psi_coeffs[n] * r_sq**n for n in psi_coeffs)

print("\nPolynomial representations built.")


def G_ij_poly(i, j):
    """G_{ij} as polynomial in x1, x2, x3."""
    return phi_poly * d(i, j) + psi_poly * xx[i] * xx[j]


def G_ij_kl_poly(i, j, k, l):
    """G_{ij,kl} = d^2 G_{ij} / dx_k dx_l as polynomial."""
    expr = G_ij_poly(i, j)
    deriv = diff(diff(expr, xx[k]), xx[l])
    return sp.expand(deriv)


def integrate_poly_over_cube(expr):
    """Integrate a polynomial in x1, x2, x3 over [-a,a]^3."""
    expr = sp.expand(expr)
    if expr == 0:
        return sp.Integer(0)
    poly = sp.Poly(expr, x1, x2, x3)
    result = sp.Integer(0)
    for powers, coeff in poly.as_dict().items():
        result += coeff * cube_moment(*powers)
    return simplify(result)


# Compute the three independent components of int G_{ij,kl} d^3x
# For cubic symmetry, the independent components are:
# (a) I_{1111}  — all indices equal
# (b) I_{1122}  — two pairs
# (c) I_{1212}  — alternating pairs

print("\nComputing int_cube G_{ij,kl} d^3x for independent index combinations...")

I_1111 = integrate_poly_over_cube(G_ij_kl_poly(0, 0, 0, 0))
I_1122 = integrate_poly_over_cube(G_ij_kl_poly(0, 0, 1, 1))
I_1212 = integrate_poly_over_cube(G_ij_kl_poly(0, 1, 0, 1))

print(f"\nI_1111 = {I_1111}")
print(f"I_1122 = {I_1122}")
print(f"I_1212 = {I_1212}")

# Verify symmetries
I_2222 = integrate_poly_over_cube(G_ij_kl_poly(1, 1, 1, 1))
I_1221 = integrate_poly_over_cube(G_ij_kl_poly(0, 1, 1, 0))
assert simplify(I_2222 - I_1111) == 0, (
    f"Cubic symmetry broken: I_2222={I_2222} != I_1111={I_1111}"
)
assert simplify(I_1221 - I_1212) == 0, "Minor symmetry broken: I_1221 != I_1212"
print("\nCubic and minor symmetries verified ✓")

# Decomposition:
# I_ijkl = A_c * d_ij d_kl + B_c * (d_ik d_jl + d_il d_jk) + C_c * E_ijkl
#
# I_1111 = A_c + 2*B_c + C_c
# I_1122 = A_c
# I_1212 = B_c

A_cube = I_1122
B_cube = I_1212
C_cube = simplify(I_1111 - I_1122 - 2 * I_1212)

print(f"\n{'=' * 70}")
print("  ★ CUBE INTEGRAL DECOMPOSITION ★")
print(f"{'=' * 70}")
print("\nA_cube (isotropic, volumetric) =")
pprint(A_cube)
print("\nB_cube (isotropic, shear) =")
pprint(B_cube)
print("\nC_cube (CUBIC ANISOTROPY) =")
pprint(C_cube)

# Check: for a sphere, C should be zero
print("\nC_cube is the cubic anisotropy correction.")
print("It vanishes for a sphere and is O(a^7) for the cube.")

# Verify: substitute into known contractions
# I_{iikl} = (3A+2B+C) d_{kl}  — trace contraction
# Note: E_{iikl} = d_{kl} (since sum_m d_im d_im d_km d_lm = d_km d_lm for i summed)
# Wait: E_{iikl} = sum_m d_im^2 d_km d_lm = sum_m d_km d_lm = d_kl
# So I_{iikl} = (3A_c + 2B_c + C_c) d_kl

trace_check = simplify(3 * A_cube + 2 * B_cube + C_cube)
I_iikl_direct = integrate_poly_over_cube(
    sum(G_ij_kl_poly(i, i, 0, 0) for i in range(3))
)
print(f"\nTrace check: 3A+2B+C = {trace_check}")
print(f"Direct I_{{ii00}} = {I_iikl_direct}")
assert simplify(trace_check - I_iikl_direct) == 0, "Trace check FAILED"
print("Trace check passed ✓")


# ================================================================
# SECTION 6: T-matrix coupling coefficients (cube)
# ================================================================

print("\n" + "-" * 70)
print("  Section 6: T-matrix coefficients (cube)")
print("-" * 70)

# S_{mnjk} = (1/2)(I_{mjkn} + I_{njkm})
# For the cube: I_{ijkl} = A_c d_ij d_kl + B_c (d_ik d_jl + d_il d_jk) + C_c E_ijkl
# S_{mnjk} = (1/2)[(A_c d_mj d_kn + B_c(d_mk d_jn + d_mn d_jk) + C_c E_mjkn)
#           + (A_c d_nj d_km + B_c(d_nk d_jm + d_nn?... )]
# This is getting complex. Let me just compute T numerically via the tensor.


def I_tensor(i, j, k, l):
    """The integral tensor, using the A, B, C decomposition."""
    iso = A_cube * d(i, j) * d(k, l) + B_cube * (d(i, k) * d(j, l) + d(i, l) * d(j, k))
    # Cubic part: E_ijkl = 1 if i=j=k=l, else 0
    cubic = C_cube * d(i, j) * d(j, k) * d(k, l)
    return iso + cubic


def S_tensor(m, n, j, k):
    """Symmetrised tensor S_{mnjk} = (1/2)(I_{mjkn} + I_{njkm})."""
    return Rational(1, 2) * (I_tensor(m, j, k, n) + I_tensor(n, j, k, m))


def Dc_tensor(j, k, l, p):
    """Isotropic stiffness contrast."""
    return Dlambda * d(j, k) * d(l, p) + Dmu * (d(j, l) * d(k, p) + d(j, p) * d(k, l))


def T_tensor(m, n, l, p):
    """T_{mnlp} = sum_{jk} S_{mnjk} Dc_{jklp}."""
    return sum(
        S_tensor(m, n, j, k) * Dc_tensor(j, k, l, p) for j in range(3) for k in range(3)
    )


# Compute independent components
T_1111_cube = simplify(T_tensor(0, 0, 0, 0))
T_1122_cube = simplify(T_tensor(0, 0, 1, 1))
T_1212_cube = simplify(T_tensor(0, 1, 0, 1))

print(f"\nT_1111 = {T_1111_cube}")
print(f"T_1122 = {T_1122_cube}")
print(f"T_1212 = {T_1212_cube}")

# Decompose: T = T1 d_mn d_lp + T2 (d_ml d_np + d_mp d_nl) + T3 E_mnlp
T1_cube = T_1122_cube
T2_cube = T_1212_cube
T3_cube = simplify(T_1111_cube - T_1122_cube - 2 * T_1212_cube)

print(f"\n{'=' * 70}")
print("  ★ T-MATRIX COEFFICIENTS (CUBE) ★")
print(f"{'=' * 70}")
print("\nT_1 (volumetric, isotropic) =")
pprint(T1_cube)
print("\nT_2 (shear, isotropic) =")
pprint(T2_cube)
print("\nT_3 (CUBIC ANISOTROPY) =")
pprint(T3_cube)


# ================================================================
# SECTION 7: Self-consistent solution — amplification factors (cube)
# ================================================================
#
# ---- THE FULL 6×6 VOIGT MATRIX AND THE EIGENVALUE SHORTCUT ----
#
# The self-consistent strain equation (Lippmann-Schwinger) is:
#   eps_{mn} = eps^0_{mn} + T_{mnlp} eps_{lp}
# with T_{mnlp} = T1 d_mn d_lp + T2 (d_ml d_np + d_mp d_nl) + T3 E_mnlp.
#
# In Voigt notation (11→1, 22→2, 33→3, 23→4, 13→5, 12→6), the
# T-tensor becomes a 6×6 matrix. The Voigt mapping introduces factors
# of 2 for the off-diagonal strain components (engineering shear strain):
#
#    ε_Voigt = [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]^T
#
# The 6×6 T-matrix in Voigt notation has the block structure:
#
#   T_Voigt = [ D  |  0 ]
#             [----|----]
#             [ 0  |  S ]
#
# where:
#
#   D (3×3 diagonal-strain block) =
#       [ T1+2T2+T3    T1         T1      ]
#       [ T1         T1+2T2+T3    T1      ]
#       [ T1           T1       T1+2T2+T3 ]
#
#     = T1 * J + (2T2+T3) * I_3
#
#     (J = matrix of ones, I_3 = 3×3 identity)
#
#   S (3×3 off-diagonal shear block) =
#       [ 2T2   0     0  ]
#       [ 0    2T2    0  ]
#       [ 0     0    2T2 ]
#
# Note: D_{mm,nn} = T1*d_mn + 2*T2*d_mn + T3*d_mn (when m=n)
#                  = T1 (when m≠n)
#       The T3 E_mnlp term contributes ONLY when all four indices
#       equal (m=n=l=p), hence it adds T3 to the diagonal of D
#       but not to S (since E has no off-diagonal components).
#
# ---- EIGENVALUE DECOMPOSITION ----
#
# The block structure means we can diagonalise T_Voigt analytically.
#
# BLOCK S: Already diagonal with eigenvalue 2T2.
#   => Amplification factor for off-diagonal shear:
#      A_e^(off) = 1/(1 - 2T2)
#
# BLOCK D: Has the form T1*J + (2T2+T3)*I, with eigenvalues:
#   (a) Eigenvector [1,1,1]: eigenvalue = 3T1 + 2T2 + T3  (dilatation mode)
#   (b) Eigenvectors [1,-1,0], [1,1,-2]: eigenvalue = 2T2 + T3  (diagonal deviatoric)
#
#   => A_theta    = 1/(1 - 3T1 - 2T2 - T3)   [dilatation]
#      A_e^(diag) = 1/(1 - 2T2 - T3)          [diagonal deviatoric]
#
# ---- WHY THE EIGENVALUE SHORTCUT MATTERS ----
#
# Instead of computing the full 6×6 matrix inverse (I - T_Voigt)^{-1}
# and then multiplying by the 6×6 Dc_Voigt matrix (which involves
# heavy symbolic algebra with simplify() calls), we exploit the
# block-diagonal structure to reduce everything to 4 SCALAR equations:
#
#   A_u       = 1/(1 - ω²Δρ Γ₀)             [displacement]
#   A_θ       = 1/(1 - 3T₁ - 2T₂ - T₃)      [dilatation]
#   A_e^(off) = 1/(1 - 2T₂)                  [off-diagonal shear]
#   A_e^(diag)= 1/(1 - 2T₂ - T₃)            [diagonal deviatoric]
#
# For the SPHERE (T₃ = 0): A_e^(off) = A_e^(diag) ≡ A_e and
# we recover the familiar 3 amplification factors.
#
# For the CUBE (T₃ ≠ 0): the two shear modes have DIFFERENT
# amplification factors — this is the cubic anisotropy.
# The splitting is proportional to T₃ ~ O(ωa)^7, i.e. it is a
# higher-order effect that grows rapidly with frequency.

print("\n" + "-" * 70)
print("  Section 7: Self-consistent solution (cube)")
print("-" * 70)

amp_u_cube = 1 / (1 - omega**2 * Drho * Gamma0_cube)
amp_theta_cube = 1 / (1 - 3 * T1_cube - 2 * T2_cube - T3_cube)
amp_e_off_cube = 1 / (1 - 2 * T2_cube)
amp_e_diag_cube = 1 / (1 - 2 * T2_cube - T3_cube)

print(f"\n{'=' * 70}")
print("  ★ AMPLIFICATION FACTORS (CUBE) ★")
print(f"{'=' * 70}")
print("\nA_u (displacement) = 1/(1 - ω²Δρ Γ₀)")
print("\nA_θ (dilatation) = 1/(1 - 3T₁ - 2T₂ - T₃)")
print("\nA_e^(off) (off-diagonal shear, ε₁₂ etc.) = 1/(1 - 2T₂)")
print("\nA_e^(diag) (diagonal deviatoric, ε₁₁-ε₂₂ etc.) = 1/(1 - 2T₂ - T₃)")
print("\nNote: A_e^(off) ≠ A_e^(diag) because T₃ ≠ 0 — this is the cubic anisotropy!")


# ================================================================
# SECTION 8: Effective contrasts (cube) — scalar computation
# ================================================================
#
# The effective stiffness contrast Dc* relates scattered stress to
# the INCIDENT strain: Dc*_{mnlp} eps^0_{lp} = Dc_{mnuv} eps*_{uv}
# where eps* = A(eps^0) is the amplified internal strain.
#
# Decomposing eps^0 into the three cubic eigenspaces and applying
# the isotropic Dc_{mnuv} = Dl d_mn d_uv + Dm (d_mu d_nv + d_mv d_nu):
#
# Dc_{mnuv} eps*_{uv}
#   = Dl d_mn [A_theta theta^0]
#     + 2 Dm [A_theta (theta^0/3) d_mn + A_e_diag e^0(diag)_mn + A_e_off e^0(off)_mn]
#
# Re-collecting by the three eigenspace projections of eps^0:
#
#   Δμ*_off  = Δμ · A_e^(off)                    [off-diagonal shear]
#   Δμ*_diag = Δμ · A_e^(diag)                   [diagonal deviatoric]
#   Δλ*      = (Δλ + 2Δμ/3) A_θ - Δμ/3 (A_e^diag + A_e^off)  [volumetric]
#   Δρ*      = Δρ · A_u                           [density]

print("\n" + "-" * 70)
print("  Section 8: Effective contrasts (cube)")
print("-" * 70)

Drho_star_cube = Drho * amp_u_cube
Dmu_star_off = Dmu * amp_e_off_cube
Dmu_star_diag = Dmu * amp_e_diag_cube
Dlambda_star_cube = (
    (Dlambda + Rational(2, 3) * Dmu) * amp_theta_cube
    - Rational(1, 3) * Dmu * amp_e_diag_cube
    - Rational(1, 3) * Dmu * amp_e_off_cube
)

print(f"\n{'=' * 70}")
print("  ★ EFFECTIVE CONTRASTS (CUBE T-MATRIX) ★")
print(f"{'=' * 70}")

print("\nΔρ* = Δρ / (1 - ω²Δρ Γ₀)")
print("\nΔμ*_off = Δμ / (1 - 2T₂)")
print("\nΔμ*_diag = Δμ / (1 - 2T₂ - T₃)")
print("\nΔλ* = (Δλ + 2Δμ/3) A_θ − Δμ/3 (A_e^diag + A_e^off)")

# Cubic anisotropy of the effective shear modulus contrast
cubic_aniso = Dmu_star_diag - Dmu_star_off

print("\nCubic anisotropy: Δμ*_diag − Δμ*_off =")
print("  = Δμ [1/(1-2T₂-T₃) − 1/(1-2T₂)]")
print("  = Δμ · T₃ / [(1-2T₂)(1-2T₂-T₃)]")


# ================================================================
# SECTION 9: Born limit verification
# ================================================================

print("\n" + "-" * 70)
print("  Section 9: Born limit verification")
print("-" * 70)

print("\nBorn limit (a → 0): all T_i → 0, so all A → 1, Dc* → Dc.")
for name, expr, expected in [
    ("Δρ*", Drho_star_cube, Drho),
    ("Δλ*", Dlambda_star_cube, Dlambda),
    ("Δμ*_diag", Dmu_star_diag, Dmu),
    ("Δμ*_off", Dmu_star_off, Dmu),
]:
    lim = limit(expr, a, 0)
    ok = simplify(lim - expected) == 0
    print(f"  lim(a→0) {name} = {lim}  {'✓' if ok else '✗ FAILED'}")

lim_aniso = limit(cubic_aniso, a, 0)
print(f"  lim(a→0) cubic anisotropy = {lim_aniso}  {'✓' if lim_aniso == 0 else '✗'}")


# ================================================================
# SECTION 10: Numerical comparison
# ================================================================

print("\n" + "=" * 70)
print("  Section 10: Numerical comparison (cube vs sphere)")
print("=" * 70)

params = {
    alpha: 5000.0,
    beta: 3000.0,
    rho: 2500.0,
    omega: 2 * np.pi * 10.0,
    a: 10.0,
    Dlambda: 2.0e9,
    Dmu: 1.0e9,
    Drho: 100.0,
}

kPa = params[omega] * params[a] / params[alpha]
kSa = params[omega] * params[a] / params[beta]

print("\nParameters: Vp=5000, Vs=3000, rho=2500, f=10Hz, a=10m (cube half-width)")
print(f"k_P*a = {kPa:.4f},  k_S*a = {kSa:.4f}")
print(
    f"\nNote: cube side = 2a = 20m, sphere would have a_s = a*(6/pi)^(1/3) = {10 * (6 / np.pi) ** (1 / 3):.2f}m for equal volume"
)

print(f"\n{'Quantity':<20} {'Value':<40}")
print("-" * 60)

for name, expr_c in [
    ("Γ₀", Gamma0_cube),
    ("A", A_cube),
    ("B", B_cube),
    ("C_cubic", C_cube),
    ("T₁", T1_cube),
    ("T₂", T2_cube),
    ("T₃", T3_cube),
    ("A_u", amp_u_cube),
    ("A_θ", amp_theta_cube),
    ("A_e(off)", amp_e_off_cube),
    ("A_e(diag)", amp_e_diag_cube),
    ("Δρ*", Drho_star_cube),
    ("Δλ*", Dlambda_star_cube),
    ("Δμ*_diag", Dmu_star_diag),
    ("Δμ*_off", Dmu_star_off),
    ("cubic aniso", cubic_aniso),
]:
    val = complex(expr_c.subs(params))
    if abs(val.imag) < 1e-15 * max(abs(val.real), 1):
        print(f"  {name:<18} {val.real:.6e}")
    else:
        sign = "+" if val.imag >= 0 else "-"
        print(f"  {name:<18} {val.real:.6e} {sign} {abs(val.imag):.6e}i")


# ================================================================
# SECTION 11: LaTeX output
# ================================================================

print("\n" + "-" * 70)
print("  Section 11: LaTeX expressions")
print("-" * 70)

for name, expr in [
    (r"\Gamma_0^{\mathrm{cube}}", Gamma0_cube),
    (r"A^{\mathrm{cube}}", A_cube),
    (r"B^{\mathrm{cube}}", B_cube),
    (r"C^{\mathrm{cube}}", C_cube),
    (r"T_1^{\mathrm{cube}}", T1_cube),
    (r"T_2^{\mathrm{cube}}", T2_cube),
    (r"T_3^{\mathrm{cube}}", T3_cube),
    (r"\Delta\rho^{*,\mathrm{cube}}", Drho_star_cube),
    (r"\Delta\lambda^{*,\mathrm{cube}}", Dlambda_star_cube),
    (r"\Delta\mu^{*,\mathrm{diag}}", Dmu_star_diag),
    (r"\Delta\mu^{*,\mathrm{off}}", Dmu_star_off),
]:
    print(f"\n% {name}")
    print(f"{name} = {latex(expr)}")


print("\n" + "=" * 70)
print("  ★ Cube T-matrix computation complete ★")
print("=" * 70)
