"""
tmatrix_analytic.py
Analytical T-Matrix for a Small Spherical Scatterer
in an Isotropic Elastic Background Medium.

Evaluates Equation 8 (Lippmann-Schwinger) of Shekhar et al. (2023)
for a sphere of radius `a` with constant isotropic contrast
(Delta_lambda, Delta_mu, Delta_rho).

ALL integrals are performed ANALYTICALLY:
  - Angular integrals use exact identities for direction-cosine products
  - Radial integrals use SymPy's symbolic calculus
  - Results are exact closed-form expressions

The key efficiency trick: rather than differentiating G_{ij} in full
Cartesian coordinates (which creates enormous expressions), we work
with the radial decomposition G_{ij} = f(r) d_{ij} + g(r) n_i n_j
and analytically derive the angular structures of all derivatives
using the chain rule. This reduces everything to 1D radial integrals
that SymPy handles instantly.

Usage:
    python tmatrix_analytic.py
"""

import numpy as np
import sympy as sp
from sympy import (
    I,
    KroneckerDelta,
    Rational,
    diff,
    exp,
    integrate,
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

print("=" * 70)
print("  Analytical T-Matrix for Spherical Elastic Scatterer (SymPy)")
print("=" * 70)

# ================================================================
# SECTION 1: Green's tensor radial decomposition
# ================================================================
#
# G_{ij}(x) = f(r) delta_{ij} + g(r) n_i n_j
#
# Near-field:  -C/r * (delta_{ij} - 3 n_i n_j)
# P far-field: X(r)/r * n_i n_j
# S far-field: V(r)/r * (delta_{ij} - n_i n_j)
#
# f(r) = -C/r + V(r)/r
# g(r) = 3C/r + X(r)/r - V(r)/r

print("\n" + "-" * 70)
print("  Section 1: Radial functions f(r), g(r)")
print("-" * 70)

C_nf = (1 - beta**2 / alpha**2) / (8 * pi * rho * beta**2)
X_of_r = exp(I * omega * r / alpha) / (4 * pi * rho * alpha**2)
V_of_r = exp(I * omega * r / beta) / (4 * pi * rho * beta**2)

f = -C_nf / r + V_of_r / r
g = 3 * C_nf / r + X_of_r / r - V_of_r / r

f = simplify(f)
g = simplify(g)

print("\nf(r) =", f)
print("g(r) =", g)

# Derivatives needed for the second-derivative integral
fp = simplify(diff(f, r))
fpp = simplify(diff(f, r, 2))
gp = simplify(diff(g, r))
gpp = simplify(diff(g, r, 2))

print("\nf'(r)  =", fp)
print("g'(r)  =", gp)


# ================================================================
# SECTION 2: Gamma_0 — Volume integral of G_{ij}
# ================================================================
#
# After angular integration:
#   int G_{ij} d^3x = delta_{ij} * int_0^a r^2 [4*pi*f + (4*pi/3)*g] dr
#                    = delta_{ij} * Gamma_0

print("\n" + "-" * 70)
print("  Section 2: Gamma_0 (volume integral of Green's tensor)")
print("-" * 70)

Gamma0_integrand = simplify(r**2 * (4 * pi * f + Rational(4, 3) * pi * g))

print("\nGamma_0 integrand =", Gamma0_integrand)
print("\nIntegrating radially from 0 to a...")

Gamma0 = integrate(Gamma0_integrand, (r, 0, a))
Gamma0 = simplify(Gamma0)

print("\nGamma_0 (exact) =")
pprint(Gamma0)

Gamma0_small = simplify(series(Gamma0, a, 0, n=5).removeO())
print("\nGamma_0 (small sphere) =")
pprint(Gamma0_small)


# ================================================================
# SECTION 3: Second derivative integral — A and B coefficients
# ================================================================
#
# int G_{ij,kl} d^3x = A d_{ij} d_{kl} + B(d_{ik} d_{jl} + d_{il} d_{jk})
#
# Strategy: Use the chain-rule decomposition of G_{ij,kl} in terms
# of radial functions f, g and their derivatives. After angular
# integration, each term reduces to a known identity times a
# radial function. We need two independent contractions to find A, B.
#
# DERIVATION OF G_{ij,kl}:
# Starting from G_{ij} = f(r) d_{ij} + g(r) x_i x_j / r^2
#
# First derivative:
#   G_{ij,k} = [f' d_{ij} + g' n_i n_j] n_k
#            + (g/r)[d_{ik} n_j + d_{jk} n_i - 2 n_i n_j n_k]
#
# Second derivative (after taking d/dx_l of the above):
#   Terms group into 5 angular structures:
#     (1) d_{ij} d_{kl}    (2) d_{ij} n_k n_l
#     (3) d_{kl} n_i n_j   (4) (d_{ik}d_{jl} + perms)
#     (5) n_i n_j n_k n_l  (6) (d_{ik} n_j n_l + perms)
#
# After angular integration over the sphere:
#   int dOmega 1 = 4*pi
#   int dOmega n_i n_j = (4*pi/3) d_{ij}
#   int dOmega n_i n_j n_k n_l = (4*pi/15)(d_{ij}d_{kl}+d_{ik}d_{jl}+d_{il}d_{jk})
#
# We extract (3A+2B) and (A+4B) from two contractions.
#
# CONTRACTION 1: I_{iikl} = (3A + 2B) d_{kl}
# -----------------------------------------------
# G_{ii} = 3f + g = h(r), a function of r only.
# h'' = 3f'' + g''
# h' = 3f' + g'
# G_{ii,kl} = h''(r) n_k n_l + (h'/r)(d_{kl} - n_k n_l)
#
# int G_{ii,kl} d^3x = d_{kl} int_0^a r^2 [(4pi/3) h'' + (8pi/3) h'/r] dr
#
# CONTRACTION 2: I_{ijjl} = (A + 4B) d_{il}
# -----------------------------------------------
# G_{ij,j} = [f' + g' + 2g/r] n_i  (divergence structure)
#          = q(r) n_i
# G_{ij,jl} = q'(r) n_i n_l + (q/r)(d_{il} - n_i n_l)
#
# int G_{ij,jl} d^3x = d_{il} int_0^a r^2 [(4pi/3) q' + (8pi/3) q/r] dr

print("\n" + "-" * 70)
print("  Section 3: A, B from second-derivative integral")
print("-" * 70)

# ---- Contraction 1: trace over (i,j) ----
h = 3 * f + g  # = G_{ii}/d_{ii} at fixed r
h = simplify(h)
hp = simplify(diff(h, r))
hpp = simplify(diff(h, r, 2))

print("\nh(r) = 3f + g =", h)

# Radial integrand for contraction 1
integrand_C1 = r**2 * (Rational(4, 3) * pi * hpp + Rational(8, 3) * pi * hp / r)
integrand_C1 = simplify(integrand_C1)

print("\nContraction 1 integrand =", integrand_C1)
print("Integrating...")

C1 = integrate(integrand_C1, (r, 0, a))
C1 = simplify(C1)
print("\n3A + 2B =")
pprint(C1)

# ---- Contraction 2: divergence structure ----
q = fp + gp + 2 * g / r  # = coefficient of n_i in G_{ij,j}
q = simplify(q)
qp = simplify(diff(q, r))

print("\nq(r) = f' + g' + 2g/r =", q)

# Radial integrand for contraction 2
integrand_C2 = r**2 * (Rational(4, 3) * pi * qp + Rational(8, 3) * pi * q / r)
integrand_C2 = simplify(integrand_C2)

print("\nContraction 2 integrand =", integrand_C2)
print("Integrating...")

C2 = integrate(integrand_C2, (r, 0, a))
C2 = simplify(C2)
print("\nA + 4B =")
pprint(C2)

# ---- Solve for A and B ----
A_s, B_s = symbols("A B")
sol = sp.solve([3 * A_s + 2 * B_s - C1, A_s + 4 * B_s - C2], [A_s, B_s])

A_exact = simplify(sol[A_s])
B_exact = simplify(sol[B_s])

print("\n" + "=" * 70)
print("  ★ EXACT CLOSED-FORM RESULTS ★")
print("=" * 70)
print("\nA =")
pprint(A_exact)
print("\nB =")
pprint(B_exact)

A_small = simplify(series(A_exact, a, 0, n=4).removeO())
B_small = simplify(series(B_exact, a, 0, n=4).removeO())
print("\nA (small sphere) =")
pprint(A_small)
print("\nB (small sphere) =")
pprint(B_small)


# ================================================================
# SECTION 4: T-matrix coupling coefficients
# ================================================================
# T_{mnlp} = sum_{jk} S_{mnjk} Dc_{jklp}
# where S_{mnjk} = (1/2)(I_{mjkn} + I_{njkm})
#
# For isotropic structures, T_{mnlp} = T1 d_{mn}d_{lp} + T2(d_{ml}d_{np}+d_{mp}d_{nl})

print("\n" + "-" * 70)
print("  Section 4: T-matrix coefficients T_1, T_2")
print("-" * 70)

d = KroneckerDelta


def S_func(m, n, j, k, Av, Bv):
    return (
        Av / 2 * (d(m, j) * d(k, n) + d(n, j) * d(k, m))
        + Bv * d(m, n) * d(j, k)
        + Bv / 2 * (d(m, k) * d(j, n) + d(n, k) * d(j, m))
    )


def Dc_func(j, k, l, p):
    return Dlambda * d(j, k) * d(l, p) + Dmu * (d(j, l) * d(k, p) + d(j, p) * d(k, l))


def T_func(m, n, l, p, Av, Bv):
    return sum(
        S_func(m, n, j, k, Av, Bv) * Dc_func(j, k, l, p)
        for j in range(3)
        for k in range(3)
    )


T1 = simplify(T_func(0, 0, 1, 1, A_exact, B_exact))
T2 = simplify(T_func(0, 1, 0, 1, A_exact, B_exact))

# Verify isotropic structure
T1111 = simplify(T_func(0, 0, 0, 0, A_exact, B_exact))
assert simplify(T1111 - T1 - 2 * T2) == 0, "T-tensor structure check FAILED"
print("T-tensor isotropic structure verified ✓")

print("\nT_1 (volumetric) =")
pprint(T1)
print("\nT_2 (shear) =")
pprint(T2)


# ================================================================
# SECTION 5: Self-consistent amplification factors
# ================================================================

print("\n" + "-" * 70)
print("  Section 5: Amplification factors")
print("-" * 70)

amp_u = simplify(1 / (1 - omega**2 * Drho * Gamma0))
amp_theta = simplify(1 / (1 - 3 * T1 - 2 * T2))
amp_e = simplify(1 / (1 - 2 * T2))

print("\nA_u (displacement) = 1 / (1 - ω² Δρ Γ₀) =")
pprint(amp_u)
print("\nA_θ (dilatation) = 1 / (1 - 3T₁ - 2T₂) =")
pprint(amp_theta)
print("\nA_e (deviatoric) = 1 / (1 - 2T₂) =")
pprint(amp_e)


# ================================================================
# SECTION 6: Effective contrasts — THE T-MATRIX
# ================================================================

print("\n" + "=" * 70)
print("  ★ Section 6: EFFECTIVE CONTRASTS (T-MATRIX) ★")
print("=" * 70)

Drho_star = simplify(Drho * amp_u)
Dmu_star = simplify(Dmu * amp_e)
Dlambda_star = simplify(
    Dlambda * amp_theta + Rational(2, 3) * Dmu * (amp_theta - amp_e)
)

print("""
Scattered field at observation point x:

  u^scat_i(x) = V × {
      H_{ijk}(x, x_s) [Δλ* θ⁰ δ_{jk} + 2Δμ* ε⁰_{jk}]
    + ω² G_{ij}(x, x_s) Δρ* u⁰_j
  }

where V = (4/3)πa³ and:
""")

print("Δρ* =")
pprint(Drho_star)
print("\nΔλ* =")
pprint(Dlambda_star)
print("\nΔμ* =")
pprint(Dmu_star)


# ================================================================
# SECTION 7: Born limit verification
# ================================================================

print("\n" + "-" * 70)
print("  Section 7: Born approximation verification")
print("-" * 70)

born_checks = {
    "Δρ*": (limit(Drho_star, a, 0), Drho),
    "Δλ*": (limit(Dlambda_star, a, 0), Dlambda),
    "Δμ*": (limit(Dmu_star, a, 0), Dmu),
}
for name, (val, expected) in born_checks.items():
    ok = simplify(val - expected) == 0
    print(f"  lim(a→0) {name} = {val}  {'✓' if ok else '✗ FAILED'}")


# ================================================================
# SECTION 8: Rayleigh (small sphere) limit
# ================================================================

print("\n" + "-" * 70)
print("  Section 8: Rayleigh limit expansions")
print("-" * 70)

for name, expr in [
    ("T₁", T1),
    ("T₂", T2),
    ("A_u", amp_u),
    ("A_θ", amp_theta),
    ("A_e", amp_e),
]:
    n_terms = 5 if "u" in name else 4
    s = simplify(series(expr, a, 0, n=n_terms).removeO())
    print(f"\n{name} ≈")
    pprint(s)


# ================================================================
# SECTION 9: Numerical evaluation
# ================================================================

print("\n" + "=" * 70)
print("  Section 9: Numerical evaluation")
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

lambda_bg = params[rho] * (params[alpha] ** 2 - 2 * params[beta] ** 2)
mu_bg = params[rho] * params[beta] ** 2
kPa = params[omega] * params[a] / params[alpha]
kSa = params[omega] * params[a] / params[beta]

print(f"\nBackground: λ={lambda_bg / 1e9:.1f} GPa, μ={mu_bg / 1e9:.1f} GPa")
print(f"k_P·a = {kPa:.4f},  k_S·a = {kSa:.4f}")

results_num = {}
for name, expr in [
    ("Γ₀", Gamma0),
    ("A", A_exact),
    ("B", B_exact),
    ("T₁", T1),
    ("T₂", T2),
    ("A_u", amp_u),
    ("A_θ", amp_theta),
    ("A_e", amp_e),
    ("Δρ*", Drho_star),
    ("Δλ*", Dlambda_star),
    ("Δμ*", Dmu_star),
]:
    val = complex(expr.subs(params))
    results_num[name] = val
    if abs(val.imag) < 1e-15 * max(abs(val.real), 1):
        print(f"  {name:4s} = {val.real:.6e}")
    else:
        print(f"  {name:4s} = {val}")


# ================================================================
# SECTION 10: Lambdified functions for fast NumPy evaluation
# ================================================================

print("\n" + "-" * 70)
print("  Section 10: Lambdified functions")
print("-" * 70)

param_list = [omega, a, alpha, beta, rho, Dlambda, Dmu, Drho]

lambdified = {}
for name, expr in [
    ("Gamma0", Gamma0),
    ("A", A_exact),
    ("B", B_exact),
    ("T1", T1),
    ("T2", T2),
    ("amp_u", amp_u),
    ("amp_theta", amp_theta),
    ("amp_e", amp_e),
    ("Drho_star", Drho_star),
    ("Dlambda_star", Dlambda_star),
    ("Dmu_star", Dmu_star),
]:
    lambdified[name] = sp.lambdify(param_list, expr, modules="numpy")

# Verify
args = [params[p] for p in param_list]
G0_test = lambdified["Gamma0"](*args)
G0_sym = complex(Gamma0.subs(params))
assert abs(G0_test - G0_sym) < 1e-20 * max(abs(G0_sym), 1), "Lambdify mismatch!"
print("Lambdified functions verified ✓")
print("\nUsage:")
print("  args = [omega, a, alpha, beta, rho, Dlambda, Dmu, Drho]")
print("  drho_eff = lambdified['Drho_star'](*args)")


# ================================================================
# SECTION 11: LaTeX export
# ================================================================

print("\n" + "-" * 70)
print("  Section 11: LaTeX expressions")
print("-" * 70)

for name, expr in [
    ("\\Gamma_0", Gamma0),
    ("A", A_exact),
    ("B", B_exact),
    ("T_1", T1),
    ("T_2", T2),
    ("A_u", amp_u),
    ("\\Delta\\rho^*", Drho_star),
    ("\\Delta\\lambda^*", Dlambda_star),
    ("\\Delta\\mu^*", Dmu_star),
]:
    print(f"\n% {name}")
    print(f"{name} = {latex(expr)}")


# ================================================================
# Store everything
# ================================================================

tmatrix_results = {
    "symbolic": {
        "Gamma0": Gamma0,
        "A": A_exact,
        "B": B_exact,
        "T1": T1,
        "T2": T2,
        "amp_u": amp_u,
        "amp_theta": amp_theta,
        "amp_e": amp_e,
        "Drho_star": Drho_star,
        "Dlambda_star": Dlambda_star,
        "Dmu_star": Dmu_star,
    },
    "lambdified": lambdified,
    "numerical": results_num,
}

print("\n" + "=" * 70)
print("  ★ Computation complete — all results in `tmatrix_results` ★")
print("=" * 70)
