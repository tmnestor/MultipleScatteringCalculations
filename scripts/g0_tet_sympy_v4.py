"""Symbolic computation of g0_tet = ∫_tet 1/|x| dV.

Clean approach using the face decomposition and careful manual integration.

Key result: g0 = ∫ 1/|x| dV = (1/2) ∮ (x̂·n̂) dS = 8·J
where J = ∫∫_{simplex} ds dt / |P(s,t)|
and P(s,t) = (1-2s-2t, -1+2t, -1+2s), |P|² = 8s²+8st-8s+8t²-8t+3.

After first integration (over t):
    J = ∫₀¹ (√2/2)·asinh(√2(1-s)/√(6s²-4s+1)) ds

After IBP (u = 1-s):
    J = (√2/2)·[asinh(√2) - √2·(-2I₁/3 + I₂)]
where I₁ = ∫₀¹ du/√(8u²-8u+3) = asinh(√2)/√2
and I₂ involves ∫ rational/√(quadratic) over [0,1].

The hardest integral reduces via complex partial fractions to
logarithms and arctangents of algebraic numbers.
"""

import math

import mpmath
from scipy.integrate import dblquad, nquad, quad
from sympy import N as Neval
from sympy import acosh, asinh
from sympy import log as Log
from sympy import pi as Pi
from sympy import sqrt as Sqrt

mpmath.mp.dps = 50

# ============================================================
# Step 0: High-precision numerical computation
# ============================================================
print("=" * 60)
print("STEP 0: High-precision numerical g0")
print("=" * 60)


def tet_integrand_scipy(l3v, l2v, l1v):
    """Integrand for g0 = 16·∫∫∫ 1/|x| dl3 dl2 dl1."""
    x = [1 - 2 * l2v - 2 * l3v, 1 - 2 * l1v - 2 * l3v, 1 - 2 * l1v - 2 * l2v]
    return 1.0 / math.sqrt(sum(xi**2 for xi in x))


# Use nested quad for precision
g0_scipy, err = nquad(
    tet_integrand_scipy,
    [
        lambda l2v, l1v: [0, 1 - l1v - l2v],
        lambda l1v: [0, 1 - l1v],
        [0, 1],
    ],
)
g0_scipy *= 16
print(f"g0 (scipy nquad)  = {g0_scipy:.15f}")
print("Expected          = 4.389580813389239")
print()

# Also compute via 2D face integral
# g0 = 8·J where J = ∫∫ 1/|P| ds dt over simplex
# |P|² = 8s²+8st-8s+8t²-8t+3
J_scipy, _ = dblquad(
    lambda t, s: 1.0 / math.sqrt(8 * s**2 + 8 * s * t - 8 * s + 8 * t**2 - 8 * t + 3),
    0,
    1,
    0,
    lambda s: 1 - s,
)
g0_from_J = 8 * J_scipy
print(f"g0 (8·J_face)     = {g0_from_J:.15f}")

# And via 1D integral after t-integration
# J = ∫₀¹ (√2/2)·asinh(√2(1-s)/√(6s²-4s+1)) ds
J_1d, _ = quad(
    lambda s: (
        math.sqrt(2)
        / 2
        * math.asinh(math.sqrt(2) * (1 - s) / math.sqrt(6 * s**2 - 4 * s + 1))
    ),
    0,
    1,
)
g0_from_1d = 8 * J_1d
print(f"g0 (8·J_1d)       = {g0_from_1d:.15f}")

# ============================================================
# Step 1: Compute the 1D integral via IBP
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: IBP decomposition")
print("=" * 60)

# After u = 1-s:
# K = ∫₀¹ asinh(√2·u/√(6u²-8u+3)) du
# IBP: K = asinh(√2) - ∫₀¹ u·f'/√(1+f²) du
# where f(u) = √2·u/√(6u²-8u+3)
# f'(u) = √2·(3-4u)/(6u²-8u+3)^{3/2}
# 1+f² = (8u²-8u+3)/(6u²-8u+3)
# u·f'/√(1+f²) = √2·u(3-4u)/((6u²-8u+3)·√(8u²-8u+3))

# Decompose: u(3-4u)/(6u²-8u+3) = -2/3 + (-7u/3+2)/(6u²-8u+3)
# IBP integral = √2·∫₀¹ [-2/3·1/√Q + (-7u/3+2)/(P·√Q)] du
# where P = 6u²-8u+3, Q = 8u²-8u+3

# Term 1: I₁ = ∫₀¹ du/√(8u²-8u+3)
# = ∫ du/√(8(u-1/2)²+1) = (1/√8)·[asinh(2√2(u-1/2))]₀¹
# = (1/2√2)·[asinh(√2)-asinh(-√2)] = (2/2√2)·asinh(√2) = asinh(√2)/√2

I1 = math.asinh(math.sqrt(2)) / math.sqrt(2)
print(f"I₁ = asinh(√2)/√2 = {I1:.15f}")

# Term 2: I₂ = ∫₀¹ (-7u/3+2)/(P·√Q) du = ∫₀¹ (-7u+6)/(3P·√Q) du
I2, _ = quad(
    lambda u: (
        (-7 * u + 6) / (3 * (6 * u**2 - 8 * u + 3) * math.sqrt(8 * u**2 - 8 * u + 3))
    ),
    0,
    1,
)
print(f"I₂ = {I2:.15f}")

ibp_int = math.sqrt(2) * (-2 / 3 * I1 + I2)
print(f"IBP integral = √2(-2I₁/3 + I₂) = {ibp_int:.15f}")

K = math.asinh(math.sqrt(2)) - ibp_int
print(f"K = asinh(√2) - IBP = {K:.15f}")

g0_check = 8 * math.sqrt(2) / 2 * K
print(f"g0 = 8·(√2/2)·K = 4√2·K = {g0_check:.15f}")

# ============================================================
# Step 2: Compute I₂ analytically using complex partial fractions
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Analytical computation of I₂")
print("=" * 60)

# I₂ = ∫₀¹ (-7u+6)/(3P·√Q) du
# P = 6u²-8u+3, Q = 8u²-8u+3
#
# Substitution v = u-1/2: P(v) = 6v²-2v+1/2, Q(v) = 8v²+1
# u = v+1/2, -7u+6 = -7v+5/2
# I₂ = ∫_{-1/2}^{1/2} (-7v+5/2)/(3(6v²-2v+1/2)√(8v²+1)) dv
#
# Split (-7v+5/2) = odd + even part:
# = 5/2 - 7v
# Over symmetric limits [-1/2, 1/2], the odd part (-7v) times even (1/(P·√Q))
# only contributes if P is not even. P(v) = 6v²-2v+1/2 has odd part -2v.
#
# Actually P(-v) = 6v²+2v+1/2, so P is not even. No simplification.

# Substitution w = 2√2·v: v = w/(2√2), Q = w²+1, √Q = √(w²+1)
# P = 6w²/8 - 2w/(2√2) + 1/2 = 3w²/4 - w/√2 + 1/2 = (3w²-2√2w+2)/4
# -7v+5/2 = -7w/(2√2)+5/2 = (-7w+5√2)/(2√2)
# dv = dw/(2√2)
# I₂ = ∫_{-√2}^{√2} [(-7w+5√2)/(2√2)] / [3·(3w²-2√2w+2)/4·√(w²+1)] · dw/(2√2)
# = ∫_{-√2}^{√2} (-7w+5√2)·4 / (2√2·3(3w²-2√2w+2)·√(w²+1)·2√2) dw
# = ∫ (-7w+5√2) / (6(3w²-2√2w+2)·√(w²+1)) dw

# Let me verify numerically
I2_check, _ = quad(
    lambda w: (
        (-7 * w + 5 * math.sqrt(2))
        / (6 * (3 * w**2 - 2 * math.sqrt(2) * w + 2) * math.sqrt(w**2 + 1))
    ),
    -math.sqrt(2),
    math.sqrt(2),
)
print(f"I₂ (w-sub check) = {I2_check:.15f}")
print(f"I₂ (direct)      = {I2:.15f}")
print(f"Match: {abs(I2 - I2_check) < 1e-12}")
print()

# Factor 3w²-2√2w+2 over C:
# roots: w = (2√2 ± √(8-24))/6 = (2√2 ± 4i)/6 = (√2 ± 2i)/3
# 3w²-2√2w+2 = 3(w - (√2+2i)/3)(w - (√2-2i)/3)

# Partial fractions: 1/((w-α)(w-ᾱ)) = [1/(α-ᾱ)]·[1/(w-α) - 1/(w-ᾱ)]
# α-ᾱ = 4i/3
# 1/P_w = (3/(4i))·[1/(w-α) - 1/(w-ᾱ)] / 3 = (1/(4i))·[1/(w-α) - 1/(w-ᾱ)]
# Wait: P_w = 3(w-α)(w-ᾱ)
# 1/P_w = 1/(3(w-α)(w-ᾱ)) = [1/(3(α-ᾱ))]·[1/(w-α) - 1/(w-ᾱ)]
# = [1/(3·4i/3)]·[...] = [1/(4i)]·[1/(w-α) - 1/(w-ᾱ)]
# = (-i/4)·[1/(w-α) - 1/(w-ᾱ)]

# (-7w+5√2)/(6P_w·√Q) = (-7w+5√2)·(-i/4)/(6√Q) · [1/(w-α) - 1/(w-ᾱ)]
# = i(7w-5√2)/(24√Q) · [1/(w-α) - 1/(w-ᾱ)]

# Now (7w-5√2)/(w-α) = 7 + (7α-5√2)/(w-α) (long division)
# 7α-5√2 = 7(√2+2i)/3 - 5√2 = (7√2+14i-15√2)/3 = (-8√2+14i)/3
# Similarly (7w-5√2)/(w-ᾱ) = 7 + (7ᾱ-5√2)/(w-ᾱ)
# 7ᾱ-5√2 = (-8√2-14i)/3

# The "7" terms cancel in [1/(w-α) - 1/(w-ᾱ)] since they give 7·[1/(w-α)-1/(w-ᾱ)]
# which integrates to 7·log ratio.

# Actually let me be more careful. We have:
# (7w-5√2)·[1/(w-α) - 1/(w-ᾱ)]
# = (7w-5√2)/(w-α) - (7w-5√2)/(w-ᾱ)
# = [7 + c/(w-α)] - [7 + c̄/(w-ᾱ)] where c = 7α-5√2
# = c/(w-α) - c̄/(w-ᾱ)

# So I₂ = (i/24)·∫ [c/(w-α) - c̄/(w-ᾱ)] / √(w²+1) dw

# Each term: ∫ dw/((w-z)√(w²+1)) where z is complex
# Use w = sinh(θ): √(w²+1) = cosh(θ), dw = cosh(θ)dθ
# ∫ cosh(θ)/((sinh(θ)-z)cosh(θ)) dθ = ∫ dθ/(sinh(θ)-z)

# Let p = e^θ: sinh(θ) = (p-1/p)/2
# ∫ dθ/((p-1/p)/2 - z) = ∫ 2p dp/(p(p²-1-2zp)) = 2∫ dp/(p²-2zp-1)
# = 2∫ dp/((p-z)²-(z²+1))
# Let β² = z²+1 (with appropriate branch):
# = 2∫ dp/((p-z-β)(p-z+β))
# = (2/(2β))·ln((p-z-β)/(p-z+β)) = (1/β)·ln((p-z-β)/(p-z+β))

# For z = α = (√2+2i)/3:
# z² = (√2+2i)²/9 = (2-4+4i√2)/9 = (-2+4i√2)/9
# z²+1 = (7+4i√2)/9
# β = √((7+4i√2)/9) = (2√2+i)/3  [denesting!]

print("Complex decomposition:")
sqrt2 = mpmath.sqrt(2)
alpha = (sqrt2 + 2j) / 3
beta_val = (2 * sqrt2 + 1j) / 3
print(f"α = (√2+2i)/3 = {alpha}")
print(f"β = √(1+α²) = (2√2+i)/3 = {beta_val}")
print(f"Check β² = {beta_val**2}, 1+α² = {1 + alpha**2}")
print(f"Match: {abs(beta_val**2 - (1 + alpha**2)) < 1e-40}")
print()

c_val = 7 * alpha - 5 * sqrt2
print(f"c = 7α-5√2 = {c_val}")
print(f"  = (-8√2+14i)/3 = {(-8 * sqrt2 + 14j) / 3}")

# p = e^θ = e^{asinh(w)} = w + √(w²+1)
# Limits: w = -√2 → p = -√2+√3; w = √2 → p = √2+√3
p_lower = -sqrt2 + mpmath.sqrt(3)
p_upper = sqrt2 + mpmath.sqrt(3)
print(f"\np_lower = √3-√2 = {p_lower}")
print(f"p_upper = √2+√3 = {p_upper}")

# α+β = (√2+2i)/3 + (2√2+i)/3 = (3√2+3i)/3 = √2+i
# α-β = (√2+2i)/3 - (2√2+i)/3 = (-√2+i)/3
apb = sqrt2 + 1j
amb = (-sqrt2 + 1j) / 3
print(f"\nα+β = √2+i = {apb}")
print(f"α-β = (-√2+i)/3 = {amb}")

# For z=α: integral = (1/β)·[ln((p-α-β)/(p-α+β))]_{p_lower}^{p_upper}
# = (1/β)·[ln(p_upper-α-β) - ln(p_upper-α+β) - ln(p_lower-α-β) + ln(p_lower-α+β)]

r1u = p_upper - apb  # p_upper - α - β = √2+√3-√2-i = √3-i
r2u = p_upper - amb  # p_upper - α + β = √2+√3+√2/3-i/3 = (4√2+3√3-i)/3
r1l = p_lower - apb  # p_lower - α - β = √3-√2-√2-i = √3-2√2-i
r2l = p_lower - amb  # p_lower - α + β = √3-√2+√2/3-i/3 = (3√3-2√2-i)/3

print(f"\nr1u = p_upper - (α+β) = {r1u}")
print("  = √3 - i")
print(f"r2u = p_upper - (α-β) = {r2u}")
print("  = (4√2+3√3-i)/3")
print(f"r1l = p_lower - (α+β) = {r1l}")
print("  = √3-2√2-i")
print(f"r2l = p_lower - (α-β) = {r2l}")
print("  = (3√3-2√2-i)/3")

# ∫_{-√2}^{√2} dw/((w-α)√(w²+1)) = (1/β)·ln[(r1u·r2l)/(r2u·r1l)]
# ... wait, it's (1/β)·[ln(r1u/r2u) - ln(r1l/r2l)]
# = (1/β)·ln[(r1u·r2l)/(r2u·r1l)]

ratio = (r1u * r2l) / (r2u * r1l)
ln_ratio = mpmath.log(ratio)
J_alpha = (1 / beta_val) * ln_ratio

print(f"\nratio = (r1u·r2l)/(r2u·r1l) = {ratio}")
print(f"ln(ratio) = {ln_ratio}")
print(f"J_α = (1/β)·ln(ratio) = {J_alpha}")

# For z=ᾱ: J_ᾱ = conj(J_α) since integrand and limits are real-symmetric
J_alpha_bar = J_alpha.conjugate()
print(f"J_ᾱ = conj(J_α) = {J_alpha_bar}")

# I₂ = (i/24)·∫ [c/(w-α) - c̄/(w-ᾱ)] / √(w²+1) dw
# = (i/24)·[c·J_α - c̄·J_ᾱ]
# Note: c̄·J_ᾱ = conj(c·J_α)
# So c·J_α - conj(c·J_α) = 2i·Im(c·J_α)
# I₂ = (i/24)·2i·Im(c·J_α) = -2·Im(c·J_α)/24 = -Im(c·J_α)/12

cJ = c_val * J_alpha
print(f"\nc·J_α = {cJ}")
print(f"Im(c·J_α) = {cJ.imag}")

I2_analytical = -float(cJ.imag) / 12
print(f"\nI₂ = -Im(c·J_α)/12 = {I2_analytical:.15f}")
print(f"I₂ (numerical)      = {I2:.15f}")
print(f"Match: {abs(I2_analytical - I2) < 1e-10}")

# ============================================================
# Step 3: Expand everything symbolically
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Symbolic expansion")
print("=" * 60)

# c = (-8√2+14i)/3
# J_α = (1/β)·ln(ratio)
# β = (2√2+i)/3
# 1/β = 3/(2√2+i) = 3(2√2-i)/((2√2)²+1) = 3(2√2-i)/9 = (2√2-i)/3

# c·(1/β) = [(-8√2+14i)/3]·[(2√2-i)/3]
# = [(-8√2+14i)(2√2-i)]/9
# Expand: -8√2·2√2 = -32
#         -8√2·(-i) = 8i√2
#         14i·2√2 = 28i√2
#         14i·(-i) = 14
# Sum: (-32+14) + i(8√2+28√2) = -18 + 36i√2
# c/β = (-18+36i√2)/9 = -2+4i√2

c_over_beta = (-18 + 36j * float(sqrt2)) / 9
print(f"c/β = (-8√2+14i)(2√2-i)/9 = -2+4i√2 = {c_over_beta}")

# So c·J_α = (c/β)·ln(ratio) = (-2+4i√2)·ln(ratio)
# I₂ = -Im[(-2+4i√2)·ln(ratio)] / 12

# ln(ratio) = ln|ratio| + i·arg(ratio) = A + iB
A_ln = float(mpmath.log(abs(ratio)))
B_ln = float(mpmath.arg(ratio))
print(f"\nln(ratio) = {A_ln:.15f} + i·{B_ln:.15f}")

# (-2+4i√2)·(A+iB) = (-2A-4√2·B) + i(4√2·A - 2B)
# Im = 4√2·A - 2B
Im_part = 4 * math.sqrt(2) * A_ln - 2 * B_ln
print(f"Im[(-2+4i√2)·ln(ratio)] = 4√2·A - 2B = {Im_part:.15f}")
print(f"I₂ = -{Im_part}/12 = {-Im_part / 12:.15f}")
print(f"I₂ (check) = {I2:.15f}")
print()

# Now we need the EXACT values of A = ln|ratio| and B = arg(ratio).

# ratio = (r1u·r2l)/(r2u·r1l)
# r1u = √3-i, |r1u|² = 3+1 = 4, arg(r1u) = atan(-1/√3) = -π/6
# r1l = √3-2√2-i, |r1l|² = (√3-2√2)²+1 = 3-4√6+8+1 = 12-4√6
# r2u = (4√2+3√3-i)/3, |r2u|² = ((4√2+3√3)²+1)/9 = (32+24√6+27+1)/9 = (60+24√6)/9
# r2l = (3√3-2√2-i)/3, |r2l|² = ((3√3-2√2)²+1)/9 = (27-12√6+8+1)/9 = (36-12√6)/9

print("Exact moduli squared:")
print("|r1u|² = 3+1 = 4")
print("|r1l|² = (√3-2√2)²+1 = 12-4√6")
r1l_sq = 12 - 4 * math.sqrt(6)
print(f"       = {r1l_sq:.10f}")

print("|r2u|² = ((4√2+3√3)²+1)/9 = (60+24√6)/9")
r2u_sq = (60 + 24 * math.sqrt(6)) / 9
print(f"       = {r2u_sq:.10f}")

print("|r2l|² = ((3√3-2√2)²+1)/9 = (36-12√6)/9 = (12-4√6)/3")
r2l_sq = (36 - 12 * math.sqrt(6)) / 9
print(f"       = {r2l_sq:.10f}")

# |ratio|² = |r1u|²·|r2l|² / (|r2u|²·|r1l|²)
# = 4·(12-4√6)/3 / ((60+24√6)/9·(12-4√6))
# = 4/(3) · 9/(60+24√6)  [the (12-4√6) cancels!]
# = 12/(60+24√6) = 12/(12(5+2√6)) = 1/(5+2√6)
# Rationalize: (5-2√6)/((5+2√6)(5-2√6)) = (5-2√6)/(25-24) = 5-2√6

print("\n|ratio|² = 4·(12-4√6)/3 / ((60+24√6)/9·(12-4√6))")
print("         = 4·9 / (3·(60+24√6))")
print("         = 12/(60+24√6)")
print("         = 1/(5+2√6)")
print("         = 5-2√6")

ratio_sq = 5 - 2 * math.sqrt(6)
print(f"         = {ratio_sq:.10f}")
print(f"Check: |ratio|² = {abs(ratio) ** 2}")

# ln|ratio| = (1/2)·ln(5-2√6)
# Note: 5-2√6 = (√3-√2)² since (√3-√2)² = 3-2√6+2 = 5-2√6!
# So ln|ratio| = (1/2)·ln((√3-√2)²) = ln(√3-√2)  [negative since √3-√2 > 0]
# = ln(√3-√2)
# Also: 1/(√3-√2) = √3+√2 (rationalizing), so ln(√3-√2) = -ln(√3+√2)

print("\nln|ratio| = ln(√3-√2) = -ln(√3+√2)")
print(f"  = {math.log(math.sqrt(3) - math.sqrt(2)):.15f}")
print(f"  = {-math.log(math.sqrt(3) + math.sqrt(2)):.15f}")
print(f"Check: = {A_ln:.15f}")
print()

# Now for arg(ratio):
# arg(ratio) = arg(r1u) + arg(r2l) - arg(r2u) - arg(r1l)
# arg(r1u) = arg(√3-i) = -atan(1/√3) = -π/6
arg_r1u = math.atan2(-1, math.sqrt(3))
print(
    f"arg(r1u) = arg(√3-i) = atan2(-1,√3) = {arg_r1u:.15f} = -π/6 = {-math.pi / 6:.15f}"
)

# arg(r2l) = arg((3√3-2√2-i)/3) = arg(3√3-2√2-i) = atan(-1/(3√3-2√2))
arg_r2l = math.atan2(-1, 3 * math.sqrt(3) - 2 * math.sqrt(2))
print(f"arg(r2l) = atan2(-1, 3√3-2√2) = {arg_r2l:.15f}")

# arg(r2u) = arg((4√2+3√3-i)/3) = arg(4√2+3√3-i) = atan(-1/(4√2+3√3))
arg_r2u = math.atan2(-1, 4 * math.sqrt(2) + 3 * math.sqrt(3))
print(f"arg(r2u) = atan2(-1, 4√2+3√3) = {arg_r2u:.15f}")

# arg(r1l) = arg(√3-2√2-i): note √3-2√2 < 0 (√3≈1.73, 2√2≈2.83)
# So real part is negative, imaginary is negative → third quadrant
# arg = -π + atan(1/(2√2-√3))
arg_r1l = math.atan2(-1, math.sqrt(3) - 2 * math.sqrt(2))
print(f"arg(r1l) = atan2(-1, √3-2√2) = {arg_r1l:.15f}")
print(f"  (√3-2√2 = {math.sqrt(3) - 2 * math.sqrt(2):.6f} < 0, so in Q3)")

arg_total = arg_r1u + arg_r2l - arg_r2u - arg_r1l
print(f"\narg(ratio) = {arg_total:.15f}")
print(f"Check B   = {B_ln:.15f}")
print(f"Match: {abs(arg_total - B_ln) < 1e-10}")

# ============================================================
# Step 4: Express arg(ratio) in closed form
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Closed form for arg(ratio)")
print("=" * 60)

# arg(r1u) = -π/6 (exact)
# arg(r1l) = atan2(-1, √3-2√2) = -π + atan(1/(2√2-√3))
#   since √3-2√2 < 0 and imaginary part -1 < 0

# For the other two, use: atan(-1/x) = -atan(1/x) for x > 0
# arg(r2l) = -atan(1/(3√3-2√2))
# arg(r2u) = -atan(1/(4√2+3√3))

# arg(ratio) = -π/6 + (-atan(1/(3√3-2√2))) - (-atan(1/(4√2+3√3))) - (-π+atan(1/(2√2-√3)))
# = -π/6 - atan(1/(3√3-2√2)) + atan(1/(4√2+3√3)) + π - atan(1/(2√2-√3))
# = 5π/6 + atan(1/(4√2+3√3)) - atan(1/(3√3-2√2)) - atan(1/(2√2-√3))

print("arg(ratio) = 5π/6 + atan(1/(4√2+3√3)) - atan(1/(3√3-2√2)) - atan(1/(2√2-√3))")
arg_check = (
    5 * math.pi / 6
    + math.atan(1 / (4 * math.sqrt(2) + 3 * math.sqrt(3)))
    - math.atan(1 / (3 * math.sqrt(3) - 2 * math.sqrt(2)))
    - math.atan(1 / (2 * math.sqrt(2) - math.sqrt(3)))
)
print(f"  = {arg_check:.15f}")
print(f"  Check: {arg_total:.15f}")
print(f"  Match: {abs(arg_check - arg_total) < 1e-10}")

# Try to simplify the atan terms using addition formulas
# Let a = 4√2+3√3, b = 3√3-2√2, c = 2√2-√3
# atan(1/a) - atan(1/b) = atan((1/a-1/b)/(1+1/(ab))) = atan((b-a)/(ab+1))
# b-a = 3√3-2√2-4√2-3√3 = -6√2
# ab = (4√2+3√3)(3√3-2√2) = 12√6-8·2+9·3-6√6 = 6√6-16+27 = 6√6+11
# ab+1 = 6√6+12 = 6(√6+2)
# atan(1/a)-atan(1/b) = atan(-6√2/(6(√6+2))) = atan(-√2/(√6+2))
# = -atan(√2/(√6+2))
# Rationalize: √2(√6-2)/((√6+2)(√6-2)) = √2(√6-2)/(6-4) = √2(√6-2)/2
# = (√12-2√2)/2 = (2√3-2√2)/2 = √3-√2
# So atan(1/a)-atan(1/b) = -atan(√3-√2)

print("\natan(1/(4√2+3√3)) - atan(1/(3√3-2√2)) = -atan(√3-√2)")
at_diff = math.atan(1 / (4 * math.sqrt(2) + 3 * math.sqrt(3))) - math.atan(
    1 / (3 * math.sqrt(3) - 2 * math.sqrt(2))
)
at_neg = -math.atan(math.sqrt(3) - math.sqrt(2))
print(f"  LHS = {at_diff:.15f}")
print(f"  RHS = {at_neg:.15f}")
print(f"  Match: {abs(at_diff - at_neg) < 1e-12}")

# So arg(ratio) = 5π/6 - atan(√3-√2) - atan(1/(2√2-√3))
# = 5π/6 - atan(√3-√2) - atan(1/c) where c = 2√2-√3

# Note: 1/(2√2-√3) = (2√2+√3)/((2√2)²-(√3)²) = (2√2+√3)/(8-3) = (2√2+√3)/5
# And √3-√2 = 1/(√3+√2) (reciprocal)

# atan(√3-√2) + atan(1/(2√2-√3)):
# Use atan(x)+atan(y) = atan((x+y)/(1-xy)) [+nπ if xy>1]
# x = √3-√2, y = (2√2+√3)/5
# xy = (√3-√2)(2√2+√3)/5 = (2√6+3-2·2-√6)/5 = (√6-1)/5
# x+y = √3-√2+(2√2+√3)/5 = (5√3-5√2+2√2+√3)/5 = (6√3-3√2)/5 = 3(2√3-√2)/5
# 1-xy = 1-(√6-1)/5 = (6-√6)/5
# atan((x+y)/(1-xy)) = atan(3(2√3-√2)/(6-√6))
# = atan(3(2√3-√2)/(6-√6))
# Rationalize: multiply by (6+√6)/(6+√6):
# numer: 3(2√3-√2)(6+√6) = 3(12√3+2√18-6√2-√12) = 3(12√3+6√2-6√2-2√3) = 3·10√3 = 30√3
# denom: (6-√6)(6+√6) = 36-6 = 30
# So atan(...) = atan(30√3/30) = atan(√3) = π/3

print("\natan(√3-√2) + atan((2√2+√3)/5) = atan(√3) = π/3")
lhs = math.atan(math.sqrt(3) - math.sqrt(2)) + math.atan(
    (2 * math.sqrt(2) + math.sqrt(3)) / 5
)
print(f"  LHS = {lhs:.15f}")
print(f"  π/3 = {math.pi / 3:.15f}")
print(f"  Match: {abs(lhs - math.pi / 3) < 1e-12}")

# So arg(ratio) = 5π/6 - π/3 = π/2!
print("\narg(ratio) = 5π/6 - π/3 = π/2!")
print(f"  Computed: {arg_total:.15f}")
print(f"  π/2     : {math.pi / 2:.15f}")
print(f"  Match: {abs(arg_total - math.pi / 2) < 1e-10}")

# ============================================================
# Step 5: Assemble the final symbolic result
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Final symbolic result")
print("=" * 60)

# We have:
# A = ln|ratio| = -ln(√3+√2) = ln(√3-√2)
# B = arg(ratio) = π/2
#
# I₂ = -Im[(-2+4i√2)·(A+iB)]/12
# (-2+4i√2)(A+iB) = (-2A-4√2B) + i(4√2A-2B)
# Im = 4√2A - 2B = 4√2·(-ln(√3+√2)) - 2·π/2
# = -4√2·ln(√3+√2) - π
# I₂ = -(-4√2·ln(√3+√2) - π)/12 = (4√2·ln(√3+√2) + π)/12

print("I₂ = (4√2·ln(√3+√2) + π) / 12")
I2_exact = (4 * math.sqrt(2) * math.log(math.sqrt(3) + math.sqrt(2)) + math.pi) / 12
print(f"   = {I2_exact:.15f}")
print(f"Check: {I2:.15f}")
print(f"Match: {abs(I2_exact - I2) < 1e-12}")

# Now assemble g0:
# K = ∫₀¹ asinh(√2·u/√(6u²-8u+3)) du
# = asinh(√2) - √2·(-2/3·I₁ + I₂)
# where I₁ = asinh(√2)/√2 = log(√2+√3)/√2
# and I₂ = (4√2·ln(√3+√2) + π)/12

# Note: asinh(√2) = ln(√2+√3) since asinh(x) = ln(x+√(x²+1)) and √(2+1)=√3

print("\n--- Assembling K ---")
print("asinh(√2) = ln(√2+√3)")
print("I₁ = ln(√2+√3)/√2")
print("I₂ = (4√2·ln(√2+√3) + π)/12")
print()

# IBP integral = √2·(-2/3·I₁ + I₂)
# = √2·[-2/(3√2)·ln(√2+√3) + (4√2·ln(√2+√3)+π)/12]
# = √2·[-2ln/(3√2) + 4√2·ln/12 + π/12]
# = √2·[-2ln/(3√2) + √2·ln/3 + π/12]
# = -2ln/3 + 2ln/3 + √2π/12
# = √2π/12
# Wait that's surprisingly simple. Let me double-check.

# √2·(-2/3·ln/(√2) + (4√2·ln+π)/12)
# = √2·(-2ln/(3√2)) + √2·(4√2·ln+π)/12
# = -2ln/3 + √2·4√2·ln/12 + √2·π/12
# = -2ln/3 + 8ln/12 + √2π/12
# = -2ln/3 + 2ln/3 + √2π/12
# = √2π/12

print("IBP integral = √2·(-2I₁/3 + I₂)")
print("  = √2·(-2ln(√2+√3)/(3√2) + (4√2·ln(√2+√3)+π)/12)")
print("  = -2ln(√2+√3)/3 + (8ln(√2+√3)+√2π)/12")
print("  = -2ln/3 + 2ln/3 + √2π/12")
print("  = √2π/12")
print("  = π/(6√2)")
print()

ibp_exact = math.sqrt(2) * math.pi / 12
print(f"IBP integral = √2π/12 = {ibp_exact:.15f}")
ibp_check = math.sqrt(2) * (-2 / 3 * I1 + I2_exact)
print(f"Check:                   {ibp_check:.15f}")
print(f"Match: {abs(ibp_exact - ibp_check) < 1e-12}")

# K = asinh(√2) - √2π/12 = ln(√2+√3) - √2π/12
K_exact = math.log(math.sqrt(2) + math.sqrt(3)) - math.sqrt(2) * math.pi / 12
print(f"\nK = ln(√2+√3) - √2π/12 = {K_exact:.15f}")
print(f"Check: {K:.15f}")
print(f"Match: {abs(K_exact - K) < 1e-12}")

# g0 = 4√2·K = 4√2·(ln(√2+√3) - √2π/12) = 4√2·ln(√2+√3) - 4·2·π/12
# = 4√2·ln(√2+√3) - 2π/3

g0_exact = 4 * math.sqrt(2) * math.log(math.sqrt(2) + math.sqrt(3)) - 2 * math.pi / 3
print(f"\n{'=' * 60}")
print("FINAL RESULT")
print(f"{'=' * 60}")
print("g0_tet = 4√2·ln(√2+√3) - 2π/3")
print(f"       = {g0_exact:.15f}")
print(f"Expected: {g0_scipy:.15f}")
print(f"Match: {abs(g0_exact - g0_scipy) < 1e-10}")

# High precision check
mpmath.mp.dps = 50
g0_mp = (
    4 * mpmath.sqrt(2) * mpmath.log(mpmath.sqrt(2) + mpmath.sqrt(3)) - 2 * mpmath.pi / 3
)
print(f"\nHigh precision: {g0_mp}")

# ============================================================
# Step 6: SymPy verification
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SymPy verification")
print("=" * 60)

g0_sympy = 4 * Sqrt(2) * Log(Sqrt(2) + Sqrt(3)) - 2 * Pi / 3
print(f"g0 = {g0_sympy}")
print(f"   = {Neval(g0_sympy, 50)}")

# Also express as: 4√2·acosh(√3) - 2π/3 since acosh(√3) = ln(√3+√2)
# Or: 4√2·asinh(√2) - 2π/3
print("\nEquivalent forms:")
print("g0 = 4√2·ln(√2+√3) - 2π/3")
print("   = 4√2·asinh(√2) - 2π/3")
print("   = 4√2·acosh(√3) - 2π/3")

g0_v2 = 4 * Sqrt(2) * asinh(Sqrt(2)) - 2 * Pi / 3
g0_v3 = 4 * Sqrt(2) * acosh(Sqrt(3)) - 2 * Pi / 3
print(f"Numerical check v2: {Neval(g0_v2, 30)}")
print(f"Numerical check v3: {Neval(g0_v3, 30)}")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print("g0_tet = ∫_tet 1/|x| dV = 4√2·ln(√2+√3) - 2π/3")
print(f"       ≈ {float(Neval(g0_sympy, 30)):.15f}")
print("")
print("Derivation chain:")
print("  1. ∫ 1/|x| dV = (1/2)∮ (x̂·n̂) dS  [∇²|x| = 2/|x|]")
print("  2. By T_d symmetry: = 4 × face contribution")
print("  3. P·n̂ = 4 (constant) on each face → reduces to ∫∫ 1/|P| ds dt")
print("  4. Inner integral: (√2/2)·asinh(√2(1-s)/√(6s²-4s+1))")
print("  5. IBP on outer integral with f(u) = √2u/√(6u²-8u+3)")
print("  6. Rational/√(quadratic) integral via complex partial fractions")
print("  7. Denesting: √(1+α²) = (2√2+i)/3 where α = (√2+2i)/3")
print("  8. |ratio|² = 5-2√6 = (√3-√2)² → ln|ratio| = -ln(√3+√2)")
print("  9. arg(ratio) = π/2 (via atan addition identities)")
print(" 10. Miraculous cancellation: IBP integral = √2π/12")
print(" 11. g0 = 4√2·(ln(√2+√3) - √2π/12) = 4√2·ln(√2+√3) - 2π/3")
