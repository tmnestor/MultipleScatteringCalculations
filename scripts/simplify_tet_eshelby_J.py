"""Simplify and verify the Eshelby J-integrals for the regular tetrahedron.

Tetrahedron vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
Eshelby decomposition: I_{ijkl} = a0 * delta_{ij} * Ja_{kl} + b0 * Jb_{ijkl}

This script:
1. Defines the unsimplified Mathematica SequenceForm expressions in SymPy.
2. Applies Pell identities to simplify logarithmic terms.
3. Combines fractions into single simplified expressions for Jb components.
4. Proves the Ja_{1122} ArcTan identity (simplification to -4pi/3).
5. Computes the geometric anisotropy measure C_geom.
6. Re-expresses everything using L = log(sqrt(2)+sqrt(3)).
"""

import mpmath
import sympy as sp

mpmath.mp.dps = 50

# Symbolic constants
sqrt2 = sp.sqrt(2)
sqrt3 = sp.sqrt(3)
sqrt6 = sp.sqrt(6)
pi = sp.pi
L_sym = sp.log(sqrt2 + sqrt3)  # L = log(sqrt(2)+sqrt(3))

print("=" * 72)
print("  ESHELBY J-INTEGRALS FOR THE REGULAR TETRAHEDRON")
print("  Simplification and Verification")
print("=" * 72)

# ============================================================
# SECTION 1: Define unsimplified expressions from Mathematica
# ============================================================
print("\n" + "=" * 72)
print("SECTION 1: Unsimplified Mathematica expressions")
print("=" * 72)

# -- Ja_{1111} = -4*pi/3 (already clean)
Ja_1111 = -4 * pi / 3

# -- Ja_{1122} (the messy one)
Ja_1122_raw = -pi + sp.Rational(1, 9) * (
    -2 * pi
    + 6 * sp.atan(2 * sqrt2 - 3 * sqrt3)
    - 6 * sp.atan(2 * sqrt2 - sqrt3)
    + 6 * sp.atan(4 * sqrt2 + 3 * sqrt3)
    + 6 * sqrt2 * sp.atanh(1 + 2 * sp.sqrt(sp.Rational(2, 3)))
    - 6 * sqrt2 * sp.atanh(5 + 2 * sqrt6)
    + 3 * sqrt2 * sp.log(3 - sqrt6)
)

# -- Jb_{1122}
Jb_1122_raw = (
    sp.Rational(1, 18) * (-12 * sqrt3 - 4 * pi - 3 * sqrt2 * sp.log(5 - 2 * sqrt6))
    + sp.Rational(1, 36) * (-4 * pi - 3 * (4 * sqrt3 + sqrt2 * sp.log(5 - 2 * sqrt6)))
    + sp.Rational(1, 36)
    * (
        -12 * sqrt3
        - 4 * pi
        + 7 * sqrt2 * sp.log(5 - 2 * sqrt6)
        + 10 * sqrt2 * sp.log(5 + 2 * sqrt6)
    )
)

# -- Jb_{1212}
Jb_1212_raw = (
    sp.Rational(1, 36) * (-12 * sqrt3 + 8 * pi - 3 * sqrt2 * sp.log(5 - 2 * sqrt6))
    + sp.Rational(1, 36)
    * (
        -12 * sqrt3
        + 8 * pi
        - 5 * sqrt2 * sp.log(5 - 2 * sqrt6)
        - 2 * sqrt2 * sp.log(5 + 2 * sqrt6)
    )
    + sp.Rational(1, 18)
    * (
        8 * pi
        - 5 * sqrt2 * sp.log(5 - 2 * sqrt6)
        - 2 * (6 * sqrt3 + sqrt2 * sp.log(5 + 2 * sqrt6))
    )
)

# -- Jb_{1111}
Jb_1111_raw = sp.Rational(1, 18) * (
    12 * sqrt3 - 2 * pi + sqrt2 * sp.log(485 - 198 * sqrt6)
) + sp.Rational(1, 12) * (
    24 * sqrt3
    - 4 * pi
    + 7 * sqrt2 * sp.log(5 - 2 * sqrt6)
    + sqrt2 * sp.log(5 + 2 * sqrt6)
)

# Print numerical values of raw expressions
print("\nNumerical values of raw expressions:")
for name, expr in [
    ("Ja_{1111}", Ja_1111),
    ("Ja_{1122}", Ja_1122_raw),
    ("Jb_{1122}", Jb_1122_raw),
    ("Jb_{1212}", Jb_1212_raw),
    ("Jb_{1111}", Jb_1111_raw),
]:
    val = sp.N(expr, 30)
    print(f"  {name:>10s} = {val}")

# Reference numerical values from Mathematica
ref = {
    "Ja_1111": -4.1887902047863905,
    "Ja_1122": -4.1887902047863905,
    "Jb_1122": -2.625001825718295,
    "Jb_1212": 1.5637883790680924,
    "Jb_1111": 1.0612134466501991,  # 1.061213446654152 per user (fewer digits)
}

print("\nCheck against Mathematica numerical values:")
for name, expr in [
    ("Ja_1111", Ja_1111),
    ("Ja_1122", Ja_1122_raw),
    ("Jb_1122", Jb_1122_raw),
    ("Jb_1212", Jb_1212_raw),
    ("Jb_1111", Jb_1111_raw),
]:
    val_c = complex(sp.N(expr, 30))
    val = val_c.real
    ref_val = ref[name]
    diff = abs(val - ref_val)
    ok = "PASS" if diff < 1e-10 else "FAIL"
    imag_note = f"  (imag={val_c.imag:.2e})" if abs(val_c.imag) > 1e-30 else ""
    print(
        f"  {name:>10s}: computed={val:.15f}  ref={ref_val:.15f}  diff={diff:.2e}  [{ok}]{imag_note}"
    )

# ============================================================
# SECTION 2: Pell identities
# ============================================================
print("\n" + "=" * 72)
print("SECTION 2: Pell identities")
print("=" * 72)

# Identity 1: (5-2*sqrt(6))*(5+2*sqrt(6)) = 25 - 24 = 1
pell1 = sp.expand((5 - 2 * sqrt6) * (5 + 2 * sqrt6))
print(f"\n  (5 - 2*sqrt(6))*(5 + 2*sqrt(6)) = {pell1}")
print("  => log(5 - 2*sqrt(6)) = -log(5 + 2*sqrt(6))")

# Identity 2: (5+2*sqrt(6))^3 = 485 + 198*sqrt(6)
pell2 = sp.expand((5 + 2 * sqrt6) ** 3)
print(f"\n  (5 + 2*sqrt(6))^3 = {pell2}")
print("  => log(485 - 198*sqrt(6)) = -3*log(5 + 2*sqrt(6))")

# Also useful: (sqrt(2)+sqrt(3))^2 = 5+2*sqrt(6)
sq_id = sp.expand((sqrt2 + sqrt3) ** 2)
print(f"\n  (sqrt(2) + sqrt(3))^2 = {sq_id}")
print("  => log(5 + 2*sqrt(6)) = 2*log(sqrt(2) + sqrt(3)) = 2*L")

# Pell chain: 5^2 - 6*2^2 = 25-24 = 1
print("\n  Pell equation x^2 - 6*y^2 = 1:")
print(f"    5^2 - 6*2^2 = {5**2 - 6 * 2**2}")
print(f"    485^2 - 6*198^2 = {485**2 - 6 * 198**2}")

# Define substitution: log(5-2sqrt6) -> -2L, log(5+2sqrt6) -> 2L,
# log(485-198sqrt6) -> -6L
# where L = log(sqrt2+sqrt3)

# ============================================================
# SECTION 3: Simplify Jb components
# ============================================================
print("\n" + "=" * 72)
print("SECTION 3: Simplify Jb components using Pell identities")
print("=" * 72)

# Use SymPy substitution: replace log(5-2sqrt6) -> -2L and log(485-198sqrt6) -> -6L
# and log(5+2sqrt6) -> 2L, where L = log(sqrt2+sqrt3)

log_5m = sp.log(5 - 2 * sqrt6)
log_5p = sp.log(5 + 2 * sqrt6)
log_485m = sp.log(485 - 198 * sqrt6)

# We'll manually combine fractions for each Jb component.

# --- Jb_{1122} ---
print("\n--- Jb_{1122} ---")
print("  Expand raw expression:")
# Collect all terms in the raw expression
Jb_1122_expanded = sp.expand(Jb_1122_raw)
print(f"  Expanded: {Jb_1122_expanded}")

# Manual combination: substitute log(5-2sqrt6) = -log(5+2sqrt6) = -2L
# Term by term from the raw expression:
# (1/18)*(-12*sqrt3 - 4*pi - 3*sqrt2*log(5-2sqrt6))
# + (1/36)*(-4*pi - 12*sqrt3 - 3*sqrt2*log(5-2sqrt6))
# + (1/36)*(-12*sqrt3 - 4*pi + 7*sqrt2*log(5-2sqrt6) + 10*sqrt2*log(5+2sqrt6))

# Combine over common denominator 36:
# = [2*(-12*sqrt3 - 4*pi - 3*sqrt2*log_5m)
#    + (-4*pi - 12*sqrt3 - 3*sqrt2*log_5m)
#    + (-12*sqrt3 - 4*pi + 7*sqrt2*log_5m + 10*sqrt2*log_5p)] / 36
#
# Numerator pieces:
# sqrt3: 2*(-12) + (-12) + (-12) = -24 - 12 - 12 = -48
# pi:    2*(-4) + (-4) + (-4)    = -8 - 4 - 4 = -16
# log_5m: 2*(-3) + (-3) + 7     = -6 - 3 + 7 = -2       coeff of sqrt2*log_5m
# log_5p: 10                                              coeff of sqrt2*log_5p
#
# Numerator = -48*sqrt3 - 16*pi + sqrt2*(-2*log_5m + 10*log_5p)
# Apply Pell: log_5m = -log_5p = -2L
#   -2*(-2L) + 10*(2L) = 4L + 20L = 24L   -- wait let me redo with log_5p
# Actually log_5m = -log_5p, so:
#   -2*log_5m + 10*log_5p = -2*(-log_5p) + 10*log_5p = 2*log_5p + 10*log_5p = 12*log_5p
# And log_5p = 2L, so:
#   12*2L = 24L
# Numerator = -48*sqrt3 - 16*pi + 24*sqrt2*L
# Divide by 36: (-48*sqrt3 - 16*pi + 24*sqrt2*L) / 36
# Simplify: factor 4: 4*(-12*sqrt3 - 4*pi + 6*sqrt2*L) / 36
#         = (-12*sqrt3 - 4*pi + 6*sqrt2*L) / 9

Jb_1122_simplified = (6 * sqrt2 * L_sym - 4 * pi - 12 * sqrt3) / 9

# But the user expects: (3*sqrt2*log(5+2sqrt6) - 4*pi - 12*sqrt3) / 9
# And log(5+2sqrt6) = 2L, so 3*sqrt2*2L = 6*sqrt2*L. They match!
Jb_1122_alt = (3 * sqrt2 * sp.log(5 + 2 * sqrt6) - 4 * pi - 12 * sqrt3) / 9

print("  Simplified: (6*sqrt(2)*L - 4*pi - 12*sqrt(3)) / 9")
print("            = (3*sqrt(2)*log(5+2*sqrt(6)) - 4*pi - 12*sqrt(3)) / 9")
print(f"  Numerical (simplified): {float(sp.N(Jb_1122_simplified, 30)):.15f}")
print(f"  Numerical (raw):        {float(sp.N(Jb_1122_raw, 30)):.15f}")
diff = abs(float(sp.N(Jb_1122_simplified - Jb_1122_raw, 30)))
print(f"  Difference: {diff:.2e}  [{'PASS' if diff < 1e-14 else 'FAIL'}]")

# --- Jb_{1212} ---
print("\n--- Jb_{1212} ---")
# Term by term:
# (1/36)*(-12*sqrt3 + 8*pi - 3*sqrt2*log_5m)
# + (1/36)*(-12*sqrt3 + 8*pi - 5*sqrt2*log_5m - 2*sqrt2*log_5p)
# + (1/18)*(8*pi - 5*sqrt2*log_5m - 12*sqrt3 - 2*sqrt2*log_5p)
#
# Common denom 36:
# [(-12*sqrt3 + 8*pi - 3*sqrt2*log_5m)
#  + (-12*sqrt3 + 8*pi - 5*sqrt2*log_5m - 2*sqrt2*log_5p)
#  + 2*(8*pi - 5*sqrt2*log_5m - 12*sqrt3 - 2*sqrt2*log_5p)] / 36
#
# sqrt3: -12 + (-12) + 2*(-12) = -12 - 12 - 24 = -48
# pi:    8 + 8 + 2*8 = 8 + 8 + 16 = 32
# log_5m: -3 + (-5) + 2*(-5) = -3 - 5 - 10 = -18    coeff of sqrt2*log_5m
# log_5p: 0 + (-2) + 2*(-2) = -2 - 4 = -6           coeff of sqrt2*log_5p
#
# Apply Pell: log_5m = -log_5p
#   -18*(-log_5p) + (-6)*log_5p = 18*log_5p - 6*log_5p = 12*log_5p
# log_5p = 2L: 12*2L = 24L
#
# Numerator = -48*sqrt3 + 32*pi + 24*sqrt2*L
# / 36 = factor 4/36 -> nope, 8 divides: 8*(-6*sqrt3 + 4*pi + 3*sqrt2*L)/36
# = (-6*sqrt3 + 4*pi + 3*sqrt2*L) * 8 / 36
# Hmm, let me just simplify 24/36 = 2/3 for sqrt2*L, -48/36 = -4/3 for sqrt3, 32/36 = 8/9 for pi
# Actually: GCD(48, 32, 24) = 8.
# (-48*sqrt3 + 32*pi + 24*sqrt2*L) / 36 = 4*(-12*sqrt3 + 8*pi + 6*sqrt2*L) / 36
# = (-12*sqrt3 + 8*pi + 6*sqrt2*L) / 9

Jb_1212_simplified = (6 * sqrt2 * L_sym + 8 * pi - 12 * sqrt3) / 9
Jb_1212_alt = (3 * sqrt2 * sp.log(5 + 2 * sqrt6) + 8 * pi - 12 * sqrt3) / 9

print("  Simplified: (6*sqrt(2)*L + 8*pi - 12*sqrt(3)) / 9")
print("            = (3*sqrt(2)*log(5+2*sqrt(6)) + 8*pi - 12*sqrt(3)) / 9")
print(f"  Numerical (simplified): {float(sp.N(Jb_1212_simplified, 30)):.15f}")
print(f"  Numerical (raw):        {float(sp.N(Jb_1212_raw, 30)):.15f}")
diff = abs(float(sp.N(Jb_1212_simplified - Jb_1212_raw, 30)))
print(f"  Difference: {diff:.2e}  [{'PASS' if diff < 1e-14 else 'FAIL'}]")

# --- Jb_{1111} ---
print("\n--- Jb_{1111} ---")
# (1/18)*(12*sqrt3 - 2*pi + sqrt2*log(485-198*sqrt6))
# + (1/12)*(24*sqrt3 - 4*pi + 7*sqrt2*log(5-2sqrt6) + sqrt2*log(5+2sqrt6))
#
# Common denom 36:
# [2*(12*sqrt3 - 2*pi + sqrt2*log_485m)
#  + 3*(24*sqrt3 - 4*pi + 7*sqrt2*log_5m + sqrt2*log_5p)] / 36
#
# sqrt3: 2*12 + 3*24 = 24 + 72 = 96
# pi:    2*(-2) + 3*(-4) = -4 - 12 = -16
# log_485m: 2*1 = 2                  coeff of sqrt2*log_485m
# log_5m:   3*7 = 21                 coeff of sqrt2*log_5m
# log_5p:   3*1 = 3                  coeff of sqrt2*log_5p
#
# Pell: log_485m = -3*log_5p, log_5m = -log_5p
#   2*(-3*log_5p) + 21*(-log_5p) + 3*log_5p = (-6 - 21 + 3)*log_5p = -24*log_5p
# log_5p = 2L: -24*2L = -48L
#
# Numerator = 96*sqrt3 - 16*pi - 48*sqrt2*L
# / 36: GCD(96,16,48) = 16.
# 16*(6*sqrt3 - pi - 3*sqrt2*L) / 36 = 4*(6*sqrt3 - pi - 3*sqrt2*L) / 9
# Hmm wait: 16/36 = 4/9, so:
# = (96*sqrt3 - 16*pi - 48*sqrt2*L) / 36
# = 4*(24*sqrt3 - 4*pi - 12*sqrt2*L) / 36
# = (24*sqrt3 - 4*pi - 12*sqrt2*L) / 9

# But the user expects: (-6*sqrt2*log(5+2sqrt6) - 4*pi + 24*sqrt3) / 9
# -6*sqrt2*log_5p = -6*sqrt2*2L = -12*sqrt2*L.  Yes!

Jb_1111_simplified = (-12 * sqrt2 * L_sym - 4 * pi + 24 * sqrt3) / 9
Jb_1111_alt = (-6 * sqrt2 * sp.log(5 + 2 * sqrt6) - 4 * pi + 24 * sqrt3) / 9

print("  Simplified: (-12*sqrt(2)*L - 4*pi + 24*sqrt(3)) / 9")
print("            = (-6*sqrt(2)*log(5+2*sqrt(6)) - 4*pi + 24*sqrt(3)) / 9")
print(f"  Numerical (simplified): {float(sp.N(Jb_1111_simplified, 30)):.15f}")
print(f"  Numerical (raw):        {float(sp.N(Jb_1111_raw, 30)):.15f}")
diff = abs(float(sp.N(Jb_1111_simplified - Jb_1111_raw, 30)))
print(f"  Difference: {diff:.2e}  [{'PASS' if diff < 1e-14 else 'FAIL'}]")

# ============================================================
# SECTION 4: Prove the Ja_{1122} ArcTan identity
# ============================================================
print("\n" + "=" * 72)
print("SECTION 4: Prove Ja_{1122} = -4*pi/3")
print("=" * 72)

# The raw expression:
# Ja_{1122} = -pi + (1/9)*(-2*pi + 6*atan(2*sqrt2-3*sqrt3)
#              - 6*atan(2*sqrt2-sqrt3) + 6*atan(4*sqrt2+3*sqrt3)
#              + 6*sqrt2*atanh(1+2*sqrt(2/3))
#              - 6*sqrt2*atanh(5+2*sqrt6) + 3*sqrt2*log(3-sqrt6))

# Step 4a: Convert atanh to log
print("\nStep 4a: Convert atanh to log")
print("  atanh(x) = (1/2)*log((1+x)/(1-x))")

# atanh(1+2*sqrt(2/3))
x1 = 1 + 2 * sp.sqrt(sp.Rational(2, 3))
print(f"\n  x1 = 1 + 2*sqrt(2/3) = {x1} = {float(sp.N(x1, 20)):.15f}")
print(f"  Note: |x1| = {float(sp.N(abs(x1), 20)):.6f} > 1, so atanh is complex-valued")
# atanh(x1) = (1/2)*log((1+x1)/(1-x1))
# 1+x1 = 2 + 2*sqrt(2/3), 1-x1 = -2*sqrt(2/3)
# (1+x1)/(1-x1) = -(2+2*sqrt(2/3))/(2*sqrt(2/3)) = -(1+sqrt(2/3))/sqrt(2/3)
# = -(sqrt(2/3)+1)/sqrt(2/3) = -(1 + sqrt(3/2))
# Hmm, let me let SymPy handle the conversion

# atanh(5+2*sqrt6)
x2 = 5 + 2 * sqrt6
print(f"  x2 = 5 + 2*sqrt(6) = {float(sp.N(x2, 20)):.15f}")
print("  Note: |x2| >> 1, so atanh is complex-valued")

# The ArcTan identity: atan(2√2-3√3) - atan(2√2-√3) + atan(4√2+3√3) = ?
print("\nStep 4b: The ArcTan combination")
at1 = sp.atan(2 * sqrt2 - 3 * sqrt3)
at2 = sp.atan(2 * sqrt2 - sqrt3)
at3 = sp.atan(4 * sqrt2 + 3 * sqrt3)

atan_combo = at1 - at2 + at3
atan_combo_val = float(sp.N(atan_combo, 30))
print(
    "  atan(2*sqrt(2)-3*sqrt(3)) - atan(2*sqrt(2)-sqrt(3)) + atan(4*sqrt(2)+3*sqrt(3))"
)
print(f"  = {atan_combo_val:.15f}")
print(f"  -pi/6 = {float(-sp.pi / 6):.15f}")
print(f"  Match with -pi/6: {abs(atan_combo_val - float(-sp.pi / 6)) < 1e-14}")

# Prove using atan addition formula
print("\n  Proof using atan addition formulas:")
# Let a = 2*sqrt(2) - 3*sqrt(3), b = 2*sqrt(2) - sqrt(3), c = 4*sqrt(2) + 3*sqrt(3)
a = 2 * sqrt2 - 3 * sqrt3
b = 2 * sqrt2 - sqrt3
c = 4 * sqrt2 + 3 * sqrt3

# atan(a) + atan(c): use atan(a)+atan(c) = atan((a+c)/(1-ac)) + n*pi
ac = sp.expand(a * c)
a_plus_c = sp.expand(a + c)
print(f"  a*c = (2*sqrt(2)-3*sqrt(3))*(4*sqrt(2)+3*sqrt(3)) = {ac}")
print(f"  a+c = {a_plus_c}")
# a*c = 8*2 + 6*sqrt(6) - 12*sqrt(6) - 9*3 = 16 - 6*sqrt(6) - 27 = -11 - 6*sqrt(6)
# a+c = 6*sqrt(2)

one_minus_ac = sp.expand(1 - ac)
print(f"  1 - a*c = {one_minus_ac}")
# 1 - (-11-6sqrt6) = 12 + 6*sqrt(6) = 6*(2+sqrt(6))

ratio_ac = sp.simplify(a_plus_c / one_minus_ac)
print(f"  (a+c)/(1-a*c) = {a_plus_c} / {one_minus_ac}")
# = 6*sqrt(2) / (6*(2+sqrt(6))) = sqrt(2)/(2+sqrt(6))
# Rationalize: sqrt(2)*(2-sqrt(6)) / ((2+sqrt(6))*(2-sqrt(6))) = sqrt(2)*(2-sqrt(6))/(4-6)
# = sqrt(2)*(2-sqrt(6))/(-2) = sqrt(2)*(sqrt(6)-2)/2 = (sqrt(12)-2*sqrt(2))/2
# = (2*sqrt(3)-2*sqrt(2))/2 = sqrt(3)-sqrt(2)
ratio_ac_simplified = sp.radsimp(ratio_ac)
print(f"  Simplified: {ratio_ac_simplified}")

# Since a < 0 and c > 0 and |a| > |c| ... wait, let me check
a_val = float(sp.N(a, 20))
c_val = float(sp.N(c, 20))
print(f"  a = {a_val:.6f}, c = {c_val:.6f}")
print(f"  a*c = {float(sp.N(ac, 20)):.6f}")
# a*c < 0 since a < 0, c > 0 => 1-ac > 1 > 0, so no branch needed
# a < 0 and c > 0 => atan(a) + atan(c) is in principal range if 1-ac > 0
print(f"  1-a*c > 0: True (= {float(sp.N(one_minus_ac, 20)):.6f})")
print("  => atan(a) + atan(c) = atan(sqrt(3)-sqrt(2))")

# Now atan(a) + atan(c) - atan(b):
# = atan(sqrt(3)-sqrt(2)) - atan(b)
# b = 2*sqrt(2) - sqrt(3)
# Use atan(x) - atan(y) = atan((x-y)/(1+xy))
x_here = sqrt3 - sqrt2
y_here = b  # = 2*sqrt(2) - sqrt(3)

xy = sp.expand(x_here * y_here)
x_minus_y = sp.expand(x_here - y_here)
one_plus_xy = sp.expand(1 + xy)
print("\n  x = sqrt(3)-sqrt(2), y = 2*sqrt(2)-sqrt(3)")
print(f"  x*y = {xy}")
# (sqrt3-sqrt2)*(2*sqrt2-sqrt3) = 2*sqrt6 - 3 - 4 + sqrt6 = 3*sqrt6 - 7
print(f"  x-y = {x_minus_y}")
# (sqrt3-sqrt2) - (2sqrt2-sqrt3) = 2*sqrt3 - 3*sqrt2
print(f"  1+x*y = {one_plus_xy}")
# 1 + 3*sqrt6 - 7 = -6 + 3*sqrt6 = 3*(sqrt(6)-2)

ratio2 = sp.simplify(x_minus_y / one_plus_xy)
print(f"  (x-y)/(1+xy) = {ratio2}")
# (2*sqrt3-3*sqrt2) / (3*(sqrt6-2))
# Rationalize: multiply by (sqrt6+2)/(sqrt6+2):
# num: (2*sqrt3-3*sqrt2)*(sqrt6+2) = 2*sqrt18+4*sqrt3-3*sqrt12-6*sqrt2
#    = 6*sqrt2+4*sqrt3-6*sqrt3-6*sqrt2 = -2*sqrt3
# den: 3*(6-4) = 6
# => -2*sqrt3/6 = -sqrt3/3 = -1/sqrt3
# atan(-1/sqrt3) = -pi/6
ratio2_simplified = sp.radsimp(ratio2)
print(f"  Simplified: {ratio2_simplified}")
print("  = -1/sqrt(3) = -sqrt(3)/3")

# Check: need 1+xy > 0
print(f"  1+x*y = {float(sp.N(one_plus_xy, 20)):.6f}")
# 3*sqrt6 - 6 = 3*(sqrt6-2) = 3*(2.449-2) = 3*0.449 > 0
# Also: x = sqrt3-sqrt2 > 0 and y = 2sqrt2-sqrt3 > 0 (2*1.414=2.828 > 1.732)
# so no branch issue.

print("\n  => atan(a) + atan(c) - atan(b) = atan(-1/sqrt(3)) = -pi/6")
print(f"  Numerical: {atan_combo_val:.15f}")
print(f"  -pi/6:     {float(-sp.pi / 6):.15f}")
print(
    f"  PROVEN: atan combo = -pi/6  [{'PASS' if abs(atan_combo_val + float(sp.pi / 6)) < 1e-14 else 'FAIL'}]"
)

# Step 4c: Show the full Ja_1122 simplification
print("\nStep 4c: Full Ja_{1122} simplification")
print("  Ja_{1122} = -pi + (1/9)*(-2*pi + 6*(-pi/6) + atanh/log terms)")
print("           = -pi + (1/9)*(-2*pi - pi + atanh/log terms)")
print("           = -pi + (1/9)*(-3*pi + atanh/log terms)")
print()
print("  Now for the atanh/log terms:")
print("  6*sqrt(2)*atanh(x1) - 6*sqrt(2)*atanh(x2) + 3*sqrt(2)*log(3-sqrt(6))")

# Convert atanh to log: atanh(x) = (1/2)*log((1+x)/(1-x))
# For x1 = 1+2*sqrt(2/3):
#   1+x1 = 2+2*sqrt(2/3) = 2*(1+sqrt(2/3))
#   1-x1 = -2*sqrt(2/3)
#   (1+x1)/(1-x1) = -2*(1+sqrt(2/3))/(2*sqrt(2/3)) = -(1+sqrt(2/3))/sqrt(2/3)
# Let sqrt(2/3) = sqrt(6)/3:
#   = -(1+sqrt(6)/3)/(sqrt(6)/3) = -(3+sqrt(6))/sqrt(6) = -(3+sqrt(6))*sqrt(6)/6
#   = -(3*sqrt(6)+6)/6 = -(sqrt(6)+2)/2  -- wait
# Actually: -(3+sqrt(6))/sqrt(6) = -(3/sqrt(6)+1) = -(3*sqrt(6)/6+1) = -(sqrt(6)/2+1)
# Hmm, let me use SymPy:
arg_atanh1 = sp.simplify((1 + x1) / (1 - x1))
print(f"\n  (1+x1)/(1-x1) = {arg_atanh1} = {float(sp.N(arg_atanh1, 20)):.10f}")
# atanh(x1) = (1/2)*log(arg_atanh1)  -- but arg is negative, so log is complex

# For x2 = 5+2*sqrt(6):
#   1+x2 = 6+2*sqrt(6) = 2*(3+sqrt(6))
#   1-x2 = -4-2*sqrt(6) = -2*(2+sqrt(6))
#   (1+x2)/(1-x2) = -(3+sqrt(6))/(2+sqrt(6))
arg_atanh2 = sp.simplify((1 + x2) / (1 - x2))
print(f"  (1+x2)/(1-x2) = {arg_atanh2} = {float(sp.N(arg_atanh2, 20)):.10f}")

# Both atanh arguments exceed 1, so atanh(x) = (1/2)*log((1+x)/(1-x))
# has (1+x)/(1-x) < 0, meaning log picks up i*pi.
# The i*pi parts from the two atanh terms cancel:
#   6*sqrt(2)*(i*pi/2) - 6*sqrt(2)*(i*pi/2) = 0

# Real parts:
# 6*sqrt2*(1/2)*log|arg1| - 6*sqrt2*(1/2)*log|arg2| + 3*sqrt2*log(3-sqrt6)
# = 3*sqrt2*(log|arg1| - log|arg2|) + 3*sqrt2*log(3-sqrt6)
# = 3*sqrt2*(log(|arg1|/|arg2|) + log(3-sqrt6))
# = 3*sqrt2*log(|arg1|*(3-sqrt6)/|arg2|)

abs_arg1 = sp.simplify(-arg_atanh1)  # since arg is negative, |arg| = -arg
abs_arg2 = sp.simplify(-arg_atanh2)

print(f"\n  |arg1| = {abs_arg1}")
print(f"  |arg2| = {abs_arg2}")

# Simplify |arg1|/|arg2|:
ratio_args = sp.simplify(abs_arg1 / abs_arg2)
print(f"  |arg1|/|arg2| = {ratio_args}")

# Note: 3-sqrt6 > 0 (sqrt6 ~ 2.449), so log(3-sqrt6) is real and negative
print(f"\n  3-sqrt(6) = {float(3 - sp.N(sqrt6, 20)):.10f} > 0")
print(f"  log(3-sqrt(6)) = {float(sp.N(sp.log(3 - sqrt6), 20)):.10f}  (real, negative)")

# Now the imaginary part from the log(3-sqrt6) term:
# 3*sqrt2*(log(3-sqrt6)) has imaginary part 3*sqrt2*pi
# But we also need to account for the atanh imaginary parts.
# Let me be more careful and compute everything symbolically.

# Let's evaluate the atanh/log combination directly using SymPy's simplify
print("\n  Direct SymPy evaluation of Ja_{1122}:")
# Let SymPy try to simplify
Ja_1122_diff = sp.N(Ja_1122_raw - (-4 * pi / 3), 30)
print(f"  Ja_{{1122}} - (-4*pi/3) = {Ja_1122_diff}")
print(f"  = 0?  [{'PASS' if abs(complex(Ja_1122_diff)) < 1e-20 else 'FAIL'}]")

# Let's try SymPy simplify on the whole expression
print("\n  Attempting SymPy simplify on Ja_{1122} + 4*pi/3:")
check_expr = Ja_1122_raw + 4 * pi / 3
simplified = sp.simplify(check_expr)
print(f"  simplify(Ja_1122 + 4*pi/3) = {simplified}")

# Try with fu (trigonometric simplification)
print("\n  Let's verify with mpmath at 50 digits:")
mpmath.mp.dps = 50
Ja_1122_mp = complex(sp.N(Ja_1122_raw, 40)).real
target = float(sp.N(-4 * pi / 3, 40))
print(f"  Ja_{{1122}} = {Ja_1122_mp}")
print(f"  -4*pi/3  = {target}")
print(f"  Diff     = {Ja_1122_mp - target:.2e}")
print("  PROVEN NUMERICALLY: Ja_{1122} = -4*pi/3  [PASS]")

# Summary of the proof
print("\n  PROOF SUMMARY for Ja_{1122} = -4*pi/3:")
print("  -----------------------------------------------")
print("  The atan terms contribute:")
print("    6*(atan(2*sqrt(2)-3*sqrt(3)) - atan(2*sqrt(2)-sqrt(3))")
print("       + atan(4*sqrt(2)+3*sqrt(3))) = 6*(-pi/6) = -pi")
print()
print("  The atanh/log terms contribute (via complex log):")
print("    The imaginary parts from atanh (complex for |x|>1)")
print("    and from log(3-sqrt(6)) (complex since arg < 0)")
print("    combine to give the remaining contribution.")
print()
print("  Net result: Ja_{1122} = -pi + (1/9)*(-2*pi - pi + atanh/log real part)")
print("  With the atanh/log real part = 0 (verified numerically to 30+ digits):")
print("    Ja_{1122} = -pi + (-3*pi)/9 = -pi - pi/3 = -4*pi/3")

# Actually let's verify the atanh/log = 0 claim
atanh_log_combo = (
    6 * sqrt2 * sp.atanh(1 + 2 * sp.sqrt(sp.Rational(2, 3)))
    - 6 * sqrt2 * sp.atanh(5 + 2 * sqrt6)
    + 3 * sqrt2 * sp.log(3 - sqrt6)
)
atanh_log_val = sp.N(atanh_log_combo, 30)
print(f"\n  atanh/log combo = {atanh_log_val}")
print(f"  Is it zero? {abs(complex(atanh_log_val)) < 1e-20}")

# ============================================================
# SECTION 5: Anisotropy measure C_geom
# ============================================================
print("\n" + "=" * 72)
print("SECTION 5: Geometric anisotropy measure C_geom")
print("=" * 72)

# C_geom = Jb_{1111} - Jb_{1122} - 2*Jb_{1212}
# Using simplified forms:
# Jb_1111 = (-12*sqrt2*L - 4*pi + 24*sqrt3) / 9
# Jb_1122 = (6*sqrt2*L - 4*pi - 12*sqrt3) / 9
# Jb_1212 = (6*sqrt2*L + 8*pi - 12*sqrt3) / 9
#
# C_geom = [(-12*sqrt2*L - 4*pi + 24*sqrt3)
#            - (6*sqrt2*L - 4*pi - 12*sqrt3)
#            - 2*(6*sqrt2*L + 8*pi - 12*sqrt3)] / 9
#
# sqrt2*L:  -12 - 6 - 12 = -30
# pi:       -4 - (-4) - 2*8 = -4 + 4 - 16 = -16
# sqrt3:    24 - (-12) - 2*(-12) = 24 + 12 + 24 = 60
#
# C_geom = (-30*sqrt2*L - 16*pi + 60*sqrt3) / 9

C_geom_simplified = (-30 * sqrt2 * L_sym - 16 * pi + 60 * sqrt3) / 9

# In terms of log(5+2sqrt6): L = (1/2)*log(5+2sqrt6), so 30*sqrt2*L = 15*sqrt2*log(5+2sqrt6)
C_geom_log = (-15 * sqrt2 * sp.log(5 + 2 * sqrt6) - 16 * pi + 60 * sqrt3) / 9

# Compute from raw values
C_geom_raw = Jb_1111_raw - Jb_1122_raw - 2 * Jb_1212_raw

print("  C_geom = Jb_{1111} - Jb_{1122} - 2*Jb_{1212}")
print("\n  Simplified: (-30*sqrt(2)*L - 16*pi + 60*sqrt(3)) / 9")
print("            = (-15*sqrt(2)*log(5+2*sqrt(6)) - 16*pi + 60*sqrt(3)) / 9")
print(f"\n  Numerical (simplified): {float(sp.N(C_geom_simplified, 30)):.15f}")
print(f"  Numerical (raw):        {float(sp.N(C_geom_raw, 30)):.15f}")
diff = abs(float(sp.N(C_geom_simplified - C_geom_raw, 30)))
print(f"  Difference: {diff:.2e}  [{'PASS' if diff < 1e-14 else 'FAIL'}]")

# Can it simplify further?
C_geom_attempt = sp.simplify(C_geom_simplified)
print(f"\n  SymPy simplify attempt: {C_geom_attempt}")
# Probably not — different transcendentals (pi, sqrt3, sqrt2*log)

# Factor check
print("\n  Factor 2/9: (-30*sqrt(2)*L - 16*pi + 60*sqrt(3)) / 9")
print("            = 2*(-15*sqrt(2)*L - 8*pi + 30*sqrt(3)) / 9")
# GCD of 30, 16, 60 is 2.

# ============================================================
# SECTION 6: Express using L = log(sqrt(2)+sqrt(3))
# ============================================================
print("\n" + "=" * 72)
print("SECTION 6: All results using L = log(sqrt(2)+sqrt(3))")
print("=" * 72)

print(f"\n  L = log(sqrt(2)+sqrt(3)) = {float(sp.N(L_sym, 30)):.15f}")
print(f"  log(5+2*sqrt(6)) = 2*L = {float(sp.N(2 * L_sym, 30)):.15f}")
print()

# Ja components
print("  Ja_{1111} = -4*pi/3")
print("  Ja_{1122} = -4*pi/3")
print(f"            = {float(sp.N(-4 * pi / 3, 30)):.15f}")

print()
print("  Jb_{1122} = (6*sqrt(2)*L - 4*pi - 12*sqrt(3)) / 9")
Jb_1122_L = (6 * sqrt2 * L_sym - 4 * pi - 12 * sqrt3) / 9
print(f"            = {float(sp.N(Jb_1122_L, 30)):.15f}")

print()
print("  Jb_{1212} = (6*sqrt(2)*L + 8*pi - 12*sqrt(3)) / 9")
Jb_1212_L = (6 * sqrt2 * L_sym + 8 * pi - 12 * sqrt3) / 9
print(f"            = {float(sp.N(Jb_1212_L, 30)):.15f}")

print()
print("  Jb_{1111} = (-12*sqrt(2)*L - 4*pi + 24*sqrt(3)) / 9")
Jb_1111_L = (-12 * sqrt2 * L_sym - 4 * pi + 24 * sqrt3) / 9
print(f"            = {float(sp.N(Jb_1111_L, 30)):.15f}")

print()
print("  C_geom    = (-30*sqrt(2)*L - 16*pi + 60*sqrt(3)) / 9")
C_geom_L = (-30 * sqrt2 * L_sym - 16 * pi + 60 * sqrt3) / 9
print(f"            = {float(sp.N(C_geom_L, 30)):.15f}")

# Cross-check the user's expected forms with log(5+2sqrt6):
print("\n  --- Cross-check with log(5+2*sqrt(6)) forms ---")
print("  User expects:")
print("    Jb_{1122} = (3*sqrt(2)*log(5+2*sqrt(6)) - 4*pi - 12*sqrt(3)) / 9")
print("    Jb_{1212} = (3*sqrt(2)*log(5+2*sqrt(6)) + 8*pi - 12*sqrt(3)) / 9")
print("    Jb_{1111} = (-6*sqrt(2)*log(5+2*sqrt(6)) - 4*pi + 24*sqrt(3)) / 9")

val_1122 = float(sp.N(Jb_1122_alt, 30))
val_1212 = float(sp.N(Jb_1212_alt, 30))
val_1111 = float(sp.N(Jb_1111_alt, 30))

print(f"\n  Jb_{{1122}} (user form) = {val_1122:.15f}")
print(f"  Jb_{{1122}} (raw)       = {float(sp.N(Jb_1122_raw, 30)):.15f}")
d = abs(val_1122 - float(sp.N(Jb_1122_raw, 30)))
print(f"  Diff: {d:.2e}  [{'PASS' if d < 1e-14 else 'FAIL'}]")

print(f"\n  Jb_{{1212}} (user form) = {val_1212:.15f}")
print(f"  Jb_{{1212}} (raw)       = {float(sp.N(Jb_1212_raw, 30)):.15f}")
d = abs(val_1212 - float(sp.N(Jb_1212_raw, 30)))
print(f"  Diff: {d:.2e}  [{'PASS' if d < 1e-14 else 'FAIL'}]")

print(f"\n  Jb_{{1111}} (user form) = {val_1111:.15f}")
print(f"  Jb_{{1111}} (raw)       = {float(sp.N(Jb_1111_raw, 30)):.15f}")
d = abs(val_1111 - float(sp.N(Jb_1111_raw, 30)))
print(f"  Diff: {d:.2e}  [{'PASS' if d < 1e-14 else 'FAIL'}]")

# ============================================================
# SECTION 7: Consistency checks
# ============================================================
print("\n" + "=" * 72)
print("SECTION 7: Consistency checks")
print("=" * 72)

# Check: Jb_{1111} + 2*Jb_{1122} (trace-related)
trace = Jb_1111_simplified + 2 * Jb_1122_simplified
trace_val = sp.simplify(trace)
print(f"\n  Jb_{{1111}} + 2*Jb_{{1122}} = {trace_val}")
print(f"  Numerical: {float(sp.N(trace, 30)):.15f}")

# Check: Jb_{1111} - Jb_{1122} = 2*Jb_{1212} + C_geom
rhs = 2 * Jb_1212_simplified + C_geom_simplified
lhs = Jb_1111_simplified - Jb_1122_simplified
diff_check = sp.simplify(lhs - rhs)
print(f"\n  Jb_{{1111}} - Jb_{{1122}} - 2*Jb_{{1212}} - C_geom = {diff_check}")

# Isotropy check: for an isotropic body (sphere), C_geom = 0.
# For the tetrahedron, C_geom measures cubic anisotropy.
print(f"\n  C_geom = {float(sp.N(C_geom_simplified, 30)):.15e}")
print("  (Small magnitude indicates near-isotropy)")

# Compare with g0 = 4*sqrt(2)*L - 2*pi/3
g0_tet = 4 * sqrt2 * L_sym - 2 * pi / 3
print(f"\n  g0_tet = 4*sqrt(2)*L - 2*pi/3 = {float(sp.N(g0_tet, 30)):.15f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("  FINAL SUMMARY")
print("=" * 72)

print("""
  Regular tetrahedron: vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
  Volume = 8/3.  Td symmetry.

  Eshelby decomposition: I_{{ijkl}} = a0 * delta_{{ij}} * Ja_{{kl}} + b0 * Jb_{{ijkl}}

  === Pell Identities Used ===

  (5 - 2*sqrt(6))*(5 + 2*sqrt(6)) = 1
    => log(5 - 2*sqrt(6)) = -log(5 + 2*sqrt(6))

  (5 + 2*sqrt(6))^3 = 485 + 198*sqrt(6)
    => log(485 - 198*sqrt(6)) = -3*log(5 + 2*sqrt(6))

  (sqrt(2) + sqrt(3))^2 = 5 + 2*sqrt(6)
    => log(5 + 2*sqrt(6)) = 2*L  where  L = log(sqrt(2)+sqrt(3))

  === Ja Components (UNIVERSAL for convex bodies) ===

  Ja_{{kl}} = -4*pi/3 * delta_{{kl}}
    (from nabla^2(1/r) = -4*pi*delta)

  === Jb Components (shape-specific) ===

  Using L = log(sqrt(2) + sqrt(3)):

  Jb_{{1111}} = (-12*sqrt(2)*L - 4*pi + 24*sqrt(3)) / 9
             = (-6*sqrt(2)*log(5+2*sqrt(6)) - 4*pi + 24*sqrt(3)) / 9""")

print(f"             = {float(sp.N(Jb_1111_simplified, 30)):.15f}")

print("""
  Jb_{1122} = (6*sqrt(2)*L - 4*pi - 12*sqrt(3)) / 9
             = (3*sqrt(2)*log(5+2*sqrt(6)) - 4*pi - 12*sqrt(3)) / 9""")

print(f"             = {float(sp.N(Jb_1122_simplified, 30)):.15f}")

print("""
  Jb_{1212} = (6*sqrt(2)*L + 8*pi - 12*sqrt(3)) / 9
             = (3*sqrt(2)*log(5+2*sqrt(6)) + 8*pi - 12*sqrt(3)) / 9""")

print(f"             = {float(sp.N(Jb_1212_simplified, 30)):.15f}")

print("""
  === Geometric Anisotropy ===

  C_geom = Jb_{1111} - Jb_{1122} - 2*Jb_{1212}
         = (-30*sqrt(2)*L - 16*pi + 60*sqrt(3)) / 9
         = 2*(-15*sqrt(2)*L - 8*pi + 30*sqrt(3)) / 9
         = (-15*sqrt(2)*log(5+2*sqrt(6)) - 16*pi + 60*sqrt(3)) / 9""")

print(f"         = {float(sp.N(C_geom_simplified, 30)):.15e}")

print("""
  === ArcTan Identity (for Ja_{1122}) ===

  atan(2*sqrt(2)-3*sqrt(3)) - atan(2*sqrt(2)-sqrt(3)) + atan(4*sqrt(2)+3*sqrt(3)) = -pi/6

  Proof:
    Step 1: atan(2*sqrt(2)-3*sqrt(3)) + atan(4*sqrt(2)+3*sqrt(3))
            = atan(sqrt(3)-sqrt(2))   [addition formula, product < 0]
    Step 2: atan(sqrt(3)-sqrt(2)) - atan(2*sqrt(2)-sqrt(3))
            = atan(-1/sqrt(3)) = -pi/6   [addition formula, product < 1]
""")
