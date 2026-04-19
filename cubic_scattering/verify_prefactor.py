"""Verify body bilinear prefactor by comparing K1at-based formula vs direct 3D quadrature."""

import numpy as np
from scipy import integrate

from cubic_scattering.compute_gerade_blocks import _k1at


def F1(u):
    """ξ-integral for axis with a=b=1: ∫_{-(1-u)}^{1-u} (ξ²-u²) dξ."""
    return 2 * (1 - u) ** 3 / 3 - 2 * u**2 * (1 - u)


def F23(u):
    """ξ-integral for axis with a=b=0: 2(1-u)."""
    return 2 * (1 - u)


def integrand(u3, u2, u1):
    """F₁(u₁)F₂(u₂)F₃(u₃) / |u|."""
    r = np.sqrt(u1**2 + u2**2 + u3**2)
    if r < 1e-15:
        return 0.0
    return F1(u1) * F23(u2) * F23(u3) / r


# 3D quadrature on [0,1]³
result, err = integrate.tplquad(integrand, 0, 1, 0, 1, 0, 1, epsabs=1e-12, epsrel=1e-12)
print(f"Direct 3D integral: {result:.10f} ± {err:.2e}")
print(f"B_A (× 32): {32 * result:.6f}")
print()

# K1at values (from the Mp table)
K0 = _k1at(0, 0, 0)
K1 = _k1at(1, 0, 0)
K2 = _k1at(2, 0, 0)
print(f"K1at(0,0,0) = {K0:.10f}")
print(f"K1at(1,0,0) = {K1:.10f}")
print(f"K1at(2,0,0) = {K2:.10f}")

# Full sum (all monomials): R = {0: 8/3, 1: -16/3, 2: -16/3}
full = (8 / 3) * K0 + (-16 / 3) * K1 + (-16 / 3) * K2
print(f"\nFull K1at sum:       {full:.10f}")
print(f"B_A (× 32, full):   {32 * full:.6f}")

# Even-only (parity-filtered): R = {0: 8/3, 2: -16/3}
even = (8 / 3) * K0 + (-16 / 3) * K2
print(f"Even-only K1at sum:  {even:.10f}")
print(f"B_A (× 32, even):   {32 * even:.6f}")

# Cross-check: compute integral R × tent / |u| on [0,1]³ directly
# R₁(u) = 2/3 - 4u/3 - 4u²/3, R₂ = R₃ = 2
# K1at integrates R × tent / |u|
R_prod_full = (8 / 3) * K0 - (16 / 3) * K1 - (16 / 3) * K2
print(f"\nR×tent/|u| integral: {R_prod_full:.10f}")
print(f"  should match direct: {result:.10f}")
print(f"  ratio: {R_prod_full / result:.6f}" if result != 0 else "")

# Also verify against explicit 6D quadrature (expensive but definitive)
print("\n--- 6D verification (r₁ r₁' / |r-r'|) ---")


def integrand_6d(x):
    """r₁ × r₁' / |r-r'| for r, r' ∈ [-1,1]³."""
    r1, r2, r3, r1p, r2p, r3p = x
    dr = np.sqrt((r1 - r1p) ** 2 + (r2 - r2p) ** 2 + (r3 - r3p) ** 2)
    if dr < 1e-15:
        return 0.0
    return r1 * r1p / dr


# Use Monte Carlo for 6D
rng = np.random.default_rng(42)
n_samples = 5_000_000
samples = rng.uniform(-1, 1, size=(n_samples, 6))
volume = 2**6  # [-1,1]^6

vals = np.array([integrand_6d(s) for s in samples[:100_000]])
mc_result = volume * np.mean(vals)
mc_err = volume * np.std(vals) / np.sqrt(len(vals))
print(f"Monte Carlo (100k): {mc_result:.4f} ± {mc_err:.4f}")
print(f"  32 × direct_3D = {32 * result:.4f}")
print(f"  32 × even_only = {32 * even:.4f}")
