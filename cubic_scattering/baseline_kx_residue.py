#!/usr/bin/env python3
"""
Baseline verification: kx residue for horizontal plane-wave decomposition.

Starting from the 3D spectral Green's tensor (Eq 1.39b of the textbook):

  G̃_ik(ξ) = -(1/μ) [ ξ_iξ_k/k_T² (1/(ξ²-k_L²) - 1/(ξ²-k_T²))
                       - δ_ik/(ξ²-k_T²) ] × exp(iξ·y)

where ξ² = ξ₁² + ξ₂² + ξ₃²,  k_L = ω/α,  k_T = ω/β.

We apply the residue theorem to the ξ₁ (kx) integral for fixed (ξ₂, ξ₃).

═══════════════════════════════════════════════════════════════════════
DERIVATION: kx residue calculus
═══════════════════════════════════════════════════════════════════════

The scalar propagator integral in kx is:

  J_α(Δx; ξ₂, ξ₃) = ∫_{-∞}^{+∞} exp(iξ₁ Δx) / (ξ₁² + ξ₂² + ξ₃² - k_α²) dξ₁

The denominator factors as:
  ξ₁² - (k_α² - ξ₂² - ξ₃²) = (ξ₁ - ξ₁_α)(ξ₁ + ξ₁_α)

where:
  ξ₁_α = √(k_α² - ξ₂² - ξ₃²)

With Im(ω) > 0, we have Im(k_α) > 0, so Im(ξ₁_α) > 0 when the argument
of the square root has positive imaginary part. For |ξ₂² + ξ₃²| < |k_α²|
(propagating), ξ₁_α is near the real axis with small positive imaginary part.
For |ξ₂² + ξ₃²| > |k_α²| (evanescent), ξ₁_α = i|κ| with κ > 0.

Poles:
  ξ₁ = +ξ₁_α  (UHP, Im > 0)   — right-going wave
  ξ₁ = -ξ₁_α  (LHP, Im < 0)   — left-going wave

For Δx > 0: close in UHP, pick up ξ₁ = +ξ₁_α:
  J_α = 2πi × exp(iξ₁_α Δx) / (2ξ₁_α) = πi exp(iξ₁_α Δx) / ξ₁_α

For Δx < 0: close in LHP, pick up ξ₁ = -ξ₁_α:
  J_α = -2πi × exp(-iξ₁_α Δx) / (-2ξ₁_α) = πi exp(iξ₁_α |Δx|) / ξ₁_α

Both cases give: J_α = πi exp(iξ₁_α |Δx|) / ξ₁_α

Now we also need integrals involving ξ₁ in the numerator (from ξ_iξ_k terms).
The relevant integrals for the kernel are:

For the directional terms ξ_iξ_k/[k_T²(ξ²-k_α²)], when i or k = 1 (x-direction),
we need ξ₁ or ξ₁² in the numerator:

  ∫ ξ₁ exp(iξ₁ Δx) / [(ξ₁-ξ₁_α)(ξ₁+ξ₁_α)] dξ₁
    = 2πi × Res at ξ₁ = +ξ₁_α  (for Δx > 0)
    = 2πi × ξ₁_α exp(iξ₁_α Δx) / (2ξ₁_α)
    = πi exp(iξ₁_α Δx)

  ∫ ξ₁² exp(iξ₁ Δx) / [(ξ₁-ξ₁_α)(ξ₁+ξ₁_α)] dξ₁
    = 2πi × ξ₁_α² exp(iξ₁_α Δx) / (2ξ₁_α)
    = πi ξ₁_α exp(iξ₁_α Δx)

But wait — the ξ₁² integral has a subtlety. The integrand goes as
ξ₁²/(ξ₁²-a²) → 1 as ξ₁ → ∞, so it doesn't decay. We need to
regularize. Write:

  ξ₁²/(ξ₁²-a²) = 1 + a²/(ξ₁²-a²)

The "1" integrates to 2π δ(Δx), and the remainder is a² × J_α.
For Δx ≠ 0, the delta drops out and:

  ∫ ξ₁² exp(iξ₁ Δx) / (ξ₁²-a²) dξ₁ = a² × πi exp(i√a |Δx|) / √a
                                         = πi a^{3/2} exp(i√a |Δx|) / √a
  Hmm, let me be more careful.

Actually, let's work directly with the full kernel. The spectral Green's
tensor from Eq (1.39b) is:

  G̃_ik = -(1/μ) { ξ_iξ_k/k_T² [1/(ξ²-k_L²) - 1/(ξ²-k_T²)] - δ_ik/(ξ²-k_T²) }

Rearranging:
  G̃_ik = (1/μ) { δ_ik/(ξ²-k_T²) - ξ_iξ_k/k_T² [1/(ξ²-k_L²) - 1/(ξ²-k_T²)] }

The three types of kx integrals we need (for Δx > 0, closing in UHP):

(a) ∫ exp(iξ₁ Δx)/(ξ₁²-a²) dξ₁ = πi exp(iξ₁_α Δx)/ξ₁_α
    where a² = k_α² - ξ₂² - ξ₃², ξ₁_α = √a²

(b) ∫ ξ₁ exp(iξ₁ Δx)/(ξ₁²-a²) dξ₁ = πi exp(iξ₁_α Δx)
    (residue at +ξ₁_α: ξ₁_α × exp(...)/(2ξ₁_α) = exp(...)/2, times 2πi)

(c) For the ξ₁² numerator case — we don't actually encounter ξ₁²/(ξ²-k²)
    directly because ξ² = ξ₁² + ξ₂² + ξ₃², so:
    ξ_iξ_k/(ξ²-k²) with both i,k = 1 gives ξ₁²/(ξ₁²+ξ₂²+ξ₃²-k²)
    = ξ₁²/(ξ₁²-a²) where a² = k²-ξ₂²-ξ₃².

    As noted: ξ₁²/(ξ₁²-a²) = 1 + a²/(ξ₁²-a²).
    For Δx ≠ 0: the "1" term gives 2πδ(Δx) = 0.
    So: ∫ ξ₁² exp(iξ₁ Δx)/(ξ₁²-a²) dξ₁ = a² × πi exp(iξ₁_α Δx)/ξ₁_α
    = πi ξ₁_α exp(iξ₁_α Δx)   [for Δx ≠ 0]

So after the kx residue (Δx > 0), each propagator 1/(ξ²-k_α²) becomes:

  πi exp(iξ₁_α Δx) / ξ₁_α

And each ξ_iξ_k/(ξ²-k_α²) becomes (for Δx > 0):

  i=1, k=1:  πi ξ₁_α exp(iξ₁_α Δx)     [ξ₁² → ξ₁_α²/ξ₁_α × πi, with delta dropped]
  i=1, k≠1:  πi ξ_k exp(iξ₁_α Δx)       [ξ₁ × ξ_k, residue gives ξ_k × πi exp]
  i≠1, k=1:  πi ξ_i exp(iξ₁_α Δx)       [by symmetry]
  i≠1, k≠1:  πi ξ_iξ_k exp(iξ₁_α Δx)/ξ₁_α  [no ξ₁ in numerator]

Wait, let me be systematic. After kx residue at ξ₁ = ξ₁_α, we replace:
  ξ₁ → ξ₁_α  in all occurrences
  exp(iξ₁ Δx) → exp(iξ₁_α Δx)
  1/(ξ₁²-a²) → 1/(2ξ₁_α) (the residue factor)
  Times 2πi for the contour integral.

So ξ_i at the pole becomes:
  ξ₁ → ξ₁_α  (for i=1)
  ξ₂ → ξ₂     (unchanged, for i=2)
  ξ₃ → ξ₃     (unchanged, for i=3)

Define k^α = (ξ₁_α, ξ₂, ξ₃) — the wave vector at the α-wave kx-pole.

Then the post-kx-residue kernel for the (ξ₂, ξ₃) integral is:

  Ĝ_ik^(kx)(ξ₂, ξ₃; Δx) = (2πi/μ) × {
      δ_ik × [exp(iξ₁_T Δx)/(2ξ₁_T)]
    - (1/k_T²) × [k^L_i k^L_k exp(iξ₁_L Δx)/(2ξ₁_L)
                 - k^T_i k^T_k exp(iξ₁_T Δx)/(2ξ₁_T)]
  }

where k^L = (ξ₁_L, ξ₂, ξ₃), k^T = (ξ₁_T, ξ₂, ξ₃),
  ξ₁_L = √(k_L² - ξ₂² - ξ₃²),
  ξ₁_T = √(k_T² - ξ₂² - ξ₃²),
with Im(ξ₁_α) > 0.

Simplify: factor out πi/μ and rearrange:

  Ĝ_ik = (πi/μ) × [
      δ_ik e_T/ξ₁_T
    - k^L_i k^L_k e_L/(k_T² ξ₁_L)
    + k^T_i k^T_k e_T/(k_T² ξ₁_T)
  ]

where e_T = exp(iξ₁_T Δx), e_L = exp(iξ₁_L Δx).

Hmm, but μ = ρβ² and k_T = ω/β, so k_T² = ω²/β², giving:
  1/(μ k_T²) = 1/(ρβ² × ω²/β²) = 1/(ρω²)

So:
  Ĝ_ik = (πi/μ) × δ_ik e_T/ξ₁_T
        - (πi/(ρω²)) × k^L_i k^L_k e_L/ξ₁_L
        + (πi/(ρω²)) × k^T_i k^T_k e_T/ξ₁_T

But wait — let me re-derive this being very careful about signs.
From Eq (1.39b) in the textbook, the exp(iξ·y) is on the right.
The inverse FT is (1/(2π)³)∫ G̃ exp(iξ·(x-y)) dξ.
So the full integrand is G̃ × exp(iξ·(x-y)) = [kernel × exp(iξ·y)] × exp(iξ·(x-y))
= kernel × exp(iξ·x).

Actually, looking at the textbook more carefully:

G̃_ik(ξ) already contains exp(iξ·y), and the spatial Green's tensor is:

  G_ik(x,y) = (1/(2π)³) ∫ G̃_ik(ξ) exp(iξ·x) dξ   ← Hmm, that doesn't seem right.

Let me re-read. The textbook says:
  I_α = 1/(2π)³ ∫ exp{iξ·(x-y)} / (ξ²-k_α²) dξ

So the full spatial Green's tensor involves integrals of the form:
  (1/(2π)³) ∫ [spectral kernel without exp(iξ·y)] × exp(iξ·(x-y)) dξ

For the kx integral, with r_x = x₁ - y₁ = Δx:

  (1/2π) ∫ [kernel(ξ₁,ξ₂,ξ₃)] exp(iξ₁ Δx) dξ₁

The remaining integrals over (ξ₂, ξ₃) carry exp(iξ₂ Δy + iξ₃ Δz).

OK let me just implement this properly now.

NOTE ON NOTATION:
  Textbook: ξ = (ξ₁, ξ₂, ξ₃), k_L = ω/α, k_T = ω/β
  Our code:  k = (kx, ky, kz),  kP = ω/α, kS = ω/β
  Mapping: ξ₁↔kx, ξ₂↔ky, ξ₃↔kz, k_L↔kP, k_T↔kS
"""

from time import time

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Physics parameters
# ═══════════════════════════════════════════════════════════════
RHO = 3.0
ALPHA = 5.0  # P-wave speed
BETA = 3.0  # S-wave speed
ETA = 0.03
OMEGA = 2 * np.pi * (1 + 1j * ETA)

MU = RHO * BETA**2  # shear modulus
KP = OMEGA / ALPHA  # P-wave number (k_L)
KS = OMEGA / BETA  # S-wave number (k_T)


# ═══════════════════════════════════════════════════════════════
#  Exact spatial Green's tensor (Ben-Menahem & Singh / textbook Eq 1.40)
# ═══════════════════════════════════════════════════════════════
def exact_greens(x, y, z, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA):
    """
    G_ij = f δ_ij + g γ_i γ_j

    Using textbook notation:
      A = 1/(4πμ),  μ = ρβ²
      Û₁ ↔ f,  Û₂ ↔ g  (radial functions)
    """
    mu = rho * beta**2
    r = np.sqrt(x**2 + y**2 + z**2 + 0j)
    gam = np.array([x, y, z], dtype=complex) / r
    kP = omega / alpha
    kS = omega / beta
    phiP = np.exp(1j * kP * r)
    phiS = np.exp(1j * kS * r)
    fac = 1.0 / (4 * np.pi * rho)
    f = fac * (
        phiS / (beta**2 * r)
        + (phiP - phiS) / (omega**2 * r**3)
        + 1j * (phiS / beta - phiP / alpha) / (omega * r**2)
    )
    g = fac * (
        phiP / (alpha**2 * r)
        - phiS / (beta**2 * r)
        + 3 * (phiS - phiP) / (omega**2 * r**3)
        - 3j * (phiS / beta - phiP / alpha) / (omega * r**2)
    )
    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            G[i, j] = f * (1 if i == j else 0) + g * gam[i] * gam[j]
    return G


# ═══════════════════════════════════════════════════════════════
#  3D spectral Green's tensor (Eq 1.39b)
# ═══════════════════════════════════════════════════════════════
def spectral_greens(kx, ky, kz, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA):
    """
    G̃_ik(ξ) = -(1/μ) { ξ_iξ_k/k_T² [1/(ξ²-k_L²) - 1/(ξ²-k_T²)]
                         - δ_ik/(ξ²-k_T²) }

    Note: the exp(iξ·y) factor is NOT included — we handle that in the
    Fourier integral.
    """
    mu = rho * beta**2
    kP = omega / alpha
    kS = omega / beta
    kP2 = kP**2
    kS2 = kS**2
    xi = np.array([kx, ky, kz])
    xi2 = kx**2 + ky**2 + kz**2

    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            # Correct form derived from Eq (1.39a) partial fractions,
            # verified to be consistent with the kz residue baseline:
            G[i, j] = (1.0 / mu) * (
                dij / (xi2 - kS2)
                + xi[i] * xi[j] / kS2 * (1.0 / (xi2 - kP2) - 1.0 / (xi2 - kS2))
            )
    return G


# ═══════════════════════════════════════════════════════════════
#  kx residue: post-kx-residue kernel for Δx > 0
# ═══════════════════════════════════════════════════════════════
def post_kx_residue_kernel(ky, kz, dx, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA):
    """
    After applying the residue theorem to the kx integral for Δx > 0
    (close in UHP, pick up the right-going pole at kx = +kx_α).

    For each propagator 1/(ξ²-k_α²), the pole is at:
      kx_α = √(k_α² - ky² - kz²)   with Im(kx_α) > 0

    The kx integral (1/2π) ∫ F(kx) exp(ikx Δx) dkx gives residue
    contributions:

    (1/2π) × 2πi × Res_{kx=kx_α} = i × F evaluated at pole

    For 1/(kx²-a²) with a² = k_α²-ky²-kz²:
      Res at kx = +kx_α:  exp(ikx_α Δx)/(2 kx_α)

    For kx/(kx²-a²):
      Res at kx = +kx_α:  kx_α exp(ikx_α Δx)/(2 kx_α) = exp(ikx_α Δx)/2

    For kx²/(kx²-a²) = 1 + a²/(kx²-a²):
      For Δx ≠ 0: δ(Δx)→0, so Res = a² exp(ikx_α Δx)/(2 kx_α)
      = kx_α exp(ikx_α Δx)/2   [since a² = kx_α²]

    General: ξ_i ξ_k/(ξ²-k_α²) at kx residue, where ξ_1 → kx_α:
      kα_vec = (kx_α, ky, kz)
      Result: kα_vec[i] × kα_vec[k] × exp(ikx_α Δx) / (2 kx_α)
      EXCEPT when both i=0 AND k=0: get kx_α × exp(...)/2 (not kx_α²/(2kx_α))
      Actually: kx_α²/(2kx_α) = kx_α/2. Yes, consistent.

    So the post-residue kernel is:

    Ĝ_ik = i × (−1/μ) × {
        (kL_i kL_k)/(kS²) × eL/(2 kx_L) − (kT_i kT_k)/(kS²) × eT/(2 kx_T)
        − δ_ik × eT/(2 kx_T)
    }

    = (i/μ) × {
        δ_ik eT/(2 kx_T)
        − kL_i kL_k eL/(kS² × 2 kx_L)
        + kT_i kT_k eT/(kS² × 2 kx_T)
    }

    where eL = exp(ikx_L Δx), eT = exp(ikx_T Δx),
    kL_vec = (kx_L, ky, kz), kT_vec = (kx_T, ky, kz).

    Compare with the kz residue kernel:
      Ĝ_ik^(kz) = (i/2ρ) × [δ_ik eS/(β² kzS) + kP_i kP_j eP/(ω² kzP)
                              − kS_i kS_j eS/(ω² kzS)]

    Note: i/μ = i/(ρβ²), and 1/(μ kS²) = 1/(ρβ² × ω²/β²) = 1/(ρω²).
    So:
      i/(μ × 2 kx_T) = i/(2ρβ² kx_T)
      i/(μ kS² × 2 kx_L) = i/(2ρω² kx_L)

    The kx residue kernel is:
      Ĝ_ik = (i/2ρ) × [δ_ik eT/(β² kx_T)
                         + kL_i kL_k eL/(ω² kx_L)     ← WAIT, sign!
                         − kT_i kT_k eT/(ω² kx_T)]

    Let me recheck: from the spectral form with minus signs:
    G̃ = -(1/μ) { ξ_iξ_k/kS² [1/(ξ²-kP²) - 1/(ξ²-kS²)] - δ_ik/(ξ²-kS²) }

    The i× from the 2πi residue, plus the -1/μ prefactor:
    After residue:
      i × (-1/μ) × { kα_i kα_k/(kS² × 2kx_α) × [for P] − [for S] − δ_ik/(2kx_S) }

    P term: i × (-1/μ) × kL_i kL_k/(kS² × 2kx_L) × eL
           = -i kL_i kL_k eL / (2μ kS² kx_L)
           = -i kL_i kL_k eL / (2ρω² kx_L)

    S term: i × (-1/μ) × [-kT_i kT_k/(kS² × 2kx_T)] × eT
           = i kT_i kT_k eT / (2μ kS² kx_T)
           = i kT_i kT_k eT / (2ρω² kx_T)

    δ term: i × (-1/μ) × [-δ_ik/(2kx_T)] × eT
           = i δ_ik eT / (2μ kx_T)
           = i δ_ik eT / (2ρβ² kx_T)

    Total:
      Ĝ_ik = (i/2ρ) × [ δ_ik eT/(β² kx_T)
                          - kL_i kL_k eL/(ω² kx_L)
                          + kT_i kT_k eT/(ω² kx_T) ]

    Wait — this has a MINUS on the P term and PLUS on the S term.
    Compare with kz residue which had PLUS on P and MINUS on S:
      Ĝ_ik^(kz) = (i/2ρ) × [ δ_ik eS/(β² kzS)
                               + kP_i kP_j eP/(ω² kzP)
                               − kS_i kS_j eS/(ω² kzS) ]

    That's strange — let me re-derive more carefully...

    Actually, the sign depends on the partial fraction.
    1/(ξ²-kP²) - 1/(ξ²-kS²) = (kS²-kP²)/[(ξ²-kP²)(ξ²-kS²)]

    Hmm, wait. I should be more careful.

    Let me just implement it both ways and check numerically.
    """
    mu = rho * beta**2
    kP = omega / alpha
    kS = omega / beta
    kP2 = kP**2
    kS2 = kS**2

    # Horizontal kx-wavenumbers at the P and S poles
    kx_L = np.sqrt(kP2 - ky**2 - kz**2)
    kx_T = np.sqrt(kS2 - ky**2 - kz**2)

    # Enforce Im >= 0 (UHP convention for right-going waves with Δx > 0)
    if np.imag(kx_L) < 0:
        kx_L = -kx_L
    if np.imag(kx_T) < 0:
        kx_T = -kx_T

    eL = np.exp(1j * kx_L * dx)
    eT = np.exp(1j * kx_T * dx)

    # Wave vectors at the poles
    kL_vec = np.array([kx_L, ky, kz])
    kT_vec = np.array([kx_T, ky, kz])

    # Post-kx-residue kernel
    # Identical structure to the verified kz residue, with kz → kx:
    #
    # Ĝ_ik = (i/2ρ) × [ δ_ik eT/(β² kxT)
    #                    + kL_i kL_k eL/(ω² kxL)
    #                    − kT_i kT_k eT/(ω² kxT) ]
    #
    # Derived from the correct spectral form:
    #   G̃ = (1/μ) × [δ_ik/(ξ²-kS²) + ξ_iξ_k/kS² × (1/(ξ²-kP²) - 1/(ξ²-kS²))]
    #
    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            G[i, j] = (1j / (2 * rho)) * (
                dij * eT / (beta**2 * kx_T)
                + kL_vec[i] * kL_vec[j] * eL / (omega**2 * kx_L)
                - kT_vec[i] * kT_vec[j] * eT / (omega**2 * kx_T)
            )

    return G


# ═══════════════════════════════════════════════════════════════
#  Numerical kx integral (for verification)
# ═══════════════════════════════════════════════════════════════
def numerical_kx_integral(
    ky, kz, dx, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA, kx_max=80, nkx=8192
):
    """
    Direct numerical integration of the kx integral:
      G_ik = (1/2π) ∫ G̃_ik(kx, ky, kz) exp(ikx Δx) dkx
    """
    kx_arr = np.linspace(-kx_max, kx_max, nkx)
    dkx = kx_arr[1] - kx_arr[0]

    G = np.zeros((3, 3), dtype=complex)
    for _idx, kx in enumerate(kx_arr):
        Gspec = spectral_greens(kx, ky, kz, omega, rho, alpha, beta)
        phase = np.exp(1j * kx * dx)
        G += Gspec * phase * dkx

    G /= 2 * np.pi
    return G


# ═══════════════════════════════════════════════════════════════
#  Full 2D (ky, kz) integral after kx residue → spatial G
# ═══════════════════════════════════════════════════════════════
def spectral_2d_integral_kx(
    dx, dy, dz, omega=OMEGA, rho=RHO, alpha=ALPHA, beta=BETA, kmax=25, nk=256
):
    """
    After kx residue, compute:
      G_ik(Δx, Δy, Δz) = (1/(2π)²) ∫∫ Ĝ_ik(ky, kz; Δx) exp(i(ky Δy + kz Δz)) dky dkz
    """
    k1d = np.linspace(-kmax, kmax, nk)
    dk = k1d[1] - k1d[0]

    G = np.zeros((3, 3), dtype=complex)
    for ky in k1d:
        for kz in k1d:
            kernel = post_kx_residue_kernel(ky, kz, abs(dx), omega, rho, alpha, beta)
            phase = np.exp(1j * (ky * dy + kz * dz))
            G += kernel * phase * dk**2

    G /= (2 * np.pi) ** 2
    return G


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("Baseline kx residue verification")
    print("=" * 72)
    print(f"  ρ={RHO}, α={ALPHA}, β={BETA}")
    print(f"  ω = {OMEGA:.6f}")
    print(f"  |kP|={abs(KP):.4f},  |kS|={abs(KS):.4f}")
    print()

    # ─── STEP 1: kx residue vs numerical kx integral ───
    print("STEP 1: kx residue vs numerical kx integral")
    print("─" * 72)

    test_cases = [
        # (ky, kz, Δx)
        (0.5, 0.3, 1.0),
        (1.0, 0.5, 0.5),
        (0.0, 0.0, 1.0),
        (2.0, 1.5, 0.8),  # evanescent regime
        (0.3, 0.7, 0.3),
    ]

    for ky, kz, dx in test_cases:
        G_res = post_kx_residue_kernel(ky, kz, dx)
        G_num = numerical_kx_integral(ky, kz, dx)

        err = np.linalg.norm(G_res - G_num) / np.linalg.norm(G_num)
        print(f"  ky={ky:5.2f}, kz={kz:5.2f}, Δx={dx:4.2f}  →  Frob err = {err:.4e}")

    print()

    # ─── STEP 2: Full 2D integral (after kx residue) vs exact ───
    print("STEP 2: 2D (ky,kz) integral after kx residue vs exact spatial formula")
    print("─" * 72)

    test_points = [
        # (Δx, Δy, Δz)
        (1.0, 0.0, 0.0),  # pure x-separation (horizontal)
        (0.5, 0.3, 0.0),  # horizontal, off-axis
        (0.8, 0.4, 0.2),  # general 3D point
        (0.0, 0.5, 0.3),  # no x-separation (Δx=0 special case)
        (1.0, 0.5, 0.5),  # general
    ]

    for nk in [128, 256]:
        kmax = 25
        print(f"\n  nk={nk}, kmax={kmax}:")
        for dx, dy, dz in test_points:
            if dx == 0.0:
                print(
                    f"  Δx={dx:5.2f}, Δy={dy:5.2f}, Δz={dz:5.2f}  →  "
                    f"SKIPPED (Δx=0 needs separate treatment)"
                )
                continue

            G_spec = spectral_2d_integral_kx(dx, dy, dz, kmax=kmax, nk=nk)
            G_exact = exact_greens(dx, dy, dz)

            err = np.linalg.norm(G_spec - G_exact) / np.linalg.norm(G_exact)
            print(
                f"  Δx={dx:5.2f}, Δy={dy:5.2f}, Δz={dz:5.2f}  →  Frob err = {err:.4e}"
            )

    print()

    # ─── STEP 3: Component detail at best point ───
    print("STEP 3: Component detail")
    print("─" * 72)
    dx, dy, dz = 1.0, 0.5, 0.5
    G_spec = spectral_2d_integral_kx(dx, dy, dz, kmax=25, nk=256)
    G_exact = exact_greens(dx, dy, dz)

    print(f"  Point: ({dx}, {dy}, {dz})")
    print(
        f"  Frobenius error: {np.linalg.norm(G_spec - G_exact) / np.linalg.norm(G_exact):.4e}"
    )
    print()
    for i in range(3):
        for j in range(3):
            s, e = G_spec[i, j], G_exact[i, j]
            if abs(e) > 1e-20:
                ce = abs(s - e) / abs(e)
                print(f"  G[{i},{j}]: residue={s:.8e}  exact={e:.8e}  err={ce:.2e}")
            else:
                print(f"  G[{i},{j}]: residue={s:.8e}  (exact≈0)")

    # ─── STEP 4: Convergence study ───
    print()
    print("STEP 4: Convergence in nk")
    print("─" * 72)
    dx, dy, dz = 1.0, 0.3, 0.4
    G_exact = exact_greens(dx, dy, dz)
    G_mag = np.linalg.norm(G_exact)

    for nk in [64, 128, 256, 512, 1024]:
        for kmax in [15, 25, 40]:
            t0 = time()
            G_spec = spectral_2d_integral_kx(dx, dy, dz, kmax=kmax, nk=nk)
            dt = time() - t0
            err = np.linalg.norm(G_spec - G_exact) / G_mag
            print(f"  nk={nk:4d}, kmax={kmax:3d}  →  err={err:.4e}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
