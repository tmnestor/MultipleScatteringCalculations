# Horizontal Green's Tensor via Cartesian Residue Decomposition

**T.M. Nestor — ANU Thesis, March 2026**

## Context

The thesis formulates seismic scattering from small cubes of heterogeneity embedded in plane-layered media. Each layer contains a uniform horizontal grid of cubes (Rayleigh regime, kS·a < 0.1) with distinct material properties (within-cube averages). The scattered field must be propagated both vertically (to plane interfaces above and below) and horizontally (inter-cube multiple scattering on the same grid).

Previous work established the **vertical propagation** via kz residues: the Green's tensor is decomposed into upgoing/downgoing plane waves, and the remaining 2D (kx, ky) integral is evaluated by IFFT. This was verified to ~10⁻⁶ against the exact Ben-Menahem & Singh spatial formula (see `greens_fft_cli.py` and `README_greens_fft.md`).

This document covers the extension to **horizontal propagation** via kx and ky residues, which decompose the Green's tensor into directional plane waves for inter-cube coupling.

## Physical Problem

All scatterers in a given layer sit at the same depth. The effective medium contrast handles intra-cube scattering self-consistently, but inter-cube coupling requires the lateral Green's tensor:

    G_ik(Δx, Δy, 0)     for all (Δx, Δy) on the cube grid

The kz residue approach is not directly applicable here because Δz = 0. Instead, we apply residue calculus to kx or ky, splitting horizontal propagation into directional plane waves — exactly analogous to the vertical case.

Three coupling cases arise on the 2D grid:

    Case 1: Δx ≠ 0         → kx residue (left/right plane waves)
    Case 2: Δx = 0, Δy ≠ 0 → ky residue (into/out-of-page plane waves)
    Case 3: Δx = Δy = 0    → self-interaction (handled by effective medium)

## Mathematical Derivation

### Starting Point: 3D Spectral Green's Tensor

From the textbook (Eq 1.39b), after partial fraction decomposition:

    G̃_ik(ξ) = (1/μ) [ δ_ik/(ξ²-kS²) + ξ_iξ_k/kS² × (1/(ξ²-kP²) - 1/(ξ²-kS²)) ]

where ξ² = ξ₁² + ξ₂² + ξ₃², kP = ω/α (P-wave), kS = ω/β (S-wave), μ = ρβ².

**Sign note:** The original textbook writes this with an overall minus and subtracted δ term. After careful expansion and verification against the working kz residue baseline, the correct form has the signs shown above: positive δ_ik and positive directional ξ_iξ_k coupling. This was confirmed numerically.

### Universal Residue Calculus

The spatial Green's tensor is:

    G_ik(r) = 1/(2π)³ ∫∫∫ G̃_ik(ξ) exp(iξ·r) d³ξ

For the residue in any Cartesian direction (call it ξ_n) with fixed remaining components, each propagator 1/(ξ²-kα²) has poles at:

    ξ_n = ± √(kα² - [sum of other ξ² terms])  ≡  ± ξn_α

With Im(ω) > 0 (causal damping), Im(ξn_α) > 0, placing +ξn_α in the upper half-plane and -ξn_α in the lower half-plane.

The residue evaluation gives, for each term:

- Scalar propagator: ∫ exp(iξn Δn)/(ξn²-a²) dξn → 2πi × exp(iξn_α Δn)/(2ξn_α)
- Linear numerator: ∫ ξn exp(iξn Δn)/(ξn²-a²) dξn → 2πi × exp(iξn_α Δn)/2
- Quadratic numerator: ∫ ξn² exp(iξn Δn)/(ξn²-a²) dξn → 2πδ(Δn) + a² × [scalar result]
  (the delta vanishes for Δn ≠ 0)

### Universal Post-Residue Kernel

All three Cartesian residue decompositions yield the same kernel structure:

    Ĝ_ik = (i/2ρ) × [ δ_ik eT/(β² kn_T)
                       + kL_i kL_k eL/(ω² kn_L)
                       − kT_i kT_k eT/(ω² kn_T) ]

where:
- kn_T, kn_L are the residue-direction components of the S and P wave vectors
- eT, eL are the exponential phase/decay factors
- kL, kT are the full 3D wave vectors evaluated at the respective poles

The specific instantiations are:

**kz residue (vertical):** kn = kz, remaining = (kx, ky)
- kzL = √(kP² - kx² - ky²), kzT = √(kS² - kx² - ky²)
- kL = (kx, ky, kzL), kT = (kx, ky, kzT)

**kx residue (lateral):** kn = kx, remaining = (ky, kz)
- kxL = √(kP² - ky² - kz²), kxT = √(kS² - ky² - kz²)
- kL = (kxL, ky, kz), kT = (kxT, ky, kz)

**ky residue (transverse):** kn = ky, remaining = (kx, kz)
- kyL = √(kP² - kx² - kz²), kyT = √(kS² - kx² - kz²)
- kL = (kx, kyL, kz), kT = (kx, kyT, kz)

### Case 1: Δx ≠ 0 (kx Residue)

After the kx residue, the remaining integral for Δz = 0 is:

    G_ik(Δx, Δy, 0) = 1/(2π)² ∫∫ Ĝ_ik(ky, kz; Δx) × exp(i ky Δy) dky dkz

The exp(ikz·0) = 1 factor means the kz integral has no oscillatory phase. This splits into:

1. **ky: 1D IFFT** — gives all Δy values on the grid simultaneously
2. **kz: trapezoidal quadrature** — no phase factor, fast convergence

### Case 2: Δx = 0, Δy ≠ 0 (ky Residue)

When Δx = 0, the kx residue is inapplicable (no directional separation to close the contour). Instead we apply residue calculus to ky:

    G_ik(0, Δy, 0) = 1/(2π)² ∫∫ Ĝ_ik(kx, kz; Δy) × exp(i kx · 0) dkx × exp(i kz · 0) dkz

Both remaining integrals have exp(i·0) = 1 — no oscillatory phase. The convergence comes entirely from the evanescent decay exp(i kyα |Δy|) for large |kx| and |kz|:

1. **kx: trapezoidal quadrature** — no phase factor (Δx = 0)
2. **kz: trapezoidal quadrature** — no phase factor (Δz = 0)

This is a double quadrature and converges more slowly than Case 1 (which uses a 1D IFFT for ky). However, for a grid of cubes, Δx = 0 only occurs along a single column, so fewer evaluations are needed.

## Implementation

### Files

| File | Purpose |
|------|---------|
| `baseline_kx_residue.py` | kx residue derivation and step-by-step verification |
| `horizontal_greens.py` | Production code: both kx and ky residue cases |

### Key Functions

**`baseline_kx_residue.py`:**
- `spectral_greens(kx, ky, kz)` — full 3D spectral Green's tensor
- `post_kx_residue_kernel(ky, kz, dx)` — scalar (ky, kz) kernel after kx residue
- `numerical_kx_integral(ky, kz, dx)` — direct numerical kx integration (for verification)
- `spectral_2d_integral_kx(dx, dy, dz)` — direct 2D (ky, kz) quadrature

**`horizontal_greens.py`:**

*Case 1 (Δx ≠ 0):*
- `post_kx_residue_kernel_vec(ky_arr, kz, dx_abs)` — vectorised kernel over ky array
- `horizontal_greens_fft(dx_abs, Nky, ky_max, kz_max, Nkz)` — kx residue + ky IFFT + kz quadrature; returns G_ik at all Δy grid points

*Case 2 (Δx = 0):*
- `post_ky_residue_kernel_vec(kx_arr, kz, dy_abs)` — vectorised kernel over kx array
- `horizontal_greens_ky_residue(dy_abs, kx_max, Nkx, kz_max, Nkz)` — ky residue + kx quadrature + kz quadrature; returns G_ik at a single (0, Δy, 0) point

*Shared:*
- `exact_greens(x, y, z)` — Ben-Menahem & Singh spatial formula (reference)
- `horizontal_greens_direct(dx_abs, dy, kmax, nk)` — direct 2D quadrature (slow, for comparison)

### FFT Conventions (Case 1)

    ky grid:   ky_n = (n - Nky/2) × dky,    dky = 2 ky_max / Nky
    Δy grid:   Δy_m = (m - Nky/2) × dy,     dy  = π / ky_max
    Period:    Ly = Nky π / ky_max

    Scaling:   G(Δy) = (dky Nky / 2π) × fftshift(ifft(ifftshift(kernel)))

    kz grid:   uniform on [-kz_max, kz_max], Nkz points, trapezoidal sum with factor dkz/(2π)

### Quadrature Conventions (Case 2)

    kx grid:   uniform on [-kx_max, kx_max], Nkx points
    kz grid:   uniform on [-kz_max, kz_max], Nkz points

    Scaling:   G = Σ_kx Σ_kz  Ĝ_ik(kx, kz; Δy) × dkx × dkz / (2π)²

## Verification Results

### Case 1: kx Residue (Δx ≠ 0)

**Step 1: kx Residue vs Numerical kx Integral**

At individual (ky, kz) points, the residue formula matches direct numerical kx integration to < 5×10⁻⁴ across propagating and evanescent regimes. This confirms the residue mechanics (pole locations, sign conventions, UHP closure).

**Step 2: Full Horizontal Green's Tensor vs Exact**

| Δx  | Δy  | Nky  | ky_max | Nkz | kz_max | Frobenius error |
|-----|-----|------|--------|-----|--------|----------------|
| 0.8 | 0.0 | 1024 | 15     | 512 | 15     | 7.7 × 10⁻⁵    |
| 0.8 | 0.2 | 1024 | 15     | 512 | 15     | 7.9 × 10⁻⁵    |
| 0.8 | 0.5 | 1024 | 15     | 512 | 15     | 8.4 × 10⁻⁵    |
| 0.8 | 1.0 | 1024 | 15     | 512 | 15     | 1.3 × 10⁻⁴    |
| 1.0 | 0.0 | 512  | 15     | 512 | 15     | 1.1 × 10⁻⁴    |

All 9 tensor components verified individually. Components involving Δz (i.e., G_i3 and G_3j for i,j ≠ 3) correctly vanish to machine precision when Δz = 0.

**Convergence Behaviour (Case 1):**
- **ky direction (IFFT):** Nky = 512 is sufficient; increasing to 1024 gives marginal improvement.
- **kz direction (quadrature):** Nkz = 512 is the key parameter. Going from 256 → 512 typically improves errors by 10×.
- **Truncation:** ky_max = kz_max = 15 is adequate for Δx ≥ 0.5. The evanescent decay exp(-κ|Δx|) ensures rapid convergence.
- **Timing:** ~0.2s for Nky=512, Nkz=512. The computation is dominated by the kz loop (512 1D IFFTs).

### Case 2: ky Residue (Δx = 0)

| Δy  | Nkx  | Nkz  | kx/kz_max | Frobenius error |
|-----|------|------|-----------|----------------|
| 0.3 | 512  | 512  | 15        | 1.4 × 10⁻²    |
| 0.5 | 512  | 512  | 15        | 8.3 × 10⁻⁴    |
| 0.8 | 512  | 512  | 15        | 1.0 × 10⁻⁴    |
| 0.8 | 768  | 768  | 15        | 1.2 × 10⁻⁵    |
| 0.8 | 1024 | 1024 | 20        | 8.7 × 10⁻⁶    |
| 1.0 | 512  | 512  | 15        | 1.2 × 10⁻⁴    |
| 1.5 | 512  | 512  | 15        | 1.9 × 10⁻⁴    |

Components G_00 = G_22 (by symmetry for Δx=Δz=0), G_11 is the distinct diagonal component, and all off-diagonals vanish to machine precision.

**Convergence Behaviour (Case 2):**
- **Double quadrature** converges more slowly than the IFFT+quadrature of Case 1, requiring N=768 or larger for errors below 10⁻⁵.
- **Truncation:** kmax = 15 is adequate for Δy ≥ 0.5; smaller Δy values need larger truncation or finer grids.
- **Timing:** ~0.1–0.3s for N=512–1024. Dominated by the double loop over kx and kz.

## Bug Found and Fixed

The initial implementation used the spectral form:

    G̃_ik = -(1/μ) { ξ_iξ_k/kS² [1/(ξ²-kP²) - 1/(ξ²-kS²)] - δ_ik/(ξ²-kS²) }

as read from the textbook. This produced ~100% errors in the 2D integral (off-diagonal components were sign-flipped). After careful re-derivation from Eq (1.39a) via partial fractions and cross-checking against the verified kz residue baseline, the correct form was found to be:

    G̃_ik = +(1/μ) × [δ_ik/(ξ²-kS²) + ξ_iξ_k/kS² × (1/(ξ²-kP²) - 1/(ξ²-kS²))]

The sign difference on the directional ξ_iξ_k term is the key correction. This was verified by requiring consistency with the kz residue kernel that was already validated against the exact spatial formula.

## Complete Scattering Framework

The three Cartesian residue decompositions form the complete framework:

| Direction  | Residue variable | Splits into              | Use case                            |
|------------|-----------------|--------------------------|-------------------------------------|
| Vertical   | kz              | Upgoing / downgoing      | Propagation to plane interfaces     |
| Lateral    | kx              | Right-going / left-going | Inter-cube coupling (Δx ≠ 0)       |
| Transverse | ky              | Into-page / out-of-page  | Inter-cube coupling (Δx = 0)       |

All three produce the same kernel structure:

    Ĝ_ik = (i/2ρ) × [ δ_ik e_S/(β² k_res_S) + k^P_i k^P_k e_P/(ω² k_res_P) − k^S_i k^S_k e_S/(ω² k_res_S) ]

with k_res being the residue-direction component of the wavenumber (kz, kx, or ky respectively).

### Practical Workflow for 2D Grid Coupling

For horizontal multiple scattering on a 2D cube grid at Δz = 0:

    For Δx ≠ 0 (each column offset):
      1. Apply kx residue for this |Δx|
      2. ky IFFT gives all Δy values simultaneously → fills one row of the coupling matrix
      3. kz quadrature (Δz = 0) is a simple trapezoidal sum
      4. Use left/right symmetry G(−Δx, Δy) = G(Δx, Δy) to halve the work

    For Δx = 0 (same column):
      1. Apply ky residue for each |Δy|
      2. kx quadrature (Δx = 0) — no phase factor
      3. kz quadrature (Δz = 0) — no phase factor
      4. Use front/back symmetry G(0, −Δy) = G(0, Δy) to halve the work

    Δx = Δy = 0 (self-interaction):
      Handled by effective medium contrast (not the Green's tensor)

This enables a propagator-style sweep across the grid — left-to-right, then right-to-left — analogous to the reflectivity method used vertically through layers, but now applied laterally.

## Parameters

    ρ = 3.0,  α = 5.0,  β = 3.0
    ω = 2π(1 + 0.03i)
    |kP| = 1.2572,  |kS| = 2.0953
