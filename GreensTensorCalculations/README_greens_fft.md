# Green's Tensor FFT Verification

Numerical verification that the spectral and spatial representations of the 3D elastodynamic Green's tensor are equivalent, via 2D inverse FFT of the post-residue spectral kernel.

## Files

| File | Purpose |
|------|---------|
| `greens_fft_cli.py` | Main CLI tool — FFT computation, error bounds, convergence sweeps |
| `baseline_kz_residue.py` | Original baseline: direct summation + kz residue verification |
| `baseline_fft_final.py` | Intermediate version establishing FFT = direct sum exactly |

## Physical Setup

Isotropic elastic half-space.  The equation of motion in the frequency domain:

```
[ρω² δ_ij − μK² δ_ij − (λ+μ) k_i k_j] G̃_jn(k) = −δ_in
```

The spectral Green's tensor (Christoffel matrix inverse via Sherman-Morrison):

```
G̃_ij(k) = −δ_ij / [ρ(ω² − β²K²)] − (α²−β²) k_i k_j / [ρ(ω²−β²K²)(ω²−α²K²)]
```

where `K² = kx² + ky² + kz²`, `α` = P-wave speed, `β` = S-wave speed.

## Method

### Step 1: kz residue (analytical)

For `z > z'` (receiver above source, or source at origin with z > 0), close the kz contour in the upper half plane.  Poles at `kz = +kzS` and `kz = +kzP` where:

```
kzS = √[(ω/β)² − kH²],    kzP = √[(ω/α)² − kH²],    kH² = kx² + ky²
```

With `Im(ω) > 0`, both `kzS` and `kzP` have `Im > 0` (upper half plane).

The post-residue kernel (kz integral evaluated):

```
Ĝ_ij(kx,ky; z) = (i/2ρ) × [δ_ij eˢ/(β² kzS)
                              + kᴾᵢ kᴾⱼ eᴾ/(ω² kzP)
                              − kˢᵢ kˢⱼ eˢ/(ω² kzS)]
```

where `eˢ = exp(i kzS z)`, `eᴾ = exp(i kzP z)`, `kˢ = (kx, ky, kzS)`, `kᴾ = (kx, ky, kzP)`.

### Step 2: 2D inverse Fourier transform (numerical, via FFT)

```
G_ij(x,y,z) = 1/(4π²) ∫∫ Ĝ_ij(kx,ky; z) exp(i[kx x + ky y]) dkx dky
```

This is a standard 2D IFT, computed exactly (for the trapezoidal quadrature) by `numpy.fft.ifft2`.

### FFT Grid Conventions

```
k-space:   k_n = (n − N/2) × dk,     dk = 2 kmax / N
x-space:   x_m = (m − N/2) × dx,     dx = π / kmax
Period:    L = N × dx = N π / kmax
```

Scaling: `G_spatial = fftshift( dk² N² / (4π²) × ifft2( ifftshift(Ĝ) ) )`

### Exact spatial formula (for comparison)

Ben-Menahem & Singh:

```
G_ij = f δ_ij + g γ_i γ_j,    γ = x/r

f = (1/4πρ) [φS/(β²r) + (φP−φS)/(ω²r³) + i(φS/β − φP/α)/(ωr²)]
g = (1/4πρ) [φP/(α²r) − φS/(β²r) + 3(φS−φP)/(ω²r³) − 3i(φS/β − φP/α)/(ωr²)]
```

where `φP = exp(ikP r)`, `φS = exp(ikS r)`, `kP = ω/α`, `kS = ω/β`.

## Error Bounds

The FFT introduces two sources of error relative to the exact integral:

### 1. Truncation Error (cutting off at |kH| = kmax)

For `kH > |ω/β|` the waves are evanescent: `kzS = iκ` with `κ ≈ kH`, so the kernel decays as `exp(−kH Δz)`.  The omitted tail is bounded by:

```
|ΔG_trunc| ≤ 1/(2π) ∫_{kmax}^∞ |Ĝ_max(kH)| kH dkH
```

In the deep evanescent regime (`kH ≫ |ω/β|`), the directional terms dominate:

```
|Ĝ_ij| ≲ kH/(ρ|ω|²) × exp(−kH Δz)
```

Integrating analytically with `p = kmax × Δz`:

```
|ΔG_trunc|/|G| ≲ exp(−p) × (p² + 2p + 2) / (2πρ|ω|² Δz³ × |G_onaxis|)
```

**Rule of thumb: need `kmax × Δz ≳ 15` for `|ΔG|/|G| < 10⁻²`, or `≳ 20` for `< 10⁻⁴`.**

The CLI computes this both analytically and via Gauss-Laguerre quadrature (tighter).

### 2. Aliasing Error (finite dk → periodic images)

The FFT evaluates a periodized Green's function with spatial period `L = Nπ/kmax`.  The aliasing error is the sum of contributions from periodic images:

```
|E_alias| = |Σ_{(m,n)≠(0,0)} G(x+mL, y+nL, z)|
```

The 8 nearest images (distance `~ L` and `√2 L`) are evaluated exactly using the Ben-Menahem & Singh formula.  Remaining images are bounded by the far-field decay `|G| ∼ exp(−Im(kS) r) / (4πρβ² r)`.

**Rule of thumb: need `Im(kS) × L ≳ 10` for `< 10⁻⁴`, where `Im(kS) = η ωr / β`.**

Since `L = Nπ/kmax`, this becomes `N > 10 kmax / (π η ωr / β)`.

### Total Error

```
|ΔG|/|G| ≤ |ΔG_trunc|/|G| + |E_alias|/|G|
```

The `--bounds` flag computes this without running the FFT (instant).

## CLI Usage

```bash
# Error bounds only (fast, no FFT):
python greens_fft_cli.py --z 0.04 --bounds

# Bounds + parameter sweep to find optimal (N, kmax):
python greens_fft_cli.py --z 0.04 --bounds --sweep

# Full tensor at default test points:
python greens_fft_cli.py --z 1.0

# Near-field, large grid:
python greens_fft_cli.py --z 0.02 --N 4096 --kmax 400

# Convergence sweep (computes FFT at each setting):
python greens_fft_cli.py --z 0.04 --sweep

# Single component (saves memory for huge N):
python greens_fft_cli.py --z 0.02 --N 4096 --kmax 400 --component 0,0

# Test at specific (x,y) points:
python greens_fft_cli.py --z 0.8 --N 4096 --kmax 25 --points "0.3,0.4;1.0,0.5"

# Save spatial map:
python greens_fft_cli.py --z 1.0 --N 4096 --kmax 25 --component 0,0 --save G00.npz

# Custom material:
python greens_fft_cli.py --z 0.5 --rho 2.7 --alpha 6.0 --beta 3.5 --eta 0.05
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--z` | required | Vertical separation Δz (must be > 0) |
| `--N` | 2048 | FFT grid size |
| `--kmax` | auto | Wavenumber truncation. Auto: `max(25, 15/z)` |
| `--rho` | 3.0 | Density |
| `--alpha` | 5.0 | P-wave speed |
| `--beta` | 3.0 | S-wave speed |
| `--omega` | 2π | Real angular frequency |
| `--eta` | 0.03 | `Im(ω)/Re(ω)` (complex frequency damping) |
| `--bounds` | off | Compute error bounds only (no FFT) |
| `--sweep` | off | Parameter sweep. With `--bounds`: sweep bounds only. Without: sweep with FFT. |
| `--component` | all | Single component `"i,j"` e.g. `"0,0"` |
| `--points` | default | Evaluation points `"x1,y1;x2,y2;..."` |
| `--save` | none | Save spatial map to `.npz` file |

### Output .npz Contents

When using `--save`, the file contains:

| Key | Description |
|-----|-------------|
| `component` | `(N,N)` complex array — `G_ij(x_m, y_n, z)` |
| `x_grid` | `(N,)` spatial coordinates |
| `z` | vertical separation |
| `i`, `j` | tensor indices |
| `N`, `kmax`, `dk`, `dx` | grid parameters |
| `omega_r`, `omega_i` | complex frequency |
| `rho`, `alpha`, `beta` | material parameters |
| `rel_error_bound` | conservative relative error bound |

## Intended Application

Scattering from small cubes of heterogeneity embedded in a plane-layered medium.  Each layer contains a uniform grid of cubes (Rayleigh regime: `kS a < 0.1`).  The scattered field from each cube is represented spectrally and propagated via the post-residue kernel to the nearest plane interface abutting the cube layer.

The critical regime is **near-interface propagation** (`Δz ≈ a`, cube half-width), where the evanescent spectrum is broad and large `kmax` is needed.  The `--bounds` mode lets you determine the required `N` and `kmax` before committing to the FFT computation.

## Verified Results

| Regime | z | kS·z | Best (N, kmax) | Actual error | Bound |
|--------|---|------|----------------|-------------|-------|
| Far-field | 1.0 | 2.09 | (2048, 25) | 2×10⁻⁷ | 5×10⁻⁵ |
| Near-field | 0.04 | 0.08 | (4096, 500) | 1×10⁻³ | 2×10⁻³ |
| Near-field | 0.02 | 0.04 | (4096, 400) | 5×10⁻⁵ (axis) | 3×10⁻³ |

Bounds are conservative by a factor of 2–50× depending on the regime.
