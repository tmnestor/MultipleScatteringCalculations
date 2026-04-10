# Plan: Full Multiple Scattering Simulation with the 27-Component Galerkin T-Matrix

## 1. Goal

Simulate plane-wave (P, SV, SH) scattering through a 3D lattice of
space-filling homogenised cubic voxels, each carrying its own elastic
contrast (Dlambda, Dmu, Drho) relative to a self-consistent CPA
effective medium.  The solution must include correct near-field
inter-voxel coupling --- not far-field approximations --- because
adjacent cubes are face-touching at separation d = 2a.

## 2. Architecture Overview

```
Plane wave u0
      |
      v
  ┌────────────────────────────────────────────────────┐
  │  Project u0 onto T27 basis for every voxel         │
  │  c0_m = M^{-1} <phi, u0(R_m + .)>                 │
  └───────────────────┬────────────────────────────────┘
                      v
  ┌────────────────────────────────────────────────────┐
  │  Foldy-Lax system:  (I - P . T) c = c0            │
  │                                                     │
  │  T_m   = 27x27 single-site T-matrix (per voxel)    │
  │  P(R)  = 27x27 inter-voxel propagator              │
  │  FFT-accelerated block-Toeplitz matvec              │
  │  Solve via GMRES (torch_gmres on GPU)               │
  └───────────────────┬────────────────────────────────┘
                      v
  ┌────────────────────────────────────────────────────┐
  │  Scattered field synthesis                          │
  │  Near-field: direct Green's tensor summation        │
  │  Far-field: multipole expansion (P, SV, SH)        │
  │  Observables: seismograms, cross sections, Q^{-1}  │
  └────────────────────────────────────────────────────┘
```

## 3. The 27-Component Basis

The trial space T27 decomposes as:

| Index range | Dim | Basis function          | Physical meaning        |
|-------------|-----|-------------------------|-------------------------|
| 1--3        | 3   | e_i (constant)          | Displacement monopole   |
| 4--9        | 6   | eps_{ij}(r) (linear)    | Symmetric strain        |
| 10--27      | 18  | r_a r_b e_i (quadratic) | Strain gradient / force quadrupole |

Under O_h, this splits into 7 irrep blocks:

- **Ungerade** (body + stiffness): T1u (4x4), T2u (2x2), A2u (1x1), Eu (1x1)
- **Gerade** (stiffness only): A1g (1x1), Eg (1x1), T2g (1x1)

The irrep T-matrices are already computed by `compute_cube_tmatrix_galerkin()`.


## 4. Components to Build

### 4.1  Full 27x27 T-matrix Assembly

**Status**: Per-irrep blocks exist.  Need to reassemble into full 27x27.

**Module**: `cubic_scattering/tmatrix_assembly.py` (new)

The symmetry-adapted basis change matrix U_sym (already in `effective_contrasts.py`
as `_build_usym_27()`) block-diagonalises the Galerkin system.  The full
T-matrix is:

```
T_27 = U_sym^T . blkdiag(T_T1u, T_T1u, T_T1u, T_T2u, T_T2u, T_T2u,
                          T_A2u, T_Eu, T_Eu, T_A1g, T_Eg, T_Eg, T_T2g,
                          T_T2g, T_T2g) . U_sym
```

where each irrep block T_rho is:
- Ungerade: T_rho = (M_rho + B^LS_rho - eps*B_body,rho)^{-1} (eps*B_body,rho - B^LS_rho)
- Gerade: T_rho = sigma_rho / (1 - sigma_rho)  (scalar, from Eshelby amplification)

The key subtlety: multiplicity placement.  T1u (dim 3, mult 4) contributes
three copies of the 4x4 block, one per Cartesian component of the triplet.
The U_sym columns tell you exactly which 27-space directions map to each
irrep copy.

**Functions**:
- `assemble_tmatrix_27(galerkin_result, ref, contrast, omega, a) -> np.ndarray(27,27)`
- `tmatrix_27_to_voigt_9(T27) -> np.ndarray(9,9)` (extract leading 9x9 for validation)
- `tmatrix_eigenvalues(T27) -> dict[str, np.ndarray]` (per-irrep check)


### 4.2  27x27 Inter-Voxel Propagator

**Status**: 9x9 propagator exists (`exact_propagator_9x9` in `horizontal_greens.py`).  Need 27x27 extension.

**Module**: `cubic_scattering/propagator_27.py` (new)

For two non-overlapping voxels centred at R_m and R_n (separation
R = R_m - R_n, |R| >= 2a), the Galerkin propagator is:

```
P_ab(R) = <phi_a^(m), G * source_b^(n)>
        = int_{V_m} phi_a(r - R_m) . G(r - r') . [body + surface sources of mode b at voxel n] dV dV'_or_dS'
```

Since voxels don't overlap, G(r-r') is smooth over V_m x V_n.
Taylor-expand G around the centre-to-centre separation R:

```
G_{ij}(r - r') = G_{ij}(R) + (r-R_m)_p d_p G_{ij}(R) - (r'-R_n)_q d_q G_{ij}(R)
               + 1/2 (r-R_m)_p (r-R_m)_q d_pq G_{ij}(R) + ...
```

Contract with the polynomial moments of phi_a and source_b over V.
For the constant+linear basis (T9), this gives:

| Test \ Source     | Constant (u_j)     | Strain (eps_{kl})     | Quadratic            |
|-------------------|--------------------|-----------------------|----------------------|
| Constant (u_i)    | G_{ij}(R)          | a d_k G_{ij}(R)      | a^2 d_{pq} G_{ij}   |
| Strain (eps_{ab}) | a d_a G_{ij}(R)    | a^2 d_{ak} G_{ij}(R) | a^3 d_{apq} G_{ij}  |
| Quadratic         | a^2 d_{pq} G_{ij}  | a^3 d_{pqk} G_{ij}   | a^4 d_{pqrs} G_{ij} |

Each entry involves G or its derivatives evaluated at R, contracted with
cube polynomial moments (S0 = V = 8a^3, S1 = 8a^3/3, S2 = 8a^5/5, etc.).

**Key requirement**: derivatives up to **4th order** of the full
elastodynamic Green's tensor.

The existing `elastodynamic_greens_deriv()` in `resonance_tmatrix.py` gives
G, G', G'' via Kupradze's radial-function decomposition.  We need to
extend this to G''' and G'''' for the quadratic-quadratic block.

For the **nearest-neighbour** voxels (|R| = 2a), the Taylor expansion
converges but slowly: the terms scale as (a/|R|)^n = (1/2)^n.  We need
enough terms to reach the target accuracy (~1% for seismic applications).
The quadratic-quadratic block at nearest-neighbour is O((a/R)^4) = 1/16
of the leading term, so the truncation at 4th derivative is adequate.

**Alternative**: Compute the inter-voxel propagator by direct numerical
integration (3D quadrature per component) as a validation, then use the
Taylor expansion for production.  The Galerkin integrals are smooth
(no singularity for non-overlapping cubes) so standard cubature works.

**Functions**:
- `greens_tensor_derivatives(R, alpha, beta, rho, order=4) -> list[np.ndarray]`
  Returns [G(R), dG(R), ddG(R), dddG(R), ddddG(R)] as tensors
- `propagator_27(R, ref, a, order=4) -> np.ndarray(27,27)`
  Full inter-voxel propagator at separation R
- `propagator_27_grid(Nx, Ny, Nz, ref, a) -> np.ndarray(Nx,Ny,Nz,27,27)`
  Propagator for all lattice separations (vectorised)


### 4.3  FFT-Accelerated Foldy-Lax Solver

**Status**: `LatticeGreens.matvec()` does FFT-accelerated 2D block-Toeplitz.
`torch_gmres` exists.  Need 3D extension and 27-component coupling.

**Module**: `cubic_scattering/foldy_lax_3d.py` (new)

The Foldy-Lax system for N = Nx * Ny * Nz voxels is:

```
(I - P . diag(T_1, ..., T_N)) c = c0
```

where:
- c is the (27*N)-vector of excited-field coefficients
- T_m is the 27x27 T-matrix of voxel m (varies per voxel)
- P is the N*N block matrix of 27x27 propagators P(R_m - R_n)
- c0 is the incident-field projection

The matvec `y = P . diag(T) . x` is computed as:
1. Elementwise: z_m = T_m . x_m  for each voxel m  [O(27^2 N)]
2. Convolution: y_m = sum_{n != m} P(R_m - R_n) . z_n  [FFT: O(27^2 N log N)]

Step 2 uses the block-Toeplitz structure: P depends only on (R_m - R_n).
Embed in a doubled grid (2Nx, 2Ny, 2Nz), FFT each of the 27*27 = 729
components, pointwise multiply, IFFT, extract.

For GPU acceleration, use the existing `torch_gmres` with a custom
`LinearOperator` that wraps this matvec.

**Memory**: For a 100^3 grid: N = 10^6 voxels, 27 components each.
- T-matrices: 27*27*N*16 bytes = ~12 GB (complex128) -- store per-phase, not per-voxel
- P grid: 27*27*(2N)^3... no, P is stored in k-space after FFT.
  P_hat: 27*27*(2Nx)*(2Ny)*(2Nz)*16 bytes.  For 200^3: 27^2 * 8M * 16 ~ 85 GB.
  **Too large.** Must exploit symmetry:
  - P is symmetric: P_{ab} = P_{ba} -> 27*28/2 = 378 unique components
  - Cubic lattice symmetry: P(R) has 48 O_h symmetries -> ~8x reduction
  - Store only unique Fourier components: ~85 / (2*8) ~ 5 GB.  Feasible.

For smaller grids (20^3 to 50^3 = 8K to 125K voxels), memory is no issue.

**Functions**:
- `FoldyLax3D` class:
  - `__init__(self, grid_shape, ref, a, tmatrix_field, propagator_fft)`
  - `matvec(x) -> y`  (FFT-accelerated)
  - `solve(c0, tol=1e-6) -> c`  (GMRES wrapper)
  - `scattered_field(c, c0) -> c_sc`
- `compute_propagator_fft(grid_shape, ref, a) -> np.ndarray`
  Pre-compute FFT of propagator on doubled grid


### 4.4  Plane-Wave Incident Field Projection

**Status**: Not implemented for T27 basis.

**Module**: `cubic_scattering/incident_field.py` (new)

For a monochromatic plane wave u0(r) = pol * exp(i k . r) with wavevector
k = omega/c * k_hat, the Galerkin projection onto basis function phi_alpha
centred at voxel R_m is:

```
<phi_alpha, u0>_V = int_{[-a,a]^3} phi_alpha(r) . [pol exp(ik.(R_m + r))] dV
                  = exp(ik.R_m) * pol_i * I_alpha(k, a)
```

where I_alpha(k, a) is the overlap integral of the polynomial basis with
exp(ik.r) over the cube.  For the cube [-a,a]^3 this factorises:

- Constant mode (phi = e_i):
  I = prod_j 2a sinc(k_j a) = V * sinc(k_1 a) sinc(k_2 a) sinc(k_3 a)

- Linear mode (phi = r_p e_i):
  I_p = [2ia(sin(k_p a) - k_p a cos(k_p a)) / (k_p a)^2] * prod_{j!=p} 2a sinc(k_j a)

- Quadratic mode (phi = r_p r_q e_i):
  I_{pq} = similar sinc-derivative products (exact)

Then c0_m = exp(ik.R_m) * M^{-1} . [pol_i * I_alpha(k,a)]

The phase factor exp(ik.R_m) varies per voxel but the shape factor
M^{-1} I_alpha is the **same for all voxels** with the same half-width a
(which is every voxel in a regular lattice).

**Wave types** (all with wavevector k = omega/c * k_hat):
- P-wave:  c = alpha,  pol = k_hat  (longitudinal)
- SV-wave: c = beta,   pol in (k_hat, z_hat) plane, perp to k_hat
- SH-wave: c = beta,   pol perp to (k_hat, z_hat) plane

**Functions**:
- `cube_overlap_integrals(k_vec, a) -> np.ndarray(27)`
  Analytical sinc-product integrals for all 27 basis functions
- `incident_field_vector(k_vec, pol, a, M_inv) -> np.ndarray(27)`
  Shape factor (same for all voxels on same lattice)
- `incident_field_grid(k_vec, pol, a, M_inv, grid_centres) -> np.ndarray(N, 27)`
  Full incident field with per-voxel phase factors
- `plane_wave_PSV_SH(k_hat, omega, ref) -> list[tuple[k_vec, pol]]`
  Generate all three wave types for a given direction and frequency


### 4.5  Scattered Field Synthesis and Observables

**Status**: Far-field exists for spheres (`mie_far_field`, `foldy_lax_far_field`).
Near-field exists in `resonance_tmatrix.py`.  Need 27-component extension.

**Module**: `cubic_scattering/scattered_field.py` (new)

The scattered field from voxel m with excited coefficients c_m is:

```
u_sc^(m)(r) = sum_beta c_sc,beta^(m) * [G convolved with source of mode beta]
```

**Multipole decomposition** (from the 27-component source):

| Component       | Source              | Radiation            | Angular order |
|-----------------|---------------------|----------------------|---------------|
| 3 constant      | Force monopole F_i  | Dipole               | l=1           |
| 6 strain        | Stress dipole s_ij  | Quadrupole           | l=2           |
| 18 quadratic    | Force quadrupole Q_ijk | Octupole          | l=3           |

The multipole amplitudes are linear functions of c_sc:
- F_i = omega^2 Drho V c_{sc,i}  (i=1..3)
- sigma_{ij} = Dc_{ijkl} eps_{kl}(u_sc)  contracted with V
- Q_{ijk} = higher-order moment

**Far field** (kr >> 1):

```
u_P^far(r) = -(1/4pi rho alpha^2) * (e^{ik_P r}/r) * [r_hat . F - ik_P r_hat.sigma.r_hat + ...]
u_S^far(r) = -(1/4pi rho beta^2)  * (e^{ik_S r}/r) * [F_perp - ik_S (sigma.r_hat)_perp + ...]
```

The ... contains the quadrupole and octupole terms from the quadratic modes.

**Near field** (for receivers inside or near the scattering region):
Direct summation of G(r - R_m) . source_m for each voxel m.  Use the
exact Kupradze Green's tensor (already in `elastodynamic_greens()`).

**Observables**:
- Scattering cross sections (total, differential) via optical theorem
- P-to-S conversion ratios
- Scattering-induced attenuation Q^{-1} (imaginary part of effective wavenumber)
- Seismograms at receiver positions
- Born scattering kernel (single-scattering approximation for imaging)

**Functions**:
- `multipole_amplitudes(c_sc, ref, contrast, omega, a) -> (F, sigma, Q)`
- `far_field_PSV_SH(r_hat, multipoles, ref, omega) -> (u_P, u_SV, u_SH)`
- `near_field(r, c_sc_grid, grid_centres, ref, omega) -> u(r)`
- `scattering_cross_section(c_sc_grid, c0_grid, ref, omega, a) -> sigma_sc`
- `optical_theorem_check(T27, k_vec, ref, omega, a) -> (sigma_ext, sigma_fwd)`


### 4.6  CPA Effective Medium (27-component extension)

**Status**: `cpa_iteration.py` exists for 9-component T-matrix.  Need extension to use 27-component Galerkin result.

The CPA condition <T> = 0 must hold for **all 27 components**, but by
O_h symmetry this reduces to 7 independent conditions (one per irrep).
The existing CPA already enforces <T1>=<T2>=<T3>=0 on the gerade sector.
For the ungerade sector, the CPA condition is:

```
sum_n phi_n * T_rho^(n) = 0   for each ungerade irrep rho
```

At leading order in ka, only the T1u dipole mode contributes to the
CPA condition (the quadratic modes are O((ka)^2) corrections).  So the
**existing 9-component CPA is correct to leading order**, and the
27-component extension is a higher-order frequency-dependent correction.

**Implementation**: For the first version, use the existing CPA to set
the background, then compute 27-component T-matrices relative to that
background.  CPA refinement at O((ka)^2) can be added later.


## 5. Implementation Phases

### Phase 1: Single-Scatterer Plane-Wave Response (2-3 weeks)

Build the infrastructure for a single cube in a homogeneous background:

1. `tmatrix_assembly.py`: Reassemble 27x27 from irrep blocks
2. `incident_field.py`: Plane-wave projection (P, SV, SH)
3. `scattered_field.py`: Far-field synthesis with multipole expansion
4. **Validation**: Compare with `compute_elastic_mie()` for a sphere
   (T27 cube results should approach Mie for ka -> 0, and the leading
   3 partial waves a0, a1, a2 should match the sphere limit).
5. **Validation**: Scattered power balance (optical theorem)

**Milestone**: Plot differential scattering cross sections for P, SV, SH
from a single cube at ka = 0.1, 0.3, 0.5.  Compare P-to-S conversion
with sphere Mie.

### Phase 2: Inter-Voxel Propagator (2-3 weeks)

1. `propagator_27.py`: Extend Green's tensor derivatives to 4th order
2. Compute 27x27 propagator for all lattice separations
3. **Validation**: For two cubes at various separations, compare with
   direct 6D numerical integration of the Galerkin bilinear form.
4. **Validation**: 27x27 propagator reduces to 9x9 propagator
   (`exact_propagator_9x9`) in the constant+linear subspace.
5. **Convergence study**: How many Taylor terms needed for nearest
   neighbours (|R|=2a) to achieve 1% accuracy?

**Milestone**: Plot propagator matrix elements vs separation for nearest,
next-nearest, and distant neighbours.  Verify 1/R decay of leading terms.

### Phase 3: FFT-Accelerated Foldy-Lax Solver (3-4 weeks)

1. `foldy_lax_3d.py`: 3D FFT matvec with 27-component blocks
2. GMRES solver wrapping `torch_gmres`
3. **Validation**: Small system (3x3x3 = 27 voxels) by direct inversion
   vs GMRES.  Compare with `compute_resonance_tmatrix()` for equivalent
   single-cube subdivision.
4. Scaling test: verify O(N log N) complexity for N up to 50^3.
5. **Homogeneous medium test**: All voxels identical -> T_eff should
   match CPA prediction.

**Milestone**: Solve 20^3 = 8000-voxel problem for a random binary
medium (two phases with ~10% contrast) at ka = 0.1.  Plot transmitted
wavefield through the slab.

### Phase 4: Full Simulation Pipeline (2-3 weeks)

1. Seismogram extraction at receiver positions
2. Born-approximation comparison (single-scattering)
3. Multiple-scattering convergence (Neumann series order analysis)
4. GPU acceleration for large grids (50^3+)
5. Connect to existing `seismic_survey.py` for shot gathers

**Milestone**: Simulate plane-wave transmission through a 50^3 random
medium at ka = 0.3.  Extract P-wave attenuation Q_P^{-1} and S-wave
attenuation Q_S^{-1}.  Compare with CPA effective-medium prediction.

### Phase 5: CPA Refinement and Applications (ongoing)

1. Frequency-dependent CPA using 27-component T-matrices
2. Anisotropic effective medium (VTI/HTI from aligned heterogeneity)
3. Surface-wave scattering (half-space geometry)
4. Coda-wave synthesis from random heterogeneity


## 6. Critical Design Decisions

### 6.1  Propagator: Taylor expansion vs direct integration

**Taylor expansion** (recommended for production):
- Pro: Fast, analytical, vectorises over all lattice separations
- Pro: Reuses existing Green's tensor derivative infrastructure
- Con: Needs 4th-order derivatives (tedious but finite)
- Con: Accuracy degrades for nearest neighbours

**Direct integration** (for validation):
- Pro: Exact to quadrature precision
- Con: Expensive (3D quadrature per propagator entry per separation)
- Use for: validating the Taylor expansion at nearest-neighbour distance

**Hybrid**: Use direct integration for |R| <= 2a*sqrt(3) (face/edge/corner
neighbours, ~26 separations), Taylor expansion for all others.  The 26
nearest-neighbour propagators can be precomputed once and cached.

### 6.2  Near-field correction for touching voxels

The touching-face case (|R| = 2a along an axis) deserves special
attention.  The Galerkin integral

```
P_{ab} = int_{V_m} int_{V_n} phi_a(r) G(r-r') source_b(r') dV dV'
```

has the integrand approaching the 1/|r-r'| singularity at the shared
face (r = (a, r2, r3), r' = (-a, s2, s3) -> |r-r'| = |(2a, r2-s2, r3-s3)|
which is >= 2a, so **no singularity is reached** for non-overlapping cubes).

However, for corner-touching neighbours (|R| = 2a*sqrt(3)), the closest
approach is |r-r'| = 0 at the shared corner --- a point singularity that
is integrable in 6D but may cause slow convergence.  The Taylor expansion
should be checked carefully at this distance.

### 6.3  Memory layout

For the FFT matvec, the field vector c has shape (Nx, Ny, Nz, 27).
The propagator FFT has shape (2Nx, 2Ny, 2Nz, 27, 27).

For GPU: use float32 (complex64) for the GMRES iteration, float64
(complex128) for the T-matrix and propagator precomputation.  This halves
memory for the solver while preserving accuracy in the physics.

### 6.4  Symmetry exploitation

For a **statistically isotropic** random medium (all orientations equally
likely), the O_h symmetry of the cube means:
- The CPA effective medium is isotropic (cubic anisotropy averages out)
- The average T-matrix <T> is isotropic
- The scattering cross section has full azimuthal symmetry

For a **layered** medium (horizontal voxel layers with different properties),
the effective medium is VTI (vertically transversely isotropic) and the
O_h symmetry reduces to D4h.

For an **aligned crack** medium (voxels with anisotropic contrasts),
the effective medium can be HTI or lower symmetry.

The 27-component framework handles all these cases: the T-matrix is
always computed in the full O_h-adapted basis, and the symmetry reduction
happens only in the averaging step.


## 7. Connections to Existing Code

| Existing module              | Role in new framework                              |
|------------------------------|---------------------------------------------------|
| `effective_contrasts.py`     | Single-site T-matrix (both Path-A and Galerkin)    |
| `resonance_tmatrix.py`      | Green's tensor + derivatives; validation target    |
| `horizontal_greens.py`      | 9x9 propagator (validation of 27x27 subblock)     |
| `lattice_greens.py`         | 2D FFT matvec (template for 3D extension)          |
| `cpa_iteration.py`          | Self-consistent effective medium background         |
| `voigt_tmatrix.py`          | 6x6 strain-space representation (observation)      |
| `sphere_scattering.py`      | Mie validation; far-field formulas                 |
| `torch_gmres.py`            | GPU-accelerated GMRES solver                       |
| `seismic_survey.py`         | Source/receiver geometry; seismogram synthesis      |
| `slab_scattering.py`        | 2D slab geometry (template for 3D finite slabs)    |

## 8. New Files

| File                              | Purpose                                    |
|-----------------------------------|--------------------------------------------|
| `tmatrix_assembly.py`             | 27x27 T-matrix from irrep blocks           |
| `incident_field.py`               | Plane-wave projection onto T27             |
| `propagator_27.py`                | 27x27 inter-voxel Green's propagator       |
| `foldy_lax_3d.py`                 | 3D FFT-accelerated multiple scattering     |
| `scattered_field.py`              | Near/far-field synthesis and observables    |
| `tests/test_tmatrix_assembly.py`  | Validation tests                           |
| `tests/test_propagator_27.py`     | Propagator convergence tests               |
| `tests/test_foldy_lax_3d.py`     | Multiple scattering solver tests           |


## 9. Estimated Computational Cost

For a grid of N = Nx * Ny * Nz voxels at frequency omega:

| Operation                     | Cost                    | N=20^3      | N=50^3     |
|-------------------------------|-------------------------|-------------|------------|
| T-matrix per phase            | O(27^3) per phase       | ~ms         | ~ms        |
| Propagator FFT (precompute)   | O(27^2 N log N)         | ~1 s        | ~30 s      |
| GMRES matvec                  | O(27^2 N log N)         | ~0.1 s      | ~3 s       |
| GMRES convergence (est.)      | 50-200 iterations       | ~10 s       | ~300 s     |
| Far-field synthesis           | O(N)                    | ~0.1 s      | ~1 s       |

Total per frequency: ~10 s (20^3) to ~5 min (50^3) on GPU.
