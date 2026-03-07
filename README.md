# CubicTmatrix: Elastic Multiple Scattering from Cubic Heterogeneities

T-matrix approach for seismic wave scattering from Rayleigh-regime cubic
inclusions in layered elastic media. The global T-matrix equation
`T = T₀ (I − G₀ T₀)⁻¹` couples single-site scattering matrices `T₀` through
inter-site Green's tensors `G₀`, solved via Krylov iteration. Originates from
Nestor (1996) PhD thesis, Australian National University.

## Physics overview

Three computational pillars underpin the multiple-scattering formulation:

| Pillar | Symbol | Role |
|--------|--------|------|
| Single-site T-matrix | `T₀` | Lippmann–Schwinger integral for each cube (analytic, Taylor-expanded Green's tensor) |
| Inter-site Green's tensor | `G₀` | Elastodynamic coupling between cube centres (spatial, spectral, or hybrid evaluation) |
| Multiple-scattering solve | `T` | Block-Toeplitz system solved by FFT-accelerated GMRES |

## Repository structure

```
CubicTmatrix/
├── cubic_scattering/                # Main Python package (T₀ + G₀)
│   ├── effective_contrasts.py       #   T₀: Lippmann–Schwinger, Eshelby, amplification
│   ├── voigt_tmatrix.py             #   T₀: 6×6 Voigt in (uz,ux,uy,tzz,txz,tyz) basis
│   ├── lattice_greens.py            #   G₀: spatial, spectral, hybrid, FCC, matvec
│   ├── horizontal_greens.py         #   G₀: exact Green's tensor at Δz=0
│   ├── greens_fft_cli.py            #   G₀: 2D FFT verification CLI
│   ├── baseline_kx_residue.py       #   Research: kx residue exploration
│   ├── baseline_kz_residue.py       #   Research: kz residue exploration
│   ├── baseline_fft_final.py        #   Research: FFT baseline
│   ├── tests/
│   │   └── test_cubic_tmatrix.py    #   T₀ tests (9 tests)
│   └── derivations/
│       ├── elastic_greens.py        #   SymPy: symbolic Green's tensor
│       ├── tmatrix_analytic.py      #   SymPy: analytical sphere T-matrix
│       └── tmatrix_cube.py          #   SymPy: analytical cube T-matrix
│
├── GreensTensorCalculations/        # LaTeX documents & Mathematica notebooks
│   ├── basic_equations.tex          #   LaTeX: fundamental equations
│   ├── cubic_tmatrix.tex            #   LaTeX: cubic T-matrix with full Green's tensor
│   └── nearfield_verification.tex   #   LaTeX: near-field verification
│
├── FFTProp.py/                      # 2.5D spectral scattering (Python port of FFTPROP.F)
│   ├── fftprop_driver.py            #   Orchestrator: compute_wavefield()
│   ├── propagation.py               #   Four-directional propagation sweeps
│   ├── spectral_arrays.py           #   Wavenumber grids, free-surface Rayleigh reflection
│   └── README.md                    #   Detailed documentation
│
├── PhD_fortran_code/                # Original Fortran 77 from Nestor (1996)
│   ├── FFTPROP.F                    #   FFT propagation (source of FFTProp.py port)
│   ├── GMRES_P.F, GMRESP.F         #   GMRES Krylov solvers
│   ├── BORN.F                       #   Born approximation
│   ├── Kennett_Reflectivity/        #   Python Kennett reflectivity package
│   └── ...                          #   ~30 Fortran source files
│
├── Mathematica/                     # Symbolic computation packages
│   ├── ElasticGreensFunction.m      #   Elastic Green's function
│   ├── SmallSphereScattering.m      #   Small sphere scattering
│   └── TMatrixAnalytic.m            #   Analytical T-matrix
│
├── envs/
│   └── seismic.yml                  # Conda environment specification
│
├── verify_nearfield.py              # Near-field verification (Wu & Ben-Menahem)
├── create_global_tmatrix_svg.py     # SVG schematic generator
└── global_tmatrix_schematic.svg     # Generated schematic figure
```

## Method implementation

### Cubic T-matrix (`cubic_scattering/effective_contrasts.py`, `voigt_tmatrix.py`)

Computes the single-site T-matrix for a cube of half-width `a` centred at the
origin via the Lippmann–Schwinger integral with the Green's tensor
Taylor-expanded about the scatterer centre.

**Key ideas:**

- The fourth-rank cube moment introduces **cubic point-group (Oₕ) symmetry**
  through the anisotropy tensor `E_{ijkl} = Σ_m δ_{im} δ_{jm} δ_{km} δ_{lm}`,
  yielding three independent coupling coefficients `T₁`, `T₂`, `T₃` instead
  of the two (`T₁`, `T₂`) for a sphere.

- **Four amplification factors** (displacement, rotation, off-diagonal strain,
  diagonal strain) dress the bare material contrasts into self-consistent
  effective contrasts `Δρ*`, `Δλ*`, `Δμ*_off`, `Δμ*_diag`.

- The result is expressed as a **6×6 Voigt matrix** in the displacement-traction
  basis `(uz, ux, uy, tzz, txz, tyz)` used by the Kennett propagator framework.

**Computation pipeline** (`compute_cube_tmatrix`):
1. `Γ₀` via 3D Gauss–Legendre quadrature of the full Green's tensor
2. `Aᶜ`, `Bᶜ`, `Cᶜ` via polynomial Taylor expansion (avoids divergent Eshelby singularity)
3. `T₁ᶜ`, `T₂ᶜ`, `T₃ᶜ` coupling coefficients from integral decomposition
4. Four amplification factors from self-consistent equations
5. Five effective contrasts

### Lattice Green's tensor (`cubic_scattering/lattice_greens.py`)

Evaluates `G_{ij}(Rₘ − Rₙ)` between all pairs of scatterer centres on a 2D
square lattice at `z = 0`. Four methods are implemented:

| Method | Description |
|--------|-------------|
| **Spatial** | Exact Ben-Menahem & Singh formula with D₄ₕ symmetry exploitation (~8× reduction) |
| **Spectral** | 2D IFFT of the post-kz-residue kernel with screened-Coulomb subtraction for convergence |
| **Hybrid** | Spatial near-field + spectral far-field, sweeps cutoff radius `r_cut` |
| **FCC** | Filon–Clenshaw–Curtis Hankel transform with angular harmonic decomposition |

**Spectral kernel at z = 0:**

```
G_ij(kx, ky) = (i / 2ρ) [ δ_ij / (β² kzS) + kP_i kP_j / (ω² kzP) − kS_i kS_j / (ω² kzS) ]
```

**Screened-Coulomb subtraction** improves spectral convergence from O(1/kH) to
O(1/kH³) by subtracting `c / √(kH² + kc²)`, FFT-ing the fast-decaying
residual, then adding back `c·exp(−kc·r) / (2πr)` analytically.

**FFT-accelerated block-Toeplitz mat-vec** exploits the translational invariance
of the lattice to perform `G₀ · u` in O(M² log M) via 2D FFT convolution.

### FFTProp (`FFTProp.py/`)

2.5D spectral scattering code — faithful Python 3.12 port of the original
Fortran `FFTPROP.F`. Features 4096-point FFT, cylindrical harmonic expansion
(m = −2 … +2), free-surface Rayleigh reflection, four-directional propagation
sweeps, and constant-Q attenuation. See [`FFTProp.py/README.md`](FFTProp.py/README.md)
for full documentation.

## Installation

```bash
# Create the conda environment
conda env create -f envs/seismic.yml

# Activate
conda activate seismic
```

### Dependencies

Python 3.12, NumPy, SciPy, Matplotlib, ObsPy, PyTorch, tqdm, typer, devito.

## Usage

### Cubic T-matrix

```python
from cubic_scattering import (
    ReferenceMedium, MaterialContrast,
    compute_cube_tmatrix, voigt_tmatrix_from_result,
)

ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2700.0)
contrast = MaterialContrast(Dlambda=1e9, Dmu=0.5e9, Drho=200.0)

result = compute_cube_tmatrix(omega=2*3.14159, a=0.05, ref=ref, contrast=contrast)

print(f"T₁ᶜ = {result.T1c:.6e}")
print(f"T₂ᶜ = {result.T2c:.6e}")
print(f"T₃ᶜ = {result.T3c:.6e}")
print(f"Cubic anisotropy = {result.cubic_anisotropy:.6e}")
```

### Lattice Green's tensor

```bash
# Run all verification methods (spatial, spectral, hybrid, FCC)
python -m cubic_scattering.lattice_greens --method all

# Spatial evaluation only, 16×16 lattice
python -m cubic_scattering.lattice_greens --method spatial -M 16

# Spectral with custom parameters
python -m cubic_scattering.lattice_greens --method spectral \
    --d 0.1 --omega 6.28 --alpha 5.0 --beta 3.0 --rho 3.0
```

**CLI parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--d` | 0.1 | Lattice spacing (m) |
| `-M` | 8 | Lattice dimension (M×M scatterers) |
| `--omega` | 2π | Angular frequency (rad/s) |
| `--rho` | 3.0 | Density (kg/m³) |
| `--alpha` | 5.0 | P-wave speed (km/s) |
| `--beta` | 3.0 | S-wave speed (km/s) |
| `--eta` | 0.03 | Attenuation ratio |
| `--method` | all | `spatial`, `spectral`, `hybrid`, `fcc`, or `all` |

### Tests

```bash
# Cubic T-matrix tests (9 tests)
python -m pytest cubic_scattering/tests/ -v

# FFTProp tests (8 tests)
pytest FFTProp.py/test_package.py -v

# Kennett Reflectivity tests
pytest PhD_fortran_code/Kennett_Reflectivity/test_package.py -v
```

## References

- **Nestor, D.P.** (1996). *Seismic wave scattering from heterogeneities using
  the T-matrix approach.* PhD thesis, Australian National University.
- **Gubernatis, J.E., Domany, E. & Krumhansl, J.A.** (1977). Formal aspects of
  the theory of the scattering of ultrasound by flaws in elastic materials.
  *J. Appl. Phys.*, 48(7), 2804–2811.
- **Eshelby, J.D.** (1957). The determination of the elastic field of an
  ellipsoidal inclusion, and related problems. *Proc. R. Soc. Lond. A*, 241,
  376–396.
- **Ben-Menahem, A. & Singh, S.J.** (1981). *Seismic Waves and Sources.*
  Springer-Verlag.
- **Wu, R.-S. & Ben-Menahem, A.** (1985). The elastodynamic near field.
  *Geophys. J. R. Astr. Soc.*, 81, 609–621.
- **Aki, K. & Richards, P.G.** (2002). *Quantitative Seismology.* 2nd edition,
  University Science Books.
- **Kennett, B.L.N.** (1983). *Seismic Wave Propagation in Stratified Media.*
  Cambridge University Press.
