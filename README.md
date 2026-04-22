# CubicTmatrix: Elastic Multiple Scattering from Cubic Heterogeneities

T-matrix approach for seismic wave scattering from cubic inclusions in layered
elastic media. The global T-matrix equation `T = T₀ (I - G₀ T₀)⁻¹` couples
single-site scattering matrices `T₀` through inter-site Green's tensors `G₀`,
solved via FFT-accelerated GMRES. Originates from Nestor (1996) PhD thesis,
Australian National University.

## Physics overview

Three computational pillars underpin the multiple-scattering formulation:

| Pillar | Symbol | Role |
|--------|--------|------|
| Single-site T-matrix | `T₀` | Lippmann-Schwinger integral for each cube |
| Inter-site Green's tensor | `G₀` | Elastodynamic coupling between cube centres |
| Multiple-scattering solve | `T` | Block-Toeplitz system solved by FFT-accelerated GMRES |

### Scattering regimes

| ka range | Module | Method |
|----------|--------|--------|
| ka < 0.3 | `effective_contrasts.py` | Analytical Rayleigh (fast, 27-component Galerkin with O_h symmetry) |
| 0.3-1.0 | `resonance_tmatrix.py` | Internal Foldy-Lax subdivision (n=2-4 sub-cells) |
| ka >= 1.0 | `resonance_tmatrix.py` | Full resonance (n >= 4 sub-cells) |

### Coordinate system

z = axis 0 (down), x = axis 1 (right), y = axis 2 (out) — right-handed.

## Repository structure

```
CubicTmatrix/
├── cubic_scattering/                 # Main Python package
│   │
│   │  # ── Single-site T-matrix (T₀) ─────────────────────────
│   ├── effective_contrasts.py        #   Rayleigh T₀: 9×9 and 27×27 Galerkin
│   ├── tmatrix_assembly.py           #   T27/T57 assembly from irrep blocks
│   ├── voigt_tmatrix.py              #   6×6 Voigt displacement-traction basis
│   ├── incident_field.py             #   Plane-wave overlap integrals
│   ├── cube_eshelby.py               #   Cube Eshelby concentration factors
│   ├── multipole_eshelby.py          #   Multipole Eshelby (sphere reference)
│   │
│   │  # ── Resonance regime ───────────────────────────────────
│   ├── resonance_tmatrix.py          #   Internal Foldy-Lax for ka ~ O(1)
│   ├── scattered_field.py            #   Far-field amplitudes, optical theorem
│   │
│   │  # ── Sphere scattering (validation) ─────────────────────
│   ├── sphere_scattering.py          #   Elastic Mie series + Foldy-Lax
│   ├── sphere_scattering_fft.py      #   FFT-accelerated sphere Foldy-Lax
│   ├── sphere_scattering_fft_gpu.py  #   GPU version (PyTorch)
│   ├── mie_asymptotic_analytic.py    #   Analytical Mie asymptotics
│   │
│   │  # ── Inter-site coupling (G₀) ──────────────────────────
│   ├── lattice_greens.py             #   Spatial, spectral, hybrid, FCC
│   ├── horizontal_greens.py          #   Exact Green's tensor at Δz=0
│   ├── inter_voxel_propagator.py     #   9×9 volume-averaged propagator
│   │
│   │  # ── Slab Foldy-Lax solver ─────────────────────────────
│   ├── slab_scattering.py            #   CPU solver + periodic R_PP + Kennett ref
│   ├── slab_scattering_gpu.py        #   GPU solver (PyTorch)
│   │
│   │  # ── Layered-medium embedding ──────────────────────────
│   ├── kennett_layers.py             #   Kennett reflectivity (PSV + SH + fluid)
│   ├── cpa_iteration.py              #   CPA effective medium (⟨T⟩ = 0)
│   │
│   │  # ── Applications ──────────────────────────────────────
│   ├── ocean_bottom.py               #   Ocean-bottom reflection (water|slab|halfspace)
│   ├── seismic_survey.py             #   Shot-gather simulation
│   ├── solver_config.py              #   YAML configuration loader
│   │
│   │  # ── GPU utilities ─────────────────────────────────────
│   ├── torch_gmres.py                #   PyTorch GMRES + device selection
│   │
│   └── tests/                        #   pytest test suite (25 test files)
│       ├── test_cubic_tmatrix.py
│       ├── test_tmatrix_assembly.py
│       ├── test_tmatrix_57.py
│       ├── test_incident_field.py
│       ├── test_cube_eshelby.py
│       ├── test_multipole_eshelby.py
│       ├── test_scattered_field.py
│       ├── test_resonance_far_field.py
│       ├── test_sphere_scattering.py
│       ├── test_sphere_scattering_fft.py
│       ├── test_sphere_scattering_fft_gpu.py
│       ├── test_mie_asymptotic_analytic.py
│       ├── test_mie_near_field.py
│       ├── test_horizontal_greens.py
│       ├── test_inter_voxel_propagator.py
│       ├── test_slab_scattering.py
│       ├── test_slab_scattering_gpu.py
│       ├── test_slab_convergence.py
│       ├── test_kennett_layers.py
│       ├── test_cpa_iteration.py
│       ├── test_ocean_bottom.py
│       ├── test_seismic_survey.py
│       ├── test_solver_config.py
│       └── test_torch_gmres.py
│
├── ocean_bottom/                     # Ocean-bottom reflection study
│   ├── README.md                     #   Physics, YAML reference, CLI docs
│   ├── run_study.py                  #   CLI script (YAML config + overrides)
│   ├── example_config.yml            #   Moderate: random, oblique, φ=0.3
│   ├── example_config_weak.yml       #   Weak: uniform, normal incidence (Born)
│   └── example_config_strong.yml     #   Strong: random, oblique, free-surface
│
├── configs/                          # YAML configuration files
│   ├── example_slab.yml
│   ├── example_sphere.yml
│   └── example_survey.yml
│
├── scripts/                          # Standalone analysis scripts
│   ├── slab_convergence_study.py     #   Slab R_PP convergence vs Kennett
│   └── ...                           #   Eshelby, Green's tensor scripts
│
├── docs/                             # LaTeX documentation (lualatex)
│   ├── cube_galerkin27.tex           #   Main document
│   ├── cube_tmatrix_closedform.tex   #   T-matrix physics and derivations
│   ├── inter_voxel_propagator.tex    #   Volume-averaged propagator
│   ├── slab_scattering_explanation.tex
│   ├── marine_survey_explanation.tex
│   └── ...                           #   Results tables, Mie derivations
│
├── Mathematica/                      # Symbolic computation (.wl scripts)
│   ├── CubeGalerkin27.wl             #   Body bilinear forms
│   ├── CubeT27Stiffness_LS.wl       #   Surface stiffness integrals
│   ├── CubeT6Block.wl               #   Quad-quad block
│   ├── InterVoxelPropagator*.wl      #   Volume-averaged propagator masters
│   ├── MieAsymptotic*.wl             #   Mie series asymptotics
│   └── ...                           #   ~50 Mathematica scripts
│
├── FFTProp.py/                       # 2.5D spectral scattering (Fortran port)
│   └── README.md
│
├── PhD_fortran_code/                 # Original Fortran 77 (Nestor 1996)
│   └── Kennett_Reflectivity/         #   Python Kennett reflectivity package
│
└── envs/
    └── seismic.yml                   # Conda environment specification
```

## Installation

```bash
conda env create -f envs/seismic.yml
conda activate seismic
```

### Dependencies

Python 3.12, NumPy, SciPy, Matplotlib, PyTorch, tqdm, typer, PyYAML.

## Usage

### Cubic T-matrix (Rayleigh regime)

```python
from cubic_scattering import (
    ReferenceMedium, MaterialContrast,
    compute_cube_tmatrix, voigt_tmatrix_from_result,
)

ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)

result = compute_cube_tmatrix(omega=150.0, a=1.0, ref=ref, contrast=contrast)
T_voigt = voigt_tmatrix_from_result(result)
```

### Slab Foldy-Lax scattering

```python
from cubic_scattering import (
    SlabGeometry, compute_slab_scattering,
    uniform_slab_material, slab_rpp_periodic,
    compute_slab_tmatrices,
)

geom = SlabGeometry(M=8, N_z=2, a=1.0)
mat = uniform_slab_material(geom, ref, contrast)
result = compute_slab_scattering(geom, mat, omega=150.0, k_hat=[1,0,0], periodic=True)

T_local = compute_slab_tmatrices(geom, mat, omega=150.0)
R_PP = slab_rpp_periodic(result, T_local, p=0.0)
```

### Ocean-bottom reflection

```bash
# Run with YAML config (seismic units: km/s, g/cm3, GPa, km, s/km)
python ocean_bottom/run_study.py ocean_bottom/example_config.yml

# With CLI overrides
python ocean_bottom/run_study.py ocean_bottom/example_config.yml --p 0.25 --free-surface
```

```python
from cubic_scattering import (
    load_ocean_bottom_config, compute_ocean_bottom_reflection, write_log,
)

cfg = load_ocean_bottom_config("ocean_bottom/example_config.yml")
result = compute_ocean_bottom_reflection(cfg, progress=True)
write_log(result, "output.log")
```

See [`ocean_bottom/README.md`](ocean_bottom/README.md) for full YAML reference
and heterogeneity parameterisation details.

### Seismic survey simulation

```bash
python -m cubic_scattering.seismic_survey configs/example_survey.yml
```

### Kennett reflectivity

```python
from cubic_scattering import (
    IsotropicLayer, LayerStack, kennett_layers,
)
import numpy as np

stack = LayerStack(layers=[
    IsotropicLayer(alpha=2000.0, beta=800.0, rho=1800.0, thickness=50.0),
    IsotropicLayer(alpha=3000.0, beta=1700.0, rho=2200.0, thickness=np.inf),
])
result = kennett_layers(stack, p=0.0, omega=np.linspace(10, 300, 100))
```

### CPA effective medium

```python
from cubic_scattering import compute_cpa_two_phase, phases_from_two_phase

phases = phases_from_two_phase(ref, contrast, phi=0.3, a=1.0, omega=150.0)
cpa_result = compute_cpa_two_phase(ref, contrast, phi=0.3, a=1.0, omega=150.0)
```

### Tests

```bash
# Full test suite
conda run -n seismic python -m pytest cubic_scattering/tests/ -v

# Single test file
conda run -n seismic python -m pytest cubic_scattering/tests/test_ocean_bottom.py -v

# FFTProp and Kennett legacy tests
conda run -n seismic pytest FFTProp.py/test_package.py -v
conda run -n seismic pytest PhD_fortran_code/Kennett_Reflectivity/test_package.py -v
```

### LaTeX documentation

```bash
# Must use lualatex (fontspec requires it)
cd docs && /usr/local/bin/lualatex -interaction=nonstopmode cube_galerkin27.tex
cd docs && /usr/local/bin/lualatex -interaction=nonstopmode cube_galerkin27.tex  # twice for xrefs
```

## Method summary

### 27-component Galerkin T-matrix

The T₂₇ uses a basis of 27 trial functions (3 displacement + 6 strain + 18
quadratic) and decomposes under O_h symmetry into 7 irreducible representations.
All body and surface integrals reduce analytically to 3D master integrals over
the unit cube with 1/r and 1/r^3 kernels.

### Resonance regime

Subdivides the cube into n^3 Rayleigh sub-cells and solves the internal
Foldy-Lax system with the full elastodynamic Green's tensor (near- +
intermediate- + far-field coupling). Reduces to the Rayleigh result at n=1.

### Slab scattering

The slab solver handles M x M x N_z grids of cubes with:
- **Linear convolution** (finite slab) or **circular convolution** (infinite periodic slab)
- **Volume-averaged propagator** for nearest-neighbour coupling (strong contrast)
- **Oblique incidence** via horizontal slowness p
- **GPU acceleration** via PyTorch (3D FFT convolution + GMRES)

### Ocean-bottom reflection

Three-layer model: water (acoustic) | heterogeneous sediment slab | elastic
halfspace. Features:
- Oblique incidence with fluid-solid coupling (Zoeppritz via Kennett recursion)
- Free-surface water-column reverberations
- Random binary heterogeneity with configurable statistical moments
- YAML configuration with seismic units

### Kennett reflectivity

Full PSV + SH propagator-matrix recursion for layered elastic media with
optional fluid layers. Supports batch frequency computation, CPA effective
medium embedding, and random velocity stack generation.

## References

- **Nestor, D.P.** (1996). *Seismic wave scattering from heterogeneities using
  the T-matrix approach.* PhD thesis, Australian National University.
- **Gubernatis, J.E., Domany, E. & Krumhansl, J.A.** (1977). Formal aspects of
  the theory of the scattering of ultrasound by flaws in elastic materials.
  *J. Appl. Phys.*, 48(7), 2804-2811.
- **Eshelby, J.D.** (1957). The determination of the elastic field of an
  ellipsoidal inclusion, and related problems. *Proc. R. Soc. Lond. A*, 241,
  376-396.
- **Kennett, B.L.N.** (1983). *Seismic Wave Propagation in Stratified Media.*
  Cambridge University Press.
- **Aki, K. & Richards, P.G.** (2002). *Quantitative Seismology.* 2nd edition,
  University Science Books.
