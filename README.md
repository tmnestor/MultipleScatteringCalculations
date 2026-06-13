# CubicTmatrix: Elastic Multiple Scattering from Cubic Heterogeneities

T-matrix approach for seismic wave scattering from cubic inclusions in layered
elastic media. The global T-matrix equation `T = T₀ (I - G₀ T₀)⁻¹` couples
single-site scattering matrices `T₀` through inter-site Green's tensors `G₀`,
solved via FFT-accelerated GMRES. Originates from Nestor (1996) PhD thesis,
Australian National University.

The slab solver produces the full specular reflection matrix — all mode-converted
channels (R_PP, R_PS, R_SP, R_SS, and SH) at arbitrary horizontal slowness,
including post-critical evanescent incidence — validated channel-by-channel
against exact Kennett reflectivity. Single-site scattering is cross-checked
against elastic Mie theory across all five non-zero channels of the sphere
scattering matrix.

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
│   ├── inter_voxel_propagator.py     #   9×9 volume-averaged propagator (complex:
│   │                                 #   reactive + radiation, physical pitch)
│   │
│   │  # ── Slab Foldy-Lax solver ─────────────────────────────
│   ├── slab_scattering.py            #   CPU solver: full reflection matrix
│   │                                 #   (R_PP/PS/SP/SS/SH) + Weyl + Kennett ref
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
│   └── tests/                        #   pytest test suite (25 files, 724 tests)
│       ├── test_cubic_tmatrix.py
│       ├── test_tmatrix_assembly.py
│       ├── test_tmatrix_57.py
│       ├── test_incident_field.py
│       ├── test_cube_eshelby.py
│       ├── test_multipole_eshelby.py
│       ├── test_scattered_field.py
│       ├── test_resonance_far_field.py
│       ├── test_sphere_scattering.py        # incl. 5-channel Mie vs Foldy-Lax
│       ├── test_sphere_scattering_fft.py
│       ├── test_sphere_scattering_fft_gpu.py
│       ├── test_mie_asymptotic_analytic.py
│       ├── test_mie_near_field.py
│       ├── test_horizontal_greens.py
│       ├── test_inter_voxel_propagator.py   # incl. radiation part, pitch, symmetry
│       ├── test_t27_coupling_calibration.py # quadrature-arbiter calibration
│       ├── test_slab_scattering.py          # incl. reflection matrix, evanescent
│       ├── test_slab_scattering_gpu.py
│       ├── test_slab_convergence.py         # incl. volume-averaged vs Kennett
│       ├── test_kennett_layers.py
│       ├── test_cpa_iteration.py
│       ├── test_ocean_bottom.py             # incl. 2×2 mode-converted recursion
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
│   ├── t27_coupling_study.py         #   Quadrature-truth inter-voxel coupling study
│   ├── face_s_rederivation.py        #   Face-S constants vs bias-free arbiters
│   ├── test_radiation_part_need.py   #   Radiation-part necessity measurement
│   └── ...                           #   Eshelby, Green's tensor scripts
│
├── docs/                             # LaTeX documentation (lualatex)
│   ├── cube_galerkin27.tex           #   Main document
│   ├── cube_tmatrix_closedform.tex   #   T-matrix physics and derivations
│   ├── inter_voxel_propagator.tex    #   Volume-averaged propagator
│   ├── point_vs_volume_tmatrix_notes.tex  # Point vs volume-averaged coupling
│   ├── slab_scattering_explanation.tex
│   ├── marine_survey_explanation.tex
│   └── ...                           #   Results tables, Mie derivations
│
├── LatexPDFs/                        # Standalone write-ups (lualatex, compiled)
│   ├── mode_converted_reflections.tex #  5-channel Mie, slab matrix, evanescent
│   ├── multipolar_mie.tex            #   Multipolar effective sources / Mie
│   └── ...                           #   Spectral, near-field, coupled Foldy-Lax
│
├── Mathematica/                      # Symbolic computation (.wl scripts)
│   ├── CubeGalerkin27.wl             #   Body bilinear forms
│   ├── CubeT27Stiffness_LS.wl       #   Surface stiffness integrals
│   ├── CubeT6Block.wl               #   Quad-quad block
│   ├── InterVoxelPropagator*.wl      #   Volume-averaged propagator masters
│   ├── CubeAnalytic.wl               #   Cube far-field P/S (Foldy-Lax)
│   ├── FiveChannelExtension.wl       #   5-channel Mie vs Foldy-Lax (+ Driver)
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

### Full specular reflection matrix (mode conversion)

The slab returns the complete 2×2 P-SV reflection matrix plus the SH channel at
arbitrary horizontal slowness `p` — including post-critical evanescent incidence.
`to_modified()` maps it to the energy-normalised Kennett convention for comparison.

```python
from cubic_scattering import (
    SlabGeometry, uniform_slab_material,
    slab_reflection_matrix, kennett_reference_matrix,
)

geom = SlabGeometry(M=8, N_z=1, a=2.0)
mat = uniform_slab_material(geom, ref, contrast)

slab = slab_reflection_matrix(geom, mat, omega=150.0, p=1.0e-4)  # oblique
R = slab.to_modified()          # 2×2: rows = outgoing P/SV, cols = incident
R_PP, R_PS, R_SP, R_SS = R[0, 0], R[0, 1], R[1, 0], R[1, 1]
R_SH = slab.R_sh

# Exact reference (all five channels)
kref = kennett_reference_matrix(ref, contrast, H=geom.d, omega=150.0, p=1.0e-4)
# kref.R_PP, kref.R_PS, kref.R_SP, kref.R_SS, kref.R_SH
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

### Elastic Mie far field (five channels)

```python
import numpy as np
from cubic_scattering import compute_elastic_mie, mie_far_field

mie = compute_elastic_mie(omega=150.0, radius=10.0, ref=ref, contrast=contrast)
theta = np.linspace(0.2, np.pi - 0.2, 40)

# incident_type selects the column of the 3×3 scattering matrix
f_P, f_SV, f_SH = mie_far_field(mie, theta, incident_type="P")    # P→P, P→SV
f_P, f_SV, f_SH = mie_far_field(mie, theta, incident_type="SV")   # SV→P, SV→SV
f_P, f_SV, f_SH = mie_far_field(mie, theta, incident_type="SH")   # SH→SH
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
- **Full specular reflection matrix** — R_PP, R_PS, R_SP, R_SS, and SH — from
  P-, SV-, and SH-incident solves, extracted by a shared Weyl lattice sum and
  validated channel-by-channel against Kennett (~0.5–1% at moderate contrast)
- **Evanescent incidence** — post-critical (p > 1/α) incident fields are the
  true exponentially decaying inhomogeneous waves, via complex slowness vectors
- **Volume-averaged propagator** for nearest-neighbour coupling (strong contrast)
- **Oblique incidence** via horizontal slowness p
- **GPU acceleration** via PyTorch (3D FFT convolution + GMRES)

### Single-site validation: five-channel Mie

The cubic/sphere T₀ is cross-checked against exact elastic Mie theory across all
five non-zero channels of the sphere scattering matrix (P→P, P→SV, SV→P, SV→SV,
SH→SH), phase-sensitively (Re and Im separately) at ka = 0.1, 0.5, 1.5. The
off-diagonal channels satisfy the reciprocity relation k_P² f_PS = −k_S² f_SP.
The same comparison is reproduced symbolically in `Mathematica/FiveChannelExtension.wl`
(headless via `FiveChannelDriver.wl`).

### Inter-voxel coupling: point vs volume-averaged

Two ways to couple the Green's tensor between cube sources:

- **Point coupling** (`volume_averaged=False`): the Green's tensor is evaluated
  at the cube *centres* (point-to-point). Correct when the separation greatly
  exceeds the cube size; degrades for touching neighbours — at face contact the
  point value departs from the true cube-averaged coupling by ~12–60% per block,
  because the 1/r near field varies sharply across the shared face.
- **Volume-averaged coupling** (`volume_averaged=True`): the propagator is the
  Green's tensor averaged over *both* cube volumes (and its derivatives for the
  strain blocks), the physically correct object for face/edge/corner neighbours.
  Implemented in closed form via O_h-symmetric master integrals, with dynamic
  ω-corrections to ω⁶. The block is **complex**: the real part is the reactive
  (near-field + even-power-ω) series; the imaginary part is the radiation
  (odd-power-ω) series from sin(kr)/r — sub-percent of the real part for the G
  block at ka ≲ 0.1, growing to O(1) by ka ≈ 0.5. The near field is insensitive
  to the radiation term; distant coupling and the far field require it. The
  propagator is correctly scaled by the physical cube pitch.

Validation against Kennett (uniform multi-layer slab): volume-averaging is the
more physically faithful propagator — verified ~2–3× closer to the quadrature
truth than point coupling on every block — and it **beats** point coupling on
R_PP at normal incidence (~0.4–0.5% tighter across ka ≤ 0.5). It does **not**
uniformly beat point coupling, however: R_PP off normal incidence, and R_SS at
*all* angles (including normal), come out modestly worse. This is **not** a
propagator inaccuracy — it is a *compensating-error* effect. The point
propagator's coupling error was partly cancelling other approximation errors in
the per-voxel representation (the single polynomial T-matrix basis per voxel);
making the coupling more correct exposes them. Block-isolation pins the strain
(S) block as the driver, with opposite-sign effects on R_PP and R_SS, and the
off-normal R_PP crossover scales with the in-plane Bloch phase per cell
(restored by mesh refinement). The residual is therefore a per-voxel
*representation* limit, not a coupling defect: closing it needs a richer
per-voxel basis (sub-voxel resolution), not a more accurate propagator. The
"beats-point" guarantee is consequently pinned only for R_PP at normal
incidence. See `docs/point_vs_volume_tmatrix_notes.tex`.

### Ocean-bottom reflection

Three-layer model: water (acoustic) | heterogeneous sediment slab | elastic
halfspace. Features:
- Oblique incidence with fluid-solid coupling (Zoeppritz via Kennett recursion)
- **2×2 P-SV sub-ocean recursion** — internal P→S→P mode conversion inside the
  sediment package feeds the observable water-column R_PP (SH cannot couple
  through the fluid and is omitted)
- Free-surface water-column reverberations
- Random binary heterogeneity with configurable statistical moments
- YAML configuration with seismic units

### Kennett reflectivity

Full PSV + SH propagator-matrix recursion for layered elastic media with
optional fluid layers. Supports batch frequency computation, CPA effective
medium embedding, and random velocity stack generation.

## References

- **Nestor, T.M.** (1996). *Seismic Wave Propagation in Heterogeneous Media: Summing the Multiple Scattering Series.* PhD thesis, Australian National University.
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
