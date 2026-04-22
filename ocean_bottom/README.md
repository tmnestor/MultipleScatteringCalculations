# Ocean-Bottom Reflection with Heterogeneous Sediment

Simulates ocean-bottom reflection through a three-layer model:

```
          free surface (optional)
  ┌──────────────────────────────────┐
  │        water (acoustic)          │  α_w, ρ_w, depth D
  ├──────────────────────────────────┤
  │  heterogeneous sediment slab     │  M×M×N_z cubes, Foldy-Lax solver
  ├──────────────────────────────────┤
  │     elastic halfspace            │  α_hs, β_hs, ρ_hs
  └──────────────────────────────────┘
```

The sediment layer is an M×M×N_z grid of cubic scatterers solved via the
Foldy-Lax multiple-scattering formalism. Each cube carries elastic contrasts
(Δλ, Δμ, Δρ) relative to a homogeneous sediment background.

## Quick start

```bash
# Weak: uniform slab, normal incidence, Born baseline
conda run -n seismic python ocean_bottom/run_study.py ocean_bottom/example_config_weak.yml

# Moderate: random heterogeneity (φ=0.3), oblique incidence (p=0.15 s/km)
conda run -n seismic python ocean_bottom/run_study.py ocean_bottom/example_config.yml

# Strong: random (φ=0.5), oblique (p=0.25 s/km), free-surface multiples,
#         volume-averaged propagator
conda run -n seismic python ocean_bottom/run_study.py ocean_bottom/example_config_strong.yml

# CLI overrides
conda run -n seismic python ocean_bottom/run_study.py ocean_bottom/example_config.yml --p 0.3
conda run -n seismic python ocean_bottom/run_study.py ocean_bottom/example_config.yml --free-surface --save result.npz
```

## Files

| File | Description |
|------|-------------|
| `example_config.yml` | YAML configuration template (seismic units) |
| `run_study.py` | CLI script: loads YAML, runs simulation, plots results |
| `cubic_scattering/ocean_bottom.py` | Core module: config, solver, YAML loader |
| `cubic_scattering/tests/test_ocean_bottom.py` | 18 tests: physics, oblique incidence, YAML loading |

## Physics

### Decomposition

The total PP reflection coefficient is built in two stages:

1. **Background** R_bg: Full Kennett recursion through the homogeneous
   [water | sediment | halfspace] stack.

2. **Heterogeneous total** R_total: The slab scattering R_slab is injected
   into the sub-ocean reflectivity and dressed by the water-sediment
   interface via a single Kennett recursion step:

```
MT = E²_sed · R_sed_hs + R_slab       (modified sub-ocean reflectivity)
R_total = R_d + T_u · MT · (1 - R_u · MT)⁻¹ · T_d   (Kennett step)
```

where E²_sed is the sediment two-way vertical phase, and R_d, R_u, T_d, T_u
are the fluid-solid interface reflection/transmission coefficients.

### Slab scattering

R_slab is computed via the periodic Weyl lattice sum:

```
R_PP(p) = -i / (2 k_z d² ρα²) × Σ_l Q_{P,l} × exp(i k_z z_l)
```

where k_z = ω η_P is the vertical P-wavenumber (η_P = √(1/α² - p²)),
d is the cube spacing, and Q_{P,l} is the far-field P-source scalar for
layer l, averaged over the M² horizontal cubes.

### Oblique incidence (p > 0)

At horizontal slowness p > 0, the incident P-wave direction in the
sediment (z,x,y coordinates) is:

```
k_hat = [η_P α,  p α,  0]     (downgoing, sagittal plane = z-x)
r_hat = [-η_P α, p α,  0]     (upgoing specular reflection)
```

P-S conversion exists in the solid layers but the PP path through the
fluid-solid boundary remains well-defined. The water column phase uses
η_water = √(1/α_w² - p²).

**Constraint:** p must satisfy p < 1/α_water to avoid evanescent P-waves
in the water column.

### Free-surface reverberations

When `free_surface: true`, water-column multiples are included via:

```
R_obs = E²_water · R / (1 + E²_water · R)
```

This generates late-arriving multiples at odd multiples of the two-way
water travel time.

## YAML configuration

The YAML file uses **seismic units** (km/s, g/cm³, GPa, km, s/km) for
numerical conditioning. The loader converts to SI internally.

| Section | Key | Unit | Description |
|---------|-----|------|-------------|
| `ocean` | `water_alpha` | km/s | Water P-wave velocity |
| | `water_rho` | g/cm³ | Water density |
| | `water_depth` | km | Water column depth |
| | `free_surface` | bool | Enable free-surface multiples |
| `sediment` | `alpha`, `beta` | km/s | Sediment P/S-wave velocity |
| | `rho` | g/cm³ | Sediment density |
| `halfspace` | `alpha`, `beta` | km/s | Halfspace P/S-wave velocity |
| | `rho` | g/cm³ | Halfspace density |
| `slab` | `M` | — | Horizontal grid size (M×M) |
| | `N_z` | — | Number of vertical layers |
| | `a` | km | Cube half-width |
| | `material` | — | `uniform` or `random` |
| | `phi` | — | Volume fraction (random only) |
| | `seed` | — | Random seed |
| `recording` | `f_peak` | Hz | Ricker wavelet peak frequency |
| | `T` | s | Record length |
| | `nw` | — | Positive frequencies |
| | `f_min`, `f_max` | Hz | Active frequency band |
| `slowness` | `p` | s/km | Horizontal slowness (0 = normal) |

### Heterogeneity parameterization

The slab section accepts exactly one of `mean` or `contrast` to specify
the material heterogeneity. Both use seismic units (GPa, g/cm³).

**`mean`** (recommended) — ensemble-mean contrast. For a random binary
medium the inclusion contrast is derived as mean/φ:

```yaml
slab:
  material: random
  phi: 0.3
  mean:
    Dlambda: 0.397       # ⟨Δλ⟩ GPa
    Dmu: 0.094           # ⟨Δμ⟩ GPa
    Drho: 0.027          # ⟨Δρ⟩ g/cm³
```

**`contrast`** — raw inclusion contrast (used directly):

```yaml
slab:
  material: random
  phi: 0.3
  contrast:
    Dlambda: 1.324       # inclusion Δλ, GPa
    Dmu: 0.312           # inclusion Δμ, GPa
    Drho: 0.09           # inclusion Δρ, g/cm³
```

For a binary Bernoulli field with volume fraction φ and inclusion
contrast Δ, the moments are:

| Moment | Expression |
|--------|-----------|
| Mean | φ Δ |
| Variance | φ(1−φ) Δ² |
| σ/μ | √((1−φ)/φ) |

For `uniform` material, every cube carries the same contrast, so
mean = contrast and variance = 0.
| `solver` | `volume_averaged` | bool | Use volume-averaged propagator |
| | `n_orders` | — | Dynamic correction orders |
| | `gmres_tol` | — | GMRES convergence tolerance |
| `output` | `save` | path | `.npz` output path or `null` |
| | `plot` | bool | Generate matplotlib plot |

## CLI reference

```
python ocean_bottom/run_study.py CONFIG_PATH [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `CONFIG_PATH` | str | Path to YAML config (required) |
| `--p` | float | Override horizontal slowness (s/km) |
| `--M` | int | Override horizontal grid size |
| `--a` | float | Override cube half-width (km) |
| `--free-surface` | flag | Enable free-surface reverberations |
| `--no-free-surface` | flag | Disable free-surface reverberations |
| `--volume-averaged` | flag | Use volume-averaged propagator |
| `--f-peak` | float | Override Ricker peak frequency (Hz) |
| `--f-min` | float | Override minimum frequency (Hz) |
| `--f-max` | float | Override maximum frequency (Hz) |
| `--save` | str | Save results to `.npz` file |
| `--no-plot` | flag | Skip matplotlib plot |

CLI overrides use the same seismic units as the YAML file.

## Python API

```python
from cubic_scattering import (
    load_ocean_bottom_config,
    compute_ocean_bottom_reflection,
)

# Load from YAML (seismic units → SI conversion)
config = load_ocean_bottom_config("ocean_bottom/example_config.yml")

# Override slowness programmatically (SI units: s/m)
config = type(config)(**{**config.__dict__, "p": 0.0002})

# Run
result = compute_ocean_bottom_reflection(config, progress=True)

# Access results
print(f"Peak |R_total| = {max(abs(result.R_total)):.6f}")
print(f"Elapsed: {result.elapsed_seconds:.1f} s")
```

### Key classes

**`OceanBottomConfig`** — all parameters for the simulation (SI units internally).

**`OceanBottomResult`** — output containing:
- `time`, `trace_total`, `trace_homogeneous` — time-domain seismograms
- `R_bg`, `R_slab`, `R_total` — frequency-domain reflection coefficients
- `omega_real` — frequency axis (rad/s)
- `n_gmres_iters` — GMRES iteration counts per frequency
- `elapsed_seconds` — wall-clock time

## Tests

```bash
# Run all 18 ocean-bottom tests
conda run -n seismic python -m pytest cubic_scattering/tests/test_ocean_bottom.py -v
```

| Test class | Tests | What it validates |
|------------|-------|-------------------|
| `TestOceanBottom` | 10 | Zero contrast, Born scaling, causality, energy bounds, Kennett embedding, coupling, free-surface multiples |
| `TestObliqueIncidence` | 4 | p=0 regression, small-p continuity, critical angle bound, oblique energy bound |
| `TestYAMLConfig` | 4 | Round-trip load, unit conversion (s/km→s/m), missing section diagnostics, file-not-found |

## Example configurations

Three configs are provided, sharing the same shallow-marine background but
with different heterogeneity strengths. Contrasts are specified as Δλ, Δμ, Δρ
perturbations to the sediment background, and are designed to give
physically consistent velocity perturbations (preserving Vp/Vs ≈ 2.5).

### Background model

| Layer | α (km/s) | β (km/s) | ρ (g/cm³) | Vp/Vs |
|-------|----------|----------|-----------|-------|
| Water | 1.500 | — | 1.025 | — |
| Sediment | 2.000 | 0.800 | 1.800 | 2.50 |
| Halfspace | 3.000 | 1.700 | 2.200 | 1.76 |

Background sediment moduli: λ = 4.896 GPa, μ = 1.152 GPa.

### Heterogeneity scenarios

| Config | Δα/α | Material | p (s/km) | Slab | Features exercised |
|--------|------|----------|----------|------|--------------------|
| `example_config_weak.yml` | 3% | uniform | 0.0 | 4×4×1 | Born limit baseline |
| `example_config.yml` | 10% | random φ=0.3 | 0.15 | 8×8×2 | Oblique incidence + random heterogeneity |
| `example_config_strong.yml` | 25% | random φ=0.5 | 0.25 | 8×8×3 | Strong contrast + oblique + free-surface multiples + volume-averaged propagator |

Contrast values (all preserve Vp/Vs ≈ 2.5):

| Config | Δλ (GPa) | Δμ (GPa) | Δρ (g/cm³) |
|--------|----------|----------|-----------|
| weak | 0.402 | 0.0946 | 0.036 |
| moderate | 1.324 | 0.312 | 0.090 |
| strong | 3.519 | 0.828 | 0.180 |

At 30 Hz with a = 1 m cubes: ka_P ≈ 0.094, ka_S ≈ 0.24 (Rayleigh regime).
