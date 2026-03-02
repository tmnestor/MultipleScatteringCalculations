# FFTProp: 2.5D Spectral Scattering via FFT-Based Plane-Wave Decomposition

A modern Python 3.12 package implementing 2.5D spectral scattering using FFT-based plane-wave decomposition with cylindrical harmonic expansions. This is a faithful conversion of the original Fortran program `FFTPROP.F`.

## Overview

The package computes scattered wavefields in a 2D heterogeneous medium embedded in a 3D homogeneous plane-layer reference medium (the "2.5D" formulation). The heterogeneity is 2D (x,z plane), while the reference medium is a 3D homogeneous elastic half-space — a homage to Kennett's plane-layer approach.

Key features:

- **Plane-wave decomposition** via 4096-point FFT in the horizontal wavenumber (kx) domain
- **Cylindrical harmonic expansion** (orders m = -2 to +2) for P and SV wave scattering
- **Free-surface Rayleigh reflection** with P-SV coupling and source-coupling W matrices
- **Four-directional propagation sweeps**: up, down, left, right through the scatterer array
- **Constant-Q attenuation** via complex wavenumbers

## Installation

The package lives inside `IntegralEquationScattering/` as a directory named `FFTProp.py/`. Because the directory name contains a dot, use importlib:

```python
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "FFTProp",
    str(Path("<workspace>") / "FFTProp.py" / "__init__.py"),
)
pkg = importlib.util.module_from_spec(spec)
sys.modules["FFTProp"] = pkg
spec.loader.exec_module(pkg)

from FFTProp import compute_wavefield, default_medium
```

## Core Components

### 1. Reference Medium (`medium.py`)

Defines the homogeneous elastic half-space with attenuation.

```python
from FFTProp import ReferenceMedium, default_medium

medium = default_medium()
# alpha=5.0, rho=3.0, Q=50.0, beta=alpha/√3 (Poisson solid)

# Complex quantities matching Fortran:
medium.complex_alpha    # a0 = alpha + i*alpha/(2Q)
medium.complex_slowness_p  # ca0 = 1/a0
medium.complex_slowness_s  # cb0 = ca0/√3
medium.ka0(freq=2.0)    # ka0 = w * a0
medium.kb0(freq=2.0)    # kb0 = ka0 * √3
```

### 2. Spectral Arrays (`spectral_arrays.py`)

Precomputes all wavenumber-domain arrays:

```python
from FFTProp import build_spectral_arrays, default_medium, default_grid

sa = build_spectral_arrays(default_medium(), default_grid(), freq=2.0)

# Wavenumber grid
sa.kxvec      # FFT-ordered kx, shape (Nk,)
sa.kzavec     # Vertical P-wavenumber, shape (Nk,)
sa.kzbvec     # Vertical S-wavenumber, shape (Nk,)

# Phase factors
sa.rtEa       # exp(i·kza·zh), half-step
sa.Eavec      # exp(i·kza·z), full-step = rtEa²

# Free-surface reflection
sa.Rpp, sa.Rsp, sa.Rss  # Rayleigh coefficients
sa.W11, sa.W12, sa.W21, sa.W22  # Source coupling

# Plane-cylindrical transform
sa.PC         # shape (Nk, 5, 2, Nscatz), m→index m+2
```

**Key function:**

- `vertical_slowness(k, kx)` → `kz = √(k² - kx²)` with Im(kz) ≥ 0

Note: Uses strict `.LT.` branch cut convention from `FFTPROP.F`, distinct from `kennetslo.f` which uses `.LE.`.

### 3. Propagation Sweeps (`propagation.py`)

Six sweep functions implementing the full scattering algorithm:

- `source_downsweep()` — Project source through free surface to scatterers (Svec)
- `receiver_downsweep()` — Project receiver through free surface (Rvec)
- `upsweep()` — Bottom-to-top sweep accumulating PSY
- `free_surface_reflect()` — Rayleigh reflection: PU,SU → PD,SD
- `downsweep()` — Top-to-bottom sweep continuing PSY
- `right_sweep()`, `left_sweep()` — Lateral sweeps within each depth layer

### 4. Main Driver (`fftprop_driver.py`)

High-level orchestrator:

```python
from FFTProp import compute_wavefield

result = compute_wavefield(freq=2.0)

result.PSY   # Scattered field at scatterers, shape (Nscatx, 5, 2, Nscatz)
result.Svec  # Source coupling, same shape
result.Rvec  # Receiver coupling, same shape
result.SY    # Scattering sources, same shape
```

## Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| alpha | 5.0 | P-wave velocity |
| rho | 3.0 | Density |
| Q | 50.0 | Quality factor |
| beta | α/√3 ≈ 2.887 | S-wave velocity (Poisson solid) |
| freq | 2.0 Hz | Frequency |
| Nk | 4096 | FFT points |
| Nscatx | 81 | Horizontal scatterers |
| Nscatz | 2 | Depth layers |
| jskip | 8 | FFT grid stride |
| Xs | 10.0 | Source position |
| Xr | 0.0 | Receiver position |
| is, ir | 1, 1 | P-source, P-receiver |

Default source: unit P-wave (m=-2 harmonic) at bottom-right corner of scatterer array.

## Algorithm

1. **Setup**: Build FFT-ordered wavenumber grid, compute vertical wavenumbers kza, kzb with branch cuts, phase factors E = exp(i·kz·z), free-surface Rayleigh reflection (Rpp, Rsp, Rss), source-coupling W matrices, and plane-wave ↔ cylindrical-harmonic PC transform arrays.

2. **Source/receiver downsweep**: Project point source through free surface into downgoing plane waves, transform k→x at each scatterer depth using PC arrays to build Svec (source) and Rvec (receiver) coupling coefficients.

3. **Upsweep**: Sweep upward from deepest scatterer layer. At each layer, accumulate incoming wavefield into PSY via PC projection (k→x), then scatter SY sources back to wavenumber domain (x→k).

4. **Free-surface reflection**: Convert upgoing (PU, SU) to downgoing (PD, SD) via Rayleigh reflection matrix.

5. **Downsweep**: Sweep downward from shallowest layer, same structure as upsweep but with opposite direction and using the reflected field.

6. **Lateral sweeps**: For each depth layer, perform right-to-left and left-to-right horizontal sweeps. These use stride-2 wavenumber sampling and direct summation (not FFT) for lateral scattering between adjacent scatterer positions.

## FFT Convention

```
Fortran FFT(X, N, +1.0) = Σ_k X(k)·exp(+i·2π·n·k/N) = N · numpy.fft.ifft(X)
Fortran FFT(X, N, -1.0) = Σ_k X(k)·exp(-i·2π·n·k/N) = numpy.fft.fft(X)
```

In the code:
- signi = +1 (k→x): `Nk * np.fft.ifft(X)`
- signi = -1 (x→k): `np.fft.fft(X)`

## Bugs Fixed from Fortran

1. **Lines 190–193**: Loop variable `i` but body references `ik` — accessing stale index from a previous loop. Fixed by vectorised computation `Eavec = rtEa**2`.

2. **Lines 148–150**: `Eavec`/`Ebvec` used in free-surface Rsp/Rss calculation BEFORE being computed at line 190. Fixed by computing phase factors before the free-surface section.

## Testing

```bash
cd <workspace>/IntegralEquationScattering
python3 FFTProp.py/test_package.py          # Quick tests (7 tests)
python3 FFTProp.py/test_package.py --full   # Include full-grid test (8 tests)
```

Test categories:
1. Reference medium properties match Fortran DATA values
2. Vertical slowness branch cut and symmetry
3. Spectral array shapes, kx grid, phase factor consistency
4. Free-surface reflection: zero P-SV coupling at normal incidence
5. Default source array placement
6. FFT round-trip verification
7. End-to-end with reduced grid (fast)
8. Full-size grid (Nk=4096, Nscatx=81, Nscatz=2)

## Files

```
FFTProp.py/
├── __init__.py            # Package initialization and exports
├── medium.py              # ReferenceMedium, GridConfig, SourceReceiverConfig
├── spectral_arrays.py     # Wavenumber grid, phase, free-surface, PC transform
├── propagation.py         # All propagation sweep algorithms
├── fftprop_driver.py      # Main orchestrator (compute_wavefield)
├── test_package.py        # Verification suite (8 tests)
└── README.md              # This file
```

## Mathematical Details

### Complex Wavenumbers

With constant-Q attenuation:
```
a0 = α + i·α/(2Q)            (complex P-velocity)
ca0 = 1/a0                    (complex P-slowness)
cb0 = ca0/√3                  (complex S-slowness, Poisson solid)
ka0 = ω · a0                  (complex P-wavenumber)
kb0 = ka0 · √3                (complex S-wavenumber)
```

### Vertical Wavenumber

```
kz = √((k + kx)(k - kx))     with Im(kz) > 0
```

### Plane-Wave ↔ Cylindrical-Harmonic Transform

```
PC(kx, m, P) = √(dkx/(π·kza)) · ((kx + i·kza)/ka0)^m
PC(kx, m, SV) = √(dkx/(π·kzb)) · ((kx + i·kzb)/kb0)^m
```

### Free-Surface Rayleigh Reflection

```
R1 = 1 - 2·cb0²·p²
Rayleigh = R1² + 4·cb0⁴·p²·qa·qb

Rpp = Ea · (-R1² + 4cb0⁴·p²·qa·qb) / Rayleigh
Rsp = -4·cb0⁴·p·R1·√(qa·qb)·Ea·Eb / Rayleigh
Rss = Eb · Rpp_base
```

### Source Coupling Through Free Surface

```
W11 = -i·√(2qa)·R1 / (Rayleigh·√ρ)
W12 = -2i·cb0²·√(2qa)·qb·p / (Rayleigh·√ρ)
W21 = -2i·cb0²·qa·√(2qb)·p / (Rayleigh·√ρ)
W22 = i·√(2qb)·R1 / (Rayleigh·√ρ)
```

## References

1. **Nestor, T.** (1996). PhD Thesis, Australian National University.
2. **Kennett, B. L. N.** (1983). "Seismic Wave Propagation in Stratified Media." Cambridge University Press.
3. **Aki, K., & Richards, P. G.** (1980). "Quantitative Seismology." W. H. Freeman.

## Version

- **Version**: 1.0.0
- **Python**: 3.12+
- **Dependencies**: NumPy 1.20+
- **Converted from**: FFTPROP.F (Fortran 77)
