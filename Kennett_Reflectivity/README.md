# Kennett_Reflectivity

Synthetic seismogram computation for plane-stratified elastic media using
Kennett's reflectivity method. Translated from the Fortran program
`kennetslo.f` to Python 3.12 with NumPy vectorisation and multiprocessing.

## Installation

The package requires a conda environment with the following dependencies:

```bash
conda env create -f envs/seismic.yml
conda activate seismic
```

The environment file (`envs/seismic.yml`) includes: Python 3.12, NumPy,
SciPy, matplotlib, ObsPy, and Devito (via pip).

## Package Structure

```
Kennett_Reflectivity/
├── __init__.py                 # Package init with lazy imports
├── layer_model.py              # LayerModel dataclass, complex/vertical slowness
├── scattering_matrices.py      # P-SV interfacial reflection/transmission coefficients
├── kennett_reflectivity.py     # Recursive reflectivity (Kennett addition formula)
├── kennett_seismogram.py       # Single-trace synthetic seismogram (fixed slowness)
├── kennett_gather.py           # Multi-offset gather (discrete wavenumber summation)
├── kennett_reflectivity_gpu.py # PyTorch GPU batched reflectivity (MPS/CUDA)
├── kennett_gather_gpu.py       # GPU-accelerated gather computation
├── source.py                   # Source wavelets (Ricker, frequency & time domain)
├── example_usage.py            # Quick-start example
└── test_package.py             # Package tests
```

## Quick Start

### Single-trace seismogram (fixed slowness)

```bash
python -m Kennett_Reflectivity.kennett_seismogram \
    -p 0.2 -T 64 -n 2048 -o seismogram.png
```

### Single-trace with free surface multiples

```bash
python -m Kennett_Reflectivity.kennett_seismogram \
    -p 0.2 -T 64 -n 2048 --free-surface -o seismogram_fs.png
```

### Multi-offset gather (discrete wavenumber summation)

```bash
python -m Kennett_Reflectivity.kennett_gather \
    --r-min 0.5 --r-max 20.0 --dr 0.5 \
    -T 64 -n 2048 --np 2048 --p-max 0.8 \
    --t-max 30 -o gather.png
```

### Head wave gather (wide aperture for linear moveout phases)

```bash
python -m Kennett_Reflectivity.kennett_gather \
    --r-min 0.5 --r-max 50.0 --dr 0.5 \
    -T 64 -n 2048 --np 4096 --p-max 1.2 \
    --free-surface --t-max 30 -o gather_headwaves.png
```

## CLI Reference

### `kennett_seismogram`

Compute a single synthetic seismogram at a fixed horizontal slowness.

| Flag | Default | Description |
|------|---------|-------------|
| `-p`, `--slowness` | 0.2 | Horizontal slowness / ray parameter (s/km) |
| `-T`, `--duration` | 64.0 | Time window (seconds) |
| `-n`, `--nw` | 2048 | Number of positive frequencies (power of 2) |
| `-o`, `--output` | `seismogram_verification.png` | Output plot filename |
| `--no-plot` | off | Skip plot, save data only |
| `--free-surface` | off | Include free surface reflections |

Output files: PNG plot + `seismogram_p{slowness}.txt` (two-column ASCII: time, amplitude).

### `kennett_gather`

Compute a multi-offset seismogram gather via discrete wavenumber summation
(τ-p integration with Bessel function weighting).

| Flag | Default | Description |
|------|---------|-------------|
| `--r-min` | 0.5 | Minimum offset (km) |
| `--r-max` | 20.0 | Maximum offset (km) |
| `--dr` | 0.5 | Offset spacing (km) |
| `-T`, `--duration` | 64.0 | Time window (seconds) |
| `-n`, `--nw` | 2048 | Number of positive frequencies (power of 2) |
| `--np` | 2048 | Number of slowness samples |
| `--p-max` | 0.8 | Maximum slowness (s/km) |
| `--gamma` | π/T | Complex frequency damping γ (rad/s) for anti-aliasing |
| `--t-max` | full window | Maximum time to display (seconds) |
| `-o`, `--output` | `seismogram_gather.png` | Output plot filename |
| `--no-plot` | off | Skip plot, save data only |
| `-j`, `--workers` | all cores | Number of parallel workers |
| `--free-surface` | off | Include free surface reflections |

Output files: PNG plot + `.npz` (NumPy archive with `time`, `offsets`, `gather` arrays).

## Python API

### LayerModel

```python
from Kennett_Reflectivity import LayerModel
import numpy as np

model = LayerModel.from_arrays(
    alpha=[1.5, 1.6, 3.0, 5.0, 2.2],     # P-wave velocity (km/s)
    beta=[0.0, 0.3, 1.5, 3.0, 1.1],       # S-wave velocity (km/s), 0 = acoustic
    rho=[1.0, 2.0, 3.0, 3.0, 1.8],        # Density (g/cm³)
    thickness=[2.0, 1.0, 1.0, 1.0, np.inf],# Layer thickness (km), inf = half-space
    Q_alpha=[20000, 100, 100, 100, 100],   # P-wave quality factor
    Q_beta=[1e10, 100, 100, 100, 100],     # S-wave quality factor
)
```

Layer 0 must be acoustic (beta=0) — this is the ocean layer. The last layer
is the half-space (thickness=inf). Layers are ordered top-down.

### compute_seismogram

```python
from Kennett_Reflectivity import compute_seismogram, default_ocean_crust_model

model = default_ocean_crust_model()
time, seis = compute_seismogram(
    model,
    p=0.2,              # horizontal slowness (s/km)
    T=64.0,             # time window (s)
    nw=2048,            # number of positive frequencies
    free_surface=True,  # include free surface multiples
)
# time: shape (4096,), seconds
# seis: shape (4096,), real-valued amplitude
```

### compute_gather

```python
from Kennett_Reflectivity.kennett_gather import compute_gather, default_ocean_crust_model
import numpy as np

model = default_ocean_crust_model()
offsets = np.arange(0.5, 50.5, 0.5)

time, offsets, gather = compute_gather(
    model,
    offsets,
    T=64.0,
    nw=2048,
    np_slow=4096,       # slowness samples (more = cleaner)
    p_max=1.2,          # max slowness — must exceed 1/v_min for all phases
    gamma=None,         # defaults to π/T for anti-aliasing
    free_surface=True,
    n_workers=None,     # defaults to all CPU cores
)
# time: shape (4096,)
# offsets: shape (100,)
# gather: shape (100, 4096)
```

### plot_gather

```python
from Kennett_Reflectivity.kennett_gather import plot_gather

plot_gather(
    time, offsets, gather,
    output="my_gather.png",
    title="My Gather",
    t_max=30.0,         # clip display at 30s
    clip=0.8,           # amplitude clip as fraction of trace spacing
)
```

### kennett_reflectivity (low-level)

```python
from Kennett_Reflectivity import kennett_reflectivity
import numpy as np

T = 64.0
nw = 2048
dw = 2 * np.pi / T
omega = np.arange(1, nw) * dw

# R(ω, p) for all frequencies at a single slowness
R = kennett_reflectivity(model, p=0.3, omega=omega, free_surface=True)
# R: shape (nw-1,), complex PP reflectivity
```

## Default Model

The built-in 5-layer ocean-crust model from `kennetslo.f`:

| Layer | Type | α (km/s) | β (km/s) | ρ (g/cm³) | h (km) | Qα | Qβ |
|-------|------|----------|----------|-----------|--------|-----|-----|
| 0 | Ocean | 1.50 | 0.00 | 1.0 | 2.0 | 20000 | 1e10 |
| 1 | Sediment | 1.60 | 0.30 | 2.0 | 1.0 | 100 | 100 |
| 2 | Crust | 3.00 | 1.50 | 3.0 | 1.0 | 100 | 100 |
| 3 | Upper mantle | 5.00 | 3.00 | 3.0 | 1.0 | 100 | 100 |
| 4 | Half-space | 2.20 | 1.10 | 1.8 | ∞ | 100 | 100 |

Critical slownesses (head wave thresholds): p = 1/α at each interface where
a faster layer underlies a slower one. Head waves appear as linear moveout
arrivals at large offsets with apparent velocity equal to the refractor velocity.

| Refractor | α (km/s) | p_critical (s/km) |
|-----------|----------|--------------------|
| Ocean | 1.50 | 0.667 |
| Sediment | 1.60 | 0.625 |
| Crust | 3.00 | 0.333 |
| Upper mantle | 5.00 | 0.200 |

To capture all head wave phases, set `--p-max` above the largest critical
slowness (0.667 s/km for this model) and use offsets well beyond the
crossover distance.

## Key Parameters and Trade-offs

### Slowness samples (`--np`)

Controls the density of the τ-p integration. More samples = cleaner gather
but longer runtime. The integration loops over slowness, calling
`kennett_reflectivity` once per sample with all frequencies vectorised.
Typical values: 2048 (quick), 4096 (clean), 8192 (publication quality).

### Maximum slowness (`--p-max`)

Must exceed the reciprocal of the minimum velocity in the model to capture
all propagating phases. For head waves, set it above 1/α_min. For the
default model, α_min = 1.5 km/s so p_max should be at least 0.667 s/km.
Using p_max = 1.2 s/km provides comfortable margin and captures evanescent
contributions near the critical angles.

### Complex frequency damping (`--gamma`)

Suppresses spatial aliasing from the discrete slowness integration by
evaluating the reflectivity at complex frequency ω + iγ, then compensating
in the time domain with exp(−γt). Default: π/T (Bouchon, 1981). Increase
if you see residual vertical stippling on traces. Trade-off: higher γ
reduces dynamic range at late times.

### Free surface (`--free-surface`)

Adds ocean surface reflections via the scalar reverberation operator:

```
R_total = E²_oc · RRd / (1 + E²_oc · RRd)
```

where R_fs = −1 (pressure-release boundary). This generates water-column
multiples at intervals of 2h_ocean/α_ocean ≈ 2.67s for the default model.
The infinite geometric series in the denominator captures all orders of
surface multiples.

## Performance

The gather computation parallelises over slowness using `multiprocessing.Pool`.
Each worker calls `kennett_reflectivity` once per slowness with all frequencies
vectorised via NumPy. No inter-process communication is needed during the
integration — embarrassingly parallel.

Typical runtimes on Apple Silicon (M-series, 10 cores):

| Configuration | Slowness samples | Frequencies | Approximate time |
|---------------|-----------------|-------------|------------------|
| Quick test | 512 | 511 | ~5s |
| Standard | 2048 | 2047 | ~30s |
| High quality | 4096 | 2047 | ~60s |
| Head waves | 4096 | 2047 | ~60s |

## GPU Acceleration (Apple MPS / CUDA)

The package includes a PyTorch-based GPU backend that batches the entire
Kennett recursion across ALL slowness samples simultaneously. Instead of
`np_slow` sequential reflectivity calls (parallelised via multiprocessing),
the GPU version runs a single batched computation over tensors of shape
`(np_slow, nfreq, 2, 2)`.

### Requirements

Add `pytorch` to your conda environment (already in `envs/seismic.yml`):

```bash
conda install pytorch -c conda-forge
```

### GPU gather CLI

```bash
python -m Kennett_Reflectivity.kennett_gather_gpu \
    --r-min 0.5 --r-max 50.0 --dr 0.5 \
    -T 64 -n 2048 --np 8192 --p-max 1.2 \
    --free-surface --t-max 30 -o gather_gpu.png
```

The `--device` flag selects the backend: `mps` (Apple Silicon), `cuda`,
or `cpu` (default: auto-detect, priority MPS > CUDA > CPU).

### GPU Python API

```python
from Kennett_Reflectivity.kennett_gather_gpu import compute_gather_gpu
from Kennett_Reflectivity.kennett_reflectivity_gpu import get_device
import numpy as np

model = default_ocean_crust_model()
offsets = np.arange(0.5, 50.5, 0.5)

print(f"Using device: {get_device()}")

time, offsets, gather = compute_gather_gpu(
    model, offsets,
    T=64.0, nw=2048, np_slow=8192, p_max=1.2,
    free_surface=True,
)
```

### Architecture

```
CPU version (kennett_gather.py):
    for each p_j in p_samples (via multiprocessing.Pool):
        R[w] = kennett_reflectivity(model, p_j, omega)    # (nfreq,)
        U[r, w] += J0(ω p_j r) · R · p_j · ω · dp

GPU version (kennett_gather_gpu.py):
    R[j, w] = kennett_reflectivity_batch_gpu(model, p_all, omega)  # (np_slow, nfreq)
    U[r, w] = Σ_j R[j,w] · J0(ω p_j r) · p_j · ω · dp           # chunked einsum
```

The Kennett recursion on GPU uses:
- Batched analytical 2×2 inverse (faster than `torch.linalg.inv`)
- `torch.matmul` for batched matrix products over `(np_slow, nfreq, 2, 2)`
- Scattering matrices precomputed on CPU, transferred once to GPU

### Long-time aliasing fix

The GPU speedup enables much denser slowness sampling (8192–16384 samples
vs 4096 on CPU) which directly suppresses the late-time aliasing artefacts
visible in wide-aperture gathers. You can also increase `--gamma` for
additional damping at the cost of reduced dynamic range at late times.

### Roadmap: Devito for 3D heterogeneity

Replace the reflectivity method with Devito's finite-difference wave
equation solver for full 3D wave propagation. This enables arbitrary
heterogeneity superimposed on the layered structure. Devito generates
optimised C/OpenACC code and supports GPU execution.

## References

1. Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified Media.* Cambridge University Press.
2. Aki, K. & Richards, P. G. (1980). *Quantitative Seismology.* W. H. Freeman.
3. Bouchon, M. (1981). A simple method to calculate Green's functions for elastic layered media. *BSSA*, 71(4), 959-971.
4. Chapman, C. H. (2004). *Fundamentals of Seismic Wave Propagation.* Cambridge University Press.
