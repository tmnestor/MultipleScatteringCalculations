# FFT Foldy-Lax vs Mie Convergence Study

Compares the FFT-accelerated GMRES voxelized-sphere solver against exact elastic Mie theory at increasing resolution.

## Usage

```bash
# Default: P-wave, ka=0.1, n_sub=3,7,13,21
python scripts/fft_convergence_study.py

# Push to large voxel counts
python scripts/fft_convergence_study.py --n-sub 3,7,13,21,27,33

# S-wave at ka=0.5 with angular pattern comparison
python scripts/fft_convergence_study.py --wave-type S --ka 0.5 --n-sub 3,7,13,21 --angular

# Custom GMRES tolerance
python scripts/fft_convergence_study.py --n-sub 3,7,13,21,27,33 --gmres-tol 1e-12
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-sub` | `3,7,13,21` | Comma-separated n_sub values |
| `--wave-type` | `P` | Incident wave type: `P` or `S` |
| `--ka` | `0.1` | Target ka_P = omega * radius / alpha |
| `--gmres-tol` | `1e-10` | GMRES relative tolerance |
| `--angular` | off | Also print angular pattern comparison at the largest n_sub |

## Voxel counts by n_sub

| n_sub | N_cells | DOF |
|-------|---------|-----|
| 3 | 19 | 171 |
| 7 | 179 | 1,611 |
| 13 | 1,189 | 10,701 |
| 21 | 4,945 | 44,505 |
| 27 | 10,395 | 93,555 |
| 33 | 18,853 | 169,677 |
