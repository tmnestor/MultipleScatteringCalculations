# Solver speedups, ranked (21 October 2026)

Measured on the M1 Pro (8 performance cores), CPU only, with the torch solvers
`sphere_scattering_fft_gpu.compute_sphere_foldy_lax_fft_gpu` and
`slab_scattering_gpu.compute_slab_scattering_gpu` run on `torch.device("cpu")`. Each case ran in its
own process with one thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), seven cases in parallel.

## Summary

- **The time is in building the sphere kernel, not in the solve.** At `n_sub = 11` the cold build takes
  619 s and the 18 GMRES solves take 1 to 2 s. Over 99% of the time is in `averaged_pair_block_9x9`.
- **complex64 GMRES is accurate enough for production** at tolerances of 1e-5 and looser: it matches
  complex128 to two significant figures in every case measured. It is 1.5 to 2 times faster on CPU, but
  that applies only to the 1% of the time spent in GMRES.
- **complex128 stays where it is needed:** the kernel construction (catastrophic cancellation as
  `k r -> 0`) and the validation gates (1e-15 agreement, GMRES tolerance 1e-11).

## Where the time goes

| Case | Cold kernel build | Same build, memoised | GMRES (all right-hand sides) |
|---|---|---|---|
| Sphere, `n_sub = 5` | 59 s | 0.0 s | 0.1 s |
| Sphere, `n_sub = 11` | 619 s | 0.2 s | 1.2 to 2.2 s |
| Sphere, `n_sub = 17` | over 25 min (stopped) | n/a | n/a |
| Slab, 32 x 32 x 10 | 1.4 s | 0.9 s | 0.8 to 2.0 s |

`_build_fft_kernel` (`sphere_scattering_fft.py`) calls `averaged_pair_block_9x9` once per distinct
offset: 9,260 offsets at `n_sub = 11`, 35,936 at `n_sub = 17`. Each call averages over the receiver cell
with a tensor Gauss rule written as three nested Python loops over the nodes
(`cell_averaged_pair._receiver_average`), and each node makes a Python call that builds lists of small
derivative tensors for both modes.

Measured cost per node, one thread:

| Offset | Neighbour type | Gauss order | Nodes | Time per block | Time per node |
|---|---|---|---|---|---|
| (1, 0, 0) | face | 16 | 4,096 | 1,122 ms | 274 µs |
| (1, 1, 0) | edge | 12 | 1,728 | 470 ms | 272 µs |
| (2, 1, 0) | near | 6 | 216 | 58 ms | 270 µs |
| (5, 3, 2) | far | 6 | 216 | 55 ms | 252 µs |

About 2.3 million nodes at 270 µs each accounts for the 619 s almost exactly. The memo
(`_BLOCK_CACHE`) is keyed on the frequency and lives only in the process, so every new frequency and
every worker of a sweep pays the full cost again.

## complex64 against complex128

Relative error of the output (sphere: `T_comp_9x9` and `psi_pw`; slab: `psi`) against a complex128
reference solved to tolerance 1e-10.

| Case | complex128, tol 1e-6 | complex64, tol 1e-6 | complex64, tol 1e-5 | GMRES iterations | complex64 GMRES speed-up |
|---|---|---|---|---|---|
| Sphere, weak, kR 0.025, `n_sub = 5` | 6.7e-8 | 6.9e-8 | 1.4e-7 | 3 | about 1 |
| Sphere, weak, kR 1, `n_sub = 11` | 1.6e-8 | 2.0e-8 | 3.4e-7 | 3 | 2.0 |
| Sphere, strong, kR 1, `n_sub = 11` | 8.2e-8 | 8.5e-8 | 2.8e-7 | 6 | 1.6 |
| Slab, weak, kd 0.1, phi 0.1 | 8.5e-8 | 8.8e-8 | 5.2e-6 | 2 | 1.5 |
| Slab, strong, kd 0.5, phi 0.3 | 1.1e-6 | 1.1e-6 | 5.8e-6 | 7 | 1.7 |
| Slab, strong, kd 1.5, phi 0.3 | 5.8e-7 | 6.0e-7 | 7.1e-6 | 16 | 1.8 |

Contrasts: weak is `Dlambda = 2e9, Dmu = 1e9, Drho = 100` (the example configurations, about 4% in
mu); strong is `Dlambda = 1.5e10, Dmu = 1.2e10, Drho = 1000` (about 50% in mu, 40% in rho).
Background `alpha = 5000, beta = 3000, rho = 2500`. Sphere radius 0.5 m; slab cube half-width 1 m.
The strong sphere at kR 3, `n_sub = 17` did not finish its kernel build within the 25-minute limit.

**Caveat:** the sphere solver's default `gmres_tol = 1e-8` is below what complex64 can reach (its
floor is about 1e-7). A complex64 run must set the tolerance to 1e-5 or looser, or every column will
warn that it did not converge.

## Speedups, ranked

### 1. Fortran with OpenMP for the cell-averaged quadrature

One routine evaluating every (offset, Gauss node) pair, reusing the derivative ladder of
`fortran/green_derivatives.f90` and returning the assembled 9 x 9 blocks. This removes the 270 µs of
Python overhead per node.

- **Expected gain:** 100 times or more on one core, and about 7 times more across 8 cores, bringing
  619 s down to a few seconds. This is an estimate that assumes a Fortran point evaluation costs a few
  microseconds; it has not been timed.
- **Gate:** agreement with `averaged_pair_block_9x9` to round-off at face, edge, corner and far
  offsets, in the style of `tests/test_kernel_fortran.py`.
- **Cheaper interim step:** vectorise `_receiver_average` over the nodes in NumPy. This removes most of
  the per-node call overhead, though not all of it.

### 2. Cube symmetry (O_h) over the offsets

Offsets related by one of the cube's 48 rotations and reflections give blocks related by a fixed
transformation of the 9-component field. Only about 1/48 of the offsets need the quadrature; the rest
follow by transformation.

- **Expected gain:** up to 48 times, on top of item 1, in any language.
- **Risk:** the transformation of the Voigt components (with engineering halving) must be got exactly
  right. Gate it by comparing every transformed block with the direct one at `n_sub = 5`.

### 3. Closed-form tail for distant offsets

`tail_pair_block_9x9` is exact through order d² and costs one scalar factor per mode. Only the 6 face
and 12 edge neighbours need high Gauss orders.

- **Expected gain:** most offsets skip quadrature entirely.
- **Condition:** use the tail only beyond the distance at which the existing gate shows it agrees with
  the quadrature to the accuracy the problem needs. Put that distance in YAML.

### 4. Kernel cache on disk

Store each built kernel keyed by (`n_sub`, d, omega, reference medium), so repeat runs and sweep
workers load it rather than rebuilding.

- **Expected gain:** a repeated build drops from minutes to the time it takes to read the file.
- **Condition:** the key must be checked exactly, as the in-memory memo already is, so that a different
  configuration can never be served a stale kernel.

### 5. All right-hand sides in one solve

The sphere solver runs 18 separate GMRES solves (9 Taylor columns, 9 plane-wave columns). Pushing all
18 through one batched FFT matrix-vector product, or using block GMRES, makes better use of each FFT.

- **Expected gain:** several times faster for the GMRES stage. Worth doing only after items 1 to 3,
  while GMRES is 1% of the total.

### 6. complex64 GMRES

Measured at 1.5 to 2 times faster on CPU, with half the memory. The gain is larger on a GPU.

- **Condition:** production runs only, with `gmres_tol` of 1e-5 or looser. Keep complex128 for the
  kernel construction and for every validation gate.

## Where Fortran will not help

The FFTs (torch and NumPy), the dense linear algebra (BLAS and LAPACK) and the GMRES loop are already
compiled library code, and together they take about 1% of the time. Rewriting them would gain little.
