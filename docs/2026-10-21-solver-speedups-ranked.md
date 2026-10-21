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

## Progress (updated 21 October 2026)

| Item | Status | Measured effect |
|---|---|---|
| 1. Fortran cell-averaged quadrature | **Done** (`fortran/cell_average.f90`, `cell_average_fortran.py`) | `n_sub = 11` build 619 s to 0.8 s (8 threads); `n_sub = 17` from over 25 min to 3.2 s |
| 1b. Vectorise what the quadrature left in Python (block assembly, Gauss orders, memo keys, receiver check) | **Done** (new; not in the original ranking) | `n_sub = 25` batched call 9.2 s to 4.1 s |
| 2. Cube symmetry over the offsets | **Done**, applied to the averaged tensors (index permutation and sign), not to the 9 x 9 blocks, so no Voigt rotation is involved | `n_sub = 25` quadrature 3.2 s to 0.45 s; batched call 4.1 s to 1.4 s |
| 3. Closed-form tail for distant offsets | **Superseded by item 8, stage 2** (the same expansion carried to higher order) | |
| 4. Kernel cache on disk | Not started; low value: it helps a sweep only when the same frequencies recur across runs. Item 8 stores the frequency-independent moments instead, which every run reuses | |
| 5. All right-hand sides in one solve | **Measured: does not pay on CPU.** One FFT product over all 18 vectors is 1.4 to 2 times SLOWER than 18 separate ones (`n_sub` 17 and 25, both precisions): the batched grids leave the cache. Revisit on a GPU | none on CPU |
| 5b. Point-major FFT kernel (one 9 x 9 per Fourier point, contiguous; `sphere_scattering_fft_gpu._point_major`) | **Done** (new) | pointwise multiply 5.2 ms to 2.3 ms per product at `n_sub = 25`; identical results |
| 6. complex64 GMRES | Available now (`dtype=torch.complex64`, `gmres_tol >= 1e-5`); not yet the default | 1.6 to 1.8 times faster GMRES |
| 7. Block GMRES (one Krylov space shared by the 18 right-hand sides) | Not started; uncertain | could cut the number of products; each solve already takes only 5 to 9 iterations |
| 8. Frequency-amortised analytic cell averages | **Stages 1 and 2 done** (`cell_average_kseries.py`, `fortran/kseries_moments.f90`, `cell_average_far.py`; `cell_average.route: kseries`). Plan and results below | `n_sub = 25`: 0.31 to 0.85 s per frequency against 1.2 s by quadrature, 1.2e-13 or better at every offset from k_S R = 1 to 12 against up to 1.9e-8; one-off coefficient build 29 s, store 40 MB |

**The FFTs are at their floor on this CPU.** Of the 30 ms one product takes at `n_sub = 25`
(complex128), 22 ms is the forward and inverse FFT. The padded size 49 is already the fastest (50, 54,
56 and 64 are all slower), and pruning the FFT to the cells that carry data saves only about 10%, the
copies taking back most of the saving. So the remaining GMRES speed-ups are fewer products (item 7) or
less precision (item 6), not a faster product.

Where the sphere solve spends its time now (strong contrast, 8 threads, `gmres_tol = 1e-5`, cold build;
the complex64 rows reuse the memoised blocks, so only their GMRES column compares):

| Case | Total, complex128 | Kernel build | GMRES (18 solves), complex128 | GMRES, complex64 | Total before this round |
|---|---|---|---|---|---|
| `n_sub = 11`, kR 1 | 0.7 s | 0.3 s | 0.3 s | 0.2 s | 1.2 s (619 s before item 1) |
| `n_sub = 17`, kR 3 | 2.2 s | 0.5 s | 1.6 s | 1.0 s | 4.8 s |
| `n_sub = 25`, kR 3 | 7.2 s | 1.7 s | 5.4 s | 3.0 s | 15.6 s |

GMRES is now three-quarters of the solve at `n_sub = 25`. In production, complex64 brings the whole
solve to 3.8 s there.

## Production sweeps many frequencies: rank by cost per sweep

Production runs solve at many frequencies, and the kernel is rebuilt at each one: the block memo is
keyed on omega, so it gives no reuse across a sweep. Every per-frequency cost above is therefore paid
once per frequency. At `n_sub = 25`, 100 frequencies cost about 2.3 min of kernel builds (1.4 s each)
and 5 to 9 min of GMRES (3.0 to 5.4 s each).

**The governing principle: analytic integrals, where possible, are amortised across frequency.** A
closed form or a frequency-independent geometric moment is computed once per geometry and serves every
frequency; only cheap scalar factors (powers of omega) are evaluated per frequency. Quadrature at each
frequency is the fallback where no analytic route exists, not the default. On that basis:

1. **Item 8 (frequency-amortised cell averages)** comes first: it removes the quadrature from the
   per-frequency cost entirely.
2. **Item 6 (complex64 GMRES)** next: its 1.6 to 1.8 times saving multiplies across the sweep, and it
   is configuration only.
3. **Item 7 (block GMRES)** after that, if GMRES still dominates.
4. **Item 4 (kernel cache on disk)** stays low: it helps only when the same frequencies recur.

## Item 8: frequency-amortised analytic cell averages (plan)

The graded-voxel blocks already work this way. ``graded_voxel.blocks.near_block_series`` writes a
block as a sum over powers of r of two scalar weights (``series_weights(m, omega, ref)``, the only
frequency dependence) times universal moments: exact closed-form box integrals of a monomial times
r^m (``graded_voxel.moments.box_integral``, 40 digits, integer arguments in units of the cell's
half-width, so universal constants that serve every frequency and every cell size). The receiver-cell
average that ``cell_average.f90`` computes by quadrature has the same structure.

**Stage 1: near offsets by the wavenumber series, with universal moments.**

- With D = g_S - g_P = (1/4 pi) sum_{t>=1} i^t (k_S^t - k_P^t) r^(t-1) / t! and
  g_S = (1/4 pi) sum_{t>=0} (i k_S)^t r^(t-1) / t!, every averaged tensor is
  <d^n f>(s; omega) = sum_t c_t(omega) <d^n r^(t-1)>(offset), with c_t the scalar weights.
- Each <d^n r^m> is a sum of monomials times r^(m - 2q) (the derivative ladder), averaged over the
  receiver cube: for odd powers, ``box_integral`` over the octant boxes into which the origin splits the
  cube; for even powers, a polynomial, exact by construction. Computed once per canonical offset
  (cube symmetry for the images) and per order, at 40 digits, and stored: universal constants.
- Per frequency: one weighted sum per offset over t. No quadrature.
- Valid while the series converges without cancellation: |k| R_far below a limit (R_far the distance
  from the source to the far corner of the cell), where the terms (|k| R)^t / t! stay near the sum. The
  limit and the series length go in YAML, as the existing ``DIFFERENCE_SERIES_LIMIT`` and
  ``DIFFERENCE_N_SERIES`` do.
- Gate: equal to the Fortran quadrature (itself gated against the Python reference) to 1e-12 at face,
  edge, corner and near offsets, for |k| d across the production range, real and complex omega. Plus
  agreement of the moments with high-order Gauss, and the test failing on a deliberate error.

**Stage 1, decisions taken while building it (21 October 2026).**

- **Reuse, not new derivation.** ``graded_voxel.derivatives.moment_table_kseries`` already sums this
  series (its table with weight W = 0 is the cube-averaged G and its first two derivatives), with its
  own truncation and its refusal past 48 powers; ``kseries_coefficients`` already defines the
  coefficients. Stage 1 supplies stored coefficients to it in place of its own.
- **The coefficients are computed in Fortran** (``fortran/kseries_moments.f90``, a term-for-term
  transcription of ``kseries_coefficients`` at W = 0, OpenMP over offsets). In Python they cost 0.13 to
  0.19 s per canonical offset, about 6.5 minutes for the 2,600 canonical offsets of `n_sub = 25`; that
  would make a first run slower than the quadrature it replaces. Compiled, about a minute of CPU.
- **They are stored once per machine, in a cache file outside the repository**
  (``cell_average.kseries_store``, by default ``~/.cache/cubic_scattering/receiver_kseries.npy``),
  created on first use and grown by the offsets each run needs. They are universal constants (integer
  offsets in cell units only), so the store is never stale for a frequency, cell size or medium.
- **The store is memory-mapped.** It is one ``.npy`` array, one row per canonical offset (three integers,
  then the coefficients flattened), opened with ``mmap_mode="r"``. Two reasons: the parallel workers of a
  frequency sweep (one process per frequency) share one physical copy through the operating system's
  page cache instead of each loading its own, and a run reads only the rows of the offsets it uses, so a
  store grown to a large radius costs nothing to open. A ``.npz`` cannot be memory-mapped, so it was
  replaced. One file rather than an index plus data means one atomic rename when it grows, so it can
  never be torn or disagree with itself; a layout change shows as a different row width and stops
  loading with a diagnostic.
- **Routing.** ``cell_average.route: kseries`` in ``numerics.yml`` takes the series for every lattice
  offset it reaches and the quadrature for the rest; ``route: quadrature`` restores the old behaviour. An
  explicit ``n_gauss`` always means the quadrature, as the convergence gates require.
- **The series is the more accurate route, and that is documented where it matters.** Against 24- and
  32-point rules it agrees to about 1e-13; the automatic Gauss orders of the quadrature leave the second
  shell wrong by up to 1.9e-8 (``cell_averaged_pair.auto_n_gauss`` has the measured table, and
  ``cell_average_fortran`` says that its tests certify the transcription, not the accuracy).

**Stage 1 results (21 October 2026, `n_sub = 25`, 117,648 offsets, 2,924 canonical, 8 threads).**

| k_S R (sphere) | Series route per frequency | Offsets the series covers | Quadrature fallback | Quadrature alone |
|---|---|---|---|---|
| 0.5, 1, 2 | 0.31 s | 100% | none | 1.22 s |
| 3 | 0.31 s | 96.3% | 0.04 s | 1.23 s |
| 5 | 0.21 s | 30.2% | 0.85 s | 1.24 s |

- One-off: the coefficients for all 2,924 canonical offsets took 29 s to build (compiled, eight threads);
  the store is 40 MB. Later runs open it memory-mapped in 0.3 s.
- The first version summed the series one offset at a time through ``moment_table_kseries`` and
  assembled every image's block; it took 1.07 s per frequency, barely ahead of the quadrature. Summing
  for all canonical offsets in one tensor product (``canonical_greens``, tested equal to
  ``moment_table_kseries`` offset by offset) and assembling only the canonical blocks, then permuting the
  finished 9 x 9 blocks to their images (``cell_average_fortran.cube_image_blocks``: a signed permutation
  of the axes is a signed permutation of the 9 components), brought it to 0.31 s.
- Coverage falls at high frequency because distant cells leave the series' reach (|k_S| r_max beyond
  about 9). At k_S R = 5 only 30% of offsets are covered, and the fallback dominates: stage 2 is what
  removes it.
- The test suite runs with its own temporary store per session (``tests/conftest.py``), so tests never
  write to the user's cache and parallel workers never share a file.
- The dense reference of ``test_sphere_scattering_fft.py::TestFFTMatvec`` now builds its blocks through
  ``averaged_pair_blocks_9x9``, the route the kernel uses. Built by the single-offset quadrature, it
  differed from the kernel by the quadrature's own 1e-8 error, and the test (at 1e-10) failed: the same
  "two different operators" trap its docstring records from an earlier change.

**Correction to stage 1 (found in stage 2): the series' accuracy depends on its reach.** Checking the
whole route on 400 random offsets showed errors up to 1.1e-11, all from the SERIES, at offsets near the
edge of its reach, against a converged reference (24- and 40-point rules agreeing to 4e-14). The terms
grow to about e^(k r) / sqrt(2 pi k r) before cancelling, which magnifies the coefficients' own ~1e-14;
the truncation bound does not see it. Measured against a 32-point rule:

| \|k_S\| r_max | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 8-9 |
|---|---|---|---|---|---|---|---|---|---|
| worst error | 2.8e-14 | 5.3e-14 | 7.7e-14 | 1.8e-13 | 4.7e-13 | 9.6e-13 | 3.5e-12 | 8.4e-12 | 1.4e-11 |

So "about 1e-13 at every offset" was true only within |k_S| r_max of about 3. The series now takes only
offsets within ``cell_average.kseries_max_reach`` (3), and stage 2's expansion takes the distant ones.
Raising the reach to 5 or 6 would move more offsets back to the amortised series, at up to 9.6e-13: a
speed-for-accuracy choice, made in the YAML.

**Stage 2 results (21 October 2026, `n_sub = 25`, 8 threads; error the worst of 400 random offsets
against a 32-point rule).**

| k_S R (k_S d) | Route per frequency | Series | Expansion | Quadrature (24 points) | Quadrature alone | Worst error |
|---|---|---|---|---|---|---|
| 1 (0.08) | 0.31 s | 98.7% | 1.3% | 0% | 1.20 s | 7.5e-14 |
| 3 (0.24) | 0.53 s | 5.6% | 94.4% | 0% | 1.21 s | 8.2e-14 |
| 5 (0.40) | 0.60 s | 1.1% | 98.9% | 0% | 1.20 s | 1.2e-13 |
| 8 (0.64) | 0.77 s | 0.2% | 99.8% | 0% | 1.20 s | 6.8e-14 |
| 12 (0.96) | 0.85 s | 0% | 99.8% | 0.2% | 1.19 s | 6.8e-14 |

- The expansion is ``moment_table_series`` (weight W = 0, one sub-cube) vectorised: one call of the
  compiled derivative ladder per distinct truncation order (a close offset needing a high order must not
  impose it on the distant ones), then a weighted sum with the cube's moments. Tested equal to
  ``moment_table_series`` offset by offset, and to a 24-point rule.
- It applies from about 2.9 cells out (``RHO_MAX`` = 0.3), so close cells beyond the series' reach (only
  at |k_S| d above about 0.8) go to the quadrature, at the converged order ``fallback_n_gauss`` (24): the
  automatic orders gave 8.4e-10 there. ``route: quadrature`` keeps the automatic orders unchanged.
- The expansion is analytic but not amortised: its derivatives depend on the frequency. At production
  frequencies (k_S R of 3 to 8) it takes almost every offset, so the per-frequency cost is 0.5 to 0.8 s,
  against 0.31 s where the series dominates.
- New YAML keys, all required with full diagnostics: ``kseries_max_reach``, ``far_expansion``,
  ``far_tol``, ``far_max_order``, ``fallback_n_gauss``. A float written without a decimal point
  (``1e-15``) is read by PyYAML as text; the diagnostic now says so and shows ``1.0e-15``.
- Tests build their configuration blocks with ``tests/numerics_yaml.cell_average_yaml``, which writes
  every key at its production value unless a test overrides or omits it, so a new key no longer means
  editing every test's YAML.

**Stage 2: far offsets by the expansion about the cell centre (the plan as written).**

- <f>(s) = sum_{a,b,c even} h^(a+b+c) d_x^a d_y^b d_z^c f(s) / ((a+1)! (b+1)! (c+1)!): the cube's
  moments (frequency-independent) times derivatives of f at the cell centre (closed form, from the
  ladder that ``fortran/green_derivatives.f90`` evaluates to any order). Converges as (h / r)^2, for
  every frequency. Item 3's d^2 tail is its first term.
- Per frequency: one derivative evaluation per distant offset, no quadrature. Analytic, though not
  amortised, since the derivatives depend on the frequency.
- Switch: by h / r, the truncation order set by a tolerance; both in YAML.
- Gate: as stage 1, at offsets beyond the switch.

**Then:** the quadrature stays as the reference and as the fallback for any offset neither route
covers (none expected), and each route is benchmarked over a realistic sweep (tens of frequencies),
reported per sweep.

## Where the time went (before the changes)

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
