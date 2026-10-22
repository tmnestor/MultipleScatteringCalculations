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

## The thesis's Chapter 7 calculation (graded sphere): a different solver

**The items above sped up the point-collocation sphere solver (`sphere_scattering_fft`), which Chapter 7
does not use.** Chapter 7's record section (`scripts/record_graded_sphere.py`) runs the GRADED-VOXEL
Galerkin solver: polynomial cells of degree p = r = 1 (36 unknowns each), Galerkin coupling blocks
K[a, c] at the canonical offsets, an FFT on the (2N - 1)^3 embedding, GMRES to 1e-8 with each voxel's
block as the preconditioner, and the field at 31 receivers from each cell's polynomial source radiated
from its Gauss nodes. Its logged cost at N = 32 (19,760 voxels, 711,360 unknowns): 370 to 480 s of solve
and 73 to 243 s of readout per frequency, 17 frequencies, 2.7 hours. The Galerkin formulation is kept
throughout: every change below is a faster route to the same Galerkin blocks and the same readout.

**Where one frequency went (N = 32, k_S a = 3, kernel in single precision memory-mapped from disk; 8 threads).**

| Stage | Before | Why |
|---|---|---|
| Expanding canonical blocks to every offset (`fft._transform`) | 326 s | 4.5 million Python calls: every offset re-indexed again for each of 18 row groups |
| Field at the receivers (`farfield.radiate_exact`) | 198 to 336 s | per receiver, the 9 x 9 block of every one of 4.3 million Gauss nodes built at once (about 5.5 GB), then contracted; on 16 GB the memory pressure made the time vary threefold |
| Canonical coupling blocks by Gauss quadrature (`coupling_block` -> `_sform`) | 66 s | per frequency, although amortised series routes exist |
| GMRES (6 products, kernel read from disk) | 23 to 67 s | |
| Contrast projection (`site.cell_contrast_coefficients`) | 23 s | independent of the frequency, yet redone at each |

**Done (21 October 2026).**

1. **Vectorised kernel assembly** (`fft.transformed_kernel`, with `_orbit_arrays`, `_signed_maps`,
   `_fill_rows`): the 48 symmetries' index maps computed once, each offset's symmetry found by one array
   operation (the same first match ``_orbit_map`` takes), each row group filled by one gather. The same
   numbers (the existing test against the every-offset path, at three row groupings, in memory and in
   single precision, to 1e-13). The solve: 584 s to 158 s; the kernel assembly 417 s to 33 s.
2. **Fused compiled readout** (`fortran/point_kernel.f90`: `apply_9x9`; `kernel_fortran.apply_9x9_fortran`;
   `farfield.radiate_exact` and `graded_field(..., strain=False)`): the propagator applied to each
   source as it is evaluated, never stored, one receiver per thread, so the result does not depend on the
   thread count. Displacement only (rows 1 to 3) skips the second derivatives, the larger part of the
   work; the record section uses displacement only. The per-point evaluation is shared with `kernel_9x9`
   (`point_block`), whose existing tests (against Python, a 50-digit reference, thread count) pass
   unchanged. Equal to the former loop to 1e-13 (`tests/test_apply_kernel_fortran.py`).

**Two pitfalls met on the way.**

- **f2py and derived types.** A first version passed the point-independent tables as a Fortran derived
  type. f2py exposes a module's derived types in its interface even when private, and the result built
  without complaint, then returned zeros (and crashed when called directly). The tables are now private
  module variables of intrinsic types, which makes a call not re-entrant (documented in the file).
- **Build only with `python -m cubic_scattering.fortran.build`.** Calling `numpy.f2py -c` by hand in a
  scratch directory produced a broken library even from the ORIGINAL source; the project's build script
  does not. The cause was not run down.

3. **Contrast projection once per sweep** (done): `site.cell_contrast_coefficients_all` keeps the
   frequency-independent projection (216 profile values per cell, 4.3 million Python calls, 23 s) and
   multiplies by each frequency's contrast operator; equal to the per-cell function exactly.
4. **Canonical blocks** (done; 70 s to 17.7 s). Touching offsets by `blocks.near_block_series` (2 to 6 ms
   against 350 to 740 ms a block, 1e-15), within its stored moments (k_S h <= 1/2, `NEAR_STORE_MAX_KH`;
   beyond, the series computes its moments, minutes each, so Gauss is kept there). The rest by
   `blocks.coupling_blocks_far`: the same s-form Gauss rule, its offset-independent nodes and weights
   built once per order (`_far_rule`), the kernel for many offsets in one compiled call, one batched
   product; equal to `coupling_block` to 1e-13. The amortised far series was measured and NOT used: its
   coefficients cost 1 to 5 s per offset, about 100 minutes for the grid's 5,980 canonical offsets, and it
   loses digits at long reach (4.8e-13 at k_S a = 8.9, refused at the far corner), the edge effect of item 8.

One frequency at N = 32 in a sweep: about 125 s (solve 139 s at the first frequency, 22 s of it the
contrast projection paid once; readout 8.7 s), against 784 s at the start.

**Next.**

5. **GMRES** (done; 54 s to 9.3 s). Each product read the whole complex64 kernel (6.5 GB) a row at a
   time through a single-threaded `einsum`: 9 s a product from disk, 3.4 s in memory. The stored layout,
   kh[row, col, Fourier point], already puts the Fourier points contiguous, so the kernel was NOT
   re-laid out point-major (which would have changed the stored format and needed a second 6.5 GB copy
   to convert): a compiled product over the existing layout (`fortran/kernel_apply.f90`, OpenMP over
   blocks of Fourier points, the kernel read in place without a copy, accumulated in double precision as
   the `einsum` did; `kernel_fortran.apply_hat_fortran`) reads it once, in order, by all threads. Equal to
   the NumPy product to 1e-14, independent of the thread count. With it the disk barely matters: 9.3 s
   from disk, 8.0 s in memory, so `--disk` stays and larger grids keep working.

One frequency at N = 32 in a sweep, kernel on disk: about 82 s (solve 96 s at the first frequency, 23 s
of it the contrast projection paid once; readout 8.9 s), against 784 s at the start: about 9.5 times
faster. A 34-frequency record: about 47 minutes, against about 5.3 hours with the original code.

What is left at N = 32: the kernel's assembly and transform (28 to 34 s, about half of it FFTs), the
canonical blocks (about 20 s, half compiled kernel evaluations, half the batched product), and GMRES
(about 9 s).
6. **Frequencies in parallel**: independent, but each holds a 6.5 GB kernel; on 16 GB at most two at a
   time. (Item 9 removes the obstacle: a 0.3 GB kernel.)

**Measured, the rerun with all of the above (21 October 2026).** The 34-frequency record at N = 32: solve
64 to 90 s (mean 71 s), readout 8.2 to 9.6 s, 45 minutes in all; every frequency's error equal to the
stored one. The ladder at k_S a = 4: solve 2.0, 5.9, 13.4 and 48.3 s for N = 8 to 24 (kernel in memory,
double precision), 101.7 and 418.7 s for N = 32 and 40 (single precision, from disk); the lean path at
N = 24, 46.5 s and 5.5 GB peak against 48.3 s and 8.9 GB.

## Item 9: the cube group in Fourier space (plan)

**The identity.** The real-space kernel satisfies K(Q o) = R(Q) K(o) R(Q)^T for the 48 signed permutations
Q, with R the signed permutation T x S on the 36 rows and U x S on the 90 columns. Q is orthogonal and
maps the circular grid to itself (npad = 2N - 1 is odd, so the centred representatives k in
[-(N-1), N-1]^3 are unique and Q k stays among them), so the transform inherits it:

    K^(Q k) = sum_o K(o) exp(-2 pi i (Q k).o / npad) = sum_o' K(Q o') exp(-2 pi i k.o' / npad)
            = R(Q) K^(k) R(Q)^T.

Checked numerically before any code (n_sub = 4, all 48 Q, every k): 1.0e-16 relative in Fourier space,
4.1e-16 in real space. The k-orbits are exactly the offsets' orbits, so the canonical k are the canonical
offsets (a >= b >= d >= 0), C(N + 2, 3) of them: 5,984 at N = 32 against 250,047 points, 1/42.

**What it saves.** The kernel is stored at the canonical k only: 36 x 90 x 5,984 complex numbers, 0.31 GB
in double precision at N = 32 (6.5 GB single, 13 GB double, before), 0.60 GB at N = 40 (12.8 GB single
before). So the kernel is always held in memory in double precision: no disk, no single-precision
rounding, and frequencies can run side by side.

**The product.** At a point k = Q k_c, with (pr, sr) the row and (pc, sc) the column signed permutations
of Q:

    yh[r, k] = sr[r] sum_j K^(k_c)[pr[r], j] s'[j],     s'[pc[c]] = sc[c] sh[c, k],

a gather with signs of the 90 sources, one canonical 36 x 90 block, a scatter with signs of the 36
results. The arithmetic is unchanged; the kernel read per GMRES step falls 42-fold and each block (51 KB)
is reused by the 24 to 48 points of its orbit. In Fortran (`kernel_apply.f90`), OpenMP over the Fourier
points, the sources and results point-major so that each point's 90 values are contiguous; the
symmetries' index tables built once in Python.

**The assembly.** Unchanged up to the transform (row groups, real-space fill by the cube group, FFT in
double precision); only the canonical k of each group's transform are kept. The FFTs remain (about half of
the 28 to 34 s); they are the next target.

**Tests.** The canonical kernel equals the full kernel at the canonical k; the canonical product equals
the full product to round-off (both precisions) and does not depend on the thread count; the solve with
it equals the former solve.

**Done (21 October 2026).** `fft.canonical_transformed_kernel`, `fft.fourier_symmetry_tables`,
`kernel_fortran.apply_canonical_fortran` (`apply_canon_c64`, `apply_canon_c128` in `kernel_apply.f90`);
the solve takes this path whenever the compiled backend is selected, and the full kernel with the reference
backend or with every offset's blocks. Tests: `tests/test_graded_voxel_fourier_symmetry.py` (the
canonical kernel equals the full one at the canonical k exactly, in any row grouping; the product equals
the full product to 1e-13 in both precisions and is independent of the thread count; the solve equals the
full-kernel solve to 1e-12).

Measured, the ladder at k_S a = 4, now always double precision in memory, the same errors to three
figures:

| N | solve before | solve now | peak before | peak now |
|---|---|---|---|---|
| 24 | 48.3 s (c128, memory) | 44.3 s | 8.9 GB | 4.4 GB |
| 32 | 101.7 s (c64, disk) | 71.5 s | 8.8 GB | 4.9 GB |
| 40 | 418.7 s (c64, disk) | 244.7 s | 9.5 GB | 6.0 GB |

The single-precision rounding (8e-11) is gone with it, and N = 32 no longer needs the disk. What is left of
the solve is the assembly (the canonical blocks and the full FFTs of every row group), which item 9 does
not touch.

**The point solver (done, 22 October 2026).** The torch solver (`sphere_scattering_fft_gpu`, the
production path) now stores its 9 x 9 kernel -P T at the canonical k only (`CanonicalKernel`,
`_canonical_kernel`, `_apply_canonical`, `_matvec_fft_canonical`). The identity needs S T S^T = T for
the cube's T-matrix, checked exact; the kernel identity holds to 6.7e-15 with T folded in, 8.7e-16 for P
alone. Two routes give the same product: on the CPU with the compiled backend, `apply_canon_*` (a new
all-single-precision variant, `apply_canon_c64s`, avoids converting complex64 sources and results);
elsewhere (a GPU) gathers and one batched matmul, with the signs folded into the gather indices, since
MPS has no complex scatter. A first pure-torch version on the CPU was 1.5 times slower than the full
product (random gathers cost more than reading the kernel), hence the compiled route there; the grids are
also packed point-major, so that no transposed copy is needed.

| n_sub | full kernel | canonical | product c128, full / canonical | c64, full / canonical |
|---|---|---|---|---|
| 17 | 46.6 MB | 1.7 MB | 8.4 / 8.6 ms | 5.0 / 5.5 ms |
| 25 | 152.5 MB | 5.2 MB | 27.8 / 26.6 ms | 15.4 / 14.7 ms |
| 32 | 324.1 MB | 10.8 MB | 61.8 / 62.9 ms | 34.0 / 35.0 ms |

On a CPU the saving is memory only; it is what bounds the grid on a GPU.

**Right-hand-side orbits (done, 22 October 2026).** (I - P T) commutes with U_Q, (U_Q psi)(Q x) = S psi(x),
on the centred sphere grid. A column with U_Q b_c = sigma b_p has psi_p = sigma U_Q psi_c, so one GMRES per
orbit: the 9 Taylor columns under all 48 symmetries in 3 orbits (displacements, normal strains, shear
strains), the 9 plane-wave columns under the symmetries fixing k (8 on an axis: 6 orbits; generically only
the identity: 9). 18 solves become 9 on an axis, 12 otherwise (`_column_orbits`, `_cell_permutation`,
`_apply_symmetry`). The generated columns equal solving every column to 1e-9 at a tolerance of 1e-12;
T_comp to round-off. Whole point solve at k_S R = 3:

| n_sub | c128, all 18 / orbits | c64, all 18 / orbits |
|---|---|---|
| 17 | 1.51 / 1.47 s | 0.92 / 0.68 s |
| 25 | 4.64 / 3.25 s | 2.63 / 1.98 s |
| 32 | 10.38 / 7.04 s | 5.83 / 4.18 s |

Less than the factor two in solves, because the kernel's construction is paid once either way. Both are
written up in the thesis's Chapter 7, section 7.4 (the cube group in the solution).

## Item 10: the point formulation against the Galerkin voxel (set up 22 October 2026)

**The question.** At equal accuracy, which formulation is cheaper? The Galerkin voxel (p = r = 1) converges
at fourth order on a smooth body against the point voxel's second, but it carries 36 unknowns a cell against
9, a 36 x 90 kernel against 9 x 9, and Galerkin blocks against a closed-form T-matrix and one
receiver-averaged propagator; and on a body with a sharp surface the staircase error of the voxelised shape
may cap both orders. Measured, not argued: error against wall time, on a smooth and on a sharp body.

**Fairness.** Each arm by its fastest route and for the one incident wave the scattered field needs
(`solve_point_plane_wave`, new: the single right-hand side on the canonical-k kernel, uniform or graded
contrast; `solve_graded_sphere_fft`); the same GMRES tolerance (1e-10), complex128 on the CPU, the same
far-field measure (max |u - exact| / peak, nine angles at 5e8 radii, as the pilots), and the readout
vectorised for the point arm (`point_far_field`, equal to the former loop to 5e-16). Each run is a child
process, so its peak memory is its own. The new point arm reproduces the pilot's stored errors exactly
(1.035e-1 and 4.655e-3 at n = 4 and 6 on the graded sphere at k_S a = 0.5), as does the Galerkin arm
(5.465e-4 at n = 4).

**Runs.** `scripts/compare_point_galerkin.py`: bodies graded (smoothstep shell, exact
`graded_mie_result`) and sharp (homogeneous, exact Mie), k_S a = 0.5 and 4, point ladder n = 4 to 64, Galerkin
ladder n = 4 to 40; rows in `scripts/data/point_vs_galerkin.jsonl`, the table by `--report`.

**Caveat on the sharp sphere.** The Galerkin cell carries the L2 projection of its contrast, computed with a
six-point Gauss rule per axis; for a step in the contrast the rule integrates a discontinuous function, so
part of the Galerkin arm's sharp-sphere error is the projection's, not the staircase's.

**Measured (22 October 2026), before item 11; superseded by the remeasurement at the end of item 11,
which changes the conclusions.** Wall time includes setup, GMRES and readout; every run from a warm
coefficient store (the first point runs at k_S a = 0.5 and n >= 32 paid its one-off build, 34 to 291 s,
and were rerun). Selected rows (all in `scripts/data/point_vs_galerkin.jsonl`):

| body, k_S a | point: n, error, wall | Galerkin: n, error, wall |
|---|---|---|
| graded, 0.5 | 12: 6.4e-4, 0.2 s; 24: 1.0e-4, 1.8 s; 64: 1.5e-5, 35 s | 4: 5.5e-4, 0.7 s; 12: 1.5e-5, 5.8 s; 40: 1.3e-7, 250 s |
| graded, 4 | 8: 5.1e-2, 0.9 s; 32: 3.3e-3, 3.3 s; 64: 8.3e-4, 43 s | 8: 3.7e-3, 1.9 s; 12: 7.7e-4, 5.7 s; 40: 6.5e-6, 252 s |
| sharp, 0.5 | 12: 7.2e-3, 0.3 s; 48: 8.9e-4, 11 s; 64: 8.0e-4, 37 s | 8: 1.2e-3, 1.9 s; 24: 2.9e-4, 46 s; 40: 1.7e-4, 244 s |
| sharp, 4 | 32: 7.0e-3, 3.3 s; 48: 5.2e-3, 12 s; 64: 1.2e-3, 39 s | 8: 5.3e-3, 1.7 s; 24: 1.7e-3, 45 s; 40: 1.0e-3, 249 s |

**What it shows.**

- Per unknown the point formulation is about six times cheaper (5.4 to 6.4 over the four cases: 1.24
  million unknowns in 35 to 43 s against 1.37 million in 244 to 252 s), as the simpler T-matrix, the 9 x 9 propagator and its symmetry
  predict. Its peak memory at a given unknown count is similar (8.5 to 9.3 GB at n = 64 against 6.2 to 7.7
  GB at Galerkin n = 40).
- On the smooth body the orders decide it: the point voxel converges at second order (1.9 to 2.0 between
  n = 16 and 64), the Galerkin at fourth (4.0 between n = 32 and 40). The point arm is cheaper only at
  coarse accuracy (about 1e-4 and above at k_S a = 0.5, about 1e-2 and above at k_S a = 4); at 1.5e-5 the
  Galerkin is six times faster (5.8 against 35 s) and at 8e-4 for k_S a = 4 seven and a half times faster
  (5.7 against 43 s), and it reaches errors (1e-7, 7e-6) the point arm cannot reach on 16 GB.
- On the sharp body neither converges cleanly: both errors move irregularly with n, the staircase of the
  surface. The Galerkin cell carries the fraction of its volume inside the surface (through its projected
  contrast), the point cell all or nothing, and at k_S a = 0.5 the Galerkin is the cheaper at about 1e-3
  (1.9 against 11 s); at k_S a = 4 the two cross near 1e-3, where the point arm is the cheaper (39 against
  249 s for 1.2e-3 and 1.0e-3). The Galerkin sharp-sphere errors include its Gauss projection of a step,
  so they bound its staircase error from above.
- The setup (blocks, kernel and its transforms) is 72 to 88% of the wall time in both arms from n = 24
  up: that, not GMRES, is the next target. (The commit that recorded these results says seven times
  cheaper per unknown; six is right.)

Written up in the thesis's Chapter 7, section 7.7 (the point and the Galerkin voxel), with the figure of
error against wall time.

## Item 11: the Galerkin setup (plan, 22 October 2026)

Measured, the setup of `solve_graded_sphere_fft` stage by stage (the same at k_S a = 0.5 and 4):

| stage | n = 24 | n = 32 | n = 40 |
|---|---|---|---|
| kernel: real-space fill and FFTs, canonical k kept | 20.8 s | 24.4 s | 132.0 s |
| canonical blocks | 9.0 s | 19.1 s | 36.7 s |
| contrast projection (Python profile calls, once a sweep) | 6.6 s | 14.8 s | 28.0 s |
| preconditioner (a 36 x 36 inverse per cell, Python loop) | 4.0 s | 8.9 s | 17.6 s |
| the rest | 0.8 s | 1.8 s | 3.5 s |

**11a. The kernel by the group again, in real space.** Every one of the 3,240 real-space arrays (row,
column) is transformed in full, and 1/42 of each transform kept. The arrays are related: an axis
permutation Q maps array (r, c) to (p_r(r), p_c(c)) with a sign, so K^(k_c)[r, c] = s K^(Q^T k_c)[r0, c0]
and one array per orbit of index pairs suffices: 576 orbits (sizes 3 and 6). And an axis flip maps every
index to itself with a sign, so each array is even or odd along each axis: its transform at the
non-negative wavenumbers is a cosine or sine sum over the non-negative octant of offsets, N^3 values
instead of (2N - 1)^3, applied as three N x N matrix products (BLAS) rather than an FFT. Q a pure
permutation keeps Q^T k_c in that octant. Expected: the 24 to 132 s stage to a few seconds. Tested against
the full kernel at the canonical k (to round-off, no longer bit for bit).

**11b. The preconditioner batched.** One stacked inverse of (M - K(0) E_n) over all cells.

**11c. The canonical blocks** and **11d. the contrast projection** (vectorised profiles) next, measured
again first.

**11a and 11b done (22 October 2026).** `canonical_transformed_kernel` now fills one array per orbit of
index pairs on the non-negative octant and transforms it by the even and odd octant sums; it equals the
full kernel at the canonical k to 1e-13 (n = 3, 4 and 5, p = r = 1 and p = r = 0), and a deliberate sign
error in the sine sum fails 8 of its tests. Kernel stage: 20.8 s to 0.63 s at n = 24, 24.4 s to 1.6 s at
n = 32, 132 s to 3.4 s at n = 40. The block-Jacobi preconditioner (`_block_jacobi`, one batched product
and one stacked inverse): 8.96 s to 1.12 s at n = 32. Setup at n = 32: 69 s to 39 s, of which the
canonical blocks are now 19.7 s and the contrast projection 14.5 s (11c, 11d).

**11c done (22 October 2026).** `coupling_blocks_far_lattice`: every canonical non-touching block from one
kernel lattice shared by neighbouring offsets (the lower Gauss piece of o is the upper piece of o - 1),
contracted axis by axis with the separable piece weights, slab by slab in o0 (which fixes the Gauss
order). Equal to `coupling_blocks_far` to 1e-13 at every distance band, a raised order, a complex
frequency and the 20- and 35-monomial bases. Blocks: 19.7 s to 8.8 s at n = 32, 36.7 s to 15.2 s at
n = 40. Of the 8.5 s left at n = 32: kernel evaluations 3.1 s (7.3 million points), contractions 3.5 s,
array building 2.0 s. The contrast projection (14.5 s) is now the largest stage (11d).

**11d done (22 October 2026).** `graded_voxel.site.BatchProfile`: a profile evaluated at many points in one
call (and still at one). The projection then evaluates every cell's 216 Gauss points in one call and
projects them in one product (equal to the point-by-point projection to 1e-14; the range check still
names the cell); the point solver evaluates its per-cell factors the same way. The scripts' graded and
sharp profiles are batch profiles (equal to the former scalar ones to 1.6e-15; the recorded comparison
errors reproduce). Contrast projection: 14.5 s to 0.18 s at n = 32.

**Setup after 11a to 11d** (k_S a = 4): n = 32, 69 s to 14.7 s; n = 40, 218 s to 34.9 s. What is left:
the canonical blocks, 9.5 s and 23 s (two-thirds), then the kernel (1.8 s, 3.9 s), the preconditioner
(1.3 s, 3.6 s) and the per-cell source expansion (1.2 s, 2.8 s).

**11e. Fast FFT sizes (done, 22 October 2026).** The embedding needs at least 2N - 1 points per axis, and
that is often prime (79 at N = 40, 127 at 64), which puts the FFT on its general-length algorithm: 66 ns a
point at 79 against 15.5 ns at 81. Both solvers now embed in `sphere_scattering_fft.fast_odd_size`, the
smallest odd size >= 2N - 1 with factors 3, 5 and 7 only (odd, so the centred wavenumbers stay unique and
the canonical-k tables hold), the kernel built for the effective N' = (size + 1) / 2 with zero blocks
beyond the real offsets. Equal solves to round-off (N = 6, 15 points against 11, both solvers). Galerkin at
N = 40: GMRES 39 s to 14 s, the whole 74 s to 50 s; point at N = 64: GMRES 10.7 s to 6.0 s.

**Item 10 remeasured after item 11 (22 October 2026).** Every error unchanged (to 9e-11); the Galerkin
times fall, and the conclusions change. The Galerkin voxel now costs only 1.30 to 1.34 times the point
voxel per unknown (n = 40: 45 to 51 s for 1.37 million unknowns; point n = 64: 31 to 34 s for 1.24 million).
Galerkin setup is 55 to 61% of its wall time (point: 79 to 85%). The size ladder solves in 1, 2, 4, 10, 20
and 42 s for n = 8 to 40; the 34-frequency record takes 17 minutes (each solve about 20 s).

| body, k_S a | point cheaper above | then, at equal error (Galerkin against point) |
|---|---|---|
| graded, 0.5 | about 3e-4 (both under a second) | 1e-4: 0.9 against 2.9 s; 1.5e-5: 2.0 against 32 s |
| graded, 4 | a tie near 3e-2 | 3e-3: 2.1 against 13 s; 1e-3: 2.1 against 34 s |
| sharp, 0.5 | about 1e-2 (both under a second) | 3e-3: 0.9 against 11.5 s |
| sharp, 4 | a tie near 3e-2 | 3e-3: 2.2 against 34 s; 1.2e-3: 24 against 34 s |

The Galerkin voxel is the method for both bodies below about 3e-4 at long wavelengths and 1e-2 at short
ones; the point voxel only at the coarsest accuracies. Chapter 7 (sections 7.3, 7.4, 7.5, 7.7, Table 7.1
and Figure 7.4) is updated to these numbers, with the kernel's construction by pair orbits and parity, the
shared lattice and the fast embedding size written into section 7.4.

## Item 12: the point solver setup (22 October 2026)

Measured first, at n = 64 (k_S a = 4, 135-point embedding): setup 33 s, of which the cell-averaged
propagator at all 2.05 million offsets 12.4 s, the canonical extraction and its tables 8.2 s (5.5 s of it
`_orbit_arrays`, comparing every offset with its 48 images), placing -P T on the grid 6.4 s, the 81 full
FFTs 5.7 s.

**Done.** (a) `_orbit_arrays` computed directly: the first matching symmetry is the first axis permutation
that fits the sorted absolute values, with the signs of the offset (pinned against the offset-by-offset
map; 5.46 s to 0.53 s at n = 68). (b) `_point_canonical_kernel`: the averaged propagator at the canonical
offsets only (about 1/48), times T, through the Galerkin construction (`_canonical_hat_from_blocks`,
split out to take plain (K, rows, cols) blocks) as a 9 x 9 cell with one test function and one source
monomial; equal to the full transform's canonical blocks to 1e-13 (T folded in or not, n = 5 and the padded
n = 6; a dropped sign fails it). (c) The gather tables of the device route are built only for that route.
Point setup: 2.8 s to 0.75 s at n = 32, 10.5 s to 2.0 s at n = 48, 28.2 s to 3.9 s at n = 64; the whole
run at n = 64, 34 s to 10 s, errors unchanged. GMRES (6 s at n = 64) is now the larger part.

**The comparison after item 12 (22 October 2026).** Point arm rerun, errors unchanged (to 9e-12). The point
voxel is again much the cheaper per unknown, 4.7 to 5.2 times (n = 64 in 8 to 9.5 s), its setup now 34 to
40% of its time at n = 64. Smooth body: the point voxel is the cheaper down to about 1e-4 at k_S a = 0.5
(a tie at about a second) and 1e-2 at k_S a = 4; below, the Galerkin voxel, 2 to 5 times faster over the
measured range (1.5e-5 at 0.5: 2.0 against 8.3 s; 1e-3 at 4: 2.1 against 9.5 s) and alone below. Sharp
body: the Galerkin voxel the cheaper at a few parts in 1e3, the point voxel near 1e-3 (3.4 against 3.9 s at
0.5; 9.0 against 24 s at 4), the Galerkin voxel alone below about 8e-4. Chapter 7 (sections 7.4 and 7.7,
Figure 7.4) updated. The point solve at n = 64 is now about 60% GMRES, at its FFT floor (0.46 s of each
0.6 s product); the Galerkin setup, chiefly its canonical blocks, is the larger remaining cost.

## Item 13: the Galerkin canonical blocks (22 October 2026)

Profiled line by line at n = 32 (9.1 s): kernel evaluations 3.1 s (1.0 s of it a copy transposing the
compiled output), contractions 3.3 s (einsum and tensordot), copying reshapes 1.2 s, accumulation and
array building 2.1 s. The kernel values, 9.5 GB over the n = 32 lattice, were written and read straight
back: the stage is bound by memory traffic, not arithmetic.

**Done.** (a) `kernel_9x9(..., transposed=True)`: the compiled routine's own layout, no copy (the lattice
carries the transposed component order and transposes each finished block). (b) The contractions as
matrix products on contiguous reshapes. (c) `point_kernel.f90`: `lattice_axis0`, the propagator on the
lattice folded into its axis-0 sums as it is evaluated, never stored (OpenMP over the output points;
equal to evaluating then contracting to 1e-14, independent of the thread count; the reference backend
keeps the evaluate-then-contract route, equal to 1e-13). Blocks: 9.1 s to 5.3 s at n = 32, 15.2 s to
9.3 s at n = 40. Tried and dropped: reordering the lattice points and merging the paired contractions
(7.1 and 8.0 to 8.5 s against 6.8 to 7.1 s, three fresh runs each).

**After item 13, remeasured (22 October 2026).** Errors unchanged (to 1.3e-11). Size ladder: 0.7, 1.5, 2.9,
7.4, 15.7 and 34.0 s for n = 8 to 40 (peak 6.4 GB at 40); the 34-frequency record 13 minutes (each solve
about 15 s). Comparison: the point voxel 4.0 to 4.9 times cheaper per unknown; the Galerkin voxel the
cheaper below a few parts in 1e4 (smooth, k_S a = 0.5: 1e-4 in 0.7 against 0.95 s) and from about 1e-2
(smooth, 4: 1e-3 in 1.7 against 9.5 s); on the sharp body a tie near 1e-3 at 0.5 (3.2 against 3.4 s) and
the point voxel cheaper there at 4 (9.0 against 19.5 s). The Galerkin setup is 46 to 52% of its time.
Chapter 7 updated (sections 7.3, 7.4, 7.5, 7.7, Table 7.1, Figure 7.4).
