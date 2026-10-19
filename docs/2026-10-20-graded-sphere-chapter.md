# The graded sphere: voxels against the exact solution, and a record section (20 October)

For the thesis's 3-D edition (`voxel-3d`), a new final Chapter 7: plane-wave scattering by the smoothly graded
sphere solved with graded voxels and compared with the exact solution, with a thorough account of the exact
method and a record section on a plane above the sphere. The code exists (`cubic_scattering/graded_voxel/`,
`scripts/crosscheck_graded_sphere.py`, `scripts/measure_graded_sphere_planes.py`); the chapter needs a
realistically large number of voxels, accepting long runs.

## The problem

- **The body** (as `LatexPDFs/GradedVoxel/GradedVoxel.tex`, section sec:sphere): radius a = 10 m, a homogeneous
  core r < a/2 with the contrast d lambda = 2 GPa, d mu = 1 GPa, d rho = 100 kg/m^3 in the background alpha =
  5 km/s, beta = 3 km/s, rho = 2500 kg/m^3; across a/2 < r < a the contrast is multiplied by the smoothstep
  10x^3 - 15x^4 + 6x^5, x = (a - r)/(a/2), so it vanishes like (a - r)^3 at the surface.
- **The exact solution**: per angular order, the radial equations of Takeuchi & Saito (1972) in the notation of
  Dahlen & Tromp (1998), the spheroidal y = (U, V, R, S) and toroidal (W, T), started from the core's regular
  (j_n) P and S solutions, integrated through the shell (DOP853, rtol 1e-13), matched at r = a to the incident
  (j_n) and scattered (h_n) fields: the per-order T-matrix, and from it the scattered displacement at any point
  outside the sphere (`crosscheck_graded_sphere.graded_mie_result`, `sphere_scattering.mie_scattered_displacement`).
- **The voxel solution**: graded voxels of degree p = r = 1 (fourth order), the coupling by FFT on the
  (2n - 1)^3 circular embedding, GMRES with block-Jacobi (`graded_voxel.fft.solve_graded_sphere_fft`); the
  field at the receivers by `graded_voxel.farfield.graded_field`.

## Decisions (each with its reason)

1. **Chapter 7 on `voxel-3d`**, a new chapter in the place of the dropped Chapter 7 (surface waves); decided
   by the author, 20 October.
2. **The core is r < a/2**, the paper's and the code's definition (`pilot_graded_sphere_vs_exact.CORE`).
   The docstring of `scripts/taup_graded_sphere.py`, which says a/10, is wrong.
3. **The record section**: a P wave travelling down (+z) onto the sphere; receivers on a line in the plane
   z = -1.25 a above it (a quarter radius clear of the sphere: on the tangent plane the readout's Gauss rule
   meets the propagator's singularity at the poles' cells, `measure_graded_sphere_planes.py`), through the
   point above the centre, recording the vertical and horizontal scattered displacement; the voxel and exact
   traces overlaid and their difference shown. Proposed to the author on 20 October; no objection.
4. **Measure the size first** (author, 20 October): `scripts/measure_graded_sphere_size.py`,
   `scripts/data/graded_sphere_size.jsonl`. At k_S a = 4, p = r = 1, on this machine (16 GB):

   | n across | cells | unknowns | solve | peak | error on the line |
   |---|---|---|---|---|---|
   | 8 | 408 | 14 688 | 4.5 s | 0.7 GB | 4.6e-3 |
   | 12 | 1 280 | 46 080 | 10.8 s | 1.9 GB | 9.1e-4 |
   | 16 | 2 776 | 99 936 | 24 s | 3.4 GB | 2.9e-4 |
   | 24 | 8 816 | 317 376 | 113 s | 6.8 GB | 5.9e-5 |

   Fourth order (apparent 4.0, 4.0, 4.0). The ladder was stopped before n = 32, which needs more memory than
   the machine has in the former path. The memory is the transformed kernel `kh`, rows x cols x
   (2n - 1)^3 complex (36 x 90 at p = 1), held twice at its peak (the dictionary of every offset's block, then
   `kh`): 1.5 GB at n = 16, 5.4 GB at 24, 13 GB at 32, 26 GB at 40.
5. **Trade speed for size** (the author: accept long runs for a realistic voxel count). Three changes to the
   FFT solve, each optional so that the existing path is unchanged:
   - only the canonical blocks are kept (one per orbit of the cube's 48 symmetries, ~1/48 of the offsets),
     and `kh` is assembled a group of rows at a time from them, transformed and stored: no dictionary of every
     offset;
   - `kh` in single precision (complex64): half the memory and disk traffic. A fixed float32 kernel is still a
     fixed linear operator, so GMRES converges as before. Measured (`tests/test_graded_voxel_fft.py`): the
     stored kernel is within 1e-7 of complex128 and the solution moves by 8e-11, the coupling being a small
     part of M - K E at this contrast: far below the discretisation error;
   - `kh` optionally memory-mapped to disk (256 GB free), so the size is bounded by the disk and each GMRES
     iteration pays a read of `kh`.

   Built: `graded_voxel.fft.canonical_blocks`, `transformed_kernel`, and `solve_graded_sphere_fft(kernel_dtype=,
   kernel_path=)`; the former path (`blocks=`, every offset's block) gives the same solution to 1e-12, the
   memory-mapped kernel likewise.

   With the kernel in single precision on disk (`--c64 --disk`), k_S a = 4:

   | n across | cells | unknowns | solve | readout | error on the line |
   |---|---|---|---|---|---|
   | 24 | 8 816 | 317 376 | 111 s | 1.0 s / receiver | 5.85e-5 (as in memory) |
   | 32 | 19 760 | 711 360 | 495 s | 6.1 s / receiver | 1.86e-5 (order 3.98) |
   | 40 | 37 920 | 1 365 120 | 1 810 s | 97 s / receiver | 7.64e-6 (order 3.99) |

   At n = 40 the readout's 97 s per receiver (6 s at n = 32) is the memory under pressure from the kernel's
   pages, not the readout's own cost.

   The reported peak (5.9, 11.8 GB) counts the kernel file's pages in the page cache, which the system
   evicts under pressure. At n = 40 the kernel file (12.8 GB) no longer fits beside the rest, and each GMRES
   iteration reads it from disk. The ladder is stopped after n = 40: n = 48 is not needed.
6. **The record section's model: n = 32** (19 760 voxels, 711 360 unknowns). The record needs ~20 solves;
   at n = 32 they take ~10-15 min each (8 min at k_S a = 4) and finish overnight, where n = 40 (the kernel
   read from disk each iteration) would take about a day. The error at n = 32 (2e-5 at k_S a = 4) is far
   below what a record section shows.
7. **The record section's parameters** (`scripts/record_graded_sphere.py`): a Ricker pulse peaking at
   k_S a = 3 (wavelength ~2a, so the top and the bottom of the body are resolved), its spectrum used to
   3 x the peak (k_S a = 9); a record of T = 12 a / beta = 40 ms, which holds every arrival on the line to
   x = 3a (the latest, S converted at the far side, by ~17 ms) after the pulse's causal delay of 1.5 periods
   (10.5 ms): 17 frequencies, k_S a = 0.52 to 8.9; 31 receivers at 0.1a on z = -1.25a, x = 0 to 3a, the
   figure every other one, mirrored; time from the incident wave's crossing of the receivers' plane. GMRES
   to 1e-8 (the discretisation error is ~1e-4 and below), the kernel in single precision on disk.
8. **Timing conditions.** The size ladders up to n = 32 ran while the slow test suite (`pytest -m slow`,
   two threads) shared the machine, so their times are slightly pessimistic. From 03.50 on 21 October the
   slow tests are paused (SIGSTOP) during the timed runs, n = 40 and the record section, and resumed after.
9. **The figures are drawn with pgfplots** from tables in the thesis's `EpsFiles/data`, as the thesis's other
   data figures are, so that type and colours match the text (Okabe-Ito: exact black, voxel vermillion
   dashed, the 100x difference blue).
