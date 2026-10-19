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
