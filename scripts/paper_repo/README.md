# Voxel discretisations of elastic scattering: reproduction repository

This repository contains everything needed to reproduce the paper

* *A Moment Hierarchy for the Voxel: Closed-Form Single-Site T-Matrices, and the Error of Elastic
  Volume-Integral Scattering Isolated on a Space-Filling Layer of Cubes* (`LatexPDFs/ContinuumLimit/`),

by T. M. Nestor. It is assembled from the author's research repository by
`scripts/assemble_paper_repo.py`, which records exactly which files a release contains.

## Layout

| Directory | Contents |
|---|---|
| `cubic_scattering/` | The Python package under test: the cube T-matrix, the cell-averaged lattice kernel, the sphere solvers and the graded voxel, with its test suite |
| `scripts/` | The cross-checks, gates, pilots and figure scripts the papers cite |
| `Mathematica/` | The symbolic derivations, as Wolfram Language scripts, with their saved outputs (`*.json`) |
| `LatexPDFs/` | The paper as compiled PDFs (`ContinuumLimit.pdf`, `JCP/ContinuumLimit_JCP.pdf`), with LaTeX source, bibliography, figures and figure data |

## Environment

    conda env create -f environment.yml
    conda activate continuum-limit
    python -m cubic_scattering.fortran.build

Python 3.12 with numpy, scipy, matplotlib, sympy, mpmath and PyTorch (the package's optional GPU
solvers import it; the CPU build is enough and nothing here uses a GPU), and gfortran, Meson and Ninja
for the compiled kernels.

The last command compiles the 9 x 9 point propagator (`cubic_scattering/fortran/point_kernel.f90`)
and the derivatives of the Green's tensor to any order that the moment hierarchy's tables use
(`cubic_scattering/fortran/green_derivatives.f90`), both Fortran 2008 with OpenMP, into extensions next to
their sources. It must be run once before anything else: `cubic_scattering/numerics.yml` selects them
(`point_kernel.backend: fortran`), and without the build the first call stops with the instruction to
build them. Each is a term-for-term transcription of a NumPy reference
(`graded_voxel.kernel.kernel_9x9_python`, `graded_voxel.derivatives.scalar_derivative_fields_python`), and
agrees with it to round-off (`cubic_scattering/tests/test_kernel_fortran.py`,
`cubic_scattering/tests/test_green_derivatives.py`). To run without compiling
anything, set `backend: python` in that file; every result is reproduced to round-off, more slowly.

## Reproducing the numbers

Two tiers. Every command is run from the repository root.

    ./reproduce.sh quick

runs every Python cross-check and gate, the measurements of the two bases and the stratified
background, and the hierarchy's checks on the layer and on one ball, and redraws the convergence
figure from the saved data. Each check prints the quantity it checks and the tolerance it meets, and
exits non-zero if it fails, which stops the run. This tier needs no Mathematica and ran in
22 minutes (1,338 s) on an Apple M-series laptop.

    ./reproduce.sh full

adds the long runs: the voxelised spheres against their exact solutions, the hierarchy as a voxel
scheme, oblique incidence in the stratified backgrounds, and the sphere's near field and impedance
march. Their cost, measured on an Apple M-series laptop, is listed below so that a reader can choose
which to repeat.

| Run | Unknowns | Time | Memory |
|---|---|---|---|
| Mean-only voxel on the graded sphere, 24 cells across | 79,344 | 330 s | small |
| Graded first-moment voxel, FFT solve, 16 cells across | 99,936 | 105 s | 3.9 GB |
| Collocation voxel on the graded sphere, 16 cells across | 19,584 | 1,560 s | small |
| Dense graded-voxel solve, 8 cells across | 14,688 | minutes | memory-bound; use the FFT solve beyond 8 |
| Hierarchy on the graded sphere, FFT solve, 4 to 8 cells across | up to 24,480 | 20 s | small |
| Hierarchy on a cube cut into n^3 cells (`measure_lattice_gradient_hierarchy.py`) | up to 1,620 | 242 s | small |
| Single cube by the hierarchy; oblique degree two; oblique stratified; two-term law at 20 degrees | | over 10 min each on one thread | small |

## Where each result comes from

Every result in the papers is derived symbolically in Mathematica and checked by an independent
Python/Fortran implementation or an exact solution. The table gives, for each result, the notebook that
derives it and the script that checks it. Mathematica is needed only to rerun the derivations;
their outputs are saved beside them, and the Python checks reproduce the numbers without it.

### The continuum-limit paper

| Result | Derivation (Mathematica) | Check (Python/Fortran) |
|---|---|---|
| Exact layer and its thin-layer series (§3) | `ContinuumLimit_Reference.wl`, `ContinuumLimit_ThinLayer.wl` | `scripts/continuum_limit_reference.py` (reference dump) |
| Specular sums of the point kernel and the surviving local term (§4) | `ContinuumLimit_SpecularSums.wl`, `ContinuumLimit_StaticLocal.wl` | `scripts/continuum_limit_specular.py` |
| Tiling identity and the exact kernel (§5) | `ContinuumLimit_AveragedSums.wl` | `scripts/continuum_limit_averaged.py`; `cubic_scattering/cell_averaged_lattice.py` |
| Moment integrals, channels, cube-versus-sphere split (§6) | `CubeMomentCore.wl` (gate `CubeMomentCoreTest.wl`), `CubeA22Block.wl`, `CubeShearSplit.wl`, `CubeMomentArchiveCheck.wl` | `scripts/gate_cube_shear_split.py` |
| Closure against exact Mie, static and dynamic (§6) | `SphereClosureDynamic.wl` | `scripts/gate_sphere_closure_vs_mie.py` |
| Package T-matrix equals the closure (§6) | `ContinuumLimit_CubeT.wl` | `scripts/continuum_limit_cubet.py` |
| Moments of every grade, and a second route to them (§6, Appendix A) | `CubeMomentHigherGrades.wl`, `CubeMomentStore.wl` (store `CubeScalarMoments.m`), `CubeMomentCanonical.wl` | `scripts/crosscheck_cube_moments_ball_shell.py` (reads `cube_higher_moments.json`) |
| What each block of the hierarchy buys, on the layer (§6.4) | `ContinuumLimit_GradientHierarchy.wl` | `scripts/measure_layer_taylor_hierarchy.py` |
| The hierarchy as the single site of a ball and of a cube (§6.5) | | `scripts/measure_ball_gradient_hierarchy.py`; `scripts/measure_cube_gradient_hierarchy.py --ref=2,3` |
| Second-order convergence and its closed-form constant (§7) | `ContinuumLimit_Chain.wl`, `ContinuumLimit_ErrorConstant.wl` | `scripts/continuum_limit_chain.py` |
| Fourth and higher order, the constant c_p, one plane and the thin-layer series (§8) | `ContinuumLimit_FourthOrder.wl`, `ContinuumLimit_FourthOrderError.wl`, `ContinuumLimit_SecondMoment.wl` | `scripts/crosscheck_second_moment_voxel.py` |
| Reduction of the 3-D first-moment voxel to 1-D (§8) | `ContinuumLimit_Reduction3D.wl` | |
| Oblique incidence, incident SV and SH (§8) | `ContinuumLimit_Oblique.wl`, `ContinuumLimit_IncidentS.wl` (exports `*Export.wl`) | `scripts/crosscheck_first_moment_voxel.py` |
| The voxel of degree two at oblique incidence and for incident S (Appendix B) | | `scripts/measure_oblique_degree.py` |
| Random stratified layer (§9) | `ContinuumLimit_Heterogeneous.wl` | `scripts/gate_heterogeneous_reference_vs_kennett.py` |
| Contrast varying within a cell, the min(2p+2, 2r+2) rule (§9) | `ContinuumLimit_GradedContrast.wl` | `scripts/crosscheck_graded_contrast.py` |
| The two bases: the Born terms and the projection-error law, and its split by cells (§9, Appendix G) | `ContinuumLimit_BornTerms.wl`, `ContinuumLimit_LayerDefect.wl` | `scripts/measure_layer_bases.py` |
| A stratified background at normal incidence (§9) | | `scripts/measure_layer_stratified_background.py` |
| Oblique incidence and a three-layer background; the two terms of the error there (§9, Appendix C) | | `scripts/measure_oblique_stratified_background.py`; `scripts/measure_oblique_two_term.py` |
| Exact graded sphere (§10) | `ContinuumLimit_GradedSphere.wl`, `TakeuchiSaito.wl` | `scripts/crosscheck_graded_sphere.py` |
| Mean-only voxel on the graded sphere, table and figure panel (d) (§10) | | `scripts/pilot_graded_voxel_sphere.py --ka=0.5 --arms=g0fft 4 6 8 12 16 20 24`, and `--ka=1.0` |
| Sharp sphere and its staircase (§10) | | `scripts/pilot_sphere_voxel_vs_mie.py` |
| The graded sphere's near field (Appendix D) | | `scripts/measure_graded_sphere_near_field.py --ka=0.5 4 6 8 12 16`, and `--ka=1.0` |
| The graded sphere against the impedance march (Appendix D) | | `scripts/measure_graded_sphere_march.py --n=20 --steps=32` (and the other grids of the table); `scripts/measure_graded_sphere_planes.py --arms=g1 6 8 16` |
| The hierarchy on a graded layer (§11) | | `scripts/measure_layer_taylor_hierarchy_graded.py` |
| The hierarchy as a voxel scheme: a cube cut into n^3 cells (§11) | | `scripts/measure_lattice_gradient_hierarchy.py` (lattice blocks in `scripts/gradient_voxel_lattice.py`) |
| The hierarchy as a voxel scheme on the graded sphere with a 9 m shell (§11) | | `scripts/measure_graded_sphere_gradient_hierarchy.py --core=1 --q=1,2 4 6 8 10`; `scripts/measure_graded_sphere_gradient_hierarchy_fft.py --core=1 --q=3 4 6 8 10 12 14`, and `--profile=sin2`; `--check` compares the FFT solve with the dense one |
| The cost of the two schemes' tables (§11) | | `scripts/measure_hierarchy_table_cost.py` |
| Convergence figure | `ContinuumLimit_Figure.wl` (data) | `scripts/plot_convergence_orders.py` |

`CubeMomentArchiveCheck.wl` compares the moments with an independent archive of earlier
derivations that is not part of this release; the notebook is included for the record of what was
compared.

### Three-dimensional voxels (the paper's appendix on the voxel solver)

| Result | Derivation (Mathematica) | Check (Python) |
|---|---|---|
| Coupling blocks: weak kernel, layer reduction, subdivision identity | `GradedVoxel_WeakKernel.wl`, `GradedVoxel_LayerReduction.wl` (integrals in `GradedVoxel_term_integrals.jsonl`) | `scripts/check_graded_voxel_subdivision.py`, `scripts/measure_graded_voxel_site.py`, `cubic_scattering/graded_voxel/` tests |
| Weak-contrast orders (Born) | | `scripts/pilot_graded_voxel_sphere.py --weak --arms=t9,g0,g1 4 6 8` at each frequency |
| Full-contrast orders, FFT solve to 16 cells | | `scripts/pilot_graded_voxel_sphere.py --arms=t9,g0,g1fft 4 6 8 10 12 14 16` |

## Mathematica

The `.wl` files are plain-text Wolfram Language scripts. Run one with

    wolframscript -file Mathematica/ContinuumLimit_Reference.wl

Each locates its inputs relative to its own location, so the repository can sit anywhere. The
notebooks' saved outputs are committed beside them, and the Python checks reproduce every number
in the papers without Mathematica.

## Building the papers

    cd LatexPDFs/ContinuumLimit && lualatex ContinuumLimit && bibtex ContinuumLimit && lualatex ContinuumLimit && lualatex ContinuumLimit

The journal version is generated from the same source by `LatexPDFs/ContinuumLimit/JCP/make_jcp.py`.

## Tests

There are two kinds of check, and both should be run.

**The package's test suite** (`cubic_scattering/tests/`, pytest). Build the compiled kernels first
(see Environment), then, from the repository root,

    pytest cubic_scattering/tests -n 5

The `-n` option runs the tests in parallel (pytest-xdist, in the environment). On a 10-core Apple
M-series laptop five workers take about 25 minutes and report 1488 passed and 1 skipped; a serial run
(`pytest cubic_scattering/tests`) takes well over an hour. Smaller selections:

    pytest cubic_scattering/tests -k graded_voxel -n 5                  # the graded voxel, a few minutes
    pytest cubic_scattering/tests/test_kernel_fortran.py \
           cubic_scattering/tests/test_green_derivatives.py              # compiled = reference, seconds
    pytest cubic_scattering/tests/test_green_derivatives.py::test_compiled_equals_reference -v

The compiled kernels are checked against their NumPy references by the last two files; to run the
whole suite on the references alone, set `point_kernel.backend: python` in
`cubic_scattering/numerics.yml` (slower). If a test fails with an `ImportError` naming
`_point_kernel` or `_green_derivatives`, the build step has not been run in this environment.

**The papers' cross-checks and gates** (`scripts/`). These are separate programs, not pytest tests:
the checks print what they compare, with the tolerance, end with a PASS or FAIL verdict and exit
non-zero on a failure; the measurements print the numbers of the paper's tables, to be compared with
them. `./reproduce.sh quick` runs those that take seconds to minutes, and
`./reproduce.sh full` adds the long runs; any one can be run alone, as the table above gives. With
OpenMP, `OMP_NUM_THREADS` sets the number of threads of the compiled kernels and of the linear algebra.

## Licence

MIT, see `LICENSE`.
