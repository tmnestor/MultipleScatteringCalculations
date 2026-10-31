# Documents

Every paper, report and note of the project, with its status and what it established. Dates are the last
change to the source. Each document builds in its own folder with `lualatex`, `bibtex`, `lualatex`,
`lualatex`.

Statuses:

- **live**: a paper of the current series, kept consistent with the code.
- **report**: a technical report; it records what was true when it was written, and the warnings
  below say where that has since changed.
- **archived**: material taken out of a paper and kept as a record, not as a paper.
- **retired**: superseded, never to be edited.
- **note**: an early working note.

## Withdrawn or corrected results

Read these before relying on any number from an older document.

- **The Galerkin cell's gain over the classical point cell, restated** (31 October 2026). Earlier versions
  of the four live papers said the cell of degree one needs 6.6 to 24 times fewer unknowns than the
  classical point cell. That point cell used the package's cube T-matrix, whose far-field form-factor
  corrections make its error 7 to 11 times larger than the uniform-strain closure itself, and sampled the
  contrast at the cell centres. Measured again with the closure done exactly (Error paper, Section 11 and
  Appendix F): given the same cell mean of the medium, the Galerkin cell needs 7.5 to 25 times fewer
  unknowns, growing as the three-quarters power of the accuracy asked; sampled at the centres, whose error
  on smooth bodies largely cancels the point cell's own, the two need about the same down to a crossover.
  At equal degree the point cell is 1.6 to 1.9 times the more accurate on a sphere.
- **Sphere comparisons before 20 September 2026** used a Mie arbiter with three sign errors. The
  verdict that followed their fixing, that the cell-average propagator is better only in the Rayleigh
  regime, did not survive the far-field fix of 29 September either. On a body without a staircase the
  matched pairing converges at second order at k_S a = 0.5 and 1 while the point propagator stalls near
  0.9% (`reports/SphereMieArbiter`, revised 31 October 2026).
- **Voxel-sphere far fields computed before 29 September 2026** through `foldy_lax_far_field` counted
  the incident plane wave's linear variation twice, an error of about (k a)^2: 1.1% at k_S a = 0.5 and
  4.6% at k_S a = 1.

## Live papers (this folder)

| Folder | Title | Date | What it establishes |
|---|---|---|---|
| `PolynomialVoxelError/` | The Error of Polynomial Voxels for Elastic Waves | 31 Oct 2026 | Order min(2p+2, 2r+2) on a layer, with closed-form constants; in 3-D a two-term error known before the solve; the gain over point matching (7.5 to 25 times fewer unknowns like for like, growing as the three-quarters power of the accuracy asked); the point cell in closed form (Appendix F) |
| `LegendreCellTMatrix/` | The Single-Site T-Matrix of a Legendre Cell in Closed Form | 31 Oct 2026 | The degree-one Legendre cell's T-matrix, every power of the frequency on six constants |
| `ExactCouplingIntegrals/` | The Coupling Integrals of Polynomial Cubes in Closed Form | 31 Oct 2026 | Coupling blocks exact and accurate to round-off, touching cells included; frequency-independent coefficients, 9.3 to 19 times cheaper than quadrature over 1024 frequencies |
| `InterfaceCouplingIntegrals/` | Coupling Integrals of Polynomial Cubes at a Welded Elastic Interface | 31 Oct 2026 | The static image in closed form and plane waves for the rest; the static and order-omega^2 parts computed once for every frequency |

## `archive/`

| Folder | Status | Date | Contents |
|---|---|---|---|
| `CartesianMultipoleHierarchyArchive/` | archived | 9 Oct 2026 | The field expanded about the cell centre, closed-form single sites of the elastic cube; superseded by the Legendre expansion. `JCP/` holds the earlier combined paper as submitted |
| `OctreeRefinementArchive/` | archived | 9 Oct 2026 | The octree material removed from the Error paper: blocks between unequal cells, refinement by the two-term error estimate |
| `GradedVoxel/` | retired | 15 Oct 2026 | Galerkin cubes with polynomial field and contrast; its results moved into the Error, Legendre and Coupling papers. Never edit |
| `TMatrixPointVsGalerkin/` | archived | 16 Jun 2026 | Point-source against body-force Galerkin single sites of one cube, judged against the equal-volume Mie sphere and the layer reflectivity, with its evidence harness (`evidence/`). Superseded by the Error paper, Section 11: its verdict (the point model suffices, the Galerkin closure is dominated) compares one cube with a sphere, uses the package cube T-matrix and predates the corrections listed above. Built with biber; the PDF is the record |

## `reports/`

Technical reports of the stratified-background programme.

| Folder | Date | What it records |
|---|---|---|
| `AdjointStateGradients/` | 30 Sep 2026 | Waveform-inversion gradients through the Foldy–Lax solve: one extra Krylov solve per source, frequency and transverse wavenumber; the Born limit is Dietrich and Kormendi's |
| `ContrastSourceVIE/` | 16 Jun 2026 | Background and literature: the contrast-source volume integral equation, FFT acceleration, the seismic grid T-matrix |
| `DirectionalSweepSolver/` | 27 Sep 2026 | The propagator's product as alternating-direction partial-wave sweeps, lateral (k_x, k_y) and in depth (Kennett) |
| `EwaldIntraPlanePropagator/` | 16 Jun 2026 | Coupling within a depth plane of a stratified medium: conditional convergence, cured by Ewald summation |
| `FirstOrderContrastOperator/` | 26 Sep 2026 | The matrix-vector wave equation as a route to the single site and to a laterally coupled Dirichlet-to-Neumann march, and what it buys |
| `GlobalMatrixMethod/` | 14 Sep 2026 | The stratified reference solver, and the source as a jump in the six-component state |
| `IntraPlaneSpectralSweep/` | 26 Jul 2026 | The same-depth lattice sum by directional residues; its equivalence with Kambe's |
| `MultiscaleBackground/` | 26 Jul 2026 | A multiresolution background inside a Krylov solve, assessed: the proposed direction conflicts with inexact-Krylov theory, and the idea has been done |
| `SolverHardwareScaling/` | 31 Oct 2026 | The FFT sphere solver measured on Graviton CPUs and an A10G: the Krylov basis was allocated in full (fixed); the GPU forms the product 15x (complex128) to 50x (complex64) faster; complex64 solves cost nothing in accuracy; the kernel stage for many CPU cores, the solve for a GPU |
| `SphereMieArbiter/` | 31 Oct 2026 | The voxel T-matrix of a sphere against exact Mie; a T-matrix and its propagator must close on the same quantity, the condition for the scheme to converge at all (revised for the sign fixes, the far-field fix and the exact single site) |
| `SpherePlaneWaveResponse/` | 19 Sep 2026 | The plane-wave content of the Mie field on planes bounding the sphere, in closed form |
| `StratifiedComposition/` | 27 Sep 2026 | Composing the stratified and lateral propagators: reciprocity gates and the source convention |
| `WrapperProblem/` | 14 Sep 2026 | The 9x9 wrapper problem resolved: three defects, the gates to 1e-15 |

## `notes/`

| Folder | Date | Contents |
|---|---|---|
| `basic_equations/` | 7 Mar 2026 | The governing equations and the free-space Green's tensor, extracted from the 1996 thesis |
| `cubic_tmatrix/` | 10 Mar 2026 | The Rayleigh-regime T-matrix of a cube with the full Green's tensor. For the uniform-strain closure exact at every frequency, see the Error paper, Appendix F |
| `coupled_foldylax/` | 26 Jul 2026 | The coupled Foldy–Lax T-matrix extended to resonance in density and both moduli |
| `mode_converted_reflections/` | 12 Jun 2026 | Mode-converted specular reflections and evanescent incidence in the periodic slab solver. Its sphere comparison predates the corrections above |
| `multipolar_mie/` | 9 Mar 2026 | Multipolar effective sources and elastic Mie theory |
| `nearfield_verification/` | 9 Mar 2026 | The elastodynamic near-field Green's tensor, derived and verified |
| `psvsh_optical_theorem/` | 25 Sep 2026 | The P-SV-SH optical theorem: forward amplitude, flux weights, extinction |
| `psvsh_reciprocity/` | 25 Sep 2026 | P-SV-SH reciprocity: mode conversion, flux normalisation, the T-matrix metric |
| `psvsh_unitarity/` | 25 Sep 2026 | P-SV-SH unitarity: energy flux, open channels, the Kennett recursion |
| `spectral_9x9/` | 9 Mar 2026 | The 9x9 lattice propagator by horizontal residues |
| `fragments/` | 7 Mar, 25 Sep 2026 | Two short write-ups (the spectral Green's tensor checked by inverse FFT; the horizontal Green's tensor by Cartesian residues) and a bibliography on the Redheffer star product |

## `references/`

Third-party papers kept beside the documents for reading. Git ignores this folder, so it is never
committed.
