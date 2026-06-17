# Intra-Plane Foldy-Lax: Planar Multiple Scattering of Spherical Voxels into Layer R/T

**Pillar 2 of the lateral-heterogeneity program.** Status: planning. Date: 2026-06-18.
**Representation: full spherical-multipole (Cruzan/Stein), all `n` (option B, DECIDED).**

## 1. Purpose

Model lateral heterogeneity superimposed on a plane-stratified (1-D depth) background.
At each depth, the lateral heterogeneity is discretised as a dense planar packing of
spherical voxels. The voxels in a plane multiply-scatter (same-depth / intra-plane), which
produces an effective **layer reflection/transmission operator R/T(p)** as a function of
horizontal slowness `p`. The Kennett recursion then stacks these scattering layers onto the
stratified background. **No CPA / effective-medium homogenisation is used.**

## 2. Where this sits

| Pillar | Object | Status |
|---|---|---|
| 1 | Single-site `T0` (full-wave sphere scattering operator) | **DONE** `Mathematica/CartesianT0.nb`, reciprocity-verified 1e-18 |
| **2** | **Intra-plane `G0` + planar Foldy-Lax -> layer R/T(p)** | **THIS PLAN** |
| 3 | Kennett vertical recursion on the stratified background | EXISTS `cubic_scattering/kennett_layers.py` |

```mermaid
flowchart TD
    H[Lateral heterogeneity field at depth z] --> V[Dense packing of spherical voxels, each a full T_n]
    V --> G[Intra-plane G0 = lattice-summed Cruzan/Stein translation]
    G --> FL[Planar Foldy-Lax:  I - G0 T0  solved per plane, spherical basis]
    FL --> RT[Layer R/T of slowness p, flux-normalized P-SV-SH]
    RT --> K[Kennett vertical recursion over stratified background]
    K --> S[Seismic response / shot gathers]
    T0src[CartesianT0.nb: full-wave sphere T_n + elastic Weyl bridge] -.per-voxel T0 + lattice-sum seed.-> V
```

## 3. Formulation

Per depth-plane (all voxels at common `z`, separations purely horizontal so `Delta z = 0`),
in the vector spherical wave (Hansen `L/M/N`) basis, retaining all multipoles `n`:

- **Foldy-Lax:** solve `(I - G0 . T0) b = a_incident` for the exciting/scattered multipole
  coefficients, where `T0` is the per-voxel full-wave T-matrix (`CartesianT0.nb`, all `n`) and
  `G0` is the intra-plane spherical-wave coupling between voxel centres.
- **Intra-plane `G0`:** the **elastic vector-spherical-wave translation-addition theorem**
  (Cruzan 1962 / Stein 1961, elastodynamic P-S form): re-expand an outgoing `L/M/N` multipole
  field centred at voxel A as regular `L/M/N` multipoles centred at voxel B. The **planar
  lattice sum** of these translation matrices over the dense packing (at fixed horizontal
  slowness `p` / Bloch vector) is the operative `G0`.
- **Lattice sum:** evaluated by the **Weyl / reciprocal-lattice angular spectrum** (the elastic
  Weyl bridge already in `CartesianT0.nb` is the seed), Ewald-accelerated for the conditionally
  convergent sum.
- **Projection to R/T(p):** the planar collective scattering is projected onto up/down P-SV-SH
  plane waves at slowness `p` in Kennett's flux-normalized basis (`sqrt(eta rho)`; `Rd, Ru, Td,
  Tu` 2x2 for P-SV plus scalar SH), and handed to `kennett_layers.py`.

## 4. Why option B (spherical-multipole) and not the 9x9 Cartesian path

- Dense near-field packing couples **high multipoles** (`n > 2`); the 9x9 (`n <= 2`) form would
  truncate exactly where the physics lives.
- It is the genuinely full-wave path and reuses `CartesianT0.nb` directly (full `T_n`).
- It delivers the **Cruzan/Stein vector-spherical-translation multiple-scattering solve** to
  benchmark against the existing Cartesian FFT / block-Toeplitz `G0`.

The existing Cartesian 9x9 `G0` / `slab_scattering` stack is **not** on this path; it is retained
as an independent **cross-check** in the Rayleigh / `n <= 2` limit (Phase 2 benchmark).

## 5. What exists to build on

| File | Provides | Use |
|---|---|---|
| `Mathematica/CartesianT0.nb` | full-wave sphere `T_n` (all `n`) + elastic Weyl angular-spectrum bridge | per-voxel `T0`; the Weyl bridge seeds the lattice sum and the R/T projection |
| `cubic_scattering/lattice_greens.py` | Ewald / screened-Coulomb + Hankel-transform lattice-sum infrastructure | adapt the acceleration to the spherical-translation lattice sum |
| `LatexPDFs/EwaldIntraPlanePropagator/` | same-depth lattice-sum / Ewald theory | derivation reference for the spherical lattice sum |
| `cubic_scattering/kennett_layers.py` | vertical R/T recursion, flux-normalized P-SV-SH | layer stacking (unchanged) |
| `cubic_scattering/horizontal_greens.py`, `slab_scattering.py` | Cartesian 9x9 `G0` + Foldy-Lax | independent Rayleigh-limit cross-check, not the primary path |

## 6. Phased implementation (tracer-bullet, each phase ends verified)

**Phase 0 - elastic translation-addition operator (Cruzan/Stein).**
Implement the elastodynamic vector-spherical-wave translation: outgoing `L/M/N` multipoles at A
-> regular `L/M/N` multipoles at B for a horizontal separation (`Delta z = 0`), with the P and S
parts. Most cleanly seeded from the Weyl/angular-spectrum representation already built.
*Accept:* re-expansion reconstructs the translated field (an outgoing multipole about A,
evaluated near B, equals the regular-multipole sum about B) to set tolerance; the translation is
reciprocal; reduces to the scalar Gegenbauer addition theorem on the L (P) part.

**Phase 1 - planar lattice sum of the translation operator.**
Sum the pairwise translation over the dense planar packing at fixed slowness `p` (Bloch vector),
via the Weyl / reciprocal-lattice angular spectrum, Ewald-split for convergence.
*Accept:* convergence of the real + reciprocal-space split; reciprocity; agreement with a direct
real-space sum in regimes where the latter converges; the scalar limit matches a known planar
lattice Green's function.

**Phase 2 - planar Foldy-Lax in the spherical basis.**
Solve `(I - G0 . T0) b = a` with the full `T_n` (`CartesianT0.nb`) and the lattice-summed `G0`.
*Accept:* single voxel returns `T0`; two voxels match a direct spherical two-body solve;
**convergence in multipole order `n`** and in packing density; reciprocity + energy of the
collective operator; reduces to the Cartesian 9x9 slab in the Rayleigh / `n <= 2` limit (the
cross-check benchmark from Section 4).

**Phase 3 - layer R/T(p).**
Project the planar collective scattering onto Kennett's flux-normalized up/down P-SV-SH at
slowness `p` (`Rd, Ru, Td, Tu`, SH scalar), via the Weyl/far-field bridge.
*Accept:* `Tu = Td^T` and `Rd` symmetric (Kennett reciprocity); for a lossless plane the
flux-normalized `|R|^2 + |T|^2 = 1`.

**Phase 4 - Kennett coupling.**
Insert the planar R/T as a scattering layer in `kennett_layers` over the stratified background.
*Accept:* a transparent (zero-contrast) plane is the identity; a uniform-contrast plane matches
an independent analytic/interface result; full survey runs.

**Phase 5 - lateral-heterogeneity model.**
Map a lateral heterogeneity field to per-voxel contrasts across planes; full multi-plane survey.
*Accept:* physical trends (reflectivity vs contrast, density, frequency); convergence in packing
and multipole order; documented.

## 7. Global verification

Reciprocity and energy conservation (lossless background) at every level: single voxel
(reciprocity 1e-18, already), the translation operator (Phase 0), the collective plane
(Phase 2/3), and the stacked response. The exact `CartesianT0.nb` Mie operator is the
single-voxel ground truth; the existing Cartesian cube-slab is the Rayleigh-limit ground truth.

## 8. Risks and open questions

- **Lattice-sum convergence** is now the central numerical risk: the planar sum of the
  translation operator is conditionally convergent; the Ewald / reciprocal-lattice split (Phase 1)
  must be done carefully. The `EwaldIntraPlanePropagator` note and `lattice_greens` Hankel/Ewald
  code are the starting points.
- **Translation-theorem region of validity:** the addition theorem re-expansion converges for
  non-overlapping spheres; dense packing (near-touching) needs **high `n`** and possibly the
  near-neighbour terms handled separately. Phase 2's `n`-convergence study quantifies this.
- **Multipole order** is a convergence parameter (not a truncation fork): track `n_max` vs
  packing density and frequency.
- **RQ1 sub-fork:** fixed-depth-planes + Kennett-between (this plan) versus full 3-D near-field.
  This plan commits to fixed-depth-planes.
- **Self / near term:** the `r = 0` and nearest-neighbour translation terms (the `G0` diagonal /
  the Eshelby delta analogue); reuse the verified cube treatment where applicable.
- **Flux-normalisation match to Kennett** (`sqrt(eta rho)`): Phase 3 reciprocity/energy validates.

## 9. Conventions and repo notes

Coordinate system `z` (down, axis 0), `x` (axis 1), `y` (axis 2); horizontal slowness `p`; Voigt
order `(zz, xx, yy, xy, zy, zx)`; conda env `seismic`; time `e^{+i w t}`, outgoing `h_n^(1)`.
The verified `Mathematica/CartesianT0.nb` is the single-site `T0` source and the elastic Weyl
bridge that seeds the lattice sum. Mathematica via
`/Applications/Wolfram.app/Contents/MacOS/wolframscript`. Self-contained lualatex per `docs/`
subdirectory for any write-up.
