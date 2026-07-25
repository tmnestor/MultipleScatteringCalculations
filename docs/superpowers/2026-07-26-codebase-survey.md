# Codebase Survey — what already exists

- **Date:** 2026-07-26
- **Why this exists:** Phase 2 (Marine3D 2½-D disorder-resolved crust) was halted after the intra-plane operator $\mathbf{P}^x$ was rebuilt four times in succession, each repair exposing a new defect. Root cause was **not** the physics: it was that the work started without an overview of the codebase, so machinery that already exists was reimplemented badly. This document is the overview that should have preceded the work.
- **Scope:** `cubic_scattering/` (35 modules, 25 test files, ~724 tests), `FFTProp.py/`, `PhD_fortran_code/`, `Mathematica/` (~50 scripts), and what `~/Desktop/Marine3D` cherry-picked from them.

---

## 1. The finding that matters most

**Four independent representations of the intra-plane / inter-site Green's tensor already exist.** Phase 2 built a fifth, by hand, from the least suitable of the four.

| Module | Representation | Periodicity | Touching pairs | Matvec |
|---|---|---|---|---|
| `lattice_greens.py` | **Spectral** kernel at z=0, 2-D IFFT; also spatial, hybrid, FCC | **Built in** — lattice sum is implicit in the spectral form | — | **FFT block-Toeplitz `matvec()`** |
| `horizontal_greens.py` | Real-space $G(\Delta x, \Delta y, 0)$ by $k_x$-residue + $k_y$ IFFT + $k_z$ quadrature | None — pairwise | Point value only | none |
| `inter_voxel_propagator.py` | **Volume-averaged** 9×9 $[[G,C],[H,S]]$, closed form via $O_h$ master integrals | None | **Correct** (face/edge/corner) | none |
| `Mathematica/IntraPlane*.wl` | Kambe/layer-KKR multipole $D[q,s]$, Ewald-accelerated | Bloch lattice sum | — | — |

`lattice_greens.LatticeGreens` provides `compute_spatial`, `compute_spectral`, `compute_hybrid`, `compute_fcc`, `_compute_spectral_9x9`, `_precompute_circulant_fft`, `matvec`, `verify`, and D4h orbit reduction. **It was cherry-picked into Marine3D as `marine3d/tmatrix/lattice_greens.py` and never used.**

Phase 2 instead built $\mathbf{P}^x$ as a pairwise real-space sum from `horizontal_greens.py` — the one representation with neither periodicity nor correct touching-pair behaviour — and then spent four rounds trying to add both back:

1. quadrature cutoffs 10× too small → 101% error;
2. replaced by the Kupradze closed form;
3. discovered periodic images were missing (10–15%) → planned an Ewald sum;
4. discovered the touching-neighbour term was wrong by 190% → planned volume-averaging.

Items 3 and 4 are *properties the spectral form and the volume-averaged form already have*.

**Caveat, stated so this document does not repeat the original error:** `LatticeGreens` is a 2-D $M\times M$ lattice with a default damping `eta=0.03`. Phase 2 is 2½-D — a 1-D $x$-lattice, $y$-invariant — and the validation ladder needs undamped operation. So it is **not** a drop-in. It is the right foundation, not a finished answer.

---

## 2. Existing solvers

| Module | Solves | Status |
|---|---|---|
| `slab_scattering.py` | $(\mathbf{I} - \mathbf{G}\mathbf{T})\psi = \psi^0$ for an $M\times M\times N_z$ cube lattice, FFT convolution in the horizontal plane, $O(N_z^2 M^2 \log M)$ matvec | Validated channel-by-channel vs Kennett (~0.5–1% at moderate contrast); full reflection matrix R_PP/PS/SP/SS/SH; evanescent incidence; volume-averaged propagator option; GPU twin |
| `FFTProp.py/` | **2.5-D** scattering: 2-D heterogeneity in $(x,z)$ inside a 3-D plane-layer reference — *the thesis geometry* | Faithful Python conversion of the thesis Fortran `FFTPROP.F`. **Four-directional propagation sweeps (up/down/left/right)** = thesis Alg 5.2/5.3. Free-surface Rayleigh reflection with P-SV coupling; constant-Q |
| `sphere_scattering_fft.py` | FFT-accelerated GMRES Foldy-Lax on a voxelised sphere | Cross-checked against elastic Mie, 5 channels |
| `cpa_iteration.py` | Self-consistent effective medium $\langle T\rangle = 0$ | — |
| GMM (`GlobalMatrix`, in Marine3D) | Block-Riccati up/down sweeps for the stratified propagator $\mathcal{Q}^\partial$ | Parity vs `kennett_layers` 4.42e-15 |

**`FFTProp.py` is the single most important omission.** Phase 2's stated goal is "reproduce the thesis 2½-D limit". A faithful port of the thesis's own 2½-D solver — including the directional sweeps that Phase 2's plan listed as a *deferred future option* ("Task C — thesis directional x-sweep (Alg 5.2)") — already exists. It is the natural arbiter for the 2½-D validation rung, and possibly the natural basis for the solver itself.

---

## 3. The two marine paths

Both are marine; they differ in completeness, and the difference explains the seabed episode.

**`seismic_survey.py` — "Towed hydrophone marine seismic reflection survey simulation".** Full shot gather. Pipeline: `build_survey_stack() → kennett_reflectivity_batch() → free_surface_reverberations() → ghosts → bessel_summation() → source spectrum → IFFT`.

Crucially, `compute_shot_gather` calls the Kennett recursion on `stack.layers[1:]` — **water excluded, by design**. The original's own test names the object `sub_ocean` and the test class `TestHalfSpaceBelowWater` ("below water"), building it from two *solid* layers. There is no fluid–solid step anywhere in the pipeline. The water column enters only as a two-way phase plus the free-surface series $E^2R/(1+E^2R)$.

**`ocean_bottom.py` — "Ocean-bottom reflection with heterogeneous sediment".** Single-point reflection coefficient for water | heterogeneous slab | half-space. Computes the **background from the full stack** via `kennett_layers` (so the fluid–solid seabed *is* included), and uses `_kennett_water_step` — $R = R_d + T_u M T (I - R_u M T)^{-1} T_d$ with `psv_fluid_solid` — to dress the *slab perturbation* through the seabed.

So `ocean_bottom.py` is the physically complete marine model; `seismic_survey.py` is a sub-ocean-reflectivity simplification wrapped in gather machinery.

---

## 4. Assessment of the two changes already shipped

### 4.1 The seabed primary (`dfde898`)

**What I claimed:** a defect — the water–crust fluid–solid primary was missing from `RRd_PP`, so water over a half-space returned an identically zero gather.

**What the survey shows:** excluding water from `kennett_reflectivity_batch` is the original's *deliberate design*, asserted in its own tests. The datum correction I removed was a Marine3D Phase-1 addition, not part of the ported design. And `ocean_bottom.py` already implements the complete treatment.

**Assessment — partially right, wrongly framed.** The zero-gather symptom is real and reproducible, and after the change the structure matches `ocean_bottom`'s (`E²R/(1+E²R)` applied to a seabed-referenced total). But I diagnosed it as a bug in `compute_shot_gather` when the actual question is architectural: **should `marine_reference_gather` have been built on `compute_shot_gather` at all, rather than on `ocean_bottom`'s machinery?** That is a Phase-1 decision my change silently entrenched instead of surfacing.

Open question, not resolved by this survey: whether the change should stand, be reverted in favour of building the marine gather on `ocean_bottom`, or whether `compute_shot_gather` should be left byte-faithful to its origin with the marine path moved elsewhere.

### 4.2 Phase 2 $\mathbf{P}^x$ (`c3f562d`…`63cd3ba`)

**Assessment — wrong foundation.** Built from `horizontal_greens.py` pairwise real-space values, when `lattice_greens.py` (spectral, periodic, FFT matvec) and `inter_voxel_propagator.py` (volume-averaged, correct at contact) were both already in the package. The four repair rounds were reconstructing properties those modules already have.

The committed code is *correct for what it is* — the closed-form kernel is exact to 1e-14 and its tests pass — but "a correct nearest-image pairwise kernel" is not the object the solver needs.

---

## 5. What Marine3D took, and what it built anyway

Phase 0 cherry-picked `lattice_greens.py`, `inter_voxel_propagator.py`, `slab_scattering.py`, `horizontal_greens.py`, `sphere_scattering*.py` into `marine3d/tmatrix/` and `marine3d/kennett/`. The `PORT_MANIFEST.md` even flags the question directly:

> revisit cutting `slab_scattering.py` when `marine_survey_3d.py` is built (Phase 2), if marine3d's own solver supplies the reflectivity instead.

So Phase 0 *knew* an existing Foldy-Lax solver was being carried and explicitly deferred to Phase 2 the decision of whether to reuse or replace it. Phase 2 never asked the question.

`FFTProp.py` was **not** ported and is not in the manifest.

---

## 6. Recommendations

1. **Do not resume Phase 2 on the current $\mathbf{P}^x$.** Tasks 2b and 2c should be dropped, not implemented; both are repairs to the wrong foundation.
2. **Answer the question Phase 0 deferred**, with evidence: does `slab_scattering.py` (+ `lattice_greens` spectral matvec) supply the 2½-D crust response directly? If so, most of the Phase 2 plan dissolves.
3. **Evaluate `FFTProp.py` as the 2½-D arbiter** before building a Wolfram reference from scratch. A faithful port of the thesis's own solver is a stronger and cheaper arbiter than a re-derivation, and it is what "reproduce the thesis 2½-D limit" literally means.
4. **Resolve the marine-path architecture** — `seismic_survey` vs `ocean_bottom` as the base for `marine_reference_gather` — before any further gather work. The seabed change stands or falls on that decision.
5. **Rewrite the Phase 2 spec and plan** against this map. The existing ones assume everything is built from scratch.

---

## 7. Process note

Project memory already carried this lesson before the work started:

> **read-source-before-proposing** — read existing Mathematica+Python source AND thesis-chapter maths BEFORE sketching any verification/build; the sweep was already coded and I proposed designs 3× before reading it.

The same failure recurred, at larger scale: the sweep was already coded (`FFTProp.py`), the lattice Green's tensor was already coded (`lattice_greens.py`), the volume-averaged contact propagator was already coded (`inter_voxel_propagator.py`), and a spec, an 8-task plan, and four rounds of repair were produced without reading any of them.
