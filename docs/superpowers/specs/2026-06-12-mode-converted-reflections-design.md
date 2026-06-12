# Mode-Converted Reflections: R_PS, R_SP, R_SS, R_SH for the Slab Solver and Ocean-Bottom Study

Date: 2026-06-12
Status: approved
Scope decisions: full reflection matrix + ocean-bottom 2×2 upgrade; CPU path only
(GPU parity is a separate later task).

## Goal

Extend the periodic-slab Foldy-Lax solver from PP-only specular reflection to
the full set of non-zero specular channels — the 2×2 P-SV reflection matrix
(R_PP, R_PS, R_SP, R_SS) plus the scalar SH channel (R_SH) — at arbitrary
horizontal slowness p, validated against Kennett reflectivity. Upgrade the
ocean-bottom study's sub-ocean recursion from scalar (PP) to the 2×2 P-SV
matrix so that P→S→P conversion inside the heterogeneous sediment feeds the
observable water-column R_PP.

## Current state

- `slab_scattering.py:slab_rpp_periodic` extracts only R_PP via the Weyl
  lattice sum: `R_PP = -(i / (2 k_z d² ρ α²)) Σ_l Q_P,l exp(i k_z z_l)` with
  `Q_P = -r̂·f - i k_z (r̂·σ·r̂)` over horizontally layer-averaged sources.
  Oblique incidence (p ≠ 0) is supported.
- `compute_slab_scattering` accepts `wave_type="P"` or `"S"` (SV polarization
  in the sagittal plane). No SH incidence.
- `kennett_layers.py` already exposes the full reference: `KennettResult.RPP/
  .RPS/.RSP/.RSS/.RSH` at arbitrary p, including the fluid-solid interface
  (`psv_fluid_solid`, 2×2 Rd/Ru/Td/Tu).
- `ocean_bottom.py` injects the scalar slab R_PP into a scalar water-step
  recursion; internal mode conversion is missing from the observable.

## Physical-realism requirements (binding)

1. **Conversion feeds the observable.** The sub-ocean composite reflectivity
   is a 2×2 matrix; suppressing its off-diagonal entries is not acceptable.
2. **Mode-specific vertical phases.** Layer phase delays use
   `diag(exp(iωη_P H), exp(iωη_S H))` — never a shared P phase.
3. **Post-critical slownesses.** η = sqrt(1/c² − p²) continues to complex
   values with Im η ≥ 0 (decaying evanescent), consistently in the slab Weyl
   sums and the Kennett reference. S-critical slownesses are exercised by
   tests for the first time.
4. **True polarization geometry.** Outgoing SV/SH amplitudes are projections
   of the layer-averaged force + stress-dipole sources onto the actual
   reflected polarization vectors (sagittal-plane SV, ŷ SH), with the
   validated T-matrix force-sign convention (`-r̂·f` term).
5. **Honest averaging.** The Weyl extraction is the specular, horizontally
   averaged (coherent) response — exactly what Kennett describes. Diffuse
   (non-specular) energy and per-realization SV↔SH coupling from random
   heterogeneity are real but are not specular observables; they are out of
   scope and documented as such, not silently discarded.

## Design

### slab_scattering.py

- `WeylAmplitudes` dataclass: `R_P`, `R_SV`, `R_SH` (complex), plus the
  slownesses used (`p`, `eta_P`, `eta_S`) for diagnostics.
- `slab_weyl_amplitudes(result, T_local, *, p) -> WeylAmplitudes` — the
  shared extractor. One pass over layer-averaged sources computes:
  - P: existing projection on the reflected P slowness direction with
    prefactor `-i/(2 ω η_P d² ρ α²)` and phase `exp(i ω η_P z_l)`;
  - SV: projection of `Q_S = -f - i k_S σ·r̂_S` onto the reflected SV
    polarization vector, prefactor `-i/(2 ω η_S d² ρ β²)`, phase
    `exp(i ω η_S z_l)`;
  - SH: same S-wave machinery projected onto ŷ.
  `slab_rpp_periodic` becomes a thin wrapper returning `.R_P` (API
  preserved; existing tests unchanged).
- `_build_slab_incident_field` gains `wave_type="SH"` (polarization ŷ).
  Unknown types keep failing fast with ValueError.
- `SlabReflectionMatrix` dataclass: `R_psv` (2×2: rows = outgoing P/SV,
  columns = incident P/SV), `R_sh` (complex), `p`, `omega`.
- `slab_reflection_matrix(geometry, material, omega, *, p, ...)` — runs the
  P-, SV-, SH-incident periodic solves at slowness p (incident direction
  `k̂ = (η c, p c, 0)` per mode) and assembles the matrix from three
  `slab_weyl_amplitudes` calls. Solver options mirror
  `compute_slab_scattering`.
- `kennett_reference_matrix(ref, contrast, H, omega, *, p)` — same 3-layer
  stack as `kennett_reference_rpp`, returns all five channels from
  `KennettResult`. `kennett_reference_rpp` delegates to it.

### ocean_bottom.py

- Per active frequency: two slab solves (P- and SV-incident; SH cannot couple
  through the fluid and is skipped).
- `R_slab` per frequency becomes the 2×2 P-SV block.
- Sub-ocean composite: `MT = E R_bg E + R_slab` with `E = diag(exp(iωη_P^sed
  H_sed-path), exp(iωη_S^sed H_sed-path))` matching the existing scalar
  phase-accounting structure, generalized per mode.
- `_kennett_water_step`: 2×2 recursion `R = Rd + Tu · MT · (I − Ru·MT)⁻¹ · Td`
  using the existing `psv_fluid_solid` coefficients. Observable remains
  `R[0, 0]` (P in water). The scalar path is removed (no legacy mode).
- No YAML schema changes. The output log gains slab-level channel
  diagnostics (R_PS, R_SP, R_SS magnitudes per frequency).

## Validation and success criteria

1. **Uniform slab vs Kennett, all 5 channels** at p = 0 and oblique p
   (one sub-critical, one beyond the S-critical slowness of the contrast):
   relative error at the established R_PP scale (~1% at moderate contrast
   with the convergence-study mesh). At p = 0 the conversion channels vanish
   (both slab and Kennett must agree they are ~0).
2. **Born scaling** at weak contrast (1e-4): each channel doubles when the
   contrast doubles (rtol 0.05, existing pattern).
3. **Reciprocity**: R_PS and R_SP related through the η_P/η_S normalization
   factors; asserted with the empirically pinned constant (cf. the sphere
   five-channel verification, where the analogous ratio was exactly −1).
4. **Ocean-bottom degenerate limit**: zero-contrast slab reproduces the full
   Kennett water|sediment|halfspace matrix result to rtol 1e-10 (existing
   test upgraded to the matrix path).
5. **Physics-gain regression**: weak contrast → 2×2 R_PP matches the old
   scalar result (conversion is second order); moderate contrast at oblique
   p → documented difference (test asserts the difference is nonzero and
   bounded, capturing the restored physics).

## Risks

- **Cross-convention normalization** (highest risk): Kennett's R_PS/R_SP
  normalization (displacement-amplitude convention in `psv_solid_solid`)
  must be matched by the slab's Weyl amplitude normalization. Mitigation:
  weak-contrast Born comparison pins the constant analytically before any
  strong-contrast comparison; the reciprocity test is an independent check.
  Precedent: the sphere five-channel work resolved the analogous m=1
  renormalization (−1/(n(n+1))) the same way.
  **RESOLVED (Task 4):** measured conversion is the diagonal similarity
  `D = diag(α·√η_P, i·β·√η_S)` (Kennett eigenvector normalization with
  velocity factors and i on SV); pinned by the reciprocity invariant
  (off-diagonal ratio product = 1). A pre-existing p=0 SV polarisation
  discontinuity in the slab incident-field builder was found and fixed in
  the same step (K_SS(0) = −K_SH(0) made it observable).
- **SV sign conventions at reflection** (Aki & Richards and Kennett
  differ): decided by the p→0 limit and the Kennett comparison, not chosen
  a priori.
- **Force-sign convention**: the slab Weyl extractor must keep the validated
  `Q_P = -r̂·f - ik σ_rr` form; the S-wave analogue is fixed by the same
  Kennett comparison (memory: "the Weyl identity does NOT carry the global
  minus from slab_reflected_field").
- **Cost**: ocean-bottom runs 2× solves per frequency. Acceptable (GMRES
  converges in ~6 iterations; the kernel build is shared per frequency where
  the implementation allows).

## Out of scope

- GPU parity (`slab_scattering_gpu.py`) — follow-up task.
- SH in the ocean-bottom embedding (physically zero through a fluid).
- Diffuse/non-specular scattered energy and ensemble statistics.
- Resonance-regime T₀ in the slab (separate roadmap item).
