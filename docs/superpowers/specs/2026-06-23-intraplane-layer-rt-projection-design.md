# Design: Phase 3a — Intra-Plane Layer R/T(p) Projection

**Date:** 2026-06-23
**Plan:** `IntraPlaneFoldyLax_Plan.md` — Phase 3 (first sub-phase, "3a"). Phase 2 is COMPLETE.
**Status:** design approved; spec for implementation.

## 1. Goal & framing

Project the Phase-2 spherical collective `T_coll(k_par)` onto **Kennett flux-normalised up/down
P-SV-SH plane waves at horizontal slowness `p`**, producing the planar layer scattering operator
**`Rd, Ru, Td, Tu`** (2×2 P-SV) + **SH scalar**, across **normal, sub-critical, and post-critical `p`**.
Validate reciprocity tightly and cross-check against the existing Cartesian/Kennett ground truth loosely.

**Energy balance `|R|²+|T|² = 1` is explicitly deferred to Phase 3b.** The Phase-2 lattice uses an
artificial damping (`Im κ = 0.25`) for convergence; damping is a loss term that **preserves reciprocity
but breaks energy conservation**. The lossless energy check requires the *undamped* vector G0
(Ewald-accelerated Kambe layer-KKR), which is itself a substantial sub-project. Phase 3a therefore makes
reciprocity the exact gate and treats the cross-checks as agreement-to-discretisation-error.

### Decisions locked in brainstorming (2026-06-23)

| Decision | Choice |
|---|---|
| Scope | Tracer: projection + reciprocity; **defer** energy balance / undamped G0 to Phase 3b |
| Slowness range | Full `p` sweep: normal (`p=0`), sub-critical (`p < 1/α`), post-critical (`p > 1/β`, complex `η`) |
| Deliverable | MMA-led `IntraPlaneRT.wl` + `.nb` twin + Python cross-check + short LaTeX note |
| Validation | Tight (machine): reciprocity, single-sphere limit, SH⊥P-SV at p=0. Loose (few %): vs Cartesian slab + Kennett |

## 2. Architecture (four units)

### (A) Spherical Weyl R/T projection — `Mathematica/IntraPlaneRT.wl` (new physics)

The genuinely new piece. Per horizontal slowness `p` and mode `m ∈ {P, SV, SH}`:

1. **Incident → multipoles.** Build the downgoing incident plane wave from the complex slowness vector
   `s_m = (η_m, p, 0)` (`η_m = √(1/c_m² − p²)`, `Im η_m > 0` past critical, `pol·pol = 1` with NO
   conjugation — the `slab_reflection_matrix` analytic-continuation convention). Expand into regular
   multipole amplitudes `a` via CartesianT0's verified incident bridge (`incP/incN/incM`), evaluated on the
   complex incidence direction.
2. **Collective scatter.** `b = T_coll · a`, with `T_coll = T0 (I − G0 T0)⁻¹` assembled at Bloch vector
   `k_par = ω·(0, p, 0)` from the Phase-2 vector-lattice machinery (`buildG0vec`/`T0vec`/`collV` from the
   item (b)/(c) scripts, damped lattice as in Phase 2).
3. **Multipoles → specular plane waves (lattice Weyl).** The periodic array's specular up/down plane-wave
   content is CartesianT0's *scattered* Weyl angular-spectrum bridge evaluated at the specular up/down
   P-SV-SH directions at fixed `(kx, ky) = ω(0, p)`, times the lattice prefactor `i / (2 η_m ω A_cell)`
   (`A_cell = aL²`, the spherical analog of `slab_weyl_amplitudes`' `i/(2 k_z d²)`). Up = reflection,
   down = unit + transmission perturbation.
4. **Assemble** the 2×2 P-SV reflection/transmission blocks (rows/cols = {P, SV}) and the SH scalar, for
   down-incidence (`Rd, Td`) and — by an **independent up-incidence solve** (incident `s_m = (−η_m, p, 0)`)
   — `Ru, Tu`. `Ru/Tu` are computed independently, NOT derived from `Rd/Td` via reciprocity, so the
   reciprocity check in §3 is a genuine test rather than a tautology.

### (B) Flux normalisation to the Kennett basis

Convert the displacement-convention R/T to Kennett's flux-normalised `√(ηρ)` basis using the pinned
diagonal similarity `D = diag(α√η_P, i·β√η_S)` (the `SlabReflectionMatrix.to_modified` convention,
`kennett_layers` line ~358). In this basis the reflection matrix is symmetric by reciprocity.

### (C) Python cross-check — `cubic_scattering/tests/test_intraplane_rt.py`

Reads the dumped JSON (`Mathematica/IntraPlaneRT_reference.json`); independently checks the tight
reciprocity invariants and the loose agreement vs the Cartesian/Kennett ground truth (see §3).

### (D) LaTeX note — `docs/intraplane_rt/intraplane_rt.tex`

Self-contained lualatex note documenting the projection (incident bridge → collective → lattice Weyl →
flux norm) and the cross-check results table. Compiled in-place per repo LaTeX convention.

## 3. Validation (two tiers)

### Tight gates (machine precision; damping preserves these)

- **Reciprocity** of the spherical layer R/T: `Tu = Td^T` and `Rd` symmetric in the flux-normalised basis;
  and the convention-independent invariant (product of the two off-diagonal P↔SV ratios `= 1`).
- **Single-sphere limit:** with `G0 = 0`, the layer R/T reduces to the lattice-Weyl projection of a single
  Mie sphere's `T0` (a known one-scatterer plane-wave response).
- **SH ⊥ P-SV decoupling at `p = 0`:** at normal incidence the off-diagonal P-SV blocks and all SH↔P-SV
  couplings vanish.

### Loose gates (few %, bounded by the item-(e) discretisation error)

- Spherical R/T(p) vs the Cartesian `slab_reflection_matrix(geometry, material, omega, p).to_modified()`
  (single-plane `N_z=1`, uniform contrast) — same physics class (periodic Foldy-Lax), n≤2, agreeing to the
  **sphere-vs-cube shape factor** (item e: ~few %).
- Spherical R/T(p) vs `kennett_reference_matrix(ref, contrast, H=2a, omega, p)` (homogeneous-layer
  reflectivity `KennettChannelReference{R_PP, R_PS, R_SP, R_SS, R_SH}`) in the dilute Rayleigh limit —
  agreeing to the discretisation error.
- Coverage: at least normal (`p=0`), one sub-critical (`p < 1/α`), and one post-critical (`p > 1/β`)
  slowness, for the moderate test contrast (the physical, well-conditioned regime). Grazing singularities
  `p = 1/α` and `p = 1/β` (`η = 0`) are excluded (the Weyl prefactor and `to_modified` are singular there).

These are honest gates: the cross-checks are *agreement-to-discretisation-error*, not exact — that is the
expected consequence of comparing a diluted sphere packing to a cube slab / homogeneous layer. Reciprocity
is the only machine-precision gate.

## 4. Deliverables

1. `Mathematica/IntraPlaneRT.wl` (+ `.nb` twin via `makeIntraPlaneNotebooks.wl`) — units (A)+(B); dumps
   `Mathematica/IntraPlaneRT_reference.json` with, per `(p, contrast)`: the flux-normalised `Rd` (2×2),
   `Td` (2×2), `R_sh`, `T_sh`, and the reciprocity residuals. Self-verifying PASS/FAIL header asserts the
   tight gates.
2. `cubic_scattering/tests/test_intraplane_rt.py` — Python cross-check (unit C).
3. `docs/intraplane_rt/intraplane_rt.tex` — the writeup (unit D).
4. Closeout: update `IntraPlaneFoldyLax_Plan.md` (Phase 3a DONE, Phase 3b = undamped G0 + energy);
   write/append memory.

## 5. Acceptance criteria

1. `IntraPlaneRT.wl` runs headless via `wolframscript`; all in-script tight-gate asserts pass; JSON dumped.
2. `.nb` twin generated and spot-verified.
3. `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_rt.py -v` passes:
   - reciprocity (`Tu=Td^T`, `Rd` symmetric, ratio-product = 1) to machine precision;
   - single-sphere and `p=0` SH-decoupling limits;
   - loose agreement vs `slab_reflection_matrix` and `kennett_reference_matrix` within the item-(e)
     discretisation tolerance, across normal / sub-critical / post-critical `p`.
4. Ruff + mypy clean on the new Python.
5. LaTeX note compiles in-place with `lualatex` (run twice).
6. Plan + memory updated.

## 6. Out of scope (Phase 3b and later)

- **Undamped vector G0** (Ewald/Kambe layer-KKR) and the lossless **energy balance `|R|²+|T|²=1`**.
- Multiple propagating Bragg orders (super-wavelength pitch) — Phase 3a assumes the sub-wavelength regime
  where only the specular order propagates.
- Phase 4 (insert the planar R/T as a scattering layer in `kennett_layers` over the stratified background)
  and Phase 5 (lateral-heterogeneity field).

## 7. Conventions

Coordinate system `z` (down, axis 0), `x` (axis 1), `y` (axis 2); horizontal slowness `p`; conda env
`seismic`; time `e^{+iωt}`, outgoing `h_n^(1)` via `SphericalHankelH1` (never `j_n + i y_n`); complex
slowness `Im η > 0`, `pol·pol = 1` (no conjugation) past critical. Mathematica via
`/Applications/Wolfram.app/Contents/MacOS/wolframscript`. LaTeX: self-contained `lualatex`, compiled
in-place per `docs/` subdir, run twice. Background α=5000, β=3000, ρ=2500; moderate contrast Δλ=+2 GPa,
Δμ=+1 GPa, Δρ=+100. The verified `CartesianT0.wl` (now self-contained — loads the bracket builders from
`ElasticMieTmatrix.nb` cell 6) is the incident/scattered Weyl-bridge source.
