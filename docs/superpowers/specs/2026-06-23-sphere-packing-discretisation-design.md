# Design: Sphere-Packing Discretisation Error (IntraPlaneFoldyLax Phase 2, item (e))

**Date:** 2026-06-23
**Plan:** `IntraPlaneFoldyLax_Plan.md` — Phase 2, item (e) (the last open Phase-2 item).
**Status:** design approved; spec for implementation.

## 1. Goal & framing

Quantify the **irreducible geometric discretisation error** of representing a
lateral heterogeneity as a diluted planar **sphere packing** (planar packing
fraction `φ ≤ π/6 ≈ 0.524` even at touching) versus the **space-filling cube
slab** (`φ = 1`, the `slab_scattering` ground truth), in the **Rayleigh limit**.

Spheres cannot fill space, so this error exists *independently of multipole
convergence* — item (c) already showed the sphere collective converges in `n`;
item (e) measures what is lost to geometry, not truncation. The error has two
physically distinct sources that the design separates:

1. **Shape factor** — sphere-vs-cube single-site effective contrast (no lattice).
2. **Packing dilution** — the `φ < 1` under-filling in the collective layer.

The **contrast renormalisation `Δ → Δ/φ`** is the proposed correction: it keeps
the sphere radius at `a = L/2` (touching, **non-overlapping**, so the
translation-addition theorem / Foldy-Lax stays valid) while scaling each voxel's
contrast so the integrated contrast `∫Δ dV` per unit cell matches the cube.
Stage B tests whether this correction collapses source (2).

### Decisions locked in brainstorming (2026-06-23)

| Decision | Choice |
|---|---|
| Error metric | Layered: single-site shape factor → collective layer `R_PP` |
| Renorm correction | Contrast renorm `Δ → Δ/φ` (keeps spheres non-overlapping) |
| Deliverable | Mathematica-led `.wl` + `.nb` twin + Python cross-check |
| Packing sweep | `aL = 6 → 2.0` (`φ`: dilute → `π/6` at touching), reusing item (c) |
| Contrast sweep | weak (`1e-4 ×` background), moderate (Δλ=+2 GPa, Δμ=+1 GPa, Δρ=+100), negative/strong stress case (−60% of background moduli & density; validate positive first per memory) |
| Frequency | Rayleigh limit only: `ka ∈ {0.05, 0.1}` |

## 2. Two-stage architecture

The two stages are independent units with a clean interface: Stage A produces
per-site effective contrasts (no lattice); Stage B produces a collective layer
reflection coefficient (lattice). Each is independently testable.

### Stage A — single-site shape factor (closed-form, Mathematica-led)

Compare **one cube's** Rayleigh effective contrast against **one
volume-renormalised sphere's**, isolating pure shape error with no lattice
involved.

- **Cube reference:** existing Eshelby / 27-Galerkin closed forms
  (`effective_contrasts.py` and the symbolic cube Eshelby in `Mathematica/`).
  The cube tiles at `φ = 1`, so no renormalisation applies to it.
- **Sphere:** closed-form Mie / Eshelby-sphere effective contrast
  (`Δκ*, Δμ*, Δρ*`) per the validated Mie extraction
  (`Δκ* = -a₀/(C_P k²)`, `Δρ* = i·a₁/(C_P ω²)`, `Δμ* = 3·a₂/(4 C_P k²)`),
  at the **same material contrast `Δ`** (no renormalisation).
- **Metric:** relative error in each of `(Δκ*, Δμ*, Δρ*)`, per contrast regime.

  > **Correction (during implementation, 2026-06-23):** `Δ→Δ/φ` is a
  > **layer-level** correction, not a single-site one — the effective-contrast
  > extraction already normalises by volume (`C_P = V/4πρα²`), so per unit
  > volume a sphere and cube of the *same* `Δ` scatter almost identically (that
  > small residual *is* the shape factor). Applying `Δ→Δ/φ` at a single site
  > just makes it `1/φ ≈ 1.9×` too strong. The renormalisation therefore lives
  > in Stage B (`layer ~ φ·single_site(Δ/φ)`), where the `φ` cancels to recover
  > the cube. Stage A reports the raw shape factor only.

### Stage B — collective layer R_PP (Mathematica sphere-packing vs Python cube)

- **Sphere side:** planar Foldy-Lax **monopole collective** built with item
  (c)'s `IntraPlaneConvergence` machinery (closed-form P-channel collective
  monopole, stable `SphericalHankelH1`), projected to the **specular `R_PP`** at
  normal incidence, with `Δ → Δ/φ`.
- **Cube ground truth:** `slab_rpp_periodic` / `kennett_reference_rpp` (Python,
  `cubic_scattering/slab_scattering.py`), normal incidence, single perturbed
  layer over background.
- **Metric:** relative error in `R_PP` vs `φ` and contrast; flag the
  **near-touching conditioning boundary** (item (c): `cond(I − G0 T0) ~ 7` at
  `aL = 2.2`, `n_max = 5`) where the sphere model degrades and the reported error
  is no longer trustworthy.

**Interface note (why the asymmetry):** Stage A is fully symbolic-friendly on
both sides (closed-form cube + closed-form sphere). Stage B's cube ground truth
is a genuinely *numerical* Python lattice sum (`slab_rpp_periodic`) with no
closed form, so the Mathematica script computes only the **sphere-packing**
`R_PP`, dumps it to JSON, and the Python test compares it against the Python cube
ground truth. The new physics stays Mathematica-led; the cross-check stays in
Python where the cube object lives.

## 3. Deliverables

### 3.1 `Mathematica/IntraPlaneDiscretisation.wl` (+ `.nb` twin)

- Stage A: closed-form sphere (Mie/Eshelby) and cube (Eshelby/Galerkin)
  effective contrasts; raw and `Δ→Δ/φ`-renormalised; relative error table over
  the contrast sweep.
- Stage B: sphere-packing monopole collective `R_PP` over the `aL` (i.e. `φ`) ×
  contrast × `ka` sweep, with per-point spectral radius / conditioning so the
  validity boundary is recorded alongside each error.
- Dumps `IntraPlaneDiscretisation_reference.json` (mirrors the item-(c) /
  item-(b) JSON-dump pattern) with the full sweep: `{aL, φ, contrast, ka, stageA
  errors, stageB sphere R_PP, cond, specrad}`.
- `.nb` twin generated via the existing `makeIntraPlaneNotebooks.wl`.
- Self-verifying header asserts (in-script, like the other Phase-2 `.wl`s):
  isolated/dilute limit reproduces the single-site contrast; `Δ→Δ/φ` reduces
  Stage A dilution error.

### 3.2 `cubic_scattering/tests/test_intraplane_discretisation.py`

Independent Python cross-check (scipy/sympy + existing modules), matching the
`test_intraplane_convergence.py` style:

- **Stage A cross-check:** independently recompute cube effective contrast
  (`effective_contrasts.py`) and sphere effective contrast
  (`sphere_scattering.py` Mie extraction); assert agreement with the dumped
  Mathematica Stage-A values to tight tolerance.
- **Stage B comparison:** load dumped sphere `R_PP`; compute the cube ground
  truth via `slab_rpp_periodic` / `kennett_reference_rpp`; assert the
  discretisation error vs `φ` and contrast.
- **Correction monotonicity:** assert `Δ→Δ/φ` reduces the dilution component of
  the error monotonically as `φ → π/6` (vs the raw, uncorrected packing).
- **Born limit:** at weak contrast (`1e-4`), assert the renormalised sphere
  layer agrees with the cube to a tight Rayleigh/Born tolerance (the irreducible
  error → small as contrast → 0; shape error is the residual).
- **Validity boundary:** assert the near-touching `cond` trend matches item
  (c)'s recorded values, marking where Stage-B error becomes untrustworthy.

### 3.3 Closeout

- Write the two forward-linked memory files referenced by the plan:
  `project_sphere_packing_discretisation.md` (the research result) and update
  `MEMORY.md`.
- Update `IntraPlaneFoldyLax_Plan.md`: mark item (e) **DONE**, update the Phase-2
  status line and the Section-6 / Risk-§8 notes.

## 4. Acceptance criteria

1. Mathematica script runs headless via `wolframscript`, all in-script asserts
   pass, JSON dumped.
2. `.nb` twin generated and round-trip spot-verified.
3. Python test passes under `conda run -n seismic pytest
   cubic_scattering/tests/test_intraplane_discretisation.py -v`, with:
   - Stage A Mathematica↔Python agreement (tight tolerance).
   - `Δ→Δ/φ` monotonically reduces dilution error.
   - weak-contrast Born limit agreement.
   - near-touching conditioning boundary reproduced.
4. Ruff + mypy clean on the new Python.
5. Plan and memory updated; item (e) closed → Phase 2 complete.

## 5. Out of scope (deferred to Phase 3+)

- Oblique-incidence / finite-`p` layer `R/T` and mode conversions (Phase 3).
- Finite-`ka` (≥ 0.3) where single-site effective contrasts blur (Phase 3).
- The full up/down P-SV-SH flux-normalised projection and energy balance
  `|R|²+|T|² = 1` (Phase 3; needs the undamped `G0`).
- Equal-volume-sphere (overlapping) correction variant — rejected in
  brainstorming because it breaks the non-overlap requirement of the addition
  theorem.

## 6. Conventions

Coordinate system `z` (down, axis 0), `x` (axis 1), `y` (axis 2); conda env
`seismic`; time `e^{−iωt}`, outgoing `h_n^(1)` via `SphericalHankelH1` (never
`j_n + i y_n`); Mathematica via
`/Applications/Wolfram.app/Contents/MacOS/wolframscript`. Test params per
`CLAUDE.md`: background α=5000, β=3000, ρ=2500; moderate contrast Δλ=+2 GPa,
Δμ=+1 GPa, Δρ=+100; weak `1e-4 ×` background.
