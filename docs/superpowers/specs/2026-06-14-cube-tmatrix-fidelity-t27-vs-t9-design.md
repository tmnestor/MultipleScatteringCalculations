# Single-Site Cube T-Matrix Fidelity: T₂₇ vs T₉ Against the Exact Elastic Sphere

**Design spec — PhD chapter, Research Questions 2 & 3**
**Date:** 2026-06-14
**Status:** approved (design) — implementation plan written

**Amendment 2026-06-14 (apples-to-apples radiation):** T9 and T27 are radiated
through the **same** finite-size full-field operator (bare physical ΔC, reconstructed
total interior field, identical finite-size phase), differing ONLY in which scattered
modes `c_sc` populates (9 vs 27). This makes the T9↔T27 far-field gap purely the
richer scattered basis — the clean RQ2 attribution. The existing `cube_far_field`
(effective Δc*×incident-strain point dipole) is retained only as a k→0 cross-check, NOT
as the study's T9. See plan §Task 6/10.

---

## 1. Purpose

Characterize **when** and **why** the 27-component Galerkin single-site cube T-matrix
(**T₂₇**) reproduces the exact elastic Mie sphere far-field more accurately than the
9-component effective-contrast T-matrix (**T₉**), and quantify the **cost-accuracy
tradeoff** between the two representations across material contrast, frequency (ka),
and wave polarization.

This addresses two of the three research questions for the thesis chapter:

- **RQ2 (primary):** Under what circumstances does the T₂₇ single-site representation
  match the analytic finite-radius, finite-contrast homogeneous-sphere scattering
  solution more accurately than the T₉ representation?
- **RQ3 (companion):** What is the cost-accuracy tradeoff between the T₂₇ and T₉
  representations across contrast and wave polarization?

**RQ1** (near-field interaction strategy: depth-fixed-plane near-field + Kennett
recursions vs. full 3-D multiple-scattering near-field) is **deferred to a separate
later study/chapter** and is out of scope here.

## 2. Framing & Physical Setup

A **single cubic scatterer** (half-width `a`, so side `2a`, volume `(2a)³`) is the
object under study. **T₉ and T₂₇ are two single-site representations of that same
cube.** The reference yardstick is the **exact elastic Mie solution** of an
equal-volume sphere — the one finite scatterer with a closed-form elastic T-matrix.

The two representations differ in their radiated multipole content:

- **T₉** (`compute_cube_tmatrix`, the Rayleigh/Galerkin/Eshelby effective-contrast
  closure, with the O((ka)²)+O((ka)⁴) form factor already merged) radiates a
  **force-monopole + stress-dipole** pattern — the standard Rayleigh point-scatterer
  far-field.
- **T₂₇** (`compute_cube_tmatrix_galerkin`, the 27-trial-function Galerkin closure: 3
  displacement + 6 strain + 18 quadratic modes, O_h irrep-decomposed) carries the
  **higher multipoles** of the strain-gradient and quadratic modes, plus the
  **finite-size phase** of the interior field profile.

> **Key relationship to prior work.** The committed `t27-lattice-verdict` measured the
> *inter-voxel* quadratic-mode coupling to be negligible (≤0.07%) and concluded the
> 27-component *lattice* solver is not justified. That is a **different question** from
> the present one. Here we study the **single-site** quadratic *radiation* into the far
> field, which the lattice-coupling measurement leaves entirely open. The form-factor
> work (`single-site-form-factor-limit`, O((ka)²)+O((ka)⁴) merged) is the immediate
> predecessor: it corrected the T₉ *effective contrasts* (monopole/dipole projection)
> against the sphere; this chapter extends the comparison to the **full angular
> far-field including higher multipoles**, where T₂₇ is expected to matter.

### Why the far-field operator is the load-bearing build

The existing `cube_far_field` (`scattered_field.py:75`) accepts the 27-component
overlap vectors but radiates **only** the leading force-monopole (from the 3
displacement components) and the stress-dipole (from the effective stiffness ×
incident strain). It does **not** radiate the higher multipoles carried by the strain
and 18 quadratic modes. As the code stands, T₉ and T₂₇ would therefore produce nearly
identical far-fields and the comparison would be null.

**The essential build is a finite-size, higher-multipole far-field radiation operator**
— the full radiation integral over the cube of the interior field profile represented
by the T-matrix's trial-function amplitudes, including all multipole orders up to the
basis and the finite-size phase factor.

## 3. The Comparison (precise definition)

For each `(contrast, ka, incident polarization)`:

1. Choose cube half-width `a` and frequency `ω` so that `ka` hits the target (use
   `ka_β = ω a / β` as the primary frequency axis; record `ka_α` too).
2. **Equal-volume sphere match (primary):** `(2a)³ = (4/3)πR³ ⟹ R = a·(6/π)^{1/3} ≈
   1.2407·a`. Compute the exact Mie far-field for this sphere with the same contrast
   (same Δλ, Δμ, Δρ relative to the same background).
3. Build the **T₉** single-site (`compute_cube_tmatrix`) and the **T₂₇** single-site
   (`compute_cube_tmatrix_galerkin`) for the cube.
4. Radiate each via `cube_multipole_far_field` (new) to get per-channel complex
   amplitudes `f_X(θ)` for `X ∈ {P→P, P→SV, SV→P, SV→SV, SH→SH}` over `θ ∈ [0, π]`.
5. Compute the exact Mie per-channel far-field `f_X^Mie(θ)` (`mie_far_field`).
6. **Accuracy metric per channel:**
   - L² relative error: `E²_X = ∫|f_X^rep − f_X^Mie|² dθ / ∫|f_X^Mie|² dθ`
   - L∞ relative error: `E∞_X = max_θ|f_X^rep − f_X^Mie| / max_θ|f_X^Mie|`
   Report both Re and Im are captured (amplitudes are complex; the metric is on the
   complex difference, so phase is included).
7. **Secondary sensitivity:** repeat the error computation with the **equal-radius**
   match `R = a` (different volume) to separate shape effects from volume matching.

**RQ2 answer object:** the map `{channel, ka, contrast, polarization} → (E_T9, E_T27)`
and the "win region" where `E_T27 < E_T9` by a meaningful margin.

**RQ3 answer object:** for each representation, a **cost vector** —
- T-matrix dimension (9 vs 27) and the dominant assembly cost
  (`compute_cube_tmatrix` vs `compute_cube_tmatrix_galerkin` wall-time / FLOP proxy),
- radiation-operator cost (number of multipole terms evaluated),
paired with the accuracy gain `E_T9 − E_T27`, yielding a **cost-accuracy Pareto**
across the sweep.

## 4. Architecture & New Components

### 4.1 `cube_multipole_far_field` (extend `cubic_scattering/scattered_field.py`)

Analytic finite-size radiation operator. Signature (final names settled in the plan):

```
cube_multipole_far_field(
    t_result,          # T9 (CubeTMatrixResult) or T27 (GalerkinTMatrixResult)
    representation,    # "T9" | "T27"
    theta, ref, contrast, omega, a,
    k_vec=None, pol=None,
) -> (f_P, f_SV, f_SH)  # complex amplitudes vs theta
```

- The scattered far-field is `f(r̂) ∝ ∫_cube [ω²Δρ·u(r′) + ∇′·(Δc:ε(r′))]·e^{−ik_sc·r′}
  d³r′`, where the interior displacement `u(r′)` and strain `ε(r′)` are reconstructed
  from the T-matrix trial-function amplitudes.
- Each trial function `φ_n(r′)` is a polynomial (constant / linear / quadratic in the
  components of `r′`) over the cube, so its radiation integral
  `∫_cube φ_n(r′) e^{−ik_sc·r′} d³r′` is a **closed-form polynomial×plane-wave cube
  integral** (products of `sinc(k_j a)` and its first/second derivatives in `k_j`).
- **T₉ path** = restrict to the 3 displacement + 6 strain modes (monopole + dipole),
  and **must reproduce the current `cube_far_field` output bit-for-bit** (regression
  pin).
- **T₂₇ path** = all 27 modes (adds quadrupole/octupole-order content + finite-size
  phase).
- Coordinate system per CLAUDE.md: z = axis 0 (down), x = axis 1, y = axis 2.

### 4.2 Mathematica derivation (`Mathematica/CubeMultipoleRadiation.wl`)

Symbolic derivation of the 27 closed-form radiation integrals → auto-generated Python
fragment (loaded by `scattered_field.py`) **and** a LaTeX fragment (for the eventual
PDF), following the existing master-integral auto-generation pattern. Use the
sequential-integration + `Chop[Re[...]]` + `ToString` wolframclient discipline from the
project memory where any singular/limit handling arises (the plane-wave kernel is
entire, so most integrals are elementary, but keep the discipline).

### 4.3 Validation arbiters

- **Direct Gauss-quadrature radiation** (`scripts/` helper): reconstruct the interior
  field on a Gauss grid from the trial-function amplitudes and integrate
  `∫ field·e^{−ik·r} d³r` numerically. Must match the analytic operator to ≤1e-10 in
  the smooth (entire-kernel) regime.
- **`resonance_far_field` cross-check:** independent finite-size far-field via sub-cell
  phases; the analytic T₂₇ operator and `resonance_far_field` must agree in the
  Rayleigh limit and both approach Mie.
- **T₉ bit-for-bit:** the 9-mode restriction reproduces `cube_far_field` exactly.
- **Reciprocity / symmetry:** apply the five-channel reciprocity relation
  (ratio = −1, per `five-channel-verification`) as a consistency check on the cube
  far-field where the channel pair allows.

### 4.4 Study driver (`scripts/cube_tmatrix_fidelity_study.py`)

CLI that sweeps the study matrix (§5), computes per-channel L²/L∞ errors for T₉ and
T₂₇ vs Mie (equal-volume primary, equal-radius secondary), measures the cost vectors,
and writes machine-readable results (CSV/JSON) plus a compact summary. **No plotting
here** — figures live in the downstream PDF deliverable. Results are the data product
this chapter produces.

### 4.5 Tests (`cubic_scattering/tests/test_cube_multipole_far_field.py`)

- Analytic == Gauss quadrature (≤1e-10) for representative modes and `k_sc`.
- T₉ restriction == `cube_far_field` (bit-for-bit).
- T₂₇ → Mie convergence in the Rayleigh limit (error → form-factor floor as ka → 0).
- `resonance_far_field` cross-check agreement in the Rayleigh limit.
- Reciprocity/symmetry consistency.
- Static/low-ka regression pins on a couple of (contrast, ka) far-field values.

## 5. Study Matrix (defaults — adjustable at plan time)

- **Background:** α = 5000 m/s, β = 3000 m/s, ρ = 2500 kg/m³ (canonical).
- **Contrast:**
  - weak: 1e-2 × background moduli/density,
  - moderate: Δλ = +2 GPa, Δμ = +1 GPa, Δρ = +100 kg/m³ (~10%),
  - strong-positive: ~+30–60%,
  - strong-negative: one negative case (~−30 to −60%, validated positive first per
    project guidance — negatives are harder).
- **Frequency:** `ka_β ∈ {0.05, 0.1, 0.3, 0.5, 1.0, 1.5}` (Rayleigh → transition →
  resonance).
- **Polarization / channels:** incidence P, SV, SH; measure all five channels
  P→P, P→SV, SV→P, SV→SV, SH→SH.
- **Angle:** full `θ ∈ [0, π]` sweep (resolution set in the plan; dense enough for the
  L² integral).
- **Match conventions:** equal-volume (primary), equal-radius (secondary).

## 6. Expected Outcomes (honest-characterization stance)

RQ2 is a characterization open to either result. The plausible prior — given the
negligible inter-voxel coupling and the form-factor results — is that **T₂₇ helps
mainly at ka ≳ 0.3–0.5, at stronger contrast, and in the mode-converted / back-scatter
channels**, with little gain at low ka where both collapse to the same monopole/dipole
form factor. The study is built to report that map **whichever way it falls**,
including the null result "T₂₇ rarely beats T₉ in the far-field for ka ≲ X."

## 7. Outputs (feed the downstream PDF)

- Per-channel error tables `E(ka, contrast)` for T₉ and T₂₇ (CSV/JSON).
- A "win-map" dataset over `(contrast, ka)` per channel.
- The cost-accuracy Pareto dataset.

These are the **inputs to the theory PDF**, which is a **separate downstream
deliverable** (its own spec + plan) covering background theory (analytic sphere, T₂₇,
T₉), academic references, tables, and TikZ figures.

## 8. Out of Scope (this chapter)

- RQ1 (near-field interaction strategy / depth-plane Kennett recursion vs full 3-D).
- The full inter-voxel T₂₇ **lattice** Foldy-Lax solver (verdict: not justified).
- The resonance subdivision as a *primary* representation (used only as a far-field
  cross-check).
- The PDF document itself (separate deliverable).

## 9. Decomposition Note

This spec is one cohesive sub-project: **the finite-size multipole far-field radiation
operator + the T₉/T₂₇-vs-Mie fidelity study**. It produces working, testable software
and a results dataset on its own. The PDF write-up is a second sub-project that
consumes this one's outputs.

## 10. Constraints (project guardrails)

- Conda env `seismic`; `conda run -n seismic` for all tooling.
- Coordinate system z=0(down)/x=1/y=2; Voigt and Fourier conventions follow it.
- Lint/format/type per CLAUDE.md (`ruff … --ignore ARG001,ARG002,F841,E741`, `ruff
  format`, `mypy --ignore-missing-imports`); line length ≤108.
- Tests committed under `cubic_scattering/tests/`.
- No silent fallbacks; fail-fast with diagnostics where config/inputs are validated.
- NO commit attribution / NO Co-Authored-By, ever.
- LaTeX (downstream): `lualatex`, compiled in-place in `docs/`.
