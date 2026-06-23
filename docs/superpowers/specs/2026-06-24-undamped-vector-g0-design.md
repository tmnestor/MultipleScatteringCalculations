# Undamped Vector `G0^vec` (Phase 3b cycle 2) — Design Spec

**Status:** approved 2026-06-24; ready for an implementation plan (`writing-plans`).

## 1. Goal & Boundary

Build the **undamped** (Im κ = 0) planar lattice vector coupling
`G0^vec_{(νμc'),(nmc)}(k_par)` in the elastic L/M/N multipole basis, by **contracting the
cycle-1 undamped scalar structure constants `D[q,s]`**, and feed it to the Phase-2 collective
Foldy–Lax solve `T_coll = T0 (I − G0^vec T0)^{-1}`.

This lifts the Phase-2 (b) **damped** vector G0 (`IntraPlaneVectorLattice.wl`, Im κ = 0.25) to a
genuinely undamped operator — the prerequisite for the cycle-3 lossless energy balance. The
damping in Phase-2 (b) is artificial loss that makes the conditionally-convergent planar sum
converge while preserving reciprocity, but it breaks energy conservation.

**In scope (cycle 2):**
- the undamped `G0^vec` object (L, M, N channels);
- the collective solve `T_coll` on it;
- correctness gates: extraction reconstruction, damped-limit agreement vs the existing
  Phase-2 (b) direct sum, η-independence, L-block reciprocity, collective sanity;
- a Python cross-check, a `.nb` twin, and plan/memory closeout.

**Out of scope (→ cycle 3):**
- the lossless energy balance `|R|² + |T|² = 1`;
- the full M/N **symplectic-flux** reciprocity (needs the thesis §3.1 flux metric, cf. Phase-2
  item (d) and Phase 3a [[thesis-energy-normalisation]]).

## 2. Core Decomposition

The lattice vector coupling is the Bloch sum of the single-pair vector translation,
`G0^vec_{...}(k_par) = Σ_{R≠0} W^{c'c}_{νμ,nm}(R) e^{i k_par·R}`. By the vector addition theorem,
the single-pair element expands as

```
W^{c'c}_{νμ,nm}(d) = Σ_q coeff_q^{c'c}(n,m,ν,μ) · h_q(κ_S |d|) · Y_q(m−μ, d̂),
```

with `coeff_q` **independent of `d`** (pure Wigner/Gaunt content; `q` runs over
`|n−ν| … n+ν`). Bloch-summing over the lattice collapses
`Σ_{R≠0} h_q(κ_S|R|) Y_q(m−μ, R̂) e^{i k_par·R}` into exactly the cycle-1 scalar structure
constant `D[q, m−μ](κ_S)`. Therefore:

- **L→L:** `g0LL = scalar-Gaunt · D(κ_P = 0.9)` — already exactly the Phase-2 (b) `g0LL`
  (`betaSep`-type Gaunt contraction), just fed the undamped `D`.
- **M/N→M/N:** `g0^{c'c}_{νμ,nm} = Σ_q coeff_q^{c'c}(n,m,ν,μ) · D[q, m−μ](κ_S = 1.5)`.

The `D[q,s]` are produced by the **cycle-1 projection machinery, parameterized by κ** (the
general-`z` scalar Ewald field + multipole projection), run at both `κ_P` and `κ_S`. The small
cycle-1 pieces are **copied** into the new file (not `Get`-ed), per the cycle-1 self-contained
pattern (`IntraPlaneKambe.wl` does not `Get` `IntraPlaneLatticeSum.wl`).

### 2.1 Coefficient extraction (the cycle-2 new content)

`coeff_q^{c'c}` are extracted **once** from the repo's already-validated single-pair vector
translation `W^{c'c}(d)` (projection method, `IntraPlaneVectorTranslation.wl` /
`IntraPlaneVectorLattice.wl` `Mw/Nw` + C/P projection) — **no literature transcription**
(decided 2026-06-24). Because `W^{c'c}(d)` is a scalar matrix element whose angular dependence
is `Σ_q coeff_q h_q(κ_S|d|) Y_q(m−μ, d̂)`, projecting `W` over source directions `d̂` (sphere
quadrature) isolates each `q` term by spherical-harmonic orthogonality; dividing by
`h_q(κ_S|d|)` recovers `coeff_q`. The extraction is gated for exactness (gate 1) and the whole
contraction is gated end-to-end against the existing damped-direct lattice sum (gate 2).

## 3. File Structure

- **Create `Mathematica/IntraPlaneKambeVector.wl`** — self-contained:
  - copies the cycle-1 scalar Ewald + multipole projection (κ-parameterized) →
    `DstructUndamped[q, s, κ, η]`;
  - copies the vector wavefunction + quadrature helpers (`Cvec`, `Bvec`, `Pvec`, `Mw`, `Nw`,
    sphere quadrature, C/P projection) from `IntraPlaneVectorLattice.wl`;
  - extracts `coeff_q^{c'c}`;
  - assembles the undamped `G0^vec` (L via `D(κ_P)`, M/N via `coeff_q · D(κ_S)`);
  - collective solve, gates, and a JSON dump;
  - `Get`s only `CartesianT0.wl` (for `T0LMN`, as Phase-2 (b) does).
- **Create `Mathematica/IntraPlaneKambeVector_reference.json`** (generated).
- **Create `cubic_scattering/tests/test_intraplane_kambe_vector.py`** — independent Python
  cross-check of the dumped `G0^vec`.
- **Modify `Mathematica/makeIntraPlaneNotebooks.wl`** — add the `.nb` twin entry.
- **Modify `IntraPlaneFoldyLax_Plan.md`** — Phase 3b cycle 2 DONE; cycle 3 remains.
- **Create** memory note; **modify** `MEMORY.md`.

## 4. Gates (mirroring cycle 1)

1. **Extraction reconstruction.** `Σ_q coeff_q h_q(κ_S d) Y_q(m−μ, d̂) == W^{c'c}(d)` at test
   directions/radii (the extraction is exact).
2. **Damped-limit method gate** (the un-fakeable one, analog of cycle-1 `[3]`). Vector G0 built
   from `coeff_q · damped-D[q,s]` **==** the existing Phase-2 (b) damped-direct
   `g0MM/g0NM/g0MN/g0NN` (copied direct routines), to ~1e-4. This validates the extracted
   coefficients and the contraction structure against the validated damped machinery.
3. **Undamped η-independence.** Build undamped `D` at η₁ = 0.7, η₂ = 1.15 → identical
   `G0^vec` (inherits the cycle-1 η-independence).
4. **L-block reciprocity + collective sanity.** β-type L-block reciprocity (as Phase-2 (b));
   `T_coll` finite; isolated limit `G0 → 0 ⇒ T_coll = T0`; coupling `‖T_coll − T0‖/‖T0‖ > 0`.
5. **Python cross-check.** Independent recompute (scipy) of `G0^vec` agrees with the dump.

## 5. Parameters

`Nmax = 2`, `aL = 2.0`, `k_par = (kx,ky) = (0.2, 0.1)`, `κ_P = 0.9`, `κ_S = 1.5` (undamped),
damped check at `κ + 0.25 i`; Ewald `η ∈ {0.7, 1.15}`, `Rc = Gc = 6`; sphere-projection radius
`ρ₀ = 0.5`. All match Phase-1 / Phase-2 (b) so the cycle-1 `D` anchor and the Phase-2 (b)
direct-sum ground truth line up directly.

## 6. Conventions & Constraints

- Time `e^{+iωt}`, outgoing `h_q^{(1)}` via `SphericalHankelH1`; regular `j_q` via
  `SphericalBesselJ`; `Y_q^s` via `SphericalHarmonicY`. Lattice in x–y (`θ_R = π/2`); z = polar
  = depth.
- Conda env `seismic`; Python tooling via `conda run -n seismic`. Line length ≤ 108; ruff
  `--ignore ARG001,ARG002,F841,E741` + ruff format + mypy `--ignore-missing-imports` after every
  Python change. B904 in except; `pathlib.Path`; Google docstrings. scipy 1.17: `sph_harm_y`
  (not `sph_harm`); complex `erfc` via Faddeeva `wofz` (`erfc(z)=exp(−z²)·w(iz)`) — both per
  [[project-undamped-kambe-structure-constants]].
- NO Claude attribution in commits. NEVER write "ATO" (use "PROD"). No heredocs in the Bash
  tool — write commit messages to a file and `git commit -F`. Long `wolframscript` runs
  auto-background; wait via a bounded waiter; ONE kernel at a time.
- Memoise the field/projection helpers (the projection re-hits the same quadrature nodes across
  all (q,s) / (n,m,ν,μ)) — cf. the cycle-1 ~20 min → few min speedup.

## 7. Risks

- **Coefficient extraction parity / q-range.** Vector M↔N couples "opposite parity" q and M→M /
  N→N "same parity"; the extraction must cover the correct `q ∈ {|n−ν|,…,n+ν}` set per channel.
  Mitigation: gate 1 (reconstruction) catches an incomplete q-set; gate 2 catches a wrong
  coefficient. Both are un-fakeable.
- **`CartesianT0.wl` load side-effects.** Phase-2 (b) `Get`s it without issue; reuse that.
- **Conditionally-convergent vector sum.** The undamped vector G0 never sums directly; it is
  *only* obtained via the `D[q,s]` contraction. The damped-direct sum is used **solely** as the
  gate-2 ground truth (where damping makes it converge).

## 8. Cross-references

[[project-undamped-kambe-structure-constants]] (cycle 1), [[kambe-validates-thesis-spectral]],
[[thesis-energy-normalisation]], [[project-intraplane-layer-rt]] (Phase 3a).
`IntraPlaneVectorLattice.wl` (Phase-2 (b) damped vector G0 — the ground truth + helper source),
`IntraPlaneVectorTranslation.wl` (single-pair vector translation — the `coeff_q` source).
