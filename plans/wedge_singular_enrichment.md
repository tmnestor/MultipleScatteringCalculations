# Wedge-singular enrichment of the cube T-matrix — analytic integration only

**Goal.** Augment the internal-field basis with the edge-singular functions of
the cubic inclusion, so that the single-site T-matrix stops being limited by a
polynomial ansatz — with every integral in closed form, no quadrature.

**Why this and not more polynomials.** The internal field of a cube has
stress singularities on its 12 edges. A Taylor expansion about the centre has
radius of convergence set by the nearest singularity: edges sit at `a√2`,
corners at `a√3`, so the series **diverges at the corners**, which are inside
the body. That is the mechanism behind the oscillating tier sequence
`T9 → T27 → T57 = 1.000 → 0.7525 → 0.9692`. Adding polynomial terms — in any
basis, Chebyshev included — cannot fix a divergence caused by a singularity
the basis cannot represent.

**Anchor.** Thesis single-site T-matrix; the closed set of `A33.nb` cell 17
("The Basic System"); `docs/moment_t9_no_eshelby.pdf` for the moment
construction this extends.

---

## Survey — what exists, what it is validated to, what is missing

| Thing | Status |
|---|---|
| `Mathematica/CubeMomentCore.wl` | moment engine; distributional face-peeling; **monomial weights only** (`wt[w_List] := Times @@ X[[#]]`) |
| `Mathematica/CubeA22Block.wl` | the 9×9 first-gradient block; three `O_h` channels in closed form |
| `scripts/gate_a22_vs_effective_contrasts.py` | A22 vs production code, agrees to 3.6e-16 |
| `Mathematica/CubeT57Masters.wl` | degree 5–6 master integrals — the pattern to follow for a new family |
| `Mathematica/CubeT6Block.wl`, `scripts/face_s_rederivation.py` | Duffy **quadrature** near the `1/r` integrand corner — *not* singular basis functions |
| **Wedge-singular basis functions** | **absent everywhere** — not in the package, not in `FFTProp.py/`, not in `GlobalMatrix`, not in the 2012 archive |

So this is genuinely new construction, and the moment engine is the right
foundation: its architecture extends, as argued in Task 2.

---

## Global constraints

- **Analytic integration only.** Read as *closed form in terms of standard
  functions* — elementary plus incomplete Beta / `2F1`. Not "elementary only":
  the edge weight provably produces hypergeometrics (Task 2), so demanding
  elementary forms would make the goal unreachable rather than rigorous.
  **No numerical quadrature anywhere in the derivation path.** Quadrature is
  permitted only as an independent arbiter in validation.
- **Two implementations.** Every closed form must be reproduced in Mathematica
  (symbolic) and Python (numeric) before it enters a `.tex`.
- **Pell-reduce every logarithm** at source. Unreduced Pell-unit logs are
  35-digit cancellations that evaluate to `Indeterminate` in double precision.
- **Source-derivative convention.** Moments are defined with `∂′ = −∂`; odd-`D`
  moments carry `(−1)^D`. Invisible to every symmetry check, so it must be set
  by convention, not discovered.
- **`ka < 0.3`** for any comparison against the analytic cube T-matrix. Print
  `ka` and refuse to fit through points above it.
- Rank-6 tensors go through their 31 invariants, never componentwise.

---

## Architecture

Three layers, each independently checkable:

1. **The exponent** `λ(α,β)` — a transcendental eigenvalue of the bimaterial
   wedge. Computed once, validated against limits with known answers.
2. **The enrichment functions** `Φ_s(x)` — edge-centred, one `O_h` orbit.
3. **The enriched moments** `∫_V Φ_s ∂^D G dV` — the new master-integral family,
   obtained by the same distributional face-peeling the engine already uses.

### Why the architecture works analytically — the load-bearing argument

The engine peels derivatives onto the cube faces. On a face the Green's kernel
is **smooth** (the origin is at distance `Δ/2`), so after peeling the only
remaining singularity is the algebraic edge weight `ρ^λ`, where `ρ` is the
2-D transverse distance to the edge. A 2-D integral of `ρ^λ` over a square
with the singularity at one corner is a standard incomplete-Beta / `2F1`
closed form. **That is the whole reason this is analytically reachable**, and
it is the first thing Task 2 must confirm rather than assume.

If that fails — if peeling leaves a genuinely two-centre integral — the plan
stops at Task 2 and the answer is "not analytically reachable in this
architecture", which is a real result and must be reported as one, not worked
around with quadrature.

---

## File structure

| File | Responsibility |
|---|---|
| `Mathematica/CubeWedgeExponent.wl` | the bimaterial-wedge eigenvalue `λ(α,β)`, with limits |
| `Mathematica/CubeMomentCore.wl` (edit) | generalise `wt` to non-monomial weights |
| `Mathematica/CubeWedgeMoments.wl` | the enriched moment family, closed form |
| `Mathematica/CubeA22Enriched.wl` | the enriched first-gradient block and its `O_h` channels |
| `scripts/gate_wedge_exponent.py` | `λ` against published limiting values |
| `scripts/gate_wedge_moments.py` | enriched moments vs independent quadrature |
| `scripts/gate_enriched_a22.py` | enriched `A22` vs the polynomial one and vs subdivision |
| `docs/wedge_enriched_tmatrix.tex` | the deliverable |

---

## Task 1 — the singular exponent

**Files:** create `Mathematica/CubeWedgeExponent.wl`, `scripts/gate_wedge_exponent.py`

**Interfaces.** Produces: `wedgeLambda[alpha_, beta_]` returning the smallest
root `λ ∈ (0,1)` of the bimaterial wedge characteristic equation, with
`alpha, beta` the Dundurs parameters. Consumed by Tasks 2–4.

- [ ] **Step 1: write the failing check first.** Three limits with answers
      known independently of this code:
      - equal materials (`α = β = 0`), interior angle 90°, no interface →
        `λ = 1` exactly (no singularity);
      - rigid inclusion limit (`α → 1`);
      - void limit (`α → −1`), which must reproduce the free 270° re-entrant
        corner exponent `λ ≈ 0.5445` (the classical Williams value).
      Assert all three. Run it. **Watch it fail** — nothing is implemented.

- [ ] **Step 2: derive the characteristic equation.** Williams eigenfunction
      expansion for a two-material wedge, interior 90° / exterior 270°,
      displacement and traction continuous across both interfaces. Solve
      symbolically for the determinant; do **not** import a published form without
      re-deriving, since the angle convention differs between sources and a
      wrong convention reproduces plausible numbers.

- [ ] **Step 3: run the limits.** Expect `λ = 1`, and `0.5445` in the void
      limit. If the void limit misses, the convention is wrong — fix that
      before proceeding, not later.

- [ ] **Step 4: tabulate `λ` over the validated contrast set** (weak, moderate,
      3× and the negative-shear case). Record whether `λ` is close enough to a
      rational over that range that a fixed exponent would suffice — this
      decides Task 2's difficulty and must be decided by measurement.

- [ ] **Step 5: commit.**

---

## Task 2 — can the enriched moment be done in closed form?

This task is a **go/no-go**, and it is deliberately placed before any
construction. Its purpose is to find out whether the architecture is viable.

**Files:** create `Mathematica/CubeWedgeMoments.wl` (probe only at this stage)

**Interfaces.** Consumes `wedgeLambda`. Produces: a yes/no with a worked
example, and if yes the closed form of the simplest enriched face integral.

- [ ] **Step 1: state the claim as a check.** The simplest non-trivial case:
      one edge, weight `ρ^λ` with `ρ² = (x−a)² + (y−a)²`, kernel `1/r`,
      `D = 2`. Assert the peeled face integral evaluates to a closed form free
      of `Integrate`.

- [ ] **Step 2: peel and inspect.** Apply the existing `outerFace` recursion
      with the new weight. Print the resulting face integrand. **Confirm the
      kernel factor is smooth on the face** — it must be, since the origin is
      at distance `Δ/2`, but confirm rather than assume.

- [ ] **Step 3: evaluate the 2-D face integral.** Expect incomplete-Beta /
      `2F1`. Assert `FreeQ[result, Integrate]`.

- [ ] **Step 4: the decision.** If Step 3 succeeds, continue to Task 3. If it
      fails after the engine's four integration-order routes, **stop and
      report** that the enrichment is not analytically reachable in this
      architecture. Do not substitute quadrature — that would silently convert
      the deliverable into something the plan explicitly excludes.

- [ ] **Step 5: commit either the closed form or the negative result.**

---

## Task 3 — generalise the moment engine to non-monomial weights

**Files:** modify `Mathematica/CubeMomentCore.wl`; create `scripts/gate_wedge_moments.py`

**Interfaces.** Consumes: Task 2's closed form. Produces:
`E$[m, ds, weightFunction]` accepting a weight expression in `x, y, z` rather
than an index list, with the monomial path unchanged.

- [ ] **Step 1: write the regression first.** Every existing moment must be
      **bit-identical** after the change. Assert `E$[-1,{p,p},{}] = −4π`, the
      three Laplacian sum rules, and the `M` cross-check to 10 digits, through
      the new code path. Run it; it fails because the path does not exist.

- [ ] **Step 2: generalise `wt`.** Replace `wt[w_List] := Times @@ X[[#]]` with
      a form accepting either an index list (existing behaviour, preserved
      exactly) or an expression. The recursion's `∂_q w` step becomes a
      symbolic derivative rather than an index-drop.

- [ ] **Step 3: re-run the whole existing gate.** `CubeMomentCoreTest.wl` must
      pass unchanged. Any drift means the monomial path was disturbed.

- [ ] **Step 4: add the enriched moments** for the one edge orbit, and
      symmetrise over the 12 edges using the existing `cubicTensorForm`
      machinery — the enriched moments are still `O_h` tensors, so the
      31-invariant reduction applies unchanged.

- [ ] **Step 5: the independent arbiter.** `scripts/gate_wedge_moments.py`
      compares each closed form against high-precision quadrature. This is the
      **only** place quadrature is allowed, and it is an arbiter, never a
      derivation path.

- [ ] **Step 6: commit.**

---

## Task 4 — the enriched first-gradient block

**Files:** create `Mathematica/CubeA22Enriched.wl`, `scripts/gate_enriched_a22.py`

**Interfaces.** Consumes the enriched moments. Produces the enriched `A22` and
its three `O_h` channel values, comparable directly with
`S_shear`, `S_diag` and the bulk channel of `docs/moment_t9_no_eshelby.pdf`.

- [ ] **Step 1: the falsifiable prediction, written before the result.** The
      enriched block must (a) reduce to the polynomial `A22` exactly when the
      enrichment amplitude is set to zero, and (b) leave the **bulk** channel
      almost unchanged, since the edge singularity is a shear phenomenon. If
      the bulk channel moves substantially, something is wrong with the
      enrichment, not with T9.

- [ ] **Step 2: assemble** with the enrichment coefficients as extra unknowns,
      solved alongside the gradient. Keep the `O_h` block structure.

- [ ] **Step 3: run (a) and (b).**

- [ ] **Step 4: the real test — an independent arbiter for the shear channel.**
      Compare against a converged subdivision computation of the cube's
      volume-averaged strain concentration. ⚠ Subdivision was previously
      recorded as an **unfit instrument** because its sub-cells coupled through
      point propagators at contact — *but that defect was fixed* (contact
      single-averaging). Re-qualify the instrument first, on the `n_sub = 1`
      control which is exact in the static limit, before trusting it as an
      arbiter.

- [ ] **Step 5: commit.**

---

## Task 5 — the deliverable

**Files:** create `docs/wedge_enriched_tmatrix.tex`; compile in-place with
`lualatex`, twice.

- [ ] **Step 1:** derivation, closed forms, validation table.
- [ ] **Step 2:** a section stating plainly what is *not* established.
- [ ] **Step 3:** update `docs/moment_t9_no_eshelby.tex` to reference the
      enrichment as the route past its own stated limitation. Updating the
      affected `.tex` is part of this task, not a follow-up.
- [ ] **Step 4:** commit `.tex` and `.pdf`.

---

## What would falsify the whole plan

- **Task 2 fails** — the peeled integral is not closed-form. Then the
  enrichment is not analytically reachable in this architecture. Report it;
  the fallback is a boundary-integral formulation, which is a different plan.
- **`λ` is strongly contrast-dependent and irrational** across the validated
  set (Task 1 Step 4). Then every contrast needs its own master integrals and
  the "compute once" economics collapse. Task 1 measures this before Task 2
  spends effort.
- **The enriched shear channel does not move toward the subdivision arbiter.**
  Then the edge singularity is not the dominant error and the polynomial
  truncation was not the limitation — which would contradict the divergence
  argument and needs to be understood before continuing.

## What this plan does *not* claim

It does not claim the enrichment will fix the **far field**. By the parity
decoupling, `A22` alone fixes `∂u` and hence the leading modulus far field —
so the enrichment helps there **only if it changes `A22` itself**, which is
exactly why Task 4 targets `A22` and not the second-gradient block. If the
enriched `A22` channels come out equal to the polynomial ones, the whole
exercise has produced a negative result, and that should be reported rather
than rescued.
