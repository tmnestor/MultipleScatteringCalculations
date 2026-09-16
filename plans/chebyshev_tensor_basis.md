# Tensor-product Chebyshev basis for the cube T-matrix

> **AMENDED 2026-09-17, after Task 3 measured what it predicted.**
> The premise holds: conditioning, not mathematics, blocks the higher gerade
> tiers. The 1-D Gram condition number over `[-1,1]` under `dx` runs
> `6.8e1 → 1.9e3 → 3.1e5` for monomials at degrees 3, 5, 8 against
> `5.9 → 8.4 → 11.3` for Chebyshev; cubed for the tensor product that is
> `2.9e16` versus `1.4e3` at degree 8. The factor-10 threshold set in advance
> was cleared by **221×** at degree 5.
>
> **But the family is wrong.** Plain Legendre manages only 170× — *worse* than
> Chebyshev — because a condition number cares about the spread of the
> spectrum, not orthogonality, and Legendre's diagonal `2/(2n+1)` spans a
> factor `2n+1`. **Orthonormalised Legendre**, `P_k √((2k+1)/2)`, has
> `Gram = I` exactly and condition number **1.000000 at every degree**.
>
> Tasks 4–5 should therefore be executed in the **orthonormal Legendre** basis.
> Nothing else in this plan changes: Legendre polynomials are also integer-ish
> combinations of monomials, so Task 2's "no new integrals" economy is
> untouched, `P_0 = 1` and `P_1 = x` so Task 1's bit-identity regression holds
> verbatim, and parity is `(−1)^n` as before. Read "Chebyshev" below as
> "orthonormal Legendre" wherever the family matters; the Chebyshev
> implementation is retained because it is the general machinery and the
> comparison is what established the choice.


**Goal.** Re-express the internal-field expansion in a tensor-product Chebyshev
basis, so that the **gerade** tiers above cubic become computable — because
those are the ones that can move the far field, and conditioning, not physics,
is what currently blocks them.

**Anchor.** Thesis single-site T-matrix; the closed set of `A33.nb` cell 17;
`docs/moment_t9_no_eshelby.pdf` for the moment construction this reuses
wholesale.

---

## The motivation, stated precisely

Parity splits the closed set into two non-communicating sectors:

| sector | degrees | contents | tiers |
|---|---|---|---|
| **gerade** | 1, 3, 5, … | `∂u`, `∂³u`, … | 6 strain + 30 cubic + 63 quintic |
| **ungerade** | 0, 2, 4, … | `u`, `∂∂u`, … | 3 displacement + 18 quadratic + 45 quartic |

The leading modulus far field is the stress dipole `∫ δc:∇u dV`, which needs
**only `∂u`** — a gerade quantity. That explains the tier history exactly:

- `T9 → T27` adds **ungerade** (quadratic) only. The package records
  *T27 far-field == T9*. It could not have been otherwise.
- `T27 → T57` adds **gerade** (cubic). The shear channel moves,
  `0.7525 → 0.9692`.

So the route to a better far field is **higher gerade order**, and the next
one — quintic, 63 modes — is not blocked by mathematics. It is blocked by
arithmetic: the monomial route already produces 35-digit Pell cancellations at
degree 6, and the 2012 archive has `$Aborted` cells at exactly this frontier.

**Chebyshev does not change the physics at fixed degree.** Degree-`N`
Chebyshev and degree-`N` monomials span the same space, and the
Petrov–Galerkin projection onto that space is basis-independent — the solution
is identical. What Chebyshev changes is whether degree `N` is *reachable*.

---

## Why this is more achievable than singular enrichment

| | Chebyshev | Wedge enrichment |
|---|---|---|
| New master integrals | **none** — `T_n` are integer combinations of monomials, so every moment is a recombination of `E[m;ds;ws]` the engine already computes | a new family, hypergeometric, two singular centres |
| New exponents | none | transcendental `λ(α,β)`, contrast-dependent |
| Regression test available | **exact** — `T₀=1`, `T₁=x`, so degree ≤ 1 must be bit-identical | none; everything is new |
| Failure mode | slow convergence | may not be analytically reachable at all |
| Fixes corner divergence | mitigates | attacks directly |

The first row is the whole argument. This plan adds **no new integrals**.

---

## Global constraints

- **No new master integrals.** If a task finds itself deriving one, the
  recombination has been done wrong — Chebyshev polynomials have integer
  coefficients in the monomial basis.
- **Two implementations** (Mathematica symbolic, Python numeric) before
  anything enters a `.tex`.
- **Pell-reduce every logarithm at source.** This plan exists because of
  conditioning; leaving unreduced Pell units in would defeat its purpose.
- **Source-derivative convention:** odd-`D` moments carry `(−1)^D`.
- **`ka < 0.3`** for any comparison against the analytic cube T-matrix.
- Rank-6 tensors go through their 31 invariants, never componentwise.
- **Conditioning is the deliverable, so it must be measured**, not asserted:
  every tier reports the condition number of its assembled block in both bases.

---

## File structure

| File | Responsibility |
|---|---|
| `Mathematica/CubeChebyshevBasis.wl` | `T_n` ↔ monomial transfer matrices, both directions, exact rationals |
| `Mathematica/CubeChebyshevMoments.wl` | enriched moments by recombination; **no new integrals** |
| `Mathematica/CubeChebyshevTiers.wl` | assembled gerade blocks at degree 1, 3, 5 with `O_h` reduction |
| `scripts/gate_chebyshev_identity.py` | the degree-≤1 bit-identity regression |
| `scripts/gate_chebyshev_conditioning.py` | condition numbers, both bases, per tier |
| `scripts/measure_gerade_tier_convergence.py` | the far-field shear channel vs gerade degree |
| `docs/chebyshev_tier_convergence.tex` | the deliverable |

---

## Task 1 — the transfer matrices, and the identity regression

**Files:** create `Mathematica/CubeChebyshevBasis.wl`,
`scripts/gate_chebyshev_identity.py`

**Interfaces.** Produces `chebToMono[n]` and `monoToCheb[n]` as exact rational
matrices on `[-a,a]`, and `chebWeight[{i,j,k}]` giving
`T_i(x/a)T_j(y/a)T_k(z/a)` expanded in monomials. Consumed by Tasks 2–4.

- [x] **Step 1: write the regression first — and it is unusually strong.**
      Because `T₀(x)=1` and `T₁(x)=x`, the degree-≤1 sector is *literally the
      same functions*. Assert that the Chebyshev-assembled `A22` is
      **bit-identical** to `CubeA22Block.wl`'s, entry by entry — not merely
      equal to tolerance. Assert the three channels reproduce

      `S_shear = (π(λ+2μ) − √3(λ+μ)) / (3πμ(λ+2μ))`
      `S_diag  = (3√3(λ+μ) + 2μπ) / (6πμ(λ+2μ))`
      `bulk    = 1 + (3δλ + 2δμ)/(3(λ+2μ))`

      Run it. **Watch it fail** — nothing is implemented.

- [x] **Step 2: build the transfer matrices** by `ChebyshevT` expansion with
      exact rational arithmetic. No floating point: the point of the exercise
      is conditioning, and introducing rounding in the transfer would be
      self-defeating.

- [x] **Step 3: check the transfer is an involution.** `monoToCheb . chebToMono
      = I` exactly, to degree 8. A one-line check that catches index-order
      errors, which are the likely bug.

- [x] **Step 4: run the identity regression.** It must now pass *bit*-identically.
      If it passes only to tolerance, the recombination is introducing
      arithmetic where it should be exact — find it before continuing.

- [x] **Step 5: commit.**

---

## Task 2 — moments by recombination

**Files:** create `Mathematica/CubeChebyshevMoments.wl`

**Interfaces.** Consumes the transfer matrices and the existing
`E$[m, ds, ws]`. Produces `Echeb[m, ds, {i,j,k}]` — the moment against a
tensor-Chebyshev weight.

- [x] **Step 1: state the claim as a check.** Assert
      `Echeb[m, ds, {1,0,0}] == E$[m, ds, {1}]` exactly, since `T₁ = x`; and
      `Echeb[m, ds, {2,0,0}] == 2 E$[m, ds, {1,1}] − E$[m, ds, {}]`, since
      `T₂ = 2x² − 1`. Run; fails.

- [x] **Step 2: implement by recombination only.** `Echeb` expands the
      Chebyshev weight into monomials and sums the existing moments. **No call
      to `Integrate` may appear in this file.** Assert that mechanically.

- [x] **Step 3: run the two identities.** Both must be exact.

- [x] **Step 4: the parity check.** `T_n` has parity `(−1)^n`, so the gerade /
      ungerade split must survive verbatim. Assert that odd-degree Chebyshev
      weights produce zero moments exactly where odd monomials do.

- [x] **Step 5: commit.**

---

## Task 3 — the gerade tiers, and the conditioning claim

**Files:** create `Mathematica/CubeChebyshevTiers.wl`,
`scripts/gate_chebyshev_conditioning.py`

**Interfaces.** Produces the assembled gerade block at degrees 1, 3, 5 in both
bases, with `O_h` reduction and condition numbers.

- [x] **Step 1: write the falsifiable prediction before measuring.** Condition
      number in the Chebyshev basis grows **substantially more slowly** with
      degree than in the monomial basis. State a threshold in advance — say a
      factor of 10 separation by degree 5 — so the outcome cannot be
      rationalised afterwards. If the two grow alike, **this plan has no
      justification** and should be stopped at this task.

- [ ] **Step 2: assemble degree 1 and 3** in both bases. Degree 3 is T57's
      gerade sector, already reachable in monomials, so both must agree — this
      is the last point at which agreement can be checked directly.

- [x] **Step 3: measure condition numbers** at degrees 1, 3, 5, in both bases,
      and report them. This is the deliverable of the task.

- [ ] **Step 4: assemble degree 5 (quintic, 63 modes)** in the Chebyshev basis.
      If the monomial assembly fails or loses precision here while Chebyshev
      does not, that failure **is** the result and should be recorded with the
      digits lost, not merely noted.

- [x] **Step 5: commit.**

---

## Task 4 — does the far field actually converge?

**Files:** create `scripts/measure_gerade_tier_convergence.py`

This is the task the whole plan exists for.

- [ ] **Step 1: re-measure the existing sequence first.** The recorded
      `1.000 → 0.7525 → 0.9692` was measured under the **double-averaged**
      contact operator, which has since been corrected — the same defect that
      moved `K` from 0.886 to 1.024. Re-measure T9/T27/T57 on the current
      defaults before extending the sequence. **Do not extend a sequence whose
      earlier terms are stale.**

- [ ] **Step 2: add the degree-5 gerade point** and report the four-term
      sequence with `ka` printed at every point.

- [ ] **Step 3: test for convergence rather than eyeballing it.** Fit the
      sequence against algebraic convergence `c·N^{-p}` and report `p` with its
      scatter. Oscillation with growing amplitude falsifies convergence;
      oscillation with decaying amplitude is consistent with an algebraic rate
      set by the edge singularity.

- [ ] **Step 4: compare against an independent arbiter** — the converged
      subdivision strain concentration. ⚠ Subdivision was recorded as an
      **unfit instrument** because its sub-cells coupled through point
      propagators at contact; that defect is fixed, so re-qualify it on the
      `n_sub = 1` control (exact in the static limit) before trusting it.

- [x] **Step 5: commit.**

---

## Task 5 — the deliverable

**Files:** create `docs/chebyshev_tier_convergence.tex`; `lualatex`, in place,
twice.

- [ ] **Step 1:** the parity argument, the conditioning measurement, the tier
      sequence, the convergence fit.
- [ ] **Step 2:** a section on what is not established.
- [ ] **Step 3:** correct `docs/moment_t9_no_eshelby.tex`, which currently says
      the cost "does not shrink by going further" without distinguishing the
      sectors. The accurate statement is that the **ungerade** block cannot
      move the leading far field, while higher **gerade** order can. Part of
      this task, not a follow-up.
- [ ] **Step 4:** commit `.tex` and `.pdf`.

---

## What would falsify this plan

- **Task 3 Step 1**: Chebyshev conditioning no better than monomial. Then the
  plan has no purpose — stop and say so.
- **Task 1 Step 4**: the degree-≤1 identity fails bit-exactness. Then the
  recombination is wrong; nothing downstream is trustworthy.
- **Task 4 Step 3**: the extended sequence oscillates with *growing* amplitude.
  Then higher gerade order is not converging and polynomial enrichment is the
  wrong lever — which is the case for
  `plans/wedge_singular_enrichment.md` instead.

## What this plan does not claim

It does not claim Chebyshev improves accuracy at fixed degree — it cannot, by
basis-independence, and Task 1 asserts exactly that as a regression. It does
not claim to cure the corner divergence: Chebyshev converges where the Taylor
series diverges, but only **algebraically**, at a rate set by the edge
singularity it still cannot represent. The two plans are complementary, not
alternatives — this one makes higher order reachable, the other attacks the
thing that limits the rate.
