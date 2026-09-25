# Closing the Imbedding / Multiple-Scattering Comparison — Plan

- **Date:** 2026-09-21
- **Why this exists:** §22 of `FirstOrderContrastOperator.tex` now compares the
  impedance march against the voxel Foldy–Lax route on the sphere, and the
  comparison has three holes that were found by being asked for the case that
  had not been run. Each is a measurement, not a derivation. The claim narrowed
  every time it was tested, so the plan is ordered by *which hole could narrow
  it again*.
- **State it builds on:** gate `gate_imbedding_vs_foldy_lax.py` 18/18; paper 94
  pages, 13/13 tallies; full suite 1061 passed.

---

## Status, 2026-09-25

- **Task 1 — DONE.** The toroidal weight is `t_n = i k_S c_n renorm(n)`, on the
  **sin φ** parity; it is determined, not bounded (Mathematica
  `ToroidalIncidentExpansion.wl` to 1e-26, a Python fit to 1e-8, the optical
  theorem for S incidence rejecting 1, −i k_S, k_S, 1/(i k_S)).  The S channels
  of the SV column now match Mie in complex value: R 0.174, T 0.223, against
  0.445 / 0.283 without the term.  March gate 36/36.
- **A scoring defect found on the way reverses §22's wide-angle verdict.** The
  Mie predictions put the sphere at the lateral origin while the march puts it
  at the grid centre; the translation phase `exp(−i(k_x x_c + k_y y_c))` was
  never applied, and it is 1 only on the specular order.  With it (plus the
  zeroed-Nyquist orders excluded, and the forward specular T scored on its
  scattered part), the march is better than the voxel route at wide angles by
  7–17× at every frequency, and at the specular orders too.  The earlier
  "1.96 / 0.41 / 0.28" and "voxel better at every row" were the scrambled
  phase.  Imbedding gate 19/19; §22 rewritten.
- **Task 2's rationale below is stale** — the wide-angle orders are no longer
  "uninterpretable"; they match the isolated sphere to a few per cent.  A
  periodic reference would still split that residual into array coupling and
  march error.
- **The SV→P outlier (T at 0.32) — RESOLVED, and it was not about SV→P.**
  `band_limited_disc` evaluated the form factor and the centring phase with
  `grid_wavenumbers`, whose Nyquist entry is zeroed; on an even grid the
  Nyquist modes got `F(0)` with phase 1 where the true content is zero.  The
  disc overshot by 75% and broke the x-mirror at order one.  Symptom: march/Mie
  differed at ±k_x.  Zeroed, T SV→P is 0.104 and the mirror holds to 6.5e-14.
- **The "1.45 ε array-coupling floor" was mostly that artefact.**  It is linear
  in ε because the artefact reaches the specular order only at second order.
  Corrected: 0.157 ε at N=8, 0.190 ε at odd N=9 (never affected), converging
  under lateral refinement (~1/N) to ≈ 0.3 ε at 2.5 diameters.  Depth order
  4.00/4.00/4.09.  March gate part 10 now produces these tables.
- **§22 corrected again:** the march wins at wide angles by 7–15× and at the
  specular orders by 3.7–13× at every frequency, including k_S a = 1 on 900 m.
  The backscatter figure (odd 9×9 grid) was unaffected; crossover 0.237.
- March gate 44/44, imbedding gate 19/19.  Both §22 figures now have producers
  (`--dump-angles`, `--dump-backscatter`).
- **Task 2, small-contrast half — DONE by second-order Born** (`scripts/gate_born2_array_coupling.py`,
  4/4).  Kernel derived in Mathematica and sympy (agree 1.7e-15; δ′/δ″ cancel,
  one q-independent contact term).  Windowed lattice-sum minus continuum,
  Q-independent to 6 figures, predicts departure/ε = 0.2880 − 0.0556i at 2.5
  diameters; the march gives −0.0558 (imag, depth-converged) and real part
  extrapolating to 0.280–0.290.  The residual is the array's double scattering.
  **Still open:** a finite-contrast periodic reference (layer-KKR, option A);
  and the isolated sphere's full Born-2 integral against Mie's ε² coefficient
  (slow ~q^-1.4 tail, not needed for the difference).
- **Mie T-matrix per order** (`mie_tmatrix_psv/sh`): a 2×2 interior-impedance
  form was built and found NO more accurate than the direct 4×4 (whose 1e33
  condition number is ill-scaling); kept the 4×4, with a 50-digit arbiter
  (`Mathematica/MieTmatrixReference.wl`) and per-order tests.

---

## What already exists — DO NOT REBUILD IT

Three things were declared "not built" during the last session and all three
were wrong. Check here before writing anything.

| Thing | Where it actually is | Status |
|---|---|---|
| m=±1 Mie plane-wave spectrum | `Mathematica/MieSphericalWaves.wl` §7, and ported to `scripts/gate_mie_spectrum_general_m.py` | **built**, 8/8 vs a 72-case reference, worst 1.5e-8 |
| Iterative solve of the voxel sphere | GMRES applies directly to `(I − P̃T̃)ψ = ψ_inc`; gate part 5 | **built**, 58→244 its as ε 0.02→0.80 |
| Block-Toeplitz FFT matvec for the sphere | `cubic_scattering/sphere_scattering_fft.py` | **built**, matches dense to 3e-9 |
| Mie coefficients for S incidence | `a_n_sv`, `b_n_sv`, `c_n` on `MieResult` | **built** |
| Periodic Foldy–Lax with plane-wave R | `slab_scattering.slab_reflection_matrix` | built, but **cannot hold a sphere** — `SlabGeometry` is a fully occupied M×M×N_z lattice with no per-cell occupancy |

🔑 **Grep for the artefact, not only the concept.** `fftn`, `circulant`,
`gmres`, `n_sub`, `_fft.py` — capabilities here are named after their mechanism,
not their physics. All three false calls came from searching for the physics.

⛔ **Never write "not built" into a gate's output, a commit message or the
`.tex` on the strength of a partial search.** It gets committed, and it is what
the next reader trusts.

---

## Task 1 — SV/SH incidence scored against the exact sphere

**Why first:** it is assembly, not derivation, and it is the only one of the
three that can still change the accuracy verdict. Every number in §22 is the P
column.

**What is missing:** the Poisson/array relation applied per incident channel,
and the *toroidal* normalisation.

**Steps**

1. In `gate_sphere_vs_impedance_march.py`, extend `mie_spectrum_s` to the
   M-type. The spheroidal part is already there and validated:
   `c_mode = (i/2πk_z) Σ coeff_n · renorm(n) · ang_n(m=1) / iⁿ`,
   `renorm(n) = −1/[n(n+1)]`.
2. **Determine the toroidal scalar honestly.** It is currently *bounded, not
   determined*: scanning it against the march rules out 1 (median agreement
   0.94→0.58) and 1/(ik_S) (destroys it), and bounds it at ≲ k_S, where it is
   invisible against the array floor. Do **not** fit it against the march —
   that is the comparison it would then be scored on, and a one-parameter
   correction fitted against its own arbiter always looks successful (the
   lattice constant `K` was chased that way and turned out to be exactly 1).
   Two non-circular routes:
   - derive it from the M-type's incident normalisation `(2n+1)iⁿ/(ik_S)`
     against the L/N families' — the relative factor is a ratio of two
     expansions, both in `compute_elastic_mie`;
   - or impose the optical theorem for S incidence, which is exact and
     asymmetric in the right way (σ_ext scales with f(0), σ_sc does not).
3. Score the SV and SH columns of `R` and `T` the way the P column is scored.

**Acceptance:** the SV column's P channel already tracks Mie at median 0.92
(R) / 1.02 (T) over 24 orders — the S→S channels should reach the same
array-coupling floor once the toroidal term is right, and the residual should
*fall* when the term is included rather than rise.

**Traps**
- ⚠ `mie_far_field(..., "SH")` is **not** an arbiter here: it drops the M-type
  as "O((ka)⁴) and negligible", which is false at k_S a = 2.4 where |c_n| runs
  1.5–2.5× the spheroidal coefficients.
- ⚠ On the specular order the azimuth is undefined. `ang_triple` and
  `mode_matrix` must take the **same** ψ=0 branch; `0/0 = 0` zeroes the whole
  specular column, and the 72-case check cannot catch it because Gauss nodes
  never land on q=0.

---

## Task 2 — a periodic reference, so the wide-angle orders mean something

**Why second:** without it the march's wide-angle `R`/`T` orders have no
external check at all. They are currently scored against an isolated sphere,
which is a different object — the array is not axisymmetric and the sphere is.

**What is missing:** an exact, or independently-converged, solution for the
*periodic array* of spheres.

**Steps**

1. Simplest credible route: **Bloch-periodic Foldy–Lax over a voxelised
   sphere.** All three ingredients exist — `sphere_sub_cell_centres` for the
   geometry, `lattice_greens` for the Bloch-summed propagator (its lattice sum
   is implicit in the spectral form), and `sphere_scattering_fft` for the FFT
   matvec. What does not exist is the combination.
2. ⚠ The lateral lattice sum **must** use the Ewald route
   (`build_slab_kernels(..., lattice_ewald=True, volume_averaged=True)`); the
   truncated sum was a months-long defect here.
3. Extract plane-wave R/T from it with the Weyl extractor that
   `slab_reflection_matrix` already uses.

**Acceptance:** at large period it must reduce to the isolated-sphere answer —
that limit is the check that the Bloch sum is right, and the march's own
period ladder (1.96 → 0.41 → 0.28 wide-angle error as L grows) is the curve it
should converge along.

**Payoff:** it converts the march's wide-angle numbers from uninterpretable to
measured, in either direction. It is the only task that can *restore* ground the
march has lost, and equally the one that could confirm the loss.

---

## Task 3 — iterative vs imbedding, with both setup costs amortised

**Why third:** it changes the *cost* verdict, not the accuracy verdict, and the
accuracy verdict is the one still moving.

**What is missing:** a like-for-like cost comparison. Measured so far:

| | setup | per solve | amortises over |
|---|---|---|---|
| voxel FFT/GMRES | 77–360 s (99.9% of the call) | 0.09–0.43 s | incident directions at **fixed ω** |
| dense Foldy–Lax | — | 1.14–6.96 s, O(n³) | 9 RHS via one factorisation |
| impedance march | — | 0.15–9.8 s | all sources (Y is source-free) |

**Steps**

1. Fix a workload that both routes actually face: N_d incident directions at
   N_f frequencies, and count total wall time for each.
2. ⚠ `_build_fft_kernel` takes `omega`, so the FFT kernel is **rebuilt per
   frequency**. Break-even against the dense solve is ≈74 directions at
   n_sub=6 and ≈27 at n_sub=8, falling as n grows. A frequency sweep is a
   straight loss for it.
3. The march's counterpart is `Y`, which is also per-frequency but reused
   across every source. Quantify that reuse on the same workload rather than
   asserting it.
4. If the FFT kernel build is the bottleneck (it is, at 99.9%), check whether it
   is algorithmic or just a Python loop over (2n−1)³ grid points × 81
   components — ~58 ms per point suggests the latter.

**Acceptance:** a table with one row per workload shape, not a single ratio.
The session's recurring failure was quoting one number as if it were general.

**Traps**
- ⚠ **Warm every size before timing it.** The first call at a given `n_sub` pays
  kernel setup; an earlier table read 18.7 s where the steady value is 0.07 s,
  and another claimed "37× faster" comparing warm FFT against cold dense.
- ⚠ Four points at small sizes give descriptive, not asymptotic, exponents —
  BLAS is not in its cubic regime at these matrix sizes.

---

## Smaller items, carried forward

- **Hulme (2004)**, the companion to Haines et al., carries the worked examples
  and numerical validation and has still not been consulted. No accuracy or
  run-time comparison against their implementation is offered anywhere.
- **The M-type normalisation** (Task 1 step 2) is the only physical quantity in
  this area currently recorded as *bounded rather than determined*.
- **`ε` is a velocity contrast**, not a modulus one: α, β and ρ all scale by
  1+ε, so the moduli scale as (1+ε)³ — at ε=0.8 the shear modulus is 5.8×
  background. Appendix A. Worth re-reading before choosing sweep ranges.

---

## The standing lesson from the last round

The comparison narrowed four times — 34× → frequency-dependent →
backscatter-only → specular-only-above-ka-1 — and every narrowing came from
someone asking for the case that had not been run. Before quoting any ratio,
ask which cases were *not* sampled and why, and whether the sampled one happens
to favour the method being argued for.

And: a method's required machinery is part of the method. The march cannot
solve an isolated sphere without periodizing, so the periodization error is the
march's, exactly as the staircase error is the voxel route's. Arguing that an
arbiter is "unfair" to a method that cannot pose the problem without extra
machinery is special pleading.
