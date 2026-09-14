# Cartesian Directional Sweeps, Stage 2 — Implementation Plan

> ## ⚠ REVISED 2026-09-14, after Task 1 measured the cost and the original plan failed
>
> The first version of this plan extended the stage-1 sweep architecture to 3-D
> by giving each sweep a transverse-wavenumber quadrature. **Task 1 measured that
> and it is not buildable.** The revision below is what the measurement says to
> build instead. The original task bodies are not preserved; what they got wrong
> is recorded here, because the error was in the architecture, not the arithmetic.
>
> **What Task 1 measured** (`scripts/measure_sweep3d_cost.py`, and the probes it
> drove):
>
> 1. **Branch points sit on the integration contour.** The post-`k_y`-residue
>    kernel carries `1/ky_L` and `1/ky_T`, which vanish on the circles
>    `|k_⊥| = kP = 7.54` and `kS = 12.57` rad/km — both inside the domain. At
>    real `ω` a midpoint rule straddles an integrable `1/√` singularity and does
>    not converge (1.9e-1 → 1.1e-1 over an 8× refinement, non-monotone). Stage 1
>    types `omega` **complex** throughout for exactly this reason. With damping
>    0.01–0.03 convergence is restored.
> 2. **The converged rule** is `kr·pitch = 30`, `Δk = 0.156` → **3.2e-7** at
>    1.85 M nodes. `kr·pitch = 40` gives an identical error, so 30 is where
>    truncation stops binding. Both constraints bind independently: at the same
>    `Δk = 0.3125`, `kr_max = 40` gives 4.9e-3 and `kr_max = 80` gives 5.8e-5.
> 3. **The sweep accumulator cannot carry a 2-D transverse rule.** In 3-D it is
>    `(n_z, n_x, n_k, 9)`:
>
>    | Rule | Accumulator, 16³ | |
>    |---|---|---|
>    | stage-1 1-D (`n_kz = 2048`) | 4.7 MB | why stage 1 works |
>    | 3-D, 5.8e-5 | 7.6 GB | two live at once |
>    | 3-D, 3.2e-7 | 68.3 GB | — |
>
>    Stage 1 escaped on two counts simultaneously — one fewer real-space axis
>    *and* a 1-D transverse rule. Stage 2 loses both at once. Storing the
>    inter-plane stack spectrally, `(n_z, n_z, 9, 9, n_k)`, is **615 GB** at 16³.
>
> **Two errors of mine along the way, recorded so the record is honest.** I read
> a 5e-3 plateau as a convention error when it was ordinary cutoff truncation;
> and the first cost projection used the node count from the non-converged table,
> understating the memory by 16×.
>
> **The original plan's tolerances were unreachable regardless:** it asserted
> 1e-8 against the pairwise sum, where the quadrature floor is 3.2e-7.

**Goal:** Make the solver fully three-dimensional, so heterogeneity may vary in
`y` rather than being invariant along it.

**Architecture:** Split the propagator by whether the pair is separated in depth.
Same-depth coupling uses the **closed-form whole-space propagator in real space**
— exact, no quadrature, no accumulator. Different-depth coupling keeps the
**stratified spectral propagator**, transformed to real space once at build time
rather than carried spectrally through the solve.

**Spec:** `docs/specs/2026-09-13-cartesian-directional-sweeps-design.md`. §6's
"stage 2 introduces exactly one new thing" and §3.1's six-sweep framing are both
amended by this plan — see Task 6.

---

## Why this decomposition, in one table

Measured at a 16³ lattice, 9×9 states, complex128:

| Class | Pairs | Instrument | Storage | Matvecs / apply |
|---|---|---|---|---|
| **B + C** | `Δz = 0`, any `Δx`, `Δy` | `exact_propagator_9x9`, real space | **1.2 MB** (961 distinct `(Δx,Δy)`) | 1.05 M |
| **A** | `Δz ≠ 0` | stratified propagator, transformed to real space at build | **149 MB** (115 320 distinct `(Δz,Δx,Δy)`) | 7.9 M |
| — | `Δ = 0` | inside `T₀` | — | — |

Under 200 MB in total, against 615 GB for the spectral route. The matvec counts
are batched `einsum` work of order 100 Mflop — not the constraint.

**Only class A needs a spectral representation at all**, because the *stratified*
propagator exists only in `(k_x, k_y)`. The whole-space propagator has a closed
form, so classes B and C never need a quadrature — which is what the original
plan missed by extending the sweep uniformly to all three directions.

**Class A splits further, and this is what makes it affordable:**

```
P_layered(Δz; Δx, Δy)  =  P_wholespace(Δz; Δx, Δy)   +   ΔG₀(Δz; Δx, Δy)
                          closed form, exact, free       spectral, but rank 3
```

`gate_dg0_absolute_magnitude.py` `[M1]` measured `ΔG₀ = D↑ · R_D · S↓` to be
genuinely rank 3 (out-of-span ≤ 4e-9). **That licenses the split; it is not what
makes it affordable.** Task 2 measured the cost directly, and corrected two
claims this plan made before measuring:

| Claim as first written | Measured |
|---|---|
| "the rank-3 property is load-bearing" | **No.** Rank 3 justifies writing the split; it does not reduce the node count. |
| "separability is what makes the build tractable" | **No.** The transform is 0.36 Gflop separable against 5.55 Gflop dense — both negligible. |
| — | **What actually helps:** `ΔG₀` converges at `kr·pitch = 10`, where the full kernel needs 30. That is 65 536 nodes against 2 365 444 — a 36× saving, from the reverberation's smaller `k`-support, not its smoothness. |
| — | **What actually costs:** `corrected_layered_9x9` at **0.88 ms/node**, already batched. Everything else is noise beside it. |

Measured convergence of `ΔG₀` at `kr·pitch = 10` (self-change between successive
rules, since no closed form exists for the reverberation): 3.4e-1 → 2.6e-2 →
1.4e-3, falling ~18× per doubling.

**Build cost, one-time and outside the Krylov loop:** 2.1 h at 16³, 8.7 h at 32³.
Affordable, but it scales as `n_z²` and is the first thing that will bite. The
obvious reduction if it does: the stratified propagator between two planes needs
only the layers between and around them, not all 60 in the model — untested, and
not to be assumed.

---

## Global constraints

- Conda env `seismic`; run as `conda run -n seismic <cmd>`.
- Seismic units (km/s, g/cm³, GPa, km); time `e^{−iωt}`; index order `z, x, y`.
- **`omega` is complex everywhere.** Task 1 measured that a real `ω` puts the
  `1/k_y` branch points on the contour and destroys convergence. Every entry
  point takes `omega: complex` and every gate asserts a damping floor rather
  than inheriting one by luck.
- State is the 9-vector `(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)`;
  no representation conversion at any boundary (spec §4).
- `exact_propagator_9x9(x, y, z, …)` takes **Cartesian `x, y, z`** while the
  state is ordered `z, x, y`. A separation of one pitch along `y` is
  `(0.0, pitch, 0.0)`. Reversing this gives a plausible wrong answer, not an
  error.
- Python 3.12 typing; `pathlib.Path`; Google docstrings; line length 108.
- Four-element diagnostics on every raise: what, where, valid example, recovery.
- After every Python change: `ruff check … --fix --ignore ARG001,ARG002,F841,E741`,
  `ruff format`, `mypy … --ignore-missing-imports`.
- **Stage 1 stays working unchanged.** `SweepGrid`, `sweep_x`, `sweep_z`,
  `apply_g0`, `build_g0_cache` and every stage-1 gate are untouched; 3-D arrives
  as new names beside them.
- Never assert a tolerance without asserting what discriminates: residuals flat
  in separation and flat in lattice size, with the distinct-source vacuity
  control from `gate_lateral_sweep_alg52.py`.

---

## File structure

| File | Responsibility |
|---|---|
| `cubic_scattering/pair_propagators.py` *(new)* | Real-space propagator tables: `same_depth_table`, `inter_plane_table`, and the `(Δz,Δx,Δy) → index` maps. Pure NumPy. |
| `cubic_scattering/directional_sweeps.py` *(modify)* | `SweepGrid3D`, `build_g0_cache_3d`, `apply_g0_3d`. `sweep_y`'s stub is **replaced by a real-space application, not a spectral sweep** — its docstring records why. |
| `cubic_scattering/tests/test_pair_propagators.py` *(new)* | Table correctness and the partition. |
| `scripts/measure_dg0_transverse_rule.py` *(new)* | Task 2 — how many transverse nodes the rank-3 reverberation actually needs. |
| `scripts/gate_g0_3d.py` *(new)* | The 3-D `G₀` against a direct pairwise sum, plus the partition support count. |
| `scripts/measure_sweep3d_cost.py` *(exists, committed)* | Task 1, done. Keep as the record of why the architecture changed. |

---

### Task 1: ✅ DONE — the cost measurement

Committed as `scripts/measure_sweep3d_cost.py`. Findings are in the header block
of this plan. **Do not repeat it**; do re-run it if the background medium or
pitch changes materially, since the node requirement scales with `kP·pitch`.

---

### Task 2: ✅ DONE — the reverberation's transverse rule and the build cost

Committed as `scripts/measure_dg0_transverse_rule.py`. Findings are in the
decomposition table above. **Do not repeat it**; do re-run it if the background
model or the plane spacing changes, since the cutoff scales with `kP·pitch`.

**The rule Task 4 uses:** `kr·pitch = 10`, `Δk ≈ 0.31`, **65 536 tensor nodes**.

**Two corrections this task forced**, both to claims made before measuring:
the rank-3 property is not what reduces the cost, and the separable transform is
not the bottleneck. Both are recorded in the table above rather than quietly
fixed, because the pattern — justifying an architecture with the most
interesting property to hand rather than the one that governs — is the same one
that produced the first version of this plan.

**A tensor grid, not the radially masked set.** Class A uses a plain tensor
product in `(k_x, k_y)`. The mask that pays at `Δz = 0` buys much less here,
since `e^{−κ|Δz|}` already suppresses the corners.

---

### Task 3: Same-depth real-space propagator table (classes B and C)

**Files:** Create `cubic_scattering/pair_propagators.py`; test in
`cubic_scattering/tests/test_pair_propagators.py`.

**Interfaces:** Produces
`same_depth_table(n_x, n_y, pitch, omega, ref) -> NDArray` of shape
`(2*n_x-1, 2*n_y-1, 9, 9)`, indexed by `(Δx + n_x - 1, Δy + n_y - 1)`, with the
`Δx = Δy = 0` entry set to **NaN** so that a self-term reaching it fails loudly
rather than contributing silently.

- [ ] **Step 1: Write the failing test**

```python
def test_same_depth_table_matches_closed_form_and_poisons_the_self_term():
    ref = ReferenceMedium(5.0, 3.0, 2.5)
    om, pitch = 2 * np.pi * 6.0 * (1 + 0.03j), 0.25
    tab = same_depth_table(4, 4, pitch, om, ref)

    for dx in (-3, -1, 0, 2):
        for dy in (-2, 0, 1, 3):
            got = tab[dx + 3, dy + 3]
            if dx == 0 and dy == 0:
                assert np.isnan(got).all(), "self-term must be poisoned, not zero"
                continue
            want = exact_propagator_9x9(dx * pitch, dy * pitch, 0.0, om, ref)
            assert np.abs(got - want).max() / np.abs(want).max() < 1e-14
```

A zero self-term would let a partition bug pass silently; NaN makes it fail.

- [ ] **Step 2:** Run it; expect FAIL, `NameError`.
- [ ] **Step 3:** Implement. Note the Cartesian-vs-state argument order above.
- [ ] **Step 4:** Run; expect PASS.
- [ ] **Step 5:** Lint, type-check, commit
  `"✨ feat: real-space same-depth propagator table"`.

---

### Task 4: Inter-plane real-space table (class A)

**Files:** Modify `cubic_scattering/pair_propagators.py`; extend its test.

**Interfaces:** Produces
`inter_plane_table(n_z, n_x, n_y, pitch, omega, ref, *, background=None) -> NDArray`
of shape `(n_z, n_z, 2*n_x-1, 2*n_y-1, 9, 9)`. With `background=None` it is the
closed form alone; with a background it is closed form **plus** the reverberation
integrated on Task 2's rule.

- [ ] **Step 1: Write the failing test** — two claims, both needed.
  *(a)* With `background=None`, every entry equals
  `exact_propagator_9x9(Δx·p, Δy·p, Δz·p, …)` to 1e-14.
  *(b)* With a **uniform** background, the table must reduce to case (a) — the
  reverberation vanishes without layering. Target 1e-12. This is the check that
  catches the reverberation being added with the wrong sign or scale, and it is
  the 3-D analogue of the reduction stage 1 pinned at 1.1e-15.
- [ ] **Step 2:** Run it; expect FAIL.
- [ ] **Step 3:** Implement. **The vertical operator does not change** — it is
  `layered_correction.corrected_layered_9x9` on the stratified background. Do
  not rebuild it; the spec's retracted "no external dependency" paragraph
  records what that error cost last time.
- [ ] **Step 4:** Run; expect PASS. **Record wall-clock build time and peak
  memory** and compare against the 149 MB projection. A large miss means Task 2's
  rule was wrong, not that the cost is acceptable.
- [ ] **Step 5:** Lint, type-check, commit
  `"✨ feat: real-space inter-plane propagator table on the layered background"`.

---

### Task 5: `apply_g0_3d`, and the partition gate

**Files:** Modify `directional_sweeps.py`; create `scripts/gate_g0_3d.py`.

**Interfaces:** Produces `SweepGrid3D`, `build_g0_cache_3d(...)`,
`apply_g0_3d(sources, cache)` on `(n_z, n_x, n_y, 9)` states.

- [ ] **Step 1: Write the failing partition test** — a **support count**, not a
  norm. Place a unit source at one site; assert every ordered pair is reached
  exactly once and the self-site never. With the NaN self-term from Task 3, a
  double-count surfaces as NaN rather than as a plausible number.
- [ ] **Step 2:** Run it; expect FAIL.
- [ ] **Step 3:** Implement as the sum of the two real-space applications.
  **Replace the `sweep_y` stub** with a docstring recording that the in-out
  coupling is carried in real space, not as a spectral sweep, and why — a reader
  finding the stub gone will otherwise assume it was simply forgotten.
- [ ] **Step 4:** Run; expect PASS, exact (a count, not a tolerance).
- [ ] **Step 5:** Write `scripts/gate_g0_3d.py` — the full `G₀` against an
  explicit `O(N²)` pairwise sum with **distinct sources at every site**, plus the
  vacuity control showing a uniform source would pass even for an
  implementation that averaged the sites. Target 1e-12 for the homogeneous case;
  for the layered case, Task 2's measured floor, stated explicitly rather than
  rounded.
- [ ] **Step 6:** Lint, type-check, commit `"✨ feat: the composed 3-D G0"`.

---

### Task 6: Amend the spec and the write-up — part of this task, not a follow-up

The documents lag the code, and that is this repository's standing failure mode.
Two of them are now wrong in ways a reader would act on.

- [ ] **Step 1: Amend the spec.** §6's *"Stage 2 then introduces exactly one new
  thing"* and §3.1's six-sweep framing are both false in 3-D. Record the
  measured reason: the sweep accumulator is `(n_z, n_x, n_k, 9)` and a 2-D
  transverse rule makes it 7.6–68 GB, so the six-direction-pure-sweep
  architecture is a **2½-D** result, not a 3-D one. State that `Δz = 0` coupling
  is carried in real space from the closed form in 3-D. Follow the file's
  existing **RETRACTED** convention rather than editing the claim away — the
  spec already retracts one wrong paragraph in place, and that is the house
  pattern.
- [ ] **Step 2: Update `LatexPDFs/DirectionalSweepSolver/DirectionalSweepSolver.tex`**
  — the validation table gains the 3-D rungs, and the "Not yet done" paragraph
  goes. §"What is distinctive" claims the forward operator is "a directional
  partial-wave summation rather than a circular-convolution FFT"; that remains
  true in `z`, but the `Δz = 0` coupling is now a direct real-space sum, and the
  section must say so.
- [ ] **Step 3:** Recompile twice, commit.

---

### Task 7: The two deferred rungs, now unblocked

- [x] **Step 1: Rung 5c — DONE, PASSES at 5e-16.**
  `scripts/gate_rung5c_cross_architecture.py`. Compared at the operator level
  with distinct sources, and localised three ways: slab's kernel against the
  real-space table (6.6e-16), slab's FFT path against a brute-force convolution
  of its own kernel (4.1e-16), the composed operators (5.1e-16).

  > **A wrong finding, recorded rather than deleted.** The first run of this
  > gate reported a 4e-4 defect inside `_slab_matvec`, and it was committed and
  > pushed as such. That was wrong, and the accusation is withdrawn. The cause
  > was **catastrophic cancellation in the gate itself**: `_slab_matvec` returns
  > `(I − G₀T)ψ`, so the gate recovered `G₀ψ` as `ψ − matvec(ψ)`, which loses
  > everything when `|G₀ψ| ≪ |ψ|` — the intermediate is stored to a relative
  > 2e-16 of `|ψ|`, so `G₀ψ` returns with an *absolute* error of `eps·|ψ|`. In
  > SI units this propagator is ~1e-13 against `ψ` ~1, which is exactly the 4e-4
  > observed. In seismic units the same quantities are ~1e+3 against ~1 and
  > everything agrees at machine precision.
  >
  > The tell was a linearity violation: superposing the FFT path's own delta
  > responses reproduced brute force at 3e-29 while differing from its direct
  > random run by 1.2e-16. A linear operator cannot fail linearity, so it was
  > round-off — and round-off that large against 1e-13 values means cancellation
  > upstream.
  >
  > **Generalises past this gate:** extracting a small quantity from
  > `(I − small)` is unsafe, and this package mixes SI and seismic units across
  > modules. The gate now leads with a `[5c-0]` guard asserting
  > `|G₀ψ|/|ψ| > 1e-3` before any other number is believed.

- [x] **Step 2: Rung 7 — DROPPED 2026-09-14, not deferred.**

  The decision and its four measured reasons are recorded in the design spec's
  §7 retraction; the short form:

  1. `FFTProp` has **no `T₀` and no solve** — it propagates given sources. The
     "cylinder Mie `T₀` vs cube `T₀` shape difference" this rung was built
     around does not exist.
  2. Its **free surface does not match** the free-surface reflection — two
     independent arbiters agree to six decimals and disagree with it
     (`scripts/gate_fftprop_free_surface.py`).
  3. Its **cylindrical-harmonic basis is the scaffolding this architecture
     replaced** — the harmonics are the shape of its cylinders entering the
     state vector, not a coordinate choice. We are Cartesian voxels throughout.
  4. **Four better arbiters already pass**: Kennett (1e-15), closed-form
     Kupradze (2.9e-16), `slab_scattering`'s FFT architecture (5.1e-16), dense
     LU and Neumann (2.4e-12).

  The rung's case was provenance. A fifth arbiter reached through a lossy basis
  conversion, against a code with a different discretisation and a suspect free
  surface, would principally measure the difference between two discretisations
  — a real quantity, but not validation of the sweeps.

  **Do not reinstate it without reading the spec retraction**, and note that the
  prerequisites built while investigating it are all keepers:
  `gate_free_surface_reverberation`, `gate_free_surface_magnitude`,
  `gate_gmm_marine_stack`, `gate_fftprop_free_surface`, and the free-surface
  flag itself.

  > **The original method, for reference only.** `docs/plans/2026-09-13-cartesian-directional-sweeps-stage1.md`,
  > Task 8, gives the whole method in five steps: the `importlib` loader (the
  > package directory is literally named `FFTProp.py`, so `import FFTProp`
  > cannot reach it — copy the loader from `gate_lateral_sweep_alg52.py` rather
  > than inventing a second), one physical model at pitches `p, p/2, p/3, p/4`
  > with the voxel count rising to hold the physical extent fixed, equal-volume
  > matching of cylinders to cubes, a four-row table of the difference plus
  > GMRES counts and wall time, and the write-up. **Follow that.** Do not write
  > a new spec for it.
  >
  > **Numbering:** the design spec calls this **rung 4**; the stage-1 plan calls
  > it **rung 7** because it inserted unnumbered rungs, and states the map at
  > its own §"the tails do not line up". Same object.

  **Corrected 2026-09-14, after a survey that had itself been wrong twice.** The
  stage-1 plan and the resume note recorded rung 7 as deferred *because the
  arbiter is 3-D and the comparison would measure geometry rather than sweeps*.
  That is wrong: `FFTProp` is 2½-D by its own README, so stage 1's solver was
  always its natural counterpart. A first correction then claimed three
  blockers; **two of those were also wrong.** What the code actually says:

  1. **No far-field projection module is needed.** Neither
     `directional_sweeps.py` nor `sweep_solver.py` has one, and none is
     required: the observable is a point source-to-receiver response, and the
     receiver leg is the same closed-form propagator the tables already use —
     `u(r_rec) = Σ_voxels P(r_rec − r_v) · T · ψ_v`. `scattered_field.py`'s
     `cube_far_field` is single-site and not the thing to reach for.
  2. **No representation conversion is needed.** `FFTProp` returns `Svec` and
     `Rvec` — source and receiver projections onto the cylindrical-harmonic
     scatterer basis, shape `(Nscatx, 5, 2, Nscatz)`. Contracting the receiver
     projection with the scattered field gives a **scalar** response in which
     the harmonics cancel internally. The 9-component ↔ harmonic bridge that
     spec §4 warns against is simply not on the path.
  3. **The free surface is present but never exercised — this is the real open
     item.** The stage-1 plan says Task 8 is unblocked because "its free surface
     is part of the layered background, which the solver now carries", and that
     is true of `layered_stack_table`. But *every gate written so far damps the
     free surface away on purpose* (`q = 2`, so the uniform reduction can hold).
     Rung 7 needs it ACTIVE and correct, which nothing has yet tested.
  4. **Cylinders against cubes** is the known difference the rung exists to
     measure, not a blocker.

  So the bounded pieces are: an incident field on the 3-D lattice (the pattern
  exists in `slab_scattering._build_slab_incident_field`), the receiver
  contraction above, and **a gate for the free surface in the layered table
  before any of it is believed**.

---

## Self-review

**Spec coverage.** §3.2 (every coupling evaluated where the pair is separated) →
preserved: the closed form is evaluated at nonzero separation and the self-term
is poisoned. §3.3 (no periodicity) → **strengthened**: a real-space table assumes
no periodicity at all, where the original plan's transverse quadrature only
avoided it. §4 (9-component state, no conversion) → every table is 9×9. §8 (fail
fast) → Task 3's NaN self-term and the damping-floor assertions. §3.1 and §6 →
**amended** by Task 6 rather than silently contradicted.

**Placeholders.** None: Tasks 2–5 carry concrete tests and named interfaces.
Tasks 6–7 are documentation and gating, where the content is the argument rather
than code.

**Type consistency.** `same_depth_table` is `(2n_x−1, 2n_y−1, 9, 9)`;
`inter_plane_table` is `(n_z, n_z, 2n_x−1, 2n_y−1, 9, 9)`; both index `Δ` with
the `+n−1` offset; states are `(n_z, n_x, n_y, 9)` throughout.

**The open risk, named.** Task 2. If the reverberation needs the same transverse
rule as the singular kernel, the 149 MB figure is wrong and the architecture
needs revisiting a second time. Task 2 Step 3 says to stop and report in that
case rather than raise the node count until the number looks acceptable — which
is the failure mode that produced the first version of this plan.
