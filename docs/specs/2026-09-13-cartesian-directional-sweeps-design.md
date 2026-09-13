# Cartesian Directional Sweeps — Design

- **Date:** 2026-09-13
- **Author:** Tod Nestor
- **Status:** design, awaiting review
- **Supersedes as an approach:** the FFT-convolution kernel path of `slab_scattering.py` for the stratified-background problem. That solver stays in place, validated, and becomes a cross-check.
- **Prior art this implements:** `LatexPDFs/DirectionalSweepSolver/DirectionalSweepSolver.tex`

---

## 1. What already exists

This section is first because the recurring failure in this project is designing
without it. Every row below was verified by grep or by running the gate.

| Piece | Where | Validated to |
|---|---|---|
| Four-directional sweeps, cylindrical 2½-D | `FFTProp.py/propagation.py`: `upsweep`, `downsweep`, `right_sweep`, `left_sweep` | faithful port of the thesis Fortran `FFTPROP.F` |
| Sweep resummation == direct pairwise sum | `scripts/gate_lateral_sweep_alg52.py` | **2.4e-16**, distinct source per site, with a control showing a uniform-source test would be vacuous |
| Up/down Kennett recursion | `cubic_scattering/kennett_layers.py` | 4.42e-15 parity vs the GMM block-Riccati |
| Lateral partial-wave kernels | `horizontal_greens.post_kx_residue_kernel_9x9_vec`, `post_ky_residue_kernel_9x9_vec` | P and S poles separate; branch pinned `Im ≥ 0` |
| Exact pairwise lateral coupling | `horizontal_greens.horizontal_greens_direct` | the arbiter for rungs 2 and 6 |
| Cube `T₀` | `effective_contrasts.py`, `resonance_tmatrix.py` | vs Mie, including the c₄ form factor |
| GMRES Foldy–Lax | `slab_scattering.py`, `sphere_scattering_fft.py` | vs Kennett (0.5–1%) and vs Mie |

**The gap:** no Cartesian directional-sweep implementation exists. `git grep -ril
"sweep"` over `cubic_scattering/` and `scripts/` returns nothing of the kind.

**The delta is small and specific.** `FFTProp` is cylindrical 2½-D and therefore
needs only four sweeps, because `y` is invariant there. Cartesian cubic voxels
need **six** — the in-out `k_y` pair is the one genuinely new piece — and `T₀`
becomes the cube T-matrix rather than a cylinder Mie coefficient. The cylindrical
apparatus of thesis Chapter 5 (`D_x`, `DFT(k_x)`, the `Θ^z_in/ou` operators of
Box 5.2) exists only because the thesis had a cylinder T-matrix; in Cartesian it
is not needed.

## 2. The operator

Foldy–Lax, solved by a Krylov method:

```
(I − G₀T₀) ψ = ψ^inc ,     T = T₀ (I − G₀T₀)⁻¹
```

`T₀` is block-diagonal and local — the cube T-matrix, self-term already closed.
All cost is the off-diagonal `G₀`. One GMRES iteration is one `T₀` followed by
one `G₀` matvec.

**`G₀` is a pure forward summation.** No inversion, no embedding, no
reverberation. Every order of multiple scattering is built and resummed by the
Krylov iterations. This division of labour is the point of the architecture: the
propagator stays a clean, stable, one-pass accumulation and the solver carries
the contrast.

## 3. `G₀` as directional sweeps

### 3.1 The three directions

1. **up-down (`k_z`)** — between depth planes, by the Kennett reflectivity
   recursion on the layered background. The only sweep carrying reflection and
   transmission coefficients, and it carries them for the *background* layering,
   not the heterogeneity. Coefficients depend on the background alone and are
   **formed once, outside the Krylov loop**.
2. **left-right (`k_x`)** — along each constant-`z` row. The in-plane propagator
   is split on the `k_x` pole into right- and left-going partial waves, each
   summed by a sweep in its own direction:

   ```
   a⁺_{n+1} = (a⁺_n + s_n) · Φ        (left to right)
   a⁻_{n−1} = (a⁻_n + s_n) · Φ        (right to left)
   ```

   with `Φ = e^{i k_pole p}` the one-pitch propagator and `p` the voxel pitch.
   P-pole and S-pole are carried separately.
3. **in-out (`k_y`)** — identical, along each constant-`(z,x)` line, on the
   `k_y` pole.

Six direction-pure passes in all.

### 3.2 Why these directions

The conventional construction sums in-plane coupling on the `k_z` pole — the
up/down decomposition — whose surviving `(k_x,k_y)` integral loses its
convergence factor `e^{−κ_⊥|Δz|}` at `Δz = 0` and **diverges at equal depth**.
Splitting laterally sums the in-plane coupling on the `k_x` and `k_y` poles,
along which distinct in-plane voxels *are* separated, so the divergent
same-depth sum is never the one evaluated. The `k_z` sweep is used only between
planes, where `Δz ≠ 0` and the recursion is exact and stable. **Every sweep
marches along an axis of nonzero separation.**

Stability is structural rather than bookkept: each partial wave is carried in
the direction in which it decays — the right-going evanescent branch
`e^{−κ_{x⊥}x}` swept right, the left-going branch left — so the growing
exponential is never formed. The existing kernels already pin the branch with
`Im ≥ 0`, which is the same choice.

### 3.3 What this is not

Not a circular-convolution FFT kernel; not a lattice sum; not a Poisson or Ewald
construction; not a Sommerfeld inverse transform. A running phase accumulation
along a line involves no convolution and therefore **carries no periodicity
assumption** — which matters, because the real Earth is not horizontally
periodic.

## 4. Representation

Every sweep carries the 9-component state `(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy,
2ε_xy, 2ε_zy, 2ε_zx)`, which is what both residue kernels already return and
what `T₀` already acts on. **No representation conversion occurs at any sweep
boundary.**

This is deliberate and costs memory: 81 complex numbers per site per accumulator,
against 3 per direction for a mode-amplitude scheme. The reason is that
conversions between representations are where this project's defects have
actually lived — all three defects resolved in the 9×9 wrapper work
(`docs/wrapper_problem_state_2026-09-13.md`) were conversion-convention errors,
and one of them survived months because a symmetry gate happily passed it. The
memory is worth the class of bug it removes.

## 5. Module layout

| File | Responsibility |
|---|---|
| `cubic_scattering/sweep_kernels.py` *(new)* | The amplitude/phase split: per pole, the distance-independent 9×9 amplitude and the scalar one-pitch phase. Pure NumPy, no solver dependencies. |
| `cubic_scattering/directional_sweeps.py` *(new)* | `sweep_x`, `sweep_z`, and `apply_g0` (the ordered composition). `sweep_y` added at stage 2. |
| `cubic_scattering/sweep_solver.py` *(new)* | GMRES around `apply_g0`, with `T₀` from the existing cube machinery. |
| `cubic_scattering/horizontal_greens.py` *(modify)* | Expose the split. Existing bundled entry points unchanged, so current callers and gates are untouched. |

Three small focused files rather than growing `slab_scattering.py`, which is
already ~1200 lines and implements the *different* architecture this supersedes.
Keeping them apart leaves that solver intact as a cross-check.

**RETRACTED 2026-09-13.** This paragraph previously read: *"No
external-repository dependency. The vertical sweep uses
`cubic_scattering.kennett_layers`, which is in-package. The stratified spectral
Green's function from `layered_correction` — and hence the sibling
`GlobalMatrix` repo — is not needed by this architecture at all."*

**That was wrong, and it was the root error of this spec.** The stratified
plane-to-plane propagator already exists, in the *identical* 9-component basis
this architecture uses:

| Object | Where | Validated to |
|---|---|---|
| `layered_greens_9x9(model, omega, kx, ky, source_iface, receiver_iface)` | `GlobalMatrix/layered_greens.py:838` | returns `(u_z,u_x,u_y,ε_zz,ε_xx,ε_yy,2ε_xy,2ε_zy,2ε_zx)` — component-for-component the sweep state |
| `corrected_layered_6x6(...)` | `cubic_scattering/layered_correction.py` | the three wrapper defects corrected; GATE D 5.1e-16, interior planes 1.7e-15 |
| `Q^∂ = (I − S_int E)⁻¹ S_int`, source-in-stack `V_inc` | thesis Ch.5 `GstratRep.tex`, Eq. `PstratDef`, `incdown`/`incup` | the derivation |

The vertical sweep therefore takes its plane-to-plane kernel from
`corrected_layered_6x6`, with zero-contrast pseudo-interfaces inserted at each
scattering-plane depth — the layer-interior case that
`assert_interface_continuous` is written to permit. The in-package whole-space
`sweep_kernels.vertical_kernel_9x9` is retained as the **homogeneous-limit
arbiter**, not as the production operator.

The entire 9×9 wrapper arc that preceded this spec existed to make that object
usable for exactly this composition. A design that deletes a dependency the
recent work was built to create is a signal to re-survey, not to proceed.

### 5.1 The amplitude/phase split

`post_kx_residue_kernel_9x9_vec(ky_arr, kz, dx_abs, …)` currently returns
amplitude × phase bundled at a given separation. A sweep needs them separated:
the polynomial-in-`k` amplitude applied once per site, and the exponential
accumulated along the line.

**This is the single most dangerous item in the design.** Accumulating a factor
that should not accumulate yields a plausible field that is wrong by a
distance-dependent factor — and a symmetry or reciprocity check will pass it,
because such checks are homogeneous of degree one and blind to exactly this.
Rung 1 of the ladder exists solely to pin it.

## 6. Sequencing

**Stage 1 — 2½-D (four sweeps).** `y` invariant, `k_y` a parameter. Up, down,
left, right. Arbitrated directly against `FFTProp`, which is a faithful
implementation of precisely this geometry.

**Stage 2 — 3-D Cartesian (six sweeps).** Add the in-out `k_y` pair, arbitrated
against the `horizontal_greens` pairwise sum in three dimensions.

Stage 1 first because every one of its parts has a validated antecedent, so a
failure localises immediately. Stage 2 then introduces exactly one new thing.

## 7. Validation ladder

Each rung states a claim with a knowable answer and names an arbiter that
already exists.

| Rung | Claim | Arbiter | Target |
|---|---|---|---|
| 1 | amplitude × phaseⁿ reproduces the bundled kernel at separation `n·p` | `horizontal_greens` itself | 1e-15 |
| 2 | lateral sweep == direct pairwise sum, **distinct source at every site** | `horizontal_greens_direct` | 1e-15 |
| 3 | vertical sweep == Kennett between the same planes | `kennett_layers` | 1e-14 |
| 4 | full 2½-D `G₀` matvec | `FFTProp` | see below |
| 5 | GMRES solve, homogeneous background | `slab_scattering` (validated vs Kennett) | ≤1% |
| 6 *(stage 2)* | in-out `k_y` sweep, 3-D lateral coupling | `horizontal_greens` pairwise, 3-D | 1e-15 |

**Rungs 1–3 and 6 are exact**: the same physics through two code paths, so
anything above ~1e-13 is a defect, not a discretisation difference.

**Rung 4 is not exact and must not be written as though it were.** `FFTProp`
discretises the heterogeneity as cylinders with a cylinder Mie `T₀`; this
implementation uses cubic voxels with the cube `T₀`. The two therefore disagree
by a genuine shape-and-scatterer difference even when both are correct. The rung
is passed by demonstrating **convergence of the difference under refinement**,
not by hitting a fixed tolerance: as the voxel pitch falls at fixed physical
contrast, the discrepancy must fall toward the known equal-volume shape
difference rather than plateauing at an arbitrary level or growing. A single
run at one pitch proves nothing here. If that refinement behaviour cannot be
demonstrated, the rung fails and the cause is the sweeps, not the shape.

**Rung 2 carries a mandatory control.** The source must differ at every site —
that is the disorder-resolved property being claimed — and the gate must also
demonstrate that a uniform-source version would pass even for an implementation
that silently averaged the sites. `gate_lateral_sweep_alg52.py` already does
both; the Cartesian gate inherits them.

**Every gate runs serially.** `frequency_sweep.sweep_frequencies` agrees with
the serial path only to round-off (~2e-16 relative), which is below these
tolerances but not by a wide margin.

## 8. Error handling

Fail fast, with the four-element diagnostic the project requires (what is wrong,
where to fix it, what a valid value looks like, how to recover):

- a sweep asked to run on a lattice of fewer than two sites along its axis;
- a pitch that is zero or negative;
- a frequency of zero, where the partial-wave decomposition is undefined;
- a request for `sweep_y` before stage 2 lands — an explicit `NotImplementedError`
  naming the stage, never a silent omission of a coupling term.

No silent fallbacks and no defaulting: a missing input raises rather than being
inferred.

## 9. Out of scope

- The kernel-build vectorisation of `_propagator_block_9x9`. That belongs to the
  FFT-convolution architecture, which this supersedes; the profile that motivated
  it is recorded in `docs/plans/2026-09-13-stratified-wrapper-correction.md`.
- Preconditioning for strong contrast. The design note anticipates it; it is a
  Krylov-side concern, transparent to the sweeps, and adds nothing until the
  sweeps are validated.
- GPU execution.
- Any change to `slab_scattering.py` beyond leaving it untouched.
