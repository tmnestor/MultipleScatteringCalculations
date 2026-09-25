# The Voxel Generator: a Riccati Equation from One Layer of Cubes — Plan

- **Date:** 2026-09-25
- **Status:** PROPOSED. Nothing below is built.
- **Idea:** The Redheffer star product (Kennett's addition rule) of a slab of thickness
  `h` with everything below it gives, at `O(h)`, a matrix Riccati equation
  whose coefficients are the slab's generator. If the slab is **one layer of
  space-filling cubic voxels**, each carrying its analytic `T0` and coupled by
  the in-plane lattice sum, the same expansion gives a *voxel* generator
  `A_vox(h)`. Comparing `A_vox(h)` with the continuum operator `Aop` entry by
  entry gives an operator-level link between the two routes this project
  already has. Today they are compared only through end results.

---

## What already exists — DO NOT REBUILD IT

Survey run 2026-09-25 across `cubic_scattering/`, `scripts/`, `Mathematica/`,
`LatexPDFs/`, `docs/`, `GlobalMatrix/` and the thesis `.tex`.

| Thing | Where | Status |
|---|---|---|
| Continuum first-order operator `Aop` (3-D Cartesian, isotropic), symplectic under `J6` | `FirstOrderContrastOperator.tex` §3–4; gate binds numerics to derived `Aop` | validated vs thesis `(Akdef)` and published matrices |
| **The continuum Riccati march**, `Y' = A21 + A22 Y − Y A11 − Y A12 Y`, in impedance (DtN) form | same doc §`sec:impedance`, `scripts/gate_sphere_vs_impedance_march.py` | validated; fourth order in depth. This **is** Haines, Hulme & Yu (2004), see §`sec:priorart` |
| Laterally coupled `Aop` (lateral grid carried inside `Y`, 2½-D and 3-D) | same doc §`sec:latop`, §`sec:threed` | internally validated; **no external arbiter** (§`sec:open` item 9) |
| Block-Riccati up/down sweeps for the stratified propagator | `~/Desktop/SeismicInversion/GlobalMatrix/riccati_solver.py`, `block_riccati_cluster.py` | parity vs `kennett_layers` 4.4e-15 |
| Kennett recursion (addition rules) for layer stacks | `cubic_scattering/kennett_layers.py`, `PhD_fortran_code/Kennett_Reflectivity/` | validated |
| Periodic voxel Foldy–Lax slab, specular **reflection** matrix | `slab_scattering.slab_reflection_matrix`, `slab_weyl_amplitudes` | validated vs Kennett at ~0.5–1% (moderate contrast) |
| Ewald lateral lattice sum + volume-averaged contact blocks | `build_slab_kernels(..., lattice_ewald=True, volume_averaged=True)`, `planar_ewald.py`, `inter_voxel_propagator.py` | validated; the truncated sum was a months-long defect, so **always** use Ewald |
| Cube `T0` (ordinary Navier route) and Eshelby concentration factors | `effective_contrasts.py`, `cube_eshelby.py` | validated |
| First-order-route `T0` (four amplification factors) | `FirstOrderContrastOperator.tex` §`sec:tmatrix`, built in the gate | measured; not shipped in the package |
| End-to-end voxel-vs-reference comparison on a **space-filling patch** | same doc §`sec:whichbetter` | `R_PP` error 2e-3 / 2e-2 / 5e-2 at weak / moderate / strong, `N_z=16`, one ω, `p=0`, unconverged lattice |
| Voxel Foldy–Lax vs march, on the Mie sphere | `scripts/gate_imbedding_vs_foldy_lax.py`, §`sec:vsfoldy` | 19/19 |

**Closest prior art — read it first.** Tromp & Snieder (1989), *Geophys. J.*
96, 447–456, in `reference_papers/` as
`The_reflection_and_transmission_of_plane_P-_and_S-.pdf`. It is this plan's
construction for a **laterally homogeneous** thin layer: the first-Born R/T of a
thin homogeneous layer, per unit thickness, plus invariant imbedding give
matrix Riccati equations for R and T. The voxel version replaces "thin
homogeneous layer, first Born" with "one layer of cubes, Foldy–Lax", and adds
lateral structure. `FirstOrderContrastOperator.tex` cites it only "via the
thesis", under year 1988; the published year is 1989.

⚠ **Their central step is the one Task 0 tests.** They state that the n-th
order Born term of the thin layer is `O(Δz^n)`, so first Born is exact at
`O(Δz)`. Their Born series (their eq. 2.7) carries `∂_z` and `∂_{z0}` of the
plane-wave Green's tensor, and `∂_z ∂_{z0} G` contains `δ(z − z0)`. So for a
**stiffness** perturbation, the second-order term appears to be `O(Δz)`, not
`O(Δz²)`. That is the same `O(1)` plate local field (normal-traction
continuity) described below. If it holds, their coefficients are linear in
`c_s` where the exact generator is not (e.g. `1/(λ+2μ)`), and their Riccati
equations are exact only for density perturbations or to first order in
stiffness contrast.

**Measured 2026-09-25 in two independent implementations:**
`scripts/gate_tromp_snieder_uniform_band.py` (21/21, reference `kennett_layers`)
and `Mathematica/TrompSniederBornGenerator.wl` (45/45, reference exact
`MatrixExp` band propagators; coefficients transcribed term by term from the
paper; eqs 3.6a–d integrated at 32 digits). The Mathematica check also
**proves the mechanism at every slowness**:
- their eqs 2.12/2.15 equal, exactly, the interaction-picture blocks of the
  contrast-linearised system matrix `A_lin`. That covers isotropic and random
  anisotropic `c_s`, with a flipped-sign control caught at 0.28.
- their band response equals the exact response of the `A_lin` band to
  ≤ 1e-20.
- `∂²A/∂ε² = 0` in the density direction and ≠ 0 in the stiffness direction.
The Mathematica errors against its propagator agree with the Python errors
against Kennett to ≤ 0.27%, which is within the rounding of the printed
3-figure values.

Results for a uniform 100 m band:
- **Density-only band:** exact, ≤ 3e-13, at three slownesses. This validates
  the harness.
- **Stiffness-only, ε → 0:** relative error ∝ ε, with log-log slope
  1.03–1.07. They are right at Born order.
- **Finite contrast: not exact.** The project's moderate contrast gives
  2.5–5% error in `RPP`/`RSS`/`RSH` and 6–7% in `RPS·RSP`. At ±20% moduli
  the error is 15–30% in the diagonal channels, and up to 160% in `RPS·RSP`.
- **Mechanism, confirmed:** at `p = 0` their result equals Kennett **exactly**
  (≤ 1.3e-13) for a band with linearised compliances,
  `M_eff = M0²/(M0 − Ms)` (for both `λ+2μ` and `μ`). Their equations are the
  exact equations of a *different medium*.
The O(1) plate local field that Task 0 tests is therefore real, and
first-order Born of a thin layer misses it.

**What is genuinely missing:**

1. The voxel generator `A_vox(h)` itself. Nothing extracts a per-unit-depth
   operator from a one-layer voxel slab.
2. **Transmission** from the periodic voxel slab. `slab_weyl_amplitudes`
   returns outgoing (reflected) amplitudes only. The forward Weyl sum and
   incidence from below are not implemented (grep: no `transmi` or `forward` in
   `slab_scattering.py`). Recovering a 6×6 propagator needs all four blocks.
3. Any operator-level comparison between the voxel route and `Aop`.

---

## The mechanism, and the prediction it makes

For a slab of thickness `h`, the state propagator is `Π(h) = I + h A + O(h²)`
in the `q = (u_z,u_x,u_y,T_zz,T_xz,T_yz)` basis. The same blocks feed both
Riccati forms: the impedance march `Y` (already built) and the reflection
Riccati `dR/dz = M_ud + M_uu R − R M_dd − R M_du R` in the wave basis
`M = D⁻¹AD − D⁻¹∂_z D`. So the comparison is made on `A`, and the choice
between `Y` and `R` never arises.

**Why the in-plane coupling does not vanish at O(h).** A cube's `T0` scales as
`h³` and there are `h⁻²` cubes per unit area, so the slab is `O(h)`, as needed.
But the neighbour interaction is `1/r³`, and its sum over the plane is `O(1)`
per cube. So `A_vox` at `O(h)` contains the cube's self-term **plus** the
whole in-plane sum. Together these must reproduce the local field of a
**thin plate**, not of an isolated cube.

**Static identity (knowable answer).** For a uniformly polarised plane of
identical cubes:

    S_self + Σ'_j Ḡ_ij  =  S_plate

where `S_self` and `Ḡ_ij` use **the same** test/source convention. This holds
because summing every cube rebuilds the plate, and the plate's static field is
**uniform inside it**, so evaluating it at the receiver's centre (collocation)
and averaging it over the receiver (Galerkin) give the same `S_plate`. Both
schemes satisfy the identity. **Mixing them does not.** If it holds, then in
the static, laterally uniform limit the Foldy–Lax solve is **exact**:
`(I − Σ'Ḡ T0)` with `T0 = ΔC (I − S_cube ΔC)⁻¹` reduces the internal field
to `(I − S_plate ΔC)⁻¹ ε0`. Isotropic thin-plate Eshelby tensor (oblate limit;
**confirm independently, do not take from here**):
`S_3333 = 1`, `S_3311 = S_3322 = ν/(1−ν)`, `S_1313 = S_2323 = 1/2`, all other
entries 0.

**Prediction.** For a laterally uniform slab of identical cubes,
`lim_{h→0} A_vox(h) = Aop(λ+Δλ, μ+Δμ, ρ+Δρ)` holds **exactly**, including
the entries that are nonlinear in the contrast (`ζ`, `χ`, `γ`,
`1/(λ+2μ)`). Any residual after `h → 0` is therefore a defect or a convention
mismatch, not physics. Finite-`h` residuals should fall as `O((kh)²)`.

**Why this matters.** §`sec:whichbetter` reports 2–5% `R_PP` error on a
patch that, by this argument, should become exact as `h → 0`. The generator
comparison splits that aggregate into (a) the static plate identity,
(b) dynamic `O((kh)²)` corrections, (c) lattice-sum truncation, and
(d) the `T0` route. It does this **per entry of `A`**, so it names the channel.
The doc places the ordinary-vs-first-order difference "entirely within the
traction channel" at `O(Δc²)`. The rows of `A_vox` should show the same thing
directly.

---

## Task 0 — the static plate identity (cheap, decisive, do first)

**Why first:** it is static, needs no extraction machinery, and has an exact
answer. If it fails, every later task inherits the failure. It is also an
**exact arbiter the project does not yet have** for the averaging-convention
question in the collocation settlement scripts.

**What the survey found (2026-09-25).** Read before building.
- The solver is a **collocation** scheme, settled in
  `scripts/settle_single_site_formulation.py` and
  `scripts/settle_collocation_everywhere.py`: the state is the point value at
  the cell centre, sources carry a volume moment. The consistent propagator is
  the **single** (source-cell) average at the receiver centre, at **every**
  separation, paired with the collocation `T0` as the self term.
- `inter_voxel_propagator.py` constants are the **double** (Galerkin)
  average, `∫∫ ∂²G`. `cell_averaged_lattice.averaged_same_plane_9x9` is the
  **single**-average same-plane Ewald sum with an analytic `O(d²)` tail.
- In `slab_scattering`, `contact_average` defaults to `'single'`, but by
  default only the **contact shell** is averaged. Beyond it the kernel is the
  bare midpoint value unless `va_all=True`. The library's own docstring calls
  the midpoint "a scale-invariant bias that refinement cannot remove".

**Prediction, per convention.** Because the slab field is uniform inside:
| self term | in-plane sum | plate identity |
|---|---|---|
| collocation `T0` self | single average at every `R` (`va_all=True` + tail) | **exact** |
| collocation `T0` self | single on contact, midpoint beyond (the default) | off by the midpoint bias, **not** reduced by refinement |
| Galerkin self | double average at every `R` | **exact** |
| mixed | mixed | off at `O(1)` |

The default row is the one that matters. If it misses `S_plate`, the default
solver cannot converge to `Aop` as `h → 0`, and the §`sec:whichbetter` 2–5%
contains a piece that no refinement removes.

**Steps 1–2 DONE 2026-09-25: `Mathematica/PlateIdentity.wl`, 14/14.**
- `μ Γ_plate` has only the traction entries: `zz = 9/25` (i.e. `μ/(λ+2μ)`),
  `yz = xz = 1/4`, all in-plane entries 0. The slab route and the thin-plate
  Eshelby values (`S_3311 = ν/(1−ν) = 7/25`, `S_1313 = 1/2`) agree exactly.
  The oblate Hill tensor converges to it at first order in aspect ratio, and
  its `e = 1` case reproduces the sphere.
- The Fourier-space sum is exact for both consistent schemes: the cube form
  factor kills every nonzero in-plane order, and `Γ̂(0,0,k_z)` is
  independent of `k_z`.
- **Real-space collocation closes.** The self term is the cube's centre-point
  field, exact: `Ψ_1122(0) = −4/√3`, `Ψ_1111(0) = 8(√3−π)/3`. Adding the
  source-averaged plane sum reproduces `Γ_plate` to 1.4e-7, which is the floor
  of the continuum tail beyond `R = 400`.
- **The default pairing does not close.** Averaging on the contact shell and
  using bare midpoints beyond leaves a static, `d`-independent bias of
  **2.1% of `Γ_plate`**. Refining the lattice spacing cannot remove it. It
  falls to 0.51% / 0.19% / 0.090% when averaging is extended to Chebyshev
  radius 2 / 3 / 4.
  The bias also puts **spurious in-plane entries** (`xx`, `yy`, `xy`,
  `xx–zz`) into an operator whose exact in-plane entries are zero. In `R` it
  enters at `O(Δc²)`, because it multiplies two `T0`s. Its share of the
  §`sec:whichbetter` error is for Python step 3 and Task 2 to measure, not
  to assume.

**Step 3 DONE 2026-09-25: `scripts/gate_plate_identity.py`, 9/9.** It tests
the solver's **own** `dz = 0` kernel at Bloch `k∥ = 0` (`build_slab_kernels`,
`M = 1`, `d = 1`, `k_S d = 1e-3`) against `Γ_plate − Γ_self`:
- **Convention:** calibrated rather than assumed. The solver's strain block
  is `−e_b × tensor`, with the engineering factor `e = (1,1,1,2,2,2)` on the
  source index. That matches at two separations to within the dynamic part
  (≤ 9e-5 from integers).
- **Ewald route, `exact_cell_average` (the auto default with
  `volume_averaged=True`): exact in the limit.** The residual is 3.7e-4 at
  the library defaults (`cell_avg_r0=2`, `cell_avg_gauss=6`). At quadrature
  order 12 it falls as 9.2e-6 → 1.7e-6 → 3.4e-7 for `cell_avg_r0 = 4, 6, 8`,
  which is the documented `O(d⁴)` tail truncation. The static limit is
  reached (5e-7).
- **`exact_cell_average=False`, contact shell only:** 2.145% bias. It
  matches Mathematica entry by entry to 5.4e-7 in `μΓ` units.
- **`va_all` to radius 2 / 3 / 4:** 0.506% / 0.189% / 0.090%, matching
  Mathematica.

**Consequence.** The current default Ewald configuration satisfies the plate
identity, so the sheet-of-voxels generator can be built on it. The 2.1% bias
lives only in configurations that switch the exact cell average off:
`exact_cell_average=False`, the non-Ewald route without `va_all`, and any
run made before the auto default existed. Whether §`sec:whichbetter`'s
measurement used such a configuration has **not been checked**. Do that
before attributing any of its 2–5% to this bias.

**Still open in Task 0:** the self term. The gate uses the exact collocation
self field (`Ψ_1122(0) = −4/√3`), not the one the solver's `T0` actually
carries. Confirm that `compute_cube_tmatrix`'s static limit implies the same
`Γ_self`. If it does not, the pairing is mixed even on the default route.

**Steps**
1. **Mathematica.** `S_plate` for an isotropic matrix, two ways that must
   agree: as the uniform field of a uniformly polarised infinite slab, from
   `Γ(n)_ijkl = sym (n_j n_l K⁻¹_ik)`, `K = c n n`; and as the oblate-spheroid
   Eshelby tensor with aspect ratio → 0.
2. **Mathematica.** The same-plane sum in the **spectral** form. Only the
   `G∥ = 0` order survives a uniform in-plane source, so the sum reduces to a
   `k_z` integral of the plane-wave `Γ̂(0,0,k_z)` weighted by the cube form
   factor: one power of `sinc` for single averaging, two for double. Both
   must return `S_plate` exactly, since `Γ̂` is degree-0 in `k`. This gives an
   independent closed form for the whole in-plane sum, with no real-space
   lattice work.
3. **Python.** The **solver's own operator**. Build the same-plane static (or
   `ω → 0`) strain block the way `slab_scattering` does, under each row of the
   table, sum it over the plane (the `k∥ = 0` component of its Bloch
   transform), and add the self term the solver pairs with it. Compare with
   `S_plate` entry by entry.
4. The **Galerkin** row, from `inter_voxel_propagator` constants plus a
   double-average tail, as a second route to the same exact answer.

**Acceptance:** Mathematica, both `S_plate` routes and both spectral sums
exact. Python: the collocation-everywhere and Galerkin rows at 1e-10 (or the
tail's stated `O(d⁴)` floor, measured); the default row's residual
**reported, not tuned**, and checked to be independent of lattice refinement.

**Traps**
- ⚠ **The self term must match the neighbour convention.** Find out exactly
  which static self term the collocation `T0` carries: the centre-point field
  of a uniform cube source, *not* the cube-averaged one. Check it against
  `settle_single_site_formulation.py` **before** interpreting a mismatch.
  §`sec:open` item 3's geometric factor `∫_V 1/r / D₀ = 1.2644` is the
  single-vs-double gap for the scalar kernel. It is the size to expect for a
  mixed-convention error.
- ⚠ `_cell_averaged_propagator` defaults to `double=True`. Omitting
  `double=False` silently reintroduces the Galerkin average; this has
  happened before (`TestVaAllAveragingConvention`).
- ⚠ The Eshelby δ-function at `r=0` (known pitfall). The self-term must carry
  it, and the neighbour sum must not double-count it.
- ⚠ Watch the Voigt engineering factor on the shear rows. It has been the
  defect here before (`H` rows exactly 0.5×).

## Task 1 — voxel slab transmission (the missing extractor)

**Steps**
1. Add a forward Weyl sum to `slab_scattering` (downgoing outgoing amplitudes
   below the slab). Mirror `slab_weyl_amplitudes`, with the sign of `k_z`
   reversed and the incident field added back on the specular order.
2. Add incidence from below: either reflect the geometry (`z → −z` on a
   laterally uniform slab), or solve with an upgoing incident field.
3. Check with a knowable answer: at `N_z = 16`, `T` and `R` from both
   sides must satisfy flux-normalised unitarity (`ω` real, no loss) and
   reciprocity. These are arbiter-free. Then score them against
   `kennett_layers` for the equivalent homogeneous layer, as §`sec:whichbetter`
   does for `R`.

**Acceptance:** unitarity and reciprocity at the lattice-error floor;
`T_PP` error vs Kennett comparable to the existing `R_PP` error.

## Task 2 — extract `A_vox(h)` for a laterally uniform slab

**Steps**
1. `N_z = 1`, space-filling `M×M` patch of identical cubes, side `h = 2a`.
   Sweep `h` over a ladder (≥5 values, `kh` from ~0.3 down to ~0.02).
2. Assemble `S(h)` from Task 1, convert to `Π(h)` in the `q` basis using the
   **background** eigenvector matrix `D`. Reuse what `layered_correction`
   and `kennett_layers` already hold; do not rederive it.
3. `A_vox(h) = log(Π(h))/h`, and separately `(Π(h) − I)/h`, as an
   extraction-error control.
4. Compare with `Aop` of the **cube's** material, entry by entry, for P-SV
   and SH, at `p = 0` and at least two `p > 0`, including one past the S
   critical angle.
5. Run with **both** `T0` routes (ordinary and first order).

**Acceptance:** Richardson-extrapolated `A_vox(0) = Aop` within the lattice
floor, with an observed order near 2 in `kh`. Report the entry-wise residual
table per `T0` route. The predicted pattern is that residuals concentrate
in the traction rows for the ordinary route.

**Traps**
- ⚠ Convention mapping (displacement vs Kennett-modified amplitudes,
  `SlabReflectionMatrix.to_modified()`, the `(−iω)` factors fixed in
  `layered_correction`) can masquerade as physics. Do the mapping through
  the existing validated wrappers, and prove it on a **zero-contrast** slab
  first: `A_vox` must equal the background `Aop` identically.
- ⚠ `T0` and the lattice must be refined together. At fixed `M` the lateral
  lattice error does not vanish as `h → 0`, so converge `M` at each `h`.

## Task 3 — laterally varying slab, against the laterally coupled `Aop`

**Why:** this is the only task that gives the laterally coupled march
(§`sec:open` item 9, "no external arbiter") something built outside the
march to agree with.

**Steps**
1. A two-material checkerboard of period `L`, one layer thick. Extract
   `A_vox(h)` as a matrix over diffraction orders `g` × (P, SV, SH),
   non-specular orders included. This needs a non-specular Weyl extractor:
   check what `gate_sphere_vs_impedance_march` already scores per order
   **before** writing one.
2. Transform the laterally coupled `Aop` of §`sec:latop` / §`sec:threed` to
   the same `g` basis.
3. Compare order-coupling blocks as `h → 0` at fixed `L`.
4. Check `J6 A_vox(k_y) = [J6 A_vox(−k_y)]^{T_x}` (§`sec:latop` eq.
   `latsymp`). This is arbiter-free and must hold at every `h`.

**Acceptance:** order-coupling blocks converge to `Aop`; the symplectic
identity holds to the solve tolerance.

## Task 4 — write it up

The deliverable is the `.tex`. Proposed: a section of
`FirstOrderContrastOperator.tex`, "The voxel generator", sitting between
§`sec:whichbetter` and §`sec:vsfoldy`, which it links. Include:
- the thin-slab star-product → Riccati derivation (Redheffer 1959/60;
  Kennett 1983 Ch. 6 for the addition rules; Bellman & Kalaba / Haines et
  al. for the continuum). Citations are in
  `LatexPDFs/redheffer_star_product.bib`.
- the static plate identity, and the entry-wise residual tables.
- an update to §`sec:open` items 4 and 9 for whatever Tasks 2–3 close.

Note for the write-up: Kennett (1983) never cites Redheffer, and treats
gradient zones with Airy-function propagators rather than a Riccati
equation. Checked by full-text search of the 2009 reissue. Present the two
as independent arrivals at the same algebra.

---

## Standing cautions

- **Grep for the mechanism, not the physics.** Extractors and sums are named
  `weyl`, `ewald`, `fft`, `circulant`, not "transmission" or "generator".
- **Never fit a correction against the arbiter it is then scored on.**
- **One ω, one `p`, one contrast is an anecdote.** Every acceptance table
  above is a ladder.
