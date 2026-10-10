# The Stratified Reference in Three Dimensions for the Legendre-Cell Voxel — Plan

**10 October 2026.** Status: plan, no code yet. This replaces the first version of this file, written
earlier the same day. That version was built on the stage-1 sweep: one point 9-vector per voxel, at a
fixed `k_y`, over two dimensions. **This plan is three-dimensional, and its unknowns are those of
Papers 1 and 2.**

Conventions, as in the package: seismic units (km/s, g/cm³, GPa, km), time `e^{−iωt}`, every vertical
wavenumber on `Im ≥ 0`, z = axis 0 (down), x = 1, y = 2, mode order (P↓, SV↓, SH↓, P↑, SV↑, SH↑),
9-component state `(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)`.

---

## Progress

**10 October 2026: G1 and G2 pass** (`scripts/gate_march_modes.py`, module
`cubic_scattering/stratified_march.py`). Mathematica is not installed in the session container; these
were verified numerically against existing exact arbiters, and the symbolic derivation of §5.1 is still
owed.

- **G1.** The whole-space plane-to-plane spectrum is exactly `Σ_m w_m e^{ik_z,m|dz|} d_m d_mᵀ M`, with
  `d_m = mode_state(k_m, e_m)`, `w = i/(2ρc²k_z)` and `M = diag(1,…,1, ½, ½, ½)`. So the source map is
  the receiver map transposed, as reciprocity says, with no other factor (`W = diag w`). The worst error
  is 3e-15 against `vertical_kernel_9x9`, over 20 cases, evanescent and complex ω included.
- **G2.** With Paper 2's closed-form moments on both ends (Legendre field moments `2 iˡ j_l`, monomial
  moments `F_e(−kh)`), the mode integral reproduces Paper 2's exact blocks (`coupling_block`) to
  2–6e-13. This holds for p = 0, 1 and 2, with 10, 20 and 35 source monomials, for cells in different,
  non-touching planes.
- **Recorded: touching planes do not converge** (errors O(10²)). Once the Galerkin moments are taken
  there is no exponential decay left, because the moments grow as `e^{κh}` on each side against
  `e^{−2κh}` of propagation.

**The consequence.** The same holds wherever a cell touches an interface, not only across it. The
reflection of such a cell, back to itself and to its neighbours, comes from an image cell that **touches**
it. So `K^rev` is not smooth for any cell adjacent to an interface, and the mode integral cannot carry
those pairs (§3, "The risk", is wider than stated there). This needs a near-interface treatment,
decided before step 3.

**Decision (user, 10 October): option 1, static images in closed form.** No restriction keeping contrast
away from interfaces.

**10 October, later: interface R/T and the static image** (commit d341856):
- **Interface R/T.** `interface_rt` and `reflected_spectrum` in the 9-state modes, checked by value: the
  SH reflection `(μ₁k_z1 − μ₂k_z2)/(μ₁k_z1 + μ₂k_z2)` at oblique incidence to 4e-16, the normal-incidence
  P impedance formulas, and the identity for equal media.
- **The static image.** The reflected kernel of a welded interface (Rongved's problem, in the
  lateral-wavenumber domain) is derived by sympy in `scripts/derive_static_interface_image.py`, twin of
  `Mathematica/StaticInterfaceImage.wl`. The dynamic reflected kernel tends to it as `(ω/βq)²`.
- **The measurement that decided it.** For a cell touching the interface coupled to its own image (p = 0),
  the change in the wavenumber integral between successive fourfold cutoffs was:
  - **total:** 7.5e-3 (Qh 10 → 40), then 1.1e-4 (40 → 160). Algebraic, about `Q⁻³`.
  - **static subtracted:** 4.8e-5, then 5.1e-8. About `Q⁻⁵`; 1e-10 near `Qh ≈ 400`.
- **Still to measure:** p = 1, 2; the strain–strain rows; transmission across the interface.

**Next: the Mathematica phase.**
1. Transform the static image kernel to space: `1/R̄` terms (Paper 2's master integrals) and Mindlin-type
   `1/(R̄ + ζ)` terms, which need new master integrals.
2. Integrate it over a cell × the mirror image of a cell in closed form, with Paper 2's s-form. The
   factors of `z` and `z′` that multiply the image terms are absorbed into the cell polynomials.
3. The same for the transmitted static field.

**Route A (user, 10 October): new master integrals for the Mindlin potentials.**

**10 October, later: the static image in space, and its blocks.**
- **In space** (`scripts/derive_static_image_space.py`, twin of `Mathematica/StaticImageSpace.wl`).
  Derivatives of `Φ₀ = 1/(2πR̄)`, `Φ₁ = −log(R̄+ζ)/(2π)` and `Φ₂ = (ζ log(R̄+ζ) − R̄)/(2π)`,
  `ζ = −(z+z′)`, with polynomial factors in `z`, `z′`. Checked against direct 2-D Fourier integration of
  the Mathematica spectral kernel to 7e-13 – 1.3e-12.
- **As canonical terms** (`scripts/derive_interface_image_terms.py`). 356 terms
  `c(materials) z^A z′^B ∂^α Φ_j` in 61 kernel families: Φ₀ under up to four derivatives, and Φ₁, Φ₂ under
  lateral derivatives only (ζ-derivatives step them down). All homogeneous. Checked against direct
  differentiation to 3e-15. At the touching corner every term is integrable: the weight vanishes as
  `ζ^(1+A+B)`, and the worst integrand is `O(R̄⁻²)`.
- **The block engine** (`cubic_scattering/interface_image.py`, gate `scripts/gate_interface_image.py`).
  - **Method.** The s-form with exact rational weights. On the piece holding the singular corner, Euler's
    identity reduces the volume to the three far faces, which are smooth. Every other piece uses Gauss
    rules in centred coordinates.
  - **Corner check.** Against tanh-sinh quadrature of the singular volume integral: 6e-16 – 4.8e-14, over
    7 cases (all three potentials, up to fourth derivatives, both box orientations).
  - **Block check.** Whole non-touching blocks against brute-force 6-D Gauss: 3e-15 – 8.5e-15, over 4
    pairs (p = 0, 1, 2; vertical, lateral, diagonal offsets).
  - **Touching block end to end.** Against the 2-D Fourier integral of the Mathematica static spectrum, the
    reference approaches the engine as it is refined: 7.9e-6 at Qh = 320, then 1.6e-6, 4.5e-7 and 2.3e-8 at
    Qh = 20480. The engine does not move.
- **The face closed forms** (`cubic_scattering/image_moments.py`, twin of `Mathematica/ImageCornerMoments.wl`).
  - **Φ₀ families.** Paper 2's master integrals (box, face, edge), extended down to `R⁻⁹`.
  - **Mindlin families, by a line of images.** `∂^αΦ₁ = ∫_ζ^∞ ∂^αΦ₀ dt` and
    `∂^αΦ₂ = ∫_ζ^∞ (t−ζ) ∂^αΦ₀ dt`. Every term is then a box or semi-infinite-column master integral, the
    edge integrals to infinity in Beta functions.
  - **Degree-zero columns** (Euler's pole) are reduced by integration by parts, to two base integrals B₁, B₂
    (one angular integral each). These are quadrature at 40 digits until the `.wl` gives their closed forms.
  - **Checks.** All 53,352 corner moments (61 families, monomials to (6, 6, 8), both orientations) against
    the Euler + Gauss reduction: 1.2e-15. B₁ and B₂ against direct 3-D tanh-sinh: 20 digits. Touching blocks
    with closed corners against the Gauss version: 1.8e-15 and 2.6e-15.
- **Next.**
  1. Closed forms of B₁ and B₂ (`ImageCornerMoments.wl`).
  2. Speed. About 25–40 s per block now; the inner loops are Python.
  3. The transmitted static image.
  4. The production apply: march over all pairs, plus a local correction `closed(static) − grid(static)`
     for the touching image pairs.

## 1. Goal

Solve the Legendre-cell voxel scheme of Papers 1 and 2 with a **stratified** reference medium. This is
the case of the thesis (Nestor 1996, Ch. 5), and the one in which the whole programme becomes
competitive.

The thesis never forms the stratified reference Green's tensor. It uses only its **up/down
factorisation** (`GstratRep.tex`):

- the matrix analogue of the Lippmann–Schwinger equation, Eq. 5.36, with the stratified propagator
  `Q^∂ = (I − S_int E)^{-1} S_int`, Eq. 5.37;
- `𝒜 = 𝒰ℒ`, Eq. 5.49, whose pivots are Kennett's net reflection operators (5.50, 5.51), applied by
  the two sweeps of Algorithm 5.1;
- the two-way march, Eqs. 5.60–5.62.

This plan does the same. Nothing in it tabulates the stratified Green's tensor, and nothing depends
on the sibling package `GlobalMatrix`.

## 2. The existing structure the sweeps must carry

| Paper | Object | Code |
|---|---|---|
| 1 | Cell unknowns: Legendre coefficients of degree `p ≤ 2` of the 9-state, `na` = 1, 4, 10 per cell | `graded_voxel/basis.py` |
| 1 | Contrast polynomial in the cell (degree `r`), source monomials 10 / 20 / 35 | `basis.source_expansion`, `site.contrast_operator` |
| 1 | Single site `T36 = F (M − K(0) E)^{-1} M` | `graded_voxel/site.py` |
| 2 | Exact Galerkin coupling blocks `K_ac(R)`, touching and self included, closed forms and multipole far series | `graded_voxel/blocks.py`, `multipole.py`, `legendre_moments.py`, `moments.py` |
| 2 | Block-Toeplitz FFT apply with the 48 signed permutations of the cube | `graded_voxel/fft.py` |
| 2 | Receiver projection, closed form: `<L_a, e^{ik·x}>` | `graded_voxel/solver.plane_wave_moments` |
| 2 | Source projection, closed form at any complex `k`: `∫ m_c e^{−ik·x}` | `graded_voxel/farfield.monomial_fourier`, `source_moments` |
| sweeps | Plane-wave modes in the 9-state at `(k_x, k_y)` | `sweep_modes.mode_basis`, `mode_state`, `modes_to_state` |
| thesis | Two-way march (P-SV, uniform reference, augmented (u, t) state) | `PhD_fortran_code/phasescreen.f` (old Mac line endings: read with `tr '\r' '\n'`) |
| thesis | Riccati / Kennett recursions | `PhD_fortran_code/qriccati.f`, `kennetslo.f`; `cubic_scattering/kennett_layers.py` |

## 3. The operator

The reference is a stack of homogeneous layers. Every cell lies wholly inside one reference layer,
so interfaces fall on cell faces. Each cell's contrast is taken against **its own layer's** medium,
and its single site T36 is built in that medium. The stratified Galerkin operator splits exactly as

```
K^strat_mn  =  δ(L_m = L_n) · K^ws_{L_m}(g_m − g_n)   +   K^rev_mn
```

- **`K^ws_L`** is Paper 2's whole-space block in layer `L`'s medium, for two cells in the same layer.
  It carries the singular, touching and self parts exactly, in closed form. It is applied by
  `fft.py`, once per layer, on that layer's slab of the grid. **It is not changed.**
- **`K^rev`** is everything else: the reflected and converted waves within a layer, and every
  coupling between different layers (transmitted, reflected, converted). It is never tabulated. It is
  applied through the modes:

```
K^rev_{ac}(m, n) = ∬ dk_x dk_y   R_a(m; k^out)ᵀ · D_ε(L_m; k^out) · 𝓡(k_x, k_y; m ← n) · S_ε(L_n; k^in) · S_c(n; k^in)
```

- `S_c(n; k)` is Paper 2's closed-form source moment of cell `n`, at the mode wavevector `k` (complex
  for evanescent modes).
- `S_ε(L; k)` is the 6×9 source map of layer `L` (§5.1).
- `𝓡` is the mode-space response of the stratified reference between the two cells' depths: phase
  steps, interface R/T, and net reflections. It contains **no direct same-layer term**, which sits in
  `K^ws`.
- `D_ε(L; k)` is `sweep_modes`' 9×6 receiver map.
- `R_a(m; k)` is the closed-form Legendre moment of the plane wave over cell `m`.

**Applying it in one GMRES iteration.** `K^rev` is applied by marching, not by pairs:

1. **Project.** For each depth plane of cells, map its source coefficients onto mode amplitudes at
   every `(k_x, k_y)` node: closed-form moments, then `S_ε`, then a lateral transform (§6, decision 1).
2. **March.** One upsweep and one downsweep through the planes (Algorithm 5.1), with phase steps
   within a layer and interface R/T and net reflections across interfaces. The cost is
   `O(n_z n_k)`, and **the live state is six amplitudes per `k` node per plane, not a stored
   inter-plane stack.** The stage-2 projection of 615 GB at 16³ was the stack.
3. **Reconstruct.** At each plane: inverse lateral transform, `D_ε`, then the closed-form Legendre
   moments `R_a`.
4. **Add** Paper 2's FFT apply of `K^ws`. Then apply each cell's contrast `E` and the Gram matrix `M`
   as now (`solver.py`).

**Why this should converge in the transverse wavenumbers.** `K^rev` has no `r → 0` singularity
except across an interface between touching cells. For cells separated by distance `d` from the
reflecting interfaces, its integrand decays like `e^{−κ·2d}`. The transverse rule that stage 2 found
expensive (1.85 M nodes for 3e-7) was needed for the **singular whole-space** kernel, which here
stays in Paper 2's closed forms. How many nodes `K^rev` needs is measured in G3 and G7, not assumed.

**The risk.** Two cells touching across an interface. Their transmitted coupling is not smooth: its
integrand decays only through the cell moments, algebraically. This is gate G5. If the decay is too
slow, the fallback is to subtract a whole-space block in the averaged medium for those pairs only.
That fallback is not to be built unless G5 fails.

## 4. What changes in the sweep code

- `sweep_modes.py`: already works at any `(k_x, k_y)`. Add the analytic source map `S_ε` (§5.1), the
  per-layer (u, ε) → (u, t) map (§5.2), and batching over `k` nodes.
- `directional_sweeps.py`: its lateral sweep and stored vertical stack are the stage-1 point-voxel
  machinery. For the Legendre-cell scheme the direct coupling is Paper 2's FFT, so neither is used.
  A new vertical apply, `apply_reverberation(cells, layers, rule)`, carries `9 × na` coefficients per
  cell, with Projection and Reconstruction by the closed forms of Paper 2. The stage-1 functions stay,
  as arbiters (G2).
- `graded_voxel/fft.py`: accepts a layer partition, applying `K^ws_L` per layer on that layer's
  slab.
- `graded_voxel/solver.py` / `site.py`: the contrast and T36 of each cell are built in its own
  layer's medium.
- `sweep_solver.py`: GMRES around `K^ws + K^rev`.

## 5. Derivations

Each is derived symbolically in Mathematica first, under `Mathematica/`, then ported. In this
repository the defects have lived in conversion conventions; see
`docs/wrapper_problem_state_2026-09-13.md`.

### 5.1 The source map `S_ε`

By reciprocity (thesis `recipDef`, `D1def`), `S_ε(k) = W(k) · D_εᵀ(−k) · M`, with `W` diagonal, of
the form `i / (2 ρ ω² k_z,c)`, and `M` the diagonal fixing the work-conjugate pairing of force and
stress polarisation with (u, ε) under engineering doubling. It must agree with the normalisation
already used by Paper 2's far field (`farfield.graded_far_field`, `source_moments`), so that the
same cell radiates the same wave by both routes.

### 5.2 Interface R/T in `sweep_modes`' normalisation

Continuity of `(u, t)`, with `t = (C : ε) · e_z`: `D_t = H_L · D_ε` per layer, and R/T from
`D_t^{above} a = D_t^{below} b`, a 6×6 solve per `k`, or a 3×3 one by the symplectic inverse once its
normalisation is carried over. Stay in `sweep_modes`' bilinear unit-polarisation normalisation
throughout. It continues into evanescence, unlike the thesis's energy normalisation (`epsdef`).

### 5.3 The free surface

The top boundary is `"radiating"` or `"free"`. `phasescreen.f` has the P-SV free-surface
coefficients in closed form, which gives a value check.

## 6. Decisions to confirm

1. **The lateral transform.** Uniform `(k_x, k_y)` grid by FFT on a padded lateral grid (fast; the
   padding sets the period and so the aliasing of `K^rev`), or a non-uniform quadrature (no
   periodicity; costs `O(n_cells · n_k)` per plane). Recommended: the FFT grid with padding as the
   default, and the quadrature as its arbiter in G3.
2. **Mode normalisation.** Recommended: `sweep_modes`' bilinear unit polarisation everywhere.
3. **Top boundary.** Recommended: both, with radiating as the default for the gates.
4. **The two-way march (Stage B).** Recommended: after Stage A's end-to-end gate, G8.

## 7. Gates

Each gate prints PASS, FAIL or SKIPPED with its measured error. **SKIPPED never counts as PASS.**

| Gate | Claim | Arbiter | Pass |
|---|---|---|---|
| **G1** wiring | a one-layer model (no interfaces, radiating) gives `K^rev = 0`, and the solve equals `solve_graded_sphere_fft` | Paper 2 solver | bitwise / 1e-14 |
| **G2** modes reproduce Paper 2 | with no interfaces, the mode integral **with the direct term kept** reproduces `blocks.coupling_block` for cells in different planes, `p = 0, 1, 2` | Paper 2's exact blocks | ≤ 1e-10 for separated planes; the convergence of the rule with separation is recorded |
| **G3** single bounce, by value | one interface: `K^rev` between two cells equals the value predicted from independent physics (closed-form moments × mode maps × `thesis_interface_rt` R/T × phases) | `scripts/thesis_interface_rt.py` | ≤ 1e-10, **including SH at oblique incidence** |
| **G4** net reflection | a stack of layers: same-type coefficients (R_PP, R_SVSV, R_SHSH) of the march | `kennett_layers` at `p = k_h/ω` | ≤ 1e-10 |
| **G5** touching across an interface | convergence of `K^rev` with the transverse rule for touching cells in different layers | itself, refined | a recorded convergence law; FAIL if it does not converge |
| **G6** reciprocity | `K^strat_{mn} = Σ K^strat_{nm}ᵀ Σ`, with Σ the pairing of §5.1 | itself | ≤ 1e-12 (necessary, not sufficient) |
| **G7** layer law in a stratified background | a laterally uniform layer of cells in a stratified reference: reflection against Kennett of the homogenised layer, with the error law of Paper 3 Part I (`app:strat`) | `kennett_layers`; `scripts/measure_layer_stratified_background.py` | Paper 3's law, `O((kh)^{2p+2})` |
| **G8** end to end | the graded sphere in a layered half-space, `p = 0, 1, 2`: GMRES converges and refines at Paper 3's order. For `p = 0`, against the tabulated route `build_vertical_stack_layered` where `GlobalMatrix` is importable | Paper 3's order; tabulated route | the order; ≤ GMRES tolerance, else SKIPPED |

**G3 is the gate that matters.** The SH impedance defect (`μ` where `μη` belongs) passed every
symmetry, reciprocity and whole-space gate at 1e-15. Only a gate that predicted the value of a
layered reflection found it. G6 alone proves nothing.

## 8. Stages

### Stage A — `K^ws + K^rev` with GMRES

1. Mathematica: `S_ε`, `W`, `M`, `H_L` (§5.1, §5.2), and the free surface (§5.3). Save outputs.
2. `sweep_modes`: the source map, batched over `k`. Then Projection and Reconstruction with Paper 2's
   closed forms. Close **G2**: it validates the maps, their normalisation and the transverse rule
   together, against objects already exact.
3. The march with no interfaces, then with interfaces, net reflections and Algorithm 5.1. Close
   **G1**, **G3**, **G4**, **G6**.
4. Per-layer `K^ws` in `fft.py`; per-layer contrasts and T36. Close **G5**.
5. End to end. Close **G7**, then **G8**.
6. Tests in `cubic_scattering/tests/`: G1, G2, G3 and G6 on small grids.

### Stage B — the two-way march as the solver (thesis Eqs. 5.60–5.62)

Port `phasescreen.f`'s `DownUp` iteration to Legendre cells:
- forward scattering is marched exactly inside the down and up sweeps;
- backscatter is lagged and iterated, the Gauss–Seidel reading of the equations, where Born is one
  Jacobi step;
- within a plane, coupling stays with T36 and Paper 2's `K^ws`.

Gate it against Stage A's converged solution, and use it as a solver or as a GMRES preconditioner,
whichever measures better (`scripts/gate_summation_stratified_reference.py` has the harness).

## 9. Out of scope

- Cells straddling an interface (each cell lies wholly in one layer).
- Anisotropic layers.
- The octree (archived; `LatexPDFs/OctreeRefinementArchive/`).
