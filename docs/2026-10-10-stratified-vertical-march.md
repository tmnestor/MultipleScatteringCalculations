# The Stratified Vertical March — Implementation Plan

**10 October 2026.** Status: plan, no code yet.

Conventions throughout, as in `cubic_scattering/sweep_modes.py`: seismic units (km/s, g/cm³, GPa, km),
time `e^{−iωt}`, every vertical wavenumber on `Im ≥ 0`, z = axis 0 (down), x = 1, y = 2, mode order
(P↓, SV↓, SH↓, P↑, SV↑, SH↑), state the 9-vector `(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)`.

---

## 1. The problem

The thesis never forms the stratified reference Green's tensor. It has only its **up/down
factorisation**: project a source onto plane-wave modes, propagate the modes, and reconstruct the
field (Nestor 1996, Ch. 5, `GstratRep.tex`):

- the matrix analogue of the Lippmann–Schwinger equation, Eq. 5.36 (`LSmat`), with the stratified
  propagator `Q^∂ = (I − S_int E)^{-1} S_int`, Eq. 5.37 (`PstratDef`);
- the block factorisation `𝒜 = 𝒰ℒ` of `𝒜 = I − S_int E`, Eq. 5.49. Its pivots are Kennett's net
  reflection operators (5.50, 5.51), and Algorithm 5.1 applies `Q^∂` by one upsweep and one downsweep;
- the two-way march, Eqs. 5.60–5.62 (`Fsplit`, `LSmatF`, `fLSeqn`): forward scattering is marched
  exactly and backscatter is iterated.

The repository's production stratified vertical operator does the opposite. It **tabulates** the
reference Green's tensor:

- `directional_sweeps.build_vertical_stack_layered` calls `layered_correction.corrected_layered_9x9`
  once for **every ordered pair of depth planes**, `n_z²` full layered Green's functions;
- `sweep_z` then sums over every pair, at `O(n_z² n_kx)` per apply;
- the stack is stored as `(n_z, n_z, 9, 9, n_kx)`. Stage 2 measured this at **615 GB for a 16³
  grid** (`docs/plans/2026-09-14-cartesian-directional-sweeps-stage2.md`);
- it needs the sibling package `GlobalMatrix` (`GlobalMatrix.layered_greens`), which is not in this
  repository. The stratified route cannot run from this repository alone, and so could not ship in a
  paper's standalone release (`docs/2026-10-10-reproduction-repository-design.md`).

The thesis code already marches. `PhD_fortran_code/phasescreen.f` (old Mac line endings: read it
with `tr '\r' '\n'`) carries running P and SV mode amplitudes `Wp`, `Ws` from screen to screen:
- **project:** each row's secondary source enters through `Bp`, `Bs` and an FFT;
- **propagate:** one phase step `Ea`, `Eb` per row;
- **reconstruct:** the field is rebuilt at each row by an inverse FFT;
- **iterate:** a down-sweep, then an up-sweep, repeated `itmax` times (subroutine `DownUp`).

Its state is the thesis's augmented (u, t), `(u_z, u_x, τ_zz, τ_zx, ∂ₓu_x)` (`B51 = ik·B21`), with
the compliance-type contrasts Δρ, Δa, Δb, Δγ, Δζ, in P-SV only, over a uniform reference with a free
surface.

**This plan ports that march into the 9-component (u, strain) sweep, over a stratified reference,
with no `GlobalMatrix` and no tabulated Green's tensor.**

## 2. What already exists and is reused

| Piece | Where | Role here |
|---|---|---|
| (u, ε) ↔ mode bridge | `sweep_modes.mode_state`, `mode_basis`, `modes_to_state` | receiver map `D_ε` (9×6) per layer |
| Whole-space factorisation `D diag(e^{ik_z\|dz\|}) S` | `sweep_modes.vertical_factorisation` | arbiter for the analytic source map (§4.1) |
| Whole-space plane-to-plane kernels | `directional_sweeps.build_vertical_stack`, `sweep_z` | homogeneous-limit arbiter (§5, G2) |
| Stratified plane-to-plane kernels | `directional_sweeps.build_vertical_stack_layered` | stratified arbiter where `GlobalMatrix` is installed (G5) |
| Lateral sweep (a running accumulation already) | `directional_sweeps.sweep_x` | unchanged |
| Interface R/T, symplectic inverse | `scripts/thesis_interface_rt.py` (twin of `Mathematica/ThesisInterfaceRT.wl`) | independent check of §4.2 (G3) |
| Kennett recursion | `cubic_scattering/kennett_layers.py` | independent check of the net reflection (G4) |
| Riccati blocks, impedance march | `scripts/gate_first_order_riccati_blocks.py`, `gate_first_order_impedance_march.py` | conditioning evidence for §4.3 |
| GMRES around `apply_g0` | `cubic_scattering/sweep_solver.py` | the outer solver for Stage A |

## 3. Architecture

A new module, `cubic_scattering/stratified_march.py`. It is independent of `GlobalMatrix`, and works
at a fixed `k_y` and per lateral node `k_x`, vectorised over the `k_x` nodes of `SweepGrid`. It
has five parts:

1. **Layer modes.** For each layer: the receiver map `D_ε` (9×6) from `sweep_modes`, and the
   analytic source map `S_ε` (6×9) of §4.1.
2. **Interfaces.** For each interface between planes: the 6×6 mode scattering matrix (R↓, T↓, R↑, T↑)
   in `sweep_modes`' normalisation, built as in §4.2. One more boundary at the top (free surface or
   radiating) and one at the bottom (radiating).
3. **Net reflections.** Built once per (ω, k_y), outside the Krylov loop: the upward recursion for
   `R↑↓^(i,∞)` (Eq. 5.51, the upsweep of Algorithm 5.1), the downward recursion for `R↓↑^(f,i)`, and
   the reverberation inverses `(I − R↓↑ R↑↓)^{-1}`, all 3×3 per `k_x`.
4. **`apply_q_march(sources)`.** The exact action of the stratified inter-plane operator, as one
   upsweep and one downsweep (Algorithm 5.1), `O(n_z n_kx)` per apply with storage `O(n_z n_kx)`. It
   replaces `sweep_z(sources, grid, build_vertical_stack_layered(...))`.
5. **Wiring.** `build_g0_cache(..., vertical="march")` selects it; `"stack"` keeps the present path.
   `apply_g0` is unchanged in form: `sweep_x + sweep_z_or_march`.

**The direct same-plane term stays in `sweep_x`.** The march never delivers a plane's own direct
radiation back to that plane. At each plane it reconstructs the field from the amplitude arriving
from above (sources strictly above, and everything reflected down) and the amplitude arriving from
below (sources strictly below, and everything reflected up). A voxel's wave that leaves, reflects
off a layer boundary and returns to its own plane is therefore included. That path is what the
diagonal of `build_vertical_stack_layered` carries today ("layered minus whole-space"), and the
march carries it without the subtraction.

## 4. The three derivations

Each derivation is done symbolically in Mathematica first, under `Mathematica/`, then ported. In
this repository conversion conventions are where the defects have lived; see the D1–D3 record in
`docs/wrapper_problem_state_2026-09-13.md`.

### 4.1 The analytic source map `S_ε`

`vertical_factorisation` recovers the source side numerically, as `pinv(D) · kernel` at one `dz`.
The march needs it per layer, in closed form. Reciprocity (`recipDef`, `D1def`) says it is the
transposed receiver map at `−k`, with a diagonal weight:

```
S_ε(k) = W(k) · D_εᵀ(−k) · M
```

- `W` is diagonal, of the form `i / (2 ρ ω² k_z,c)`, up to the Voigt and polarisation normalisation.
- `M` is diagonal and fixes the work-conjugate pairing of the 9-state. The source is (force, stress
  polarisation) and the state is (u, ε with engineering doubling), so the strain rows pick up factors
  of 1 or ½.

`W` and `M` are to be derived, not fitted.

### 4.2 Interface scattering in `sweep_modes`' normalisation

Continuity at a welded interface is continuity of `(u, t)`, with `t = (C : ε) · e_z`. So the (u, t)
mode matrix of a layer is `D_t = H_layer · D_ε`, where `H_layer` maps (u, ε) to (u, t) through that
layer's moduli. Interface R/T follow from `D_t^{above} a = D_t^{below} b`: a 6×6 solve per `k_x`, or a
3×3 one by the symplectic inverse (`D1def`) once its normalisation is carried over.

**Do not rescale the modes to energy flux.** The thesis normalises its eigenvectors by energy
(`epsdef`); `sweep_modes` uses bilinear unit polarisation, which continues into evanescence. The
march must stay in `sweep_modes`' normalisation from end to end. Different-type coefficients (P↔SV)
differ between the two by a known ratio, and same-type ones do not, which is what G4 exploits.

### 4.3 The free surface

The top boundary condition is selectable: `"radiating"`, or `"free"` (traction-free: `R = −D_t,↑⁻¹ D_t,↓`
on the traction rows). `phasescreen.f` builds the P-SV free-surface coefficients (`Rpp`, `Rsp`, `Rss`)
in closed form, which gives a P-SV value check.

## 5. Gates

Each gate prints PASS, FAIL or SKIPPED with its measured error. **SKIPPED never counts as PASS.**

| Gate | Claim | Arbiter | Pass |
|---|---|---|---|
| **G1** source map | `S_ε` of §4.1 equals the numeric source side | `vertical_factorisation` at several `dz`, `k_x`, both directions, propagating and evanescent | ≤ 1e-12 |
| **G2** homogeneous march | uniform model, radiating boundaries: `apply_q_march` equals the stack | `sweep_z(build_vertical_stack(...))` on random sources | ≤ 1e-12 |
| **G3** single bounce, by value | one interface: the reflected field at each receiver plane equals the field predicted from independent physics (source map × phase × interface R × phase × receiver map) | `thesis_interface_rt.interface_rt`, converted to `sweep_modes`' normalisation | ≤ 1e-12, **including SH at oblique incidence** |
| **G4** net reflection | a stack of layers: same-type coefficients (R_PP, R_SVSV, R_SHSH) of the march's net reflection | `kennett_layers` at `p = k_x/ω`, normal and oblique incidence | ≤ 1e-10 |
| **G5** full operator | stratified model: `apply_q_march` equals the stratified stack | `build_vertical_stack_layered` | ≤ 1e-12 when `GlobalMatrix` is importable, else SKIPPED |
| **G6** reciprocity | `P(x₂; x₁; k_y) = S Pᵀ(x₁; x₂; k_y) S`, with S the y-mirror | self | ≤ 1e-12 (necessary, not sufficient) |
| **G7** cost | time and memory linear in `n_z`; the stack's grow as `n_z²` | measured at `n_z` = 8, 16, 32, 64 | slopes ≈ 1 and ≈ 2 |
| **G8** end to end | `sweep_solver` with `vertical="march"` matches `"stack"` on a small layered scatterer | the existing path | to the GMRES tolerance |

**G3 is the gate that matters.** The SH impedance defect (`μ` where `μη` belongs) passed every
symmetry, reciprocity and whole-space gate at 1e-15. Only a gate that predicted the value of a
layered reflection found it. G3 predicts values; G6 alone would prove nothing.

## 6. Stages

### Stage A — exact `apply_g0` over a stratified reference

Tasks, in order, each closing on its gate:

1. Mathematica: `S_ε`, `W`, `M` (§4.1) and the (u, ε) → (u, t) map `H_layer` (§4.2). Save outputs.
2. `stratified_march.py`: layer modes and the source map. Close **G1**.
3. The homogeneous march: phase steps only, no interfaces. Close **G2**.
4. Interfaces, net reflections and Algorithm 5.1 sweeps. Close **G3**, **G4**, **G6**, then **G5**
   where `GlobalMatrix` is available.
5. The free surface. Check the P-SV coefficients against `phasescreen.f`'s closed forms.
6. Wire `vertical="march"` into `build_g0_cache`. Close **G7** and **G8**.
7. Tests in `cubic_scattering/tests/test_stratified_march.py`: G1, G2, G3 and G6 on small grids, so
   that they run in the package suite.

Outcome: GMRES around an exact stratified `G0` at `O(n_z n_kx)` per apply, with no `GlobalMatrix`.

### Stage B — the two-way march as the solver (thesis Eqs. 5.60–5.62)

Stage A keeps GMRES as the outer solver. Stage B ports `phasescreen.f`'s `DownUp` iteration in the
9-state:
- the scatterers' forward scattering is marched exactly inside the down and up sweeps (`P↓↓`, `P↑↑`,
  block bidiagonal, no inversions);
- backscatter is lagged (`B↓↓`, `B↑↑`, `A↓↑`) and iterated, the Gauss–Seidel reading of the
  equations, where Born is one Jacobi step;
- within a row, side scattering comes from the Paper 1 single-site T-matrix and the lateral sweep,
  the role ΔTᵖ = ΔC(I − Pˣ ΔC)⁻¹ plays in Box 5.2.

It is gated against the converged Stage A solution, and used either as a solver or as a GMRES
preconditioner, whichever measures better. Its claimed advantage is convergence on smooth media
with dominant forward scattering. That is a measured question, and
`scripts/gate_summation_stratified_reference.py` has the harness for it.

## 7. Decisions to confirm

1. **Mode normalisation.** Recommended: `sweep_modes`' bilinear unit polarisation everywhere, with the
   thesis's energy-normalised R/T converted to it at the gate only (§4.2).
2. **Top boundary.** Recommended: both `"radiating"` and `"free"`, with radiating as the default for
   the gates.
3. **The tabulated route.** Recommended: keep `build_vertical_stack_layered` as an optional arbiter
   (G5) and do not delete it. The march becomes the default once G1–G8 pass.
4. **Stage B scope.** Recommended: start it only after Stage A is complete and the end-to-end
   comparison (G8) is recorded.

## 8. Out of scope

- 3-D (`k_y` quadrature). The march is per `(k_x, k_y)`, so it carries over unchanged. The stage-2
  memory problem was the stored inter-plane stack, which the march removes.
- Planes on material interfaces. As now, planes lie in layer interiors
  (`scripts/gate_stratified_correction.py`).
- Anisotropic layers.
