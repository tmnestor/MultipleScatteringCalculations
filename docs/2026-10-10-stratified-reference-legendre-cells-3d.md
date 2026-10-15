# The Stratified Reference in Three Dimensions for the Legendre-Cell Voxel — Plan

**10 October 2026.** Status: in progress (see Progress); the next steps are at the end of Progress. This replaces the first version of this file, written
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
- **The transmitted static image** (`scripts/derive_static_interface_transmission.py`, twin
  `Mathematica/StaticInterfaceTransmission.wl`; terms by `derive_interface_image_terms.py transmitted`).
  - **The field.** For a source in A and a receiver in B, the whole static coupling is
    `e^{−q(z−z′)}/q × poly`, the same potentials with `ζ = z − z′`. It has 275 terms in 46 families,
    integrable at the touching corner.
  - **The blocks.** The vertical weight is a correlation. The other two arrangements (receiver in A with a
    source in B, and both cells in B) are the mirror `z → −z` with the media exchanged.
  - **Identical media.** Against Paper 2's touching blocks: 1.9e-15 – 7.6e-15 (face, edge, corner, mirrored,
    p = 2, 35 sources). The image is identically zero.
  - **Different media.** Against 2-D Fourier integration: 7.8e-13 and 6.6e-13. Against brute-force 6-D Gauss:
    1.7e-13 – 5.0e-13.
  - **The dynamic limit.** The dynamic transmitted kernel tends to the static one as `(ω/βq)²`.
  - Gates T1 and T2 are in `scripts/gate_interface_image.py`, and the material is written up (first in Paper 2 §11, since 11 October 2026 in its own paper, `LatexPDFs/InterfaceCouplingIntegrals`).
- **The Mathematica runs.**
  - **`StaticInterfaceTransmission.wl`.** Spectral kernel against sympy: 4.2e-16. Its spatial form against
    `NIntegrate` of the Fourier integral: 3.9e-16. The transmitted terms regenerated from it are identical
    to the sympy-derived ones (275 terms, coefficient difference exactly 0).
  - **`ImageCornerMoments.wl`, first run.**
    - The line-of-images identities hold symbolically.
    - B₁ came out as Catalan's constant and dilogarithms, plus one remaining angular integral
      `∫_{π/4}^{π/2} log((1+√(1+csc²θ))/2) dθ`. Its value agrees with ours to 20 digits.
    - B₂ gave no closed form.
    - **Its `NIntegrate` references were wrong** (1e-12 to 5e-7), and so was B₂ (1.7e-8). Against them stand
      three independent routes that agree with ours to 30 digits: Paper 2's master integral, the closed form,
      and exact 1-D integration plus 2-D tanh-sinh. The cause is a direct 3-D `NIntegrate` with the singular
      point at a corner, or over a semi-infinite column, silently missing its goal.
    - The `.wl` now integrates one variable exactly first and uses Duffy coordinates.
  - **`ImageCornerMoments.wl`, rerun.** All 13 references and B₂ agree with ours to 1.6e-16 – 4.9e-23,
    and B₁ to 6e-32. That is `NIntegrate`'s accuracy: the exactly known reference is off by 1.4e-17, while
    ours do not move between 40 and 70 digits (1e-39).
- **Next.**
  1. Closed forms of B₁ and B₂ (`ImageCornerMoments.wl`).
  2. Speed. About 25–40 s per block now; the inner loops are Python.
  3. The production apply: march over all pairs, plus a local correction `closed(static) − grid(static)`
     for the touching image and transmission pairs.

**11 October 2026: beyond the static image, and the frequency sweep.**
- **Convergence, corrected.** The plane-wave remainder of a touching block converges as `Q⁻²` without the
  static image and `Q⁻⁴` with it (the earlier `Q⁻³ → Q⁻⁵` was the change between fourfold cutoffs, not
  the order). Six-shell sweep at 40 digits: `scripts/measure_interface_static_subtraction.py`.
- **The `O(ω²)` term.** `G = G0 + ω² G1 + O(ω⁴)` at large q, by perturbation of the static problem
  (`Mathematica/InterfaceOmega2.wl`, twin `scripts/derive_interface_omega2.py`). Checks: identical media;
  the residual against the 50-digit exact kernels falls 16× per halving of ω; the two derivations agree to
  5e-51. Its spatial form would need Φ₃, Φ₄ (finite parts only), so it is used in the wavenumber domain.
- **The split.** `K(ω) = D(ω) + A0 + ω² A1`, with
  - `D = ∫_{q<Q0} Y`, per frequency, in double precision;
  - `A0 = K0 − ∫_{q<Q0} Y0` and `A1 = ∫_{q>Q0} Y1`, once per arrangement of cells.

  The dropped remainder converges as `Q0⁻⁶`. For the measurements, reflected and transmitted, see
  `scripts/measure_interface_omega2_subtraction.py [R|T]`: Q0h = 10 leaves 4e-10 at ωh/β = 1/6, and
  Q0h = 40 leaves 6e-11 at ωh/β = 1.
- **The package.** `cubic_scattering/interface_spectral.py`:
  - **API.** `frequency_independent_parts`, `lateral_harmonics`, `truncated_blocks`, `interface_block` and
    `sweep`.
  - **Structure.** The static and `ω²` integrands are Laurent polynomials in q times elementary vertical
    moments. The rotation to the wave direction has nine harmonics only. The azimuthal integrals `S_n(q)` of
    the cells' lateral moments are tabulated once per lateral offset (Chebyshev). Each frequency costs one
    6 × 6 interface solve per radial node.
  - **Cost.** 10–300 ms per block per frequency, against 1–16 s for the mode sum at every (q, φ), which is
    kept as `_truncated_blocks_direct`.
- **Gates.** `scripts/gate_interface_spectral.py --full`:
  - double frames against 40 digits;
  - identical media against Paper 2's `coupling_block`: 3e-10 and 5e-11 (face, edge, corner, mirrored);
  - different media against the 40-digit references (`scripts/data/interface_omega2_reference_{R,T}.json`):
    R 3.6e-10 / 5.9e-11, T 2.2e-10 / 3.9e-11;
  - the two routes of D.

  Unit tests: `cubic_scattering/tests/test_interface_spectral.py`.
- **Cost still to fix.** A0 and A1 take about 10–17 s per pair, and every touching pair needs them at
  every lateral offset. That does not scale to a grid along the interface (step 1 below).

**Next (decided 11 October): step 1, then step 2.** Step 1's window and locality measurement are done (12 October, below the step 1 plan).

**Step 1 — make the near-interface corrections local.**
- **Why.** A0 and A1 are high-pass parts of kernels that are smooth away from the touching point. With the
  sharp cutoff `θ(q − Q0)` they ring in space (Gibbs) and decay slowly with lateral offset.
- **The change.** Replace the cutoff by a smooth window `χ(q)` (0 below `Q0/2`, 1 above `Q0`, `C^∞` or a
  high-order polynomial ramp), in both A0/A1 and D, so that `D + A0 + ω² A1` is unchanged:
  - `K = ∫ [Y − χ (Y0 + ω² Y1)] + (K0 − ∫ (1 − χ) Y0) + ω² ∫ χ Y1`.
- **Measure.**
  - the decay of A0 and A1 with lateral offset (0 … 16 cells), sharp against smooth window;
  - the remainder `E` against the window width: the window must not spoil the `Q0⁻⁶` law;
  - hence the radius beyond which no correction is needed at a given tolerance.
- **Then.**
  - Corrections only for pairs within that radius. D for every offset at once on the shared lateral grid
    (FFT in (k_x, k_y), or the polar `S_n` route per offset if the radius is small).
  - Gate: the full gate unchanged; plus a row of cells along the interface against per-pair
    `interface_block`.
- **Deliverables.** A windowed `frequency_independent_parts` and `truncated_blocks`; the measurement script
  `scripts/measure_interface_correction_locality.py`; the gate extended.

**12 October 2026: step 1, the window and its locality.**
- **The code.** `interface_spectral.Window(start, shape)` (C^∞ step `e^{−1/t}/(e^{−1/t} + e^{−1/(1−t)})`, or
  the polynomial step `I_t(m+1, m+1)`), accepted by `truncated_blocks`, `frequency_independent_parts`,
  `interface_block` and `sweep` (default: the sharp cutoff, unchanged). The windowed parts are the sharp ones
  plus the ramp integrals `∫χY0`, `∫χY1` (`window_parts`, cheap: q < Q0), which D gives back; so the window
  costs nothing beyond the sharp parts, and E is the same by construction (the `Q0⁻⁶` law cannot be spoiled).
- **Quadrature.** Order-12 Gauss on D's panels left 5e-11 on the C^∞ flanks; the ramp is now 16 panels in D
  and in `window_parts` (rounding). A ramp starting inside the branch-point panels is refused.
- **Identity.** Windowed against sharp blocks: 1e-14 at ωh/β = 1; at ωh/β = 1/6, 3e-11 (R, D's known
  rounding, now on different nodes) and 8e-14 (T). Unit test: D − D_χ = ramp integrals to 1e-11.
- **Locality** (`scripts/measure_interface_correction_locality.py`, cached in
  `scripts/data/interface_correction_locality.json`). Row of cells touching the interface, offset d along x,
  full 4 × 10 × 9 × 9 blocks, Q0h = 10, ωh/β = 1/6, relative to max |K0| of the touching pair:

  | d/h | sharp A0 | C^∞ 0.5 A0 | C^∞ 0.25 A0 | C^∞ 0.25 ω²A1 | poly3 0.5 A0 |
  |---|---|---|---|---|---|
  | 4 | 2.7e-4 | 9.8e-5 | 1.2e-4 | 2.7e-7 | 7.9e-5 |
  | 8 | 1.0e-4 | 1.7e-6 | 1.2e-6 | 5.1e-9 | 5.4e-7 |
  | 16 | 3.2e-5 | 1.5e-8 | 5.9e-9 | 2.7e-11 | 9.8e-9 |
  | 24 | 2.4e-5 | 5.5e-10 | 9.1e-11 | 4.4e-13 | 8.8e-10 |
  | 32 | 1.4e-5 | 3.2e-11 | 4.7e-12 | 3.8e-14 | 1.4e-10 |

  (Reflected shown; transmitted within a factor 1.3.) The sharp cutoff decays as about `d^−1.5` (Gibbs); the
  C^∞ windows decay faster than any power, the wider ramp (start 0.25) faster; the polynomial step is
  algebraic and stalls near 1e-10. **Radius** (C^∞ 0.25, beyond which |A0| + ω²|A1| < tol): 12 h for 1e-6,
  16 h for 1e-8, 24 h for 1e-10 (6, 8 and 12 cells), R and T alike.
- **The cost consequence.** ω²A1, the expensive part (octaves to Q1, azimuthal work ∝ q × offset: 25 s at
  d = 0, 540–820 s at 32 h for the sharp parts), is below 1e-10 beyond 12 h; farther out only A0 is needed,
  which is the closed-form K0 and an integral below Q0. A0, A1 depend only on the two depths and the lateral
  offset, so a grid along the interface needs them once per (depth pair, offset within the radius).
- **Gate S5** (`scripts/gate_interface_spectral.py`): a row of five cells along the interface (0 … 8 h), D for all pairs in one call, against per-pair sharp
  blocks: with every correction R 1.9e-11, T 3.9e-14; without the corrections beyond 4 h R 1.0e-5,
  T 1.3e-5 (the measured A0 at 6 h). The windowed row 16 s, the sharp parts pair by pair 130–145 s.
- **Still open in step 1.** D for every offset at once (FFT on the shared lateral grid, or the `S_n` route per
  offset); A1 only within its own, smaller radius (and A0 alone beyond); whether a still wider ramp (start
  0.1, the lowest the branch points allow at Q0h = 10, ωh/β = 1/6) shrinks the radius further.
- **Checked, not built: the Jacobi–Anger form of S_n.** With `e^{ik·Δ} = Σ_p i^p J_p(q|Δ|) e^{ip(φ−φ_Δ)}`,
  `S_n(q; Δ) = (1/2π) Σ_p i^p J_p(q|Δ|) e^{−ipφ_Δ} c_{−n−p}(q)`, `c_m` the azimuthal Fourier coefficients of
  the cells' lateral moments, band-limited by the cell (|m| ≤ 3.5qh + 20 at 1e-12, qh = 1 … 40), not by the offset. Against
  the direct azimuthal rule, 4e-16 … 3e-13 for qh = 0.7 … 9.9 and offsets to 40 h. No per-offset tables.

**Decision (user, 12 October): analytic first, as in Paper 2.** The wavenumber split (Q0, D, A0, A1, the
window, Jacobi–Anger) departs from the master-integral approach of the rest of the codebase; it began when
the ω² term was moved to the wavenumber domain because its spatial form needs Φ₃, Φ₄ (11 October). It is
retired from production and kept as the **verification route** (it shares no step with the closed forms).
The structure is Paper 2's (`graded_voxel/blocks.py`):
- **Touching image and transmission pairs:** `K0 + ω² K1` in closed form by master integrals; K0 done
  (`interface_image.image_block`, 356 / 275 terms); the remainder `G − G0 − ω² G1`, bounded at the touching
  point, by Gauss / Duffy in space, as Paper 2's dynamic remainder.
- **Non-touching pairs:** Gauss quadrature in space of the full reflected / transmitted kernel. With P–S
  conversion the reflected kernel depends on z and z′ separately (not z + z′ only): tensor Gauss in (u, u′)
  as `far_block`, not the 3-D s-form.
- **The kernel at a point** has no closed form: a 1-D Sommerfeld integral in q (the existing `interface_rt`
  and mode states), the counterpart of `e^{ikr}/r` in Paper 2: a kernel evaluation, not a cell integration.
- **The paper.** Jacobi–Anger and the window's locality go in the Verification section, as part of the
  independent wavenumber route, not as the method; the ω² section is restructured once K1 is in closed form.

**Next: K1 in space (the Mathematica phase, with Python twins).**
1. **The families.** Rotate G1's frame kernel to fixed axes and list the terms
   `c z^A z′^B ∂^α Φ_j` as for K0: which j (now up to 4: `q^{−4}`, `q^{−5}`) and with how many lateral
   derivatives. A term with n_lat < j has an integrand ~ q^{n_lat−1−j} at q → 0: its transform is a finite
   part (item 2). `derive_interface_image_terms.py <kind> 1` (order 0 reproduces the K0 data byte for byte);
   checked by `check_interface_terms_spectrum.py` (exact on K0; an injected 1e-3 coefficient error shows).
   **Done (13 October), with rational media** (`... <kind> 1 media1|media2`, 13–21 s; the symbolic run,
   six material symbols under `sympy.simplify`, had not finished after an hour). Both media give the same
   families, so none is an accident of the media:

   | G1 | terms | families | under Φ₀ / Φ₁ / Φ₂ / Φ₃ / Φ₄ | finite part needed (n_lat < j) |
   |---|---|---|---|---|
   | reflected | 744 | 76 | 235 / 232 / 172 / 80 / 25 | 81 terms, 16 families |
   | transmitted | 582 | 61 | 123 / 182 / 172 / 80 / 25 | 81 terms, 16 families |

   The finite-part families: Φ₁ with no lateral derivative (12 terms); Φ₂ with α = (0,0), (0,1), (1,0) (27);
   Φ₃ with |α| ≤ 2 (26 terms, 5 families); Φ₄ with |α| ≤ 3 (16 terms, 7 families); the same in both kinds.
   So the finite parts are genuinely needed, and the closed forms of item 2 carry them. The spectral check
   passes exactly on all 81 entries, both kinds, both media.
2. **Φ_j as Hadamard finite parts — done (12 October).** The series' theme (user, 12 October): finite parts
   are evaluated analytically, so that no asymptotic wavenumber tail is ever integrated; nothing is left to a
   cutoff or pushed into a numerical remainder. With the Laplace–Hankel transform
   `(1/2π)∫ q^{m+1} e^{−qζ} J₀(qρ) dq = Γ(m+2) R^{−m−2} P_{m+1}(ζ/R)/2π`, Φ_j (m = −1−j) is the ε⁰ Laurent
   coefficient at the pole of Γ: the analytic continuation in the power of q, as `image_moments.box`
   continues in the degree. With n = j−1,
   `Φ_j = (−1)ⁿ/(n! 2π) {[ψ(j) − 2(ψ(2n+1) − ψ(n+1)) − log((R+ζ)/2)] RⁿP_n(ζ/R) − 2 Σ_{k<n} (−1)^{n+k}(2k+1)/((n−k)(n+k+1)) RⁿP_k(ζ/R)}`:
   polynomial in (R, ζ) times log(R + ζ), plus a polynomial. The residue is the solid harmonic RⁿP_n(ζ/R), the
   ambiguity a cutoff would leave free; the finite part fixes its coefficient (γ, log 2).
   - `scripts/derive_finite_part_potentials.py` (twin `Mathematica/FinitePartPotentials.wl`, which takes the
     Laurent coefficient itself, no degree-derivative formula supplied; not yet run): the ladder
     `∂_ζ Φ_{j+1} = −Φ_j` and harmonicity symbolically, j ≤ 6; against the finite part computed from the
     integral itself (Taylor series of the entire head, quadrature of the tail), 0 … 3e-40 at 40 digits.
     Data: `scripts/data/finite_part_potentials.json`.
   - The paper's Φ₁, Φ₂ differ by `(log 2 − γ)/2π` and `ζ (γ − log 2)/2π`, annihilated by every K0 term's
     lateral derivatives (n_lat ≥ j): K0 unchanged.
   - **Recorded:** the degree derivative of P_ν at an integer needs the diagonal term
     `2(ψ(2n+1) − ψ(n+1)) P_n`; without it Φ₂ … Φ₄ were wrong by 20–50 %. Checked against the numerical
     ν-derivative to 2e-41. And mpmath's `legenp` near a negative integer degree is not a usable reference
     (errors 1e-11 that move with the precision); the finite part from the integral is.
   - **Consistency.** `K = FP∫Y0 + ω² FP∫Y1 + FP∫(Y − Y0 − ω²Y1)`: the finite part of a convergent integral is
     the integral and the finite part is linear, so the split is exact with **no cutoff Q0**, provided the
     remainder's small-q finite part uses the same continuation (its subtracted Laurent terms are integrated
     analytically near q = 0). The first two terms are closed form for every pair; the Gibbs problem of the
     window step does not arise.
   - **The line of images is another convention (13 October).** For a finite-part family the line
     `∫_ζ^∞ (t−ζ)^{j−1}/(j−1)! ∂^αΦ₀ dt` diverges at t → ∞; its finite part there differs from Φ_j by a
     polynomial H_j: `H1 = −γ/2π`, `H2 = γζ/2π`, `H3 = ((2γ−3)ρ² − 4γζ²)/16π`,
     `H4 = −ζ((6γ−9)ρ² − 4γζ²)/48π` — for j ≥ 3 not harmonic, so not a potential. Moments built on
     `image_moments`' line of images would be silently wrong for these families unless ∂^αH_j is added.
   - **The finite-part kernels are bounded.** Every finite-part family has |α| = n_lat ≤ j − 1, so
     ∂^αΦ_j has degree j − 1 − |α| ≥ 0: no corner singularity, only log(R + ζ). Their box moments are ordinary
     integrals of explicit closed forms, and 40-digit cubature of the closed form is a direct reference.
3. **K1 blocks — first version done (13 October), direct route (user's choice).**
   `interface_image.image_block(..., order=1)` gives K1 (K = K0 + ω²K1 + …) from the ω² terms
   (`interface_omega2_terms_*.json`, symbolic coefficients, densities included) and the finite-part potentials:
   - `kernel_function` uses Φ_j^FP for j ≥ 1 (to j = 4); K0 is unchanged to 5.7e-16 (7 pairs, both bases,
     both corner routes, against the previous module).
   - **Log-homogeneity.** ∂^αΦ_j(λx) = λ^d [∂^αΦ_j(x) + log λ · g(x)], g = ∂^α g₀,
     g₀ = −(−1)ⁿRⁿP_n(ζ/R)/(2π n!) (`kernel_log_coefficient`; zero for every K0 family and for 8 of the 16
     finite-part families). Corner pieces: the Duffy pyramids add ∫₀¹ ρ^e log ρ P(ρ) dρ exactly (shifted-Legendre
     coefficients of P, `∫ρ^m log ρ = −1/(m+1)²`); Euler's identity (power basis) adds −∫xⁿg. The two routes
     agree to 7e-16 … 9e-16 on K1 blocks. The block's h-scaling gains log h ∫g (`family_log_integrals`): a
     finite part depends on the unit of length (km), as does the wavenumber remainder's.
   - `image_moments`' closed forms (line of images) are used only where they apply (j ≤ 2, n_lat ≥ j).
   - **Gate** `scripts/gate_interface_omega2_blocks.py`: K1 against FP∫Y1 d²k from `interface_spectral`'s ω²
     integrand (q⁻² × entire at q → 0: Chebyshev head with exact modal moments and log q_s; Gauss panels
     above), no step shared: **5.1e-14, 6.0e-14** (full 4 × 10 × 9 × 9 blocks, two pairs whose images do not
     touch). q_s moved: 6e-14, 9e-14.
   - Identical media: reflected K1 = 0 to 2e-17 relative.
   - **Corner moments** (`scripts/gate_finite_part_corner_moments.py`, ~5 CPU-hours, run in 4 workers):
     26 moments in 9 families (Φ₀–Φ₄, d = −3 … 1, with and without a log term, two static-type controls)
     against 25-digit Duffy quadrature with tanh-sinh in ρ: worst 8.3e-16 (Euler), 4.3e-16 (pyramids). A
     monomial with 3 + |n| + d ≤ 0 has no integral; both routes refuse it (Euler NaN, pyramids raise).
   - **Transmitted (13 October).** The symbolic transmitted terms (582 in 61 families; about 3.5 h, checkpointed,
     after one run was lost to a container restart): spectral check exact on all 81 entries, structure identical
     to the rational runs. Gate: (3,0,0) ← (−1,0,0) **5.4e-14**, (3,2,0) ← (−1,0,0) **3.6e-14**.
   - **Touching pairs (13 October), the high-q tail analytically** (`scripts/gate_interface_touching_blocks.py`).
     The vertical moments split exactly, `E_k(c) = P_k(1/c) + e^{−2c} Q_k(1/c)`; only cells on the interface have an
     O(1) prefactor, so `Y = Y_alg + O(e^{−2qh})` with `Y_alg = Σ_p q^p Σ_n e^{inφ} H_p[n] × lateral × phase` (p from
     −8 to 1). `FP∫Y_alg` over the whole plane in closed form: Weber–Schafheitlin
     `(1/4π²)FP∫ q^p e^{inφ} e^{ik·ρ} = (iⁿe^{inθ}/2π) 2^{p+1}Γ((|n|+p+2)/2)/Γ((|n|−p)/2) ρ^{−p−2}` (continuation in p),
     then the lateral s-form weights in 2-D, polar about the touching corner, the radial integral exact with its pole /
     log, the Laurent series of Γ-ratio, radial integral and h^{−ε} multiplied and the constant term taken. Its head
     below q_s also in closed form (Taylor series of J_n; the Chebyshev version lost 3e-10 to a cancellation of two
     terms 20× the block). Y − Y_alg by Gauss to qh = 24. No step shared with the 3-D closed form.
     **K0 and K1, four touching pairs** (self-image, edge, corner, across), full blocks: 3.4e-14 … 4.9e-13, each at
     the reference's own precision (q_s moved: 3.8e-14 … 4.3e-13).
4. **Previously: their cell moments.** Extend `image_block`'s master integrals (homogeneity, the line of images) to
   Φ₃, Φ₄. Gate against the 40-digit wavenumber integrals of `∫ Y1` (the verification route).
4. **The remainder and the far pairs** in space (pointwise Sommerfeld kernel, Gauss / Duffy); gate the whole
   block against `interface_spectral` (sharp cutoff, `--full`).
5. **The remainder R(ω) = FP∫(Y − Y0 − ω²Y1) (started 13 October).** For pairs that do not touch, the block
   converges exponentially in q and the split adds nothing; R matters for touching pairs. Self-image pair,
   R / K = 1.3e-2 at ωh/β = 1/6 and 1.17 at ωh/β = 1; truncating R's tail at Q0h = 10 / 20 changes it by
   3.5e-10 / 6e-11 of K (ωh/β = 1/6) and 3e-7 / 5e-9 (ωh/β = 1): the Q0⁻⁶ law of the dropped E.
   **Decision (user): k-space, the tail carried by the ω⁴ term.**
   - **Done: A1 exact.** `frequency_independent_parts(..., a1_route="closed")` (default):
     `A1 = K1 − FP∫_{q<Q0} Y1` (`fp_low_integral`: Chebyshev head with exact modal finite parts, Gauss panels to
     Q0). The octave route (to Q1, then the q⁻⁴ law) converges to it: 4.8e-7 → 8.0e-10 of A1 for Q1h 160 → 640
     (self-image), so the old A1 carried ~5e-7 of itself (6e-13 of K at ωh/β = 1/6, 2e-11 at 1). Gate S3 unchanged
     (3.6e-10, 6.3e-11; T 2.2e-10, 4.2e-11): what remains is the dropped E.
   - **The ω⁴ term (13 October).** `scripts/derive_interface_omega4.py` (any order; exact rational media, 17 s;
     symbolic twin `Mathematica/InterfaceOmega4.wl`, not run here): residual after ω⁴ falls 64× per halving of ω;
     identical media R_n = 0, T_n = Kelvin's. Canonical terms (order 2): R 1245 in 91 families, T 1002 in 76,
     potentials to Φ₆, ~550 finite-part terms each; spectral check exact. (Then for the gate media only; any
     media since 14 October, below.)
   - **K2 gate.** Touching pairs against the plane waves (order 2): reference-limited, 5.4e-12 at q_s h = 3.
   - **A2 and the split to ω⁴.** `frequency_independent_parts(..., omega4=True)` → (A0, A1, A2),
     `A2 = K2 − FP∫_{q<Q0}Y2`; `interface_block` adds ω⁴A2. A2 is ~1e-7 of K2 (seven digits cancel). The small-q
     head is q⁻⁴ × entire: **the sampled head has a narrow window** (q_s h = 3: 24 nodes left 6e-9 of K2, 40 nodes
     4e-11, 48 nodes 5e-10); found because K2 − low disagreed with the octaves of Y2 by 5% of A2 while K2 agreed
     with the touching gate, whose head shared the same 24-node sampling. Default now 40 nodes for N = 4.
   - **Against the 40-digit references**, to ω² → to ω⁴ (relative to the block):
     R, ωh/β = 1/6: Q0h 10 3.6e-10 → 1.1e-11; ωh/β = 1: Q0h 10 3.1e-7 → 3.1e-9, 20 4.8e-9 → 5.4e-11, 40 6.3e-11 → 4.6e-11.
     T, ωh/β = 1/6: Q0h 10 2.2e-10 → 1.4e-13, 20 3.5e-12 → 8.0e-14; ωh/β = 1: 10 2.0e-7 → 1.5e-9, 20 3.2e-9 → 6.9e-11,
     40 4.2e-11 → 6.5e-11. The ~5e-11 floor at ωh/β = 1 is the head's: ω⁴K2 ≈ 4 K there, so 4e-11 of K2 shows.
   - **The analytic small-q head — done (13 October).** `cubic_scattering/interface_lateral.py`: the frame's
     Laurent polynomial × the vertical moments as exact power series in q × the lateral moments by the Bessel series
     against the exact weights; every power and harmonic's finite part exact; q_s = min(1/h, 4/R) so the series lose
     few digits. Default in `fp_low_integral` (`head="sampled"` kept). Checks: order 0 (integrable) against plain
     Gauss 1.6e-15, 1.4e-14; order 1 against the sampled head 3e-14 … 6e-14; **order 2: K2 = analytic low + octaves
     of Y2 to 1.9e-15 (R), 1.2e-15 (T) of K2** (sampled: 4e-11 at best, 6e-9 at 24 nodes).
     Found: the sampled head's order-0 case read the log coefficient from co[-1] (N = 0); fixed (never on the
     production path, where A0 uses plain panels).
   - **Against the 40-digit references, to ω⁴ with the analytic head:** ωh/β = 1: R Q0h 10 3.1e-9, 20 1.2e-11,
     40 8.8e-13; T 1.6e-9, 6.1e-12, 8.1e-14 — the floor gone and the Q0⁻⁸ law (×260 per doubling of Q0).
     ωh/β = 1/6: T 6.3e-14, 1.2e-14; R 1.1e-11, 2.6e-11, the known rounding of D's mode sums at that frequency.
   - **The ω⁴ term with symbolic media — done (14 October).** `Mathematica/InterfaceOmega4.wl` run by the user;
     `InterfaceOmega4.json` agrees **identically** with the python twin at its rational media (all 18 entries,
     exact difference 0; the twin's medium B is the binary rational of 6.5, 3.6, 2.8, hence ~2e-17 at 13/2, 18/5,
     14/5), and R2 = 0 for identical media. Each entry is X·poly/(q⁵ D_e), D_e a product of μ_A, μ_B, λ+2μ,
     μ_A+μ_B and two mixed factors: `derive_interface_image_terms.py ... 2` (symbolic) collects the terms over the
     common denominator on sparse rings over QQ_I (Expr `diff` was hours), cancelling at the end by exact division
     (sympy `factor`/`cancel` timed out at 60 min); 5 min per kind on four workers. Same 1245/1002 terms as the
     gate's; spectrum check exact for both kinds. `_general(kind, 2)` reads it; the media guards are gone; the gate
     term files kept as `*_gate.json`. **`scripts/gate_interface_omega4_blocks.py`**: K2 closed form vs analytic
     head + Gauss panels, gate media and (4, 2.25, 2)/(7, 4, 3): 4.6e-16 … 2.9e-15, ALL PASS.
     Coefficient loading compiled directly (`interface_image._coefficient`): 1.3 s instead of ~12 min per ω⁴ kind,
     equal to lambdify to 2.7e-16.
6. **Then step 2** (the scattering solve), on these blocks.

**Step 2 — a scattering solve near an interface.**
- **Done (14 October): the dense solve and its gate.** `cubic_scattering/interface_solver.py`:
  `InterfaceGrid` (cells (iz, ix, iy), centres h(2iz+1, 2ix, 2iy), A for iz < 0), `InterfaceCouplings`
  (frequency-independent parts once per lateral symmetry class — signed permutations fixing z, `fft._transform` —
  in parallel), `blocks(ω)` (interface D batched by `truncated_blocks`, sources in B mirrored; whole-space blocks
  per cube orbit: Gauss for touching, `far_block_series` otherwise), `reference_waves` / `reference_moments`
  (incident + R_d in A, T_d in B, complex k), `cell_contrasts` (each cell against its own side), `assemble`,
  `apply_coupling` (the matvec), `solve_interface`. Tests `test_interface_solver.py` (9): reference field
  continuous (P, SV, SH), symmetry map vs direct 8e-16, batched blocks vs pair by pair 1e-11, identical-media
  solve, matvec vs assembled.
  **Gate `scripts/gate_interface_solve.py`** (8 cells straddling the interface, 10% contrast, 4 receiver cells,
  ωh/β_A = 1/6, oblique P): identical media vs Paper 2's homogeneous solve 3.1e-12 (reference field 2.9e-15);
  Born — scattered field solved directly, (M − KE)s = KEψ₀, Richardson in ε — against couplings by an independent
  plane-wave (mode) integral from `interface_rt` alone: 4.9–5.6e-11 (one block 1.7e-12); p: |p0 − p2| 8.1e-3,
  |p1 − p2| 2.2e-4; cost (n_source 20, n_test 10, 78 classes): set-up 871 s on 4 workers, first frequency 48 s,
  each further 29 s (≈16 s touching whole-space Gauss blocks, ≈13 s interface D), solves 0.1–0.3 s.
  Found on the way: `interface_lateral.rect_poly_moments` wrapped in int64 for numpy-integer corners (callers with
  Python ints unaffected); `frame_laurent` failed on an identically zero kernel (identical media);
  `near_block_series` is slower than Gauss per frequency (8–21 s vs 1.5–3 s; minutes on first use).
- **The near series per frequency (15 October).** `near_block_series` re-contracted the family tables with the
  universal moments at every frequency (8–21 s per block); now `near_series_coefficients(m, offset, …)` contracts
  each term once (grouped by derivative count, h^(6+m−d)), and a frequency is a weighted sum: 5.8 ms (self),
  8.3 ms (edge) once its terms exist; new vs old ≤ 5.2e-15, vs Gauss 1.2e-15, 4.3e-15; Paper 2's series tests
  (51) pass. **But a higher frequency needs more terms m**, and each new odd m costs minutes per touching orbit:
  even m are polynomials (Gauss exact, ms), odd m go through the closed master integrals (3–10 s per derivative
  multi-index, ~41 per m). Gate cost with the series (n_source 20, n_test 10): first frequency 3683 s, a 10%
  higher one 387 s. The moments are universal (no medium, h or ω). **Decision (user): package data.**
  `scripts/build_near_series_moments.py` (resumable, 4 workers, ~2 h) writes `scripts/data/near_series_moments.npz`:
  4284 moments, m to 31 (ωh/β ≤ 1/2), the four touching orbits, n_source 35 / n_test 10 (smaller bases are exact
  slices), 5.1 MB; `universal_moment` reads it, else computes (`universal_moment_direct`). Store vs computed
  1.4e-15; the four orbits at n_source 20 cost 0.3–0.55 s per frequency from the first one. **Gate cost now:
  set-up 804 s, first frequency 31 s, each further 13.4 s** (was 29 s) — the rest is the interface D.
- **The ω⁴ split and Q0 (15 October).** `scripts/measure_interface_q0_omega4.py`: **D's time per pair does not
  depend on Q0** (0.03–0.06 s for Q0h = 3 … 10: fixed work, not radial panels), so lowering Q0 saves nothing. The
  ω⁴ split buys accuracy at the same Q0 (worst of three pairs, Q0h = 10: ωh/β = 1/6 3.5e-10 → 4.8e-11, the rounding
  floor, reached at Q0h = 5; 1/2: 2.5e-8 → 6.2e-11). The time was in `_interpolate` (S_n at the radial nodes, per
  pair): now once per (lateral offset, rule) and one matrix product per panel. `InterfaceCouplings` defaults to the
  ω⁴ split (no window). **Gate: set-up 2062 s, first frequency 24 s, each further 3.5 s**; Born, p unchanged.
- **Next in step 2.** The per-frequency cost (touching whole-space blocks: Paper 2's frequency-independent near
  parts; interface D: fewer radial nodes with the ω⁴ split's lower Q0); GMRES with `apply_coupling` and a
  block-Jacobi preconditioner for larger scatterers; then Stage A (layer partition).
- **The problem.** A Legendre-cell scatterer (Paper 1's single-site T-matrix, Paper 2's direct blocks)
  touching or near a welded interface. The reflected and transmitted couplings come from step 1. Solve with
  GMRES (Stage A below) over a frequency sweep: setup once, then each frequency.
- **Checks.**
  - **Identical media:** the solution equals the homogeneous solver's to its tolerance.
  - **A weak scatterer, different media:** against the Born approximation computed independently with the
    reflectivity plane waves (`kennett_layers` / `stratified_march` spectra).
  - **Convergence in p** (0, 1, 2) at Paper 3's order.
  - **Cost** per frequency against setup.
- **Then.** Stage A/B below (layer stacks: reflections from interfaces the cells do not touch are smooth
  and go to the plane waves), the two-way march.

**Paper status (`LatexPDFs/InterfaceCouplingIntegrals`, 18 pp; to be resumed later).**
- **Contents.** Static image and transmission in closed form (356 / 275 terms, B₁, B₂), Legendre/pyramid
  blocks, Table 1 (Q⁻² → Q⁻⁴), the `ω²` section with Table 4 (Q⁻⁶, two frequencies, reflected and
  transmitted), the sweep implementation and its verification rows.
- **Open for the paper.**
  - ~~step 1's locality result~~ — done 14 October: Table tab:locality and the radius (12/16/24 h for
    1e-6/1e-8/1e-10) in the plane-wave-route paragraph of the Verification section;
  - whether to include step 2's end-to-end solve as an application section;
  - one stated limitation (base integrals for cubic corner boxes only).
- **Older open items.**
  - Scan Papers 1 and 3 for conclusions resting on numpy `leggauss` weights.
  - The same-sign ~1e-15 bias in Paper 2's 3-D pieces.
  - An Ooura–Mori check of the touching-image block.

---

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
