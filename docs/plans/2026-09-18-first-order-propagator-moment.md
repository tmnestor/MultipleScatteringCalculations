# The First-Order Propagator Moment — Implementation Plan

> ## ✅ THIRD REVISION, 2026-09-18 — the design question is settled; both faults closed
>
> `scripts/gate_first_order_schwinger.py`, **10/10**. Read this before the second
> banner below, which it supersedes on every point except the warnings.
>
> ### The settlement — route 2, and it is exact
>
> The Born term and the Schwinger term are **not** the same situation, which is
> why they take different forms:
>
> - **Born.** The contrast's indicator and the test function sit at the *same*
>   point. `∂α(1_V c)` there would multiply a surface layer by an indicator,
>   which is undefined. The transfer is not a convenience — it is the only
>   well-posed form, and `ΔC_eff` is right.
> - **Schwinger.** The two indicators are separated by `Γ`. Nothing coincides,
>   the surface layer is harmless, and in the lateral Fourier domain it never has
>   to be formed at all:
>
>   ```
>   FT[ ∂α (1_V c) ] = i k_α FT[ 1_V c ]        exact
>   ```
>
>   The derivative rides on the **transform**; the indicator stays inside the
>   form factor, undifferentiated.
>
> ▶ **Route 1 (surface terms on the faces) is not needed** — same number, harder.
>
> Every derivative in `A` is lateral, so all four derivative slots of one entry
> are `ik_α` at the single wavenumber: left-outer → test polynomial (the
> augmentation, unchanged), left-inner → propagated field, right-outer → source
> transform, right-inner → trial polynomial.
>
> Checked, not asserted: `∫f ∂₁1_S = −∫_S ∂₁f` via the k-route to **1.0e-13**,
> zero imaginary residue, opposite sign rejected by 2×.
>
> ### Fault 1 (basis) — fixed
>
> `amat_paper` / `amat_paper_batch`, transcribed from the validated Mathematica
> blocks. `A_paper = S⁻¹ A_thesis S` to **1.2e-26**, spectra to 8.6e-12.
>
> 🔑 **The quasi-Hamiltonian relation is `A(−k)ᵀJ₆ = −J₆A(k)`, not the fixed-k
> form.** The operator statement carries `∂ᵀ = −∂`, so the transpose flips the
> wavenumber. The fixed-k version fails by 4.7e6 arithmetic floors — it is now a
> negative control. An easy and silent slip.
>
> ### Fault 2 (one ΔA where two are needed) — fixed and verified
>
> The kernel connects the component the **left** operator READS to the component
> the **right** operator WRITES: `kern[(m,n)][colL, rowR]`. The old assembly used
> `[idx, col]`, which is the single-operator object.
>
> Two new arbiters, neither owing anything to the moment machinery:
>
> 1. **Γ inverted laterally IS the Kelvin tensor** — `7.3e-4` and converging,
>    off-diagonal leakage `2.9e-4`. This fixes the source normalisation, which
>    the bridge gate could not reach (it compares two depths, so it tests
>    propagation, not the strength of the jump).
> 2. **The density channel** reproduces `iω⁵Δρ²M` with `M = (a₀+b₀/3)D₀` to
>    `6.9e-5`, `D₀ = ∫∫1/r` computed in real space by the autocorrelation weight
>    plus a Duffy substitution (Jacobian `t²` cancels the `1/r` outright).
>
> ### 🔍 What that comparison found in code already committed
>
> `propagator_moment` is a **collocation** object; the Schwinger form is
> **Galerkin**. The gap factorises exactly (residual 3.6e-5):
>
> ```
> [ΔC G ΔC]₀₀ / (iω⁵Δρ²M) = iω · ( ∫_V 1/r + i·radiation ) / D₀
> ∫_V 1/r = 2.3800773640   D₀ = 1.8823126444   ratio = 1.2644431684
> ```
>
> The geometric factor is consistent with the project's settled finding that the
> single site is collocation — but the two **cannot be mixed without it**.
>
> ▶▶ **STILL OPEN: the `iω`.** `ΔC 𝒢 ΔC` carries two `J₆` test factors where
> `⟨J₆φ|ΔAΓΔA|ψ⟩` carries one, so a composing `𝒢` must absorb the trial-space
> normalisation. The natural candidate — an inverse Gram `⟨J₆ψ_k, ψ_l⟩` — is
> **degenerate on the rigid block** (the `J₆` contraction of a pure-velocity
> field with itself is zero), so it cannot simply be inverted. Settle this before
> quoting any first-order `T` as consistent.
>
> ### Quadrature: two opposite cases, do not swap them
>
> - **The propagator alone → POLAR.** It depends on `|k|` and on direction
>   separately, so it is not smooth at the origin and a Cartesian tensor rule
>   converges as `1/n` (4.0e-2 → 1.2e-2 over a 9× refinement, against 5.4e-3 →
>   7.3e-4 in polar).
> - **The lateral moment → CARTESIAN.** Its form factors are separable sinc
>   products, which polar cannot resolve. (Unchanged from the second banner.)
> - **Panel it either way.** A single Gauss rule puts its nodes at the *ends*,
>   which is the opposite of where these integrands keep their mass. This alone
>   accounted for a 24% error that looked like a physics discrepancy.
>
> ⚠ The density-channel quadrature agrees at `1e-4` but is **not monotone**
> (6.9e-5 → 1.2e-3 as the box widens); treat it as converged to `1e-3`, no better.
>
> ### Next
>
> 1. The `iω` above.
> 2. The **derivative-carrying channels** — the density channel's contrast
>    operator is multiplicative, so it does not exercise the `ik` rule *at all*.
>    `Δλ`, `Δμ` are what test the thing Part 1 settled.
> 3. Then the remaining eight trial functions, and read `B` off.

> ## ⚠⚠ SECOND REVISION, 2026-09-18 — the assembly computed the wrong object, in the wrong basis
>
> `scripts/gate_first_order_lateral_moment.py` assembles the lateral integral and
> **its result is not usable.** The integrand is `O(1e9)` and the integral came
> out `O(1e-2)` — eleven orders of cancellation — and the polar and Cartesian
> assemblies disagreed by ~`1e4`. Both faults were found by the density-channel
> diagnostic, the smallest case with a known closed-form answer.
>
> ### Fault 1 — basis mismatch (fixable, fix known)
>
> `qfield_of` and `delta_a_terms` work in the **paper** basis
> `(−τ₁₃, −τ₂₃, −τ₃₃, v₁, v₂, v₃)`. `amat` in the lateral script builds `A` in the
> **thesis** basis `(u_z, u_x, u_y, T_zz, T_xz, T_yz)`. A rigid translation lands
> in slot 3, which is a *velocity* in one basis and `T_zz` in the other.
>
> ▶ Fix: `K_paper = S⁻¹ K_thesis S`, with `S` the constant map already verified in
> `MatrixVectorWaveEquation.wl` (`b = S q`: `u = v/(−iω)`, `τ₃ = −q₁₋₃`, ordering
> `(1,2,3)→(3,1,2)`). Spectra checked to 1.9e-10 under the transform.
>
> ⚠ This is precisely the assumption the very first bridge gate was written to
> avoid making, and it was made anyway one layer further down.
>
> ### Fault 2 — one ΔA where the object needs two (NOT just a missing loop)
>
> The assembly loops once over the transferred terms, so it computes
> `⟨J₆ψ_k, Γ ΔA ψ_l⟩`. The Schwinger object is `⟨J₆ψ_k, ΔA Γ ΔA ψ_l⟩`.
>
> ▶▶ **And the augmentation does not carry over unchanged.** The transferred
> bilinear `b[φ,ψ] = Σ coef (D^a φ)_i (D^b ψ)_j` *is* `⟨φ, J₆ΔAψ⟩` — a single
> `ΔA`, with the derivatives already distributed between the two sides. That is
> what makes `ΔC_eff` free of surface deltas. The sandwiched form needs `ΔAψ_l`
> as an actual **source between two propagator legs**, i.e. `∂α(Δc·1_V·ψ)` — and
> differentiating the indicator is exactly what the transfer was introduced to
> avoid.
>
> 🛑 **This is a design question, not a bug: how does the augmentation compose
> with the Schwinger form?** Two candidate routes, neither yet worked:
> 1. put the surface terms back explicitly and evaluate them on the cube faces;
> 2. transfer within each leg separately, so each propagator sees a smooth source.
>
> **Settle that before writing any more quadrature.** Three rounds of quadrature
> engineering — scalar, vectorised, angular-reduced — were spent on an integrand
> that was both mis-based and mis-defined. The quadrature was never the problem.
>
> ### What survives, and is verified
>
> - Task 2's **angular reduction works**: `A(k cosθ, k sinθ) = R(θ)A(k,0)R(θ)ᵀ`
>   to **2.7e-16**, kernel to **6e-12**; one eigendecomposition per *radius*.
>   **13 s** where the Cartesian route timed out at 600 s. Angular convergence
>   clean (`2.3993e-3` at nth = 240, 480, 960).
> - It also removes a real defect: `k_zS` is doubly degenerate, `eig` splits that
>   eigenspace arbitrarily, and independently computed kernels disagree at ~1e-7.
>   Computing once and rotating makes that impossible by construction.
>   ⚠ Do **not** try to fix it by averaging near-degenerate eigenvalues — that was
>   tried and made the rotation residual *worse* (2.14), almost certainly by
>   disturbing the up/down classification.
> - Everything in steps 1–3e stands: `Γ`, the triangular `z` integrals, the
>   scaling, the asymptote as an `O(1/k)` rate, the `D₄` vanishing of the
>   principal-value part, and the closure test.
>
> ### The diagnostic that should have been run first
>
> The **density channel**: with `Δλ = Δμ = 0` only the multiplicative `iωΔρ` term
> is active, the answer is known in closed form (`A_u = 1/(1 − ω²ΔρΓ₀)`), and it
> exercises the same `J₆` contraction, `T[m,n]` weights and form factors. It
> found both faults in a single run. Start there next time.

> ## ⚠ REVISED 2026-09-18, after step 3d measured the tail integral
>
> **Task 1 as written said to "add the subtracted piece back in closed form",
> and meant in `k`-space. That cannot be done.** The measurement:
>
> ```
> Int sinc^2(k a) dk                =  Pi/a                      finite
> Int k^2 sinc^2(k a) e^-eps|k| dk  =  4/(4 a^2 eps + eps^3)
>                                   =  1/(a^2 eps) - eps/(4 a^4)
> ```
>
> **a pure pole with no finite part.** So `∫I_∞ d²k` is not a number, and adding
> it back restores exactly the divergence the subtraction was introduced to
> remove. A regulator does not rescue this; there is nothing finite to keep.
>
> **The cause, and why the method survives.** Forming `|Ŝ|²` first and then
> integrating term by term destroys the Parseval structure. A `K̂` tending to a
> constant `C` contributes `C∫_V f g dx` — local, finite, needing no regulator
> at all (verified in 1-D, both sides `1/2a`). The subtraction is exact; the
> **tail must be evaluated in real space**, and only the remainder stays in `k`.
>
> **The revised split is by representation, not magnitude:**
>
> | piece | real-space object | evaluation |
> |---|---|---|
> | isotropic, `−ν₁` and `Pδ_αβ` | delta | local, elementary |
> | traceless `k̂_αk̂_β − ½δ_αβ` | 2-D PV dipole, degree −2 | conditionally convergent |
> | remainder `I − I_∞` | — | numerically, in `k` |
>
> ▶ The traceless piece has vanishing angular average — the same conditional
> convergence the 3-D moment engine handles by peeling derivatives onto the cube
> faces, now in two dimensions. Task 1's remaining work is that piece over the
> square cross-section, plus the numerical remainder.
>
> **The kill criterion was NOT triggered.** It read: *if the delta has to be
> fitted to recover the known `B`, abandon the route.* Nothing was fitted —
> `−ν₁` and `μ` came out of the limit unprompted, and `ν₁` is the operator's own
> constant from `A₁₂`, not a new one.
>
> **What steps 1–3c established, unchanged by this revision:** `Γ` from spectral
> projectors (13 checks); the triangular `z` integrals and the assembly (5); the
> scaling, the power table, and the closed-form asymptote verified as an `O(1/k)`
> rate (5). Never form `MatrixExp[A dz]` — it is Thomson–Haskell and shows up as
> a decay rate going negative.


**Goal.** Derive the cube's self propagator moment from the first-order
matrix-vector wave equation itself, so that the single-site T-matrix is
first-order in *both* factors — coupling **and** propagator. This is the last
missing piece: the coupling is already derived from `ΔA`, but the propagator
moment is still taken from the ordinary Kelvin/Navier Green's tensor.

**Decisions taken before writing this plan** (Tod, 2026-09-18):

- the **sandwiched** Schwinger form, not plain Galerkin on `q`;
- **split-and-subtract** for the lateral wavenumber integral, not a bare window;
- **whole-space first**, layered second.

---

## What already exists — do not rebuild it

| Object | Where | Status |
|---|---|---|
| `A`, the operator matrix | `Mathematica/MatrixVectorWaveEquation.wl` | 25/25; two independent arbiters |
| `ΔA`, contrast operator, augmentation, `J₆` test space, `ΔC_eff` | `Mathematica/FirstOrderContrastOperator.wl` | 44/44 |
| `Γ`, the first-order Green's matrix, **bridged to this `A`** | `GlobalMatrix/layered_greens.py`, wrapped by `cubic_scattering/layered_correction.py`; gated by `scripts/gate_first_order_propagator_bridge.py` | corrected object propagates by `expm(A dz)` to the arithmetic floor; raw one fails by 14 orders |
| Cube self moments by integration, static **and** dynamic | `Mathematica/CubeSelfMomentExport.wl` on `CubeMomentCore.wl` | closed forms below; trace identity gated |
| The T-matrix assembly and its calibration | `scripts/gate_first_order_tmatrix.py` | 13/13 |

**The targets Task 1–3 must hit**, already in closed form by integration:

```
B = sqrt3 (lam+mu) / (6 pi mu (lam+2mu))
A = B - 1/(3 mu)
C = -(5 - 2 pi/sqrt3) B
G = (a0 + b0/3) (Del/2)^2 Int_{[-1,1]^3} dV/r
```

with `∫dV/r = −2(π + log 64 − 12 log(1+√3)) = 9.520309455918214…`.

Two independent routes agreeing is the evidence standard here. The k-space
route is the second route; these closed forms are the first.

---

## ⚠ THE TRAP THIS PROJECT HAS ALREADY PAID FOR, AND WILL HIT AGAIN

`docs/plans/2026-09-14-cartesian-directional-sweeps-stage2.md`, Task 1:

> **Branch points sit on the integration contour.** The kernel carries
> `1/ky_L` and `1/ky_T`, which vanish on the circles `|k_⊥| = kP = 7.54` and
> `kS = 12.57` rad/km — both inside the domain. At real `ω` a midpoint rule
> straddles an integrable `1/√` singularity and does not converge
> (1.9e-1 → 1.1e-1 over an 8× refinement, non-monotone). Stage 1 types `omega`
> **complex** throughout for exactly this reason. With damping 0.01–0.03
> convergence is restored.

**This applies here unchanged.** `Γ(z,z';k)` is built from `k_{z,c} =
sqrt(k_c² − k²)`, which vanishes at `|k| = k_P` and `|k| = k_S`, and the jump
condition puts `k_z` in a denominator. Our radial integral crosses both circles.

▶ **Type `ω` complex from the first line of Task 1.** Do not discover this at
Task 3. Damping 0.01–0.03, and report the undamped limit as a separate
convergence study, never as the working configuration.

---

## The object

```
[ DeltaC G DeltaC ]_kl  =  < J6 psi_k | DeltaA Gamma DeltaA | psi_l >
T = DeltaC (I - G DeltaC)^-1
```

Plain Galerkin on `q` is not available: its mass matrix `⟨J₆ψ_k, ψ_l⟩` is
antisymmetric, and an antisymmetric matrix in **odd** dimension (9) is singular.

In the mixed `(k₁, k₂, z)` representation the vertical integral is **exact** —
`Γ` is exponential in `|z−z'|`, so

```
Int Int_{-a}^{a} dz dz' exp(-q |z - z'|) = 2 (2 a q - 1 + exp(-2 a q)) / q^2
```

with no quadrature in `z` at all. The entire difficulty is lateral.

### Why the lateral integral diverges, and what the divergence *is*

As `|k| → ∞` the eigenvalues tend to `±|k|`, `Γ ~ exp(−|k||z−z'|)`, the
`z`-integral gives `~1/|k|`, and the box form factor gives `~1/k²`. With the
radial measure the displacement channel is `dk/k²` — convergent. The **strain**
channel carries two more powers of `k` and becomes `~dk` — divergent.

**That divergence is the Eshelby delta in k-space.** In real space the moment
engine keeps it by peeling derivatives onto the cube faces; here it is the
`k`-integral of a constant. A bare cutoff discards exactly the term whose
omission flips the sign of the depolarisation. Hence split-and-subtract: remove
the `k → ∞` asymptote analytically, integrate the convergent remainder, and add
the subtracted piece back **in closed form**, where the delta lives.

---

## Tasks

Each task ends in an executable gate. `plans/*.md` in this repo lag the code, so
the plan's status must be readable by *running* something, not by reading this
file.

### Task 1 — ONE NUMBER, END TO END. The task that can kill the plan.

Reproduce the **static shear moment `B`** alone, through the first-order k-space
route with split-and-subtract. Nothing else. `B` is the smallest object that
contains the entire difficulty — UV divergence and delta content both — and it
has an exact closed-form target.

Deliberately *not* the easy channel first: the displacement channel converges on
its own and would prove nothing about the hard part.

**Gate.**
1. `B` to 1e-10 relative against `√3(λ+μ)/(6πμ(λ+2μ))`.
2. The remainder integral converges **at a measured rate** under refinement, not
   merely to a plausible number.
3. The subtracted closed-form piece is shown to carry the delta: the trace
   identity `Σ_p M_11,pp = −(2λ+5μ)/(3μ(λ+2μ))` holds, and a version with the
   subtraction omitted **fails** it.

**Kill criteria.** If the remainder will not converge at a rate with `ω` complex
and damping 0.01–0.03, or if the delta does not come out of the closed-form
piece, stop and revise this plan in place with what the measurement showed.
Do not proceed to Task 2 on a Task 1 that "looks about right".

### Task 2 — the angular reduction

`Γ` depends on `k` only through `|k|`, while the box form factor is separable,
`sinc(k₁a)·sinc(k₂a)`, and the angular tensor structure is polynomial in `k̂`.
So the angular integral is analytic and the 2-D integral collapses to a **1-D
radial** integral with precomputed angular coefficients.

**Gate.** Reproduces Task 1's `B` at materially lower cost; the angular
coefficients verified against direct 2-D quadrature on at least two of them.

### Task 3 — the full static tensor

`A`, `B`, `C` and `G`.

**Gate.** All four against the closed forms; the trace identity; and the result
is `O_h`-symmetric — the k-space route must not break a symmetry the real-space
one respects. That last one is free and would catch an angular-coefficient slip
that the four numbers alone would not.

### Task 4 — the dynamic moment

Finite `ω`.

**Gate, as rates rather than bars.** `A` and `B` approach their static values as
`O(ω²)` — 100× per decade; `G` approaches its as `O(ω)` — 10× per decade,
because the zeroth moment carries the radiation reaction. `Im G` against
`V ω (1/α³ + 2/β³)/(12πρ)`, with the gap falling as `O(ka²)`.

⚠ `A₂₁ ~ iω` while `A₁₂` carries `1/iω`, so `Γ` has ω-scaling of both signs
across its blocks. Check the `ω→0` limit **block by block on its own scale**,
never as a norm. This repo has already lost twelve digits by `k_S r ~ 1e-6` to
exactly that cancellation.

### Task 5 — the fully first-order T

Assemble `T = ΔC(I − 𝒢ΔC)⁻¹` with `𝒢` from Task 4.

**Gate.** Born limit unchanged; density channel exactly `1/(1 − ω²ΔρΓ₀)`; and
the whole T compared channel-by-channel against the Kelvin-route T already in
`scripts/gate_first_order_tmatrix.py`. In a whole-space background the two
*must* agree — that is the point of doing whole-space first.

### Task 6 — the layered background

The payoff. `Γ` becomes the stratified propagator, which already exists and is
bridged to `A`. There is no whole-space `G` here, so the moment method has **no
competitor** and this route is the only one.

**Gate.** The uniform limit of the layered assembly reduces to Task 5. Any
further claim needs an arbiter chosen in advance, not after seeing the number.

---

## Open questions, to settle at the task that needs them

1. **`k_par → 0` degeneracy** (Task 1). At normal incidence the P–SV
   eigenvectors degenerate and `k_zS = k_zH`. Handle by Mathematica limit, or by
   never placing a node there? The first is cleaner; the second may suffice.
2. **Where the subtraction is anchored** (Task 1). Subtract the asymptote of the
   full integrand, or of the `z`-integrated kernel? The second is cheaper; only
   the first is obviously exact.
3. **Whether the angular coefficients depend on `ω`** (Task 2). If not, they are
   computed once for all of Tasks 3–5.

---

## What would make this plan wrong

Recorded now, so a later reader can tell refutation from drift:

- if the UV subtraction cannot be made exact and the delta has to be *fitted* to
  recover the known `B`, the route is not a derivation and should be abandoned
  rather than tuned;
- if Task 5 shows the two routes disagreeing in a whole-space background, one of
  them is wrong and the plan does not proceed to Task 6 until it is known which;
- if the complex-`ω` damping needed for convergence is large enough to move the
  answer at the tolerance of Task 1, the contour treatment needs rethinking, not
  a looser tolerance.
