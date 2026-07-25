# Marine3D Phase 2 — 2½-D Disorder-Resolved Crust — Design

- **Date:** 2026-07-25
- **Status:** Draft for review
- **Author:** Tod Nestor (with Claude Code)
- **Topic:** Per-voxel $T_0$ screens and intra-plane coupling on a 2½-D crust, solved by a matrix-free GMRES matvec, validated against an independently re-derived Wolfram reference, and recorded in LaTeX.
- **Parent:** [`2026-06-26-disorder-resolved-3d-marine-scattering-design.md`](2026-06-26-disorder-resolved-3d-marine-scattering-design.md) (Phase 2 of its roadmap)
- **Predecessor:** Phase 0 + Phase 1 complete; seabed primary fixed 2026-07-25 (Marine3D `dfde898`)

---

## 1. Objective

Give the crust lateral structure and resolve the resulting multiple scattering.

The Phase-1 crust is a stack of identical transparent planes — a homogeneous block with a numerical $z$-grid. Phase 2 puts a per-voxel contrast field on that grid and solves the resulting Lippmann–Schwinger system by GMRES, with the matrix–vector product built from the thesis Chapter 5 operator splitting.

Scope is deliberately **2½-D**: heterogeneity varies in $(z,x)$ and is invariant in $y$. This is not a code limitation — `horizontal_greens.py` is already 3-D — but a validation choice. The 2½-D restriction is the only regime where the thesis provides an independent formulation to check against. Full 3-D $\mathbf{P}^x(x,y)$ remains Phase 3.

**Three deliverables, not one.** Per the project methodology (`CLAUDE.md`, *Research Methodology*), the LaTeX note is the product; the Wolfram reference and the Python solver are what license it to be written.

## 2. Anchor: thesis Chapter 5

Inherited unchanged from `GstratRep.tex`:

- The matrix Lippmann–Schwinger equation (`LSmat`)
  $$\mathcal{V}_{int} = \mathcal{V}_{inc} + \mathcal{Q}^\partial\,\Delta\mathcal{S}\,\mathcal{V}_{int},
  \qquad \mathcal{Q}^\partial \equiv (\mathbf{I}-\mathcal{S}_{int}\mathcal{E})^{-1}\mathcal{S}_{int} .$$
- The standard splitting $\mathcal{S}_{layer} = \mathcal{E} + \Delta\mathcal{S}$ (`Ssplit`) — appropriate here because the heterogeneity is weak relative to a reference that changes sharply at the three real interfaces.
- The $\mathcal{Q}^\partial$ apply as two triangular solves (`ULsweeps`): an up-sweep by back substitution (`Upsweep`) and a down-sweep by forward substitution (`Downsweep`).
- The efficiency property: $\mathcal{Q}^\partial$ depends only on the **reference** medium, so the Riccati operators are built once per frequency and every GMRES iterate is only the two sweeps plus a local apply.

Extended by this phase:

- $\Delta\mathcal{C}_{eff}$ becomes a **per-voxel** field on each crust plane rather than one scatterer per interface.
- $\mathbf{P}^x$, the intra-plane coupling, is introduced. The cherry-picked `interlayer_ms.py` sets it to zero (`if i_idx == j_idx: continue` in `build_interlayer_greens_matrix_9x9`), which its own docstring names as an open research question: *"Intralayer (side) scattering is neglected — the research question is when this is valid."*

## 3. What Phase 2 delivers, and where it plugs in

**The deliverable is a heterogeneity-dressed PP reflectivity at the seabed datum** — the same object `kennett_reflectivity_batch` returns, shape `(np_slow, nfreq)`, same datum, same convention. Everything downstream is Phase-1 code reused unchanged: water-column phase, the closed-form multiple series $E^2R/(1+E^2R)$, source and receiver ghosts, Bessel summation, damped IFFT.

Two reasons this seam is the right one:

1. The heterogeneous gather inherits the seabed physics fixed and verified on 2026-07-25, rather than re-deriving it.
2. Every validation arbiter lives at the reflectivity level, so Phase 2 can be closed without touching the gather pipeline at all.

## 4. Model

The Phase-1 stack with disorder added to the crust:

- **Ocean** — `FluidLayer`, unchanged. Free surface, fluid–solid seabed.
- **Crust** — $N_z$ planes, each $N_x$ voxels of pitch $a$ along $x$, invariant in $y$. Every voxel carries its own $(\Delta\lambda, \Delta\mu, \Delta\rho)$ **relative to the crust mean**, mapped to a 9×9 $(u,\varepsilon)$ $T_0$. The planes remain transparent in the *reference*: all crust structure, lateral and vertical, lives in the screens.
- **Half-space** — unchanged, stiffer and denser than the crust.

Reference-reflecting interfaces remain exactly three: free surface, seabed, crust base.

### 4.1 The $x$-lattice is periodic, deliberately

$L = N_x a$ with lattice harmonics $k_x = 2\pi n / L$. This is a modelling commitment, not an implementation detail:

- FFT convolution for $\mathbf{P}^x$ is exact rather than approximate.
- $\mathbf{P}^z$ is diagonal in $k_x$, which is what makes the matvec cheap.
- The periodic-plane arbiter (§7 rung 4) becomes directly usable.

Two costs, both gated in §8: wrap-around unless $L$ exceeds the survey aperture, and **open diffraction orders** once a harmonic becomes propagating.

### 4.2 Two distinct slowness grids — Bloch incidence, lattice harmonics

These must not be conflated, and the distinction drives the solver's outer loop.

- **Incident slowness $p$** — the gather's own grid, `p_samples` from `compute_shot_gather` ($p = k\,\mathrm{d}p$, `np_slow` values). This is what the Bessel summation consumes and what the delivered reflectivity must be indexed by.
- **Lattice harmonics $n$** — the periodic crust scatters an incident Bloch wave $k_x = \omega p$ into $k_x + 2\pi n/L$. There are exactly $N_x$ of them, and they are the solver's *internal* degrees of freedom.

So the solve is **per incident $p$**: fix the Bloch wavenumber $k_x = \omega p$, solve the $N_x$-harmonic system, and extract the **specular ($n=0$)** reflected amplitude as `RRd_PP[p, ω]`. The delivered object then lands directly on the gather's grid with no interpolation, and the Phase-1 pipeline is reused verbatim.

This is also *why* the diffraction gate is load-bearing rather than cosmetic. While only $n=0$ propagates, the disordered crust presents an effectively laterally-invariant response to the far field and a single specular coefficient is a complete description. The moment an $n \neq 0$ order opens, energy leaves in a direction the gather's slowness decomposition cannot represent, and a specular-only reflectivity is silently wrong rather than approximate. Same conclusion reached in the Phase 3b intra-plane energy-balance work.

## 5. Architecture

### 5.1 The matvec

One outer solve per incident slowness $p$ (§4.2). Within it, each GMRES iterate alternates domains. $\psi$ is the exciting field, indexed by (plane, voxel, 9-component):

```
ψ  (real space in x, per voxel)
 → apply ΔC_eff     local, block-diagonal per voxel        O(N_z · N_x · 81)
 → FFT  x → k_x
 → apply P^z        diagonal in k_x; Riccati up/down sweep O(N_z · N_x · 81)
 → IFFT k_x → x
 → apply P^x        circulant in x, applied by FFT         O(N_z · N_x log N_x · 81)
```

The $\mathbf{P}^z$ cost is linear in $N_z$, not quadratic: it is applied as the two triangular sweeps (`Upsweep` / `Downsweep`), never as the dense $N_z \times N_z$ block Green's matrix. That distinction is the whole point of the thesis splitting, and it is what the retained dense multi-p arbiter (§7 rung 6) is checked against.

The residual $(\mathbf{I} - [\mathbf{P}^z + \mathbf{P}^x]\Delta\mathcal{C}_{eff})\psi$ is never assembled as a matrix. The Riccati operators, which depend only on the reference, are built once per frequency and reused across all iterates.

**Born is one iterate** — the first Jacobi step of the same operator, obtained by replacing the exciting field with the incident field. It is computed by the same code path, not a separate implementation.

### 5.2 Modules

New in `Marine3D/marine3d/`. Each has one purpose, a stated interface, and is testable without the others.

| Module | Owns | Depends on |
|---|---|---|
| `crust_field.py` | per-voxel $(\Delta\lambda,\Delta\mu,\Delta\rho)$ → 9×9 $T_0$; validity-floor enforcement | `tmatrix.effective_contrasts` |
| `intraplane_px.py` | $\mathbf{P}^x$ apply: circulant-in-$x$ intra-plane $G$, FFT-applied | `tmatrix.horizontal_greens` |
| `vertical_pz.py` | $\mathbf{P}^z$ apply: inter-plane **and** self-plane reverberation, diagonal in $p$ | `gmm.interlayer_ms`, `gmm.layered_greens` |
| `matvec_25d.py` | $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})$ apply, GMRES driver, Born | the three above |
| `reflectivity_25d.py` | outer loop over incident $p$ and $\omega$; specular ($n=0$) extraction → dressed `RRd_PP(p, ω)` at the seabed datum; diffraction gate | `matvec_25d`, Phase-1 Kennett |

### 5.3 The $G_{ii}$ contract

The zeroed diagonal block in `build_interlayer_greens_matrix_9x9` is doing two physically distinct jobs at once, and Phase 2 must separate them explicitly:

$$G_{ii}^{\text{strat}} \;=\; \underbrace{G_{ii}^{\text{direct}}}_{\textstyle \mathbf{P}^x} \;+\; \underbrace{G_{ii}^{\text{reverb}}}_{\textstyle \mathbf{P}^z}$$

- $G_{ii}^{\text{direct}}$ — coupling between voxels in the same plane through the homogeneous whole-space crust. This is what `horizontal_greens.py` supplies and what "$\mathbf{P}^x$" conventionally means.
- $G_{ii}^{\text{reverb}}$ — energy leaving a plane, reflecting off the free surface, seabed, or crust base, and returning to the **same** plane. This is a $\mathbf{P}^z$ effect, already available as `layered_greens_9x9(source_iface=i, receiver_iface=i)` minus its whole-space direct part.

Restoring only the first silently discards seabed and free-surface reverberation onto the crust — a large omission in a marine model whose seabed reflection coefficient is ≈0.79. Restoring both without the subtraction double-counts the direct term. **The subtraction is the contract between `intraplane_px` and `vertical_pz`, and it gets a dedicated test (§7 rung 3) asserting the two pieces sum to the unsubtracted stratified $G_{ii}$.** This is the most likely site of a silent factor-of-two in the whole phase.

## 6. The three tracks

### Track L — LaTeX (the product)

`LatexPDFs/Marine25DDisorderResolved/Marine25DDisorderResolved.tex`, following the established companion-note convention: self-contained LuaLaTeX article, compiled in place with two passes.

Records:

- the anchor to thesis Ch5, stating precisely what is inherited and what is extended;
- the 2½-D disorder-resolved specialisation — per-voxel $T_0$ screens on a periodic $x$-lattice;
- the operator split, with the $G_{ii}$ contract of §5.3 written as an equation rather than prose;
- the domain-alternating matvec as a numbered algorithm, with costs;
- Born as one iterate, and GMRES as the resummation of the full multiple-scattering series;
- the diffraction-order gate as an explicit inequality in $L$, $p$, $\omega$;
- a validation table.

**The validation table is written last, carrying measured residuals from Tracks W and P — not projected ones.** If a gate fails, the table records the failure. Per `CLAUDE.md`, a formula reaching the `.tex` without a passing check behind it is unvalidated and must be labelled as such.

### Track W — Wolfram (the arbiter)

`Mathematica/Marine25DReference.wl`.

Assembles the full Lippmann–Schwinger system directly from the stratified Green's function and the per-voxel $T_0$, and solves it by direct high-precision inversion on small configurations ($N_z \in \{1,2,3\}$, $N_x \in \{2,4,8\}$, a handful of slownesses).

**No splitting, no sweeps, no FFT.** Sharing no algorithmic structure with Track P is the entire point; an arbiter that reuses the method under test proves nothing. It emits a fixture file (JSON, high precision) that the Python tests load, so the cross-check is a regression rather than a live re-run.

Applicable pitfalls from prior work: `NIntegrate` at `PrecisionGoal → 12, AccuracyGoal → 12, MaxRecursion → 25` on singular integrands; `Chop[Re[N[expr, 20]]]` then `ToString` before returning through `wolframclient`.

### Track P — Python (the implementation)

The five modules of §5.2, plus the existing dense multi-p solver retained as a small-case arbiter.

## 7. Validation ladder

Ordered so a failure localises. Rungs 1–3 need no solver.

| # | Gate | What it tests | Tolerance |
|---|---|---|---|
| 1 | $\Delta = 0$ → Phase-1 reference reflectivity | plumbing, datum | machine precision |
| 2 | Laterally uniform plane → Kennett stack with an effective layer | $p$-diagonal path, zero lateral coupling | ≤ single-site $T_0$ accuracy |
| 3 | $G^{\text{direct}}_{ii} + G^{\text{reverb}}_{ii} = G^{\text{strat}}_{ii}$ | the §5.3 module contract | machine precision |
| 4 | Periodic plane → Kambe layer $R/T$ (existing, S-matrix-unitary) | $\mathbf{P}^x$ specifically | quadrature floor of the Kambe reference |
| 5 | Small disordered case → Wolfram direct solve | the whole operator, independently | high precision |
| 6 | GMRES matvec ≡ dense multi-p solver | matrix-free path vs dense | machine precision |
| 7 | Born = one GMRES iterate == single scattering | the splitting | machine precision |
| 8 | Reciprocity; energy balance undamped | physics invariants | reciprocity ≲ 1e-7, energy ≲ 1e-4 (Phase 3b achieved 4.1e-8 / 1.2e-5) |
| 9 | ≈8 iterations at 5%, ≈28 at 10% heterogeneity | thesis Ch6 convergence signature | order-of-magnitude |

Rung 4 reuses the validated Kambe layer $R/T$ from the intra-plane energy-balance work in `MultipleScatteringCalculations`. It is a reference **result**, consumed as stored numbers — not a cross-repo import.

Tests are local-only (`Marine3D/tests/`, gitignored), `pytest`, run via `conda run -n seismic`.

## 8. Fail-fast gates

Every one carries the four-element diagnostic: what is wrong, where (absolute path + dotted YAML key), a valid example, and the recovery step.

- **Open diffraction orders.** Hard `Abort` naming the offending $(p, \omega, L)$ and the order that opened. Precedent: this gate was load-bearing in the Phase 3b intra-plane energy balance — a periodic lattice is not a laterally invariant medium, and specular-only reasoning silently breaks once an order opens.
- **Wrap-around.** $L$ must exceed the survey aperture; validated at construction, and covered by a test asserting invariance under doubling $N_x$ at fixed physical content.
- **Validity floor.** $|\Delta| < \varphi \cdot \text{background}$ enforced when the field is built, not mid-solve. Anchored to the sphere-packing renormalisation study.
- **GMRES stagnation.** Reported with the full iterate history. Never silently truncated to a fixed count.

YAML is the single source of truth; every key required; no Python-side defaults.

## 9. Non-goals

- No 3-D $\mathbf{P}^x(x,y)$ — Phase 3.
- No forward-scattering / two-way marching splitting (thesis §`fscat`, Alg 5.2 directional sweeps). It is an attractive $O(N_x)$ alternative to the FFT convolution, but it is a *different* splitting and needs its own proof that it reproduces the exact intra-plane coupling. Phase 3 efficiency variant, validated against the Phase 2 result.
- No GPU/torch path — Phase 4. Module boundaries keep it reachable.
- No inversion, no ensemble/statistical modelling, no anisotropic background, no irregular interfaces.
- No change to the Phase-1 gather pipeline.

## 10. Risks

- **The $G_{ii}$ split (§5.3)** is the highest-risk item: both failure modes (dropped reverberation, double-counted direct term) produce plausible-looking output. Mitigated by making rung 3 a gate that runs before any solve.
- **Wolfram reference cost.** Track W is real work, not a checkbox, and it blocks rung 5. It is scoped small deliberately ($N_x \le 8$) — its job is to be independent, not large.
- **GMRES conditioning** at larger contrast. Crust-mean referencing keeps $|\Delta|$ small; rung 9 is the early-warning signal.
- **Single-site $T_0$ fidelity** bounds rung 2 from below. Known limit: the finite-$ka$ form-factor accuracy of the effective-contrast $T_0$. Do not attribute that residual to the Phase-2 operators.
- **Periodicity as physics.** A periodic $x$-lattice is a genuine modelling commitment. It must be stated in the LaTeX note, not buried in the FFT.

## 11. References

- Nestor (1996), PhD thesis, ANU — Ch 5 `GstratRep.tex` (`LSmat`, `Ssplit`, `ULsweeps`, `Upsweep`, `Downsweep`, `PstratDef`, §`fscat`); Ch 6 `VariationalSum.tex` (GMRES).
- Chin, Hedstrom & Thigpen (1984); Schmidt & Tango (1986); Wu (1994).
- Companion notes: `EwaldIntraPlanePropagator`, `ContrastSourceVIE`, `DirectionalSweepSolver`, `IntraPlaneSpectralSweep`.
- Marine3D: `gmm/interlayer_ms.py`, `gmm/layered_greens.py`, `gmm/riccati_solver.py`, `tmatrix/horizontal_greens.py`, `tmatrix/effective_contrasts.py`, `kennett/seismic_survey.py`.
