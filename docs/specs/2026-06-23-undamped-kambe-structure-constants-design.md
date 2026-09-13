# Design: Undamped Kambe Multipole Structure Constants `D[q,s]` (Phase 3b cycle 1)

**Date:** 2026-06-23
**Plan:** `IntraPlaneFoldyLax_Plan.md` — Phase 3b (undamped G0 + energy balance), **cycle 1 of 3**.
**Status:** design approved; spec for implementation.

**Strategic framing (2026-06-23, per Tod).** Route A is the *standard* Kambe / layer-KKR multipole
structure constant — textbook-established, hence trustworthy as an **independent ground-truth reference**.
It is built now so that a LATER investigation can validate the thesis's own **non-standard `z/x/y`-split
spectral lattice propagator (§3.3)** against it: the thesis method is expected to be much faster, and Route
A is the rigorous yardstick to certify it. Source of the `D[q,s]` formulas: **Kambe (1967) / layer-KKR
literature (Pendry/Modinos/Williams), NOT thesis §3.3** (§3.3 implements the spectral `D_z Q^z D_z^T`
propagator, which is the future target, not the multipole structure constants).

## 1. Goal & framing

Compute the **undamped** (`κ` real, no artificial absorption) planar lattice **structure constants**

```
D[q,s](k_par) = Sum_{R != 0}  h_q(κ|R|)  Y_q^s(R̂)  e^{i k_par · R}
```

for `q = 0 … 2·Nmax`, `s = -q … q`, where the lattice `{R}` lies in the x–y plane (`θ_R = π/2`) and
`h_q = h_q^(1)` is the outgoing spherical Hankel function. The undamped sum is conditionally convergent;
it is evaluated exactly via the **Kambe layer-KKR Ewald split**. These structure constants are the
foundation for the undamped vector `G0` (cycle 2, via the existing Gaunt contraction
`G0_{nm,νμ} = 4π(-1)^m Σ_q i^{ν+q-n}(-1)^q D[q,m-μ] gaunt(n,m,ν,-μ,q,μ-m)`) and ultimately the lossless
energy balance `|R|²+|T|²=1` (cycle 3). The Phase-1 scalar Ewald (TB2 in `IntraPlaneLatticeSum.wl`,
validated η-independent to 2e-16) is the `q=0` anchor.

This replaces the damped direct sum currently used by `buildG0vec` / `g0LLfun` (the `Im κ = 0.25`
damping is artificial loss that preserves reciprocity but breaks energy conservation).

### Decisions locked in brainstorming (2026-06-23)

| Decision | Choice |
|---|---|
| Route | A — Kambe multipole `D[q,s]` (lands directly in the multipole L/M/N `G0` basis) |
| Deliverable | New `Mathematica/IntraPlaneKambe.wl` (+ `.nb` twin) + Python cross-check |
| Split | recip-space + real-space + `q=0` central term (η-independent total) |
| Validation | η-independence; damped-sum agreement as `Im κ → 0`; `q=0` ≡ TB2; G0 reciprocity |

## 2. Architecture (the Kambe split)

`D[q,s] = D^recip[q,s] + D^real[q,s] + D^(0)[q,s]`, with the total independent of the Ewald splitting
parameter `η`.

### (A) Reciprocal-space half `D^recip[q,s]`
Sum over the 2D reciprocal lattice `{G}` of the in-plane lattice. Each term carries the spherical
harmonic `Y_q^s` of the propagation direction of the plane-wave order `(k_par+G, k_zG)` — with
`k_zG = √(κ² − |k_par+G|²)` (real propagating, `i√(…)` evanescent, `Im > 0`) — times an incomplete-Γ /
`erfc(k_zG / 2iη)` regulator and the cell-area / phase prefactor. Gaussian-fast in `|G|`
(`~ e^{-|k_par+G|²/4η²}`). Reduces, for `q=0`, to TB2's `ewaldRecip`.

### (B) Real-space half `D^real[q,s]`
Sum over the direct lattice `{R≠0}` of `Y_q^s(R̂) e^{i k_par·R}` times the incomplete-Γ representation of
the Ewald-split multipole Hankel `h_q(κ|R|)` (the upper-incomplete-Γ "complementary" piece). Gaussian-fast
in `|R|` (`~ e^{-η²|R|²}`). Reduces, for `q=0`, to TB2's `ewaldReal`.

### (C) Central term `D^(0)`
The `q=s=0` self/forward contribution from the `R→0` regularisation of the split (the
`erfc`-complementary constant of the scalar Ewald). Nonzero only for `q=s=0`.

The exact closed forms are the **standard layer-KKR structure constants** (Kambe 1967; Pendry,
*Low Energy Electron Diffraction*; Modinos; Williams & Maradudin) — NOT from thesis §3.3, which gives the
spectral propagator instead. The `q=0` scalar case is pinned by the already-validated `ewaldTotal` (TB2).
The plan transcribes the specific layer-KKR `D^recip`/`D^real`/`D^(0)` expressions (the `Y_q` factor of the
order direction, the incomplete-Γ arguments, the prefactors) and the gates below verify the transcription;
η-independence in particular cannot be satisfied by a mis-transcribed formula.

## 3. Deliverables

1. `Mathematica/IntraPlaneKambe.wl` (+ `.nb` twin via `makeIntraPlaneNotebooks.wl`) — `D[q,s]` via the
   three-part Kambe split; self-verifying PASS/FAIL header asserts (the gates in §4). Dumps
   `Mathematica/IntraPlaneKambe_reference.json`.
2. `cubic_scattering/tests/test_intraplane_kambe.py` — independent Python cross-check.
3. Closeout: update the plan (cycle 1 DONE; cycles 2–3 remain), memory.

`IntraPlaneKambe.wl` reuses the lattice geometry and the damped direct `D[q,s]` (for the agreement
gate) from the `IntraPlaneLatticeSum.wl` / Phase-2 conventions, but is a **new self-contained file**
(it does not run the Phase-1 study on load). Conventions inherited: time `e^{−iωt}`, outgoing `h_q^(1)`
via `SphericalHankelH1`; lattice in x–y, `θ_R = π/2`; Bloch vector `k_par`.

## 4. Validation (tight gates; the TB2 pattern extended to all `q`)

- **η-independence** (the defining correctness property): `D[q,s]` computed at two splitting parameters
  `η₁ ≠ η₂` agree to machine precision (~1e-12 or better) for every `(q,s)` in range. The real or
  reciprocal half *alone* is η-dependent (RED); only the total is η-independent (GREEN).
- **`q=0` ≡ scalar Ewald**: `D[0,0]` equals TB2's `ewaldTotal` (the validated undamped scalar lattice
  sum) to machine precision.
- **Damped-sum agreement**: the undamped Kambe `D[q,s]` matches the damped direct sum
  `Σ_{R≠0} h_q(κ_damped|R|) Y_q^s(R̂) e^{ik_par·R}` in the limit `Im κ → 0` (extrapolated, or matched in
  the convergent regime), for each `(q,s)`.
- **G0 reciprocity**: the Gaunt-contracted `G0` from the undamped `D[q,s]` satisfies the
  symplectic-J reciprocity (item (d) metric) — `D-conj G0` ≈ machine zero.

## 5. Acceptance criteria

1. `IntraPlaneKambe.wl` runs headless via `wolframscript`; all in-script gate asserts pass; JSON dumped.
2. `.nb` twin generated and spot-verified.
3. `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_kambe.py -v` passes:
   independent η-independence, `q=0`≡TB2, damped-sum agreement, and G0 reciprocity checks.
4. Ruff + mypy clean on the new Python.
5. Plan + memory updated (cycle 1 DONE; cycle 2 = undamped vector `G0`, cycle 3 = energy balance).

## 6. Out of scope (cycles 2 and 3)

- **Cycle 2:** assemble the undamped vector `G0` (L/M/N) from `D[q,s]` and swap it into the collective;
  agreement with the damped `buildG0vec` as damping→0; full reciprocity + convergence.
- **Cycle 3:** energy balance — `T_coll` with the undamped `G0` → R/T(p) via the Phase-3a projection →
  verify `|R|²+|T|²=1` over the **propagating** channels (a mode evanescent past its critical slowness
  carries no flux and is excluded from the balance).
- The strictly real-`κ` *vector* M/N Ewald via differential operators (route B) — not used; route A
  (multipole `D[q,s]`) is the committed path.

## 7. Conventions

Time `e^{−iωt}`, outgoing `h_q^(1)` via `SphericalHankelH1` (never `j_q + i y_q`); lattice in the x–y
plane (`θ_R = π/2`), Bloch vector `k_par = ω(0,p,0)` (project frame; p along x); conda env `seismic`;
Mathematica via `/Applications/Wolfram.app/Contents/MacOS/wolframscript`. Reference JSON in
`Mathematica/`; complex numbers serialised as `[re, im]`. Background α=5000, β=3000, ρ=2500;
`κ_P = ω/α`, `κ_S = ω/β` (real, undamped). The validated TB2 `ewaldTotal` (`IntraPlaneLatticeSum.wl`)
is the `q=0` ground truth; the symplectic-J metric is from [[project_intraplane_reciprocity_metric]].
