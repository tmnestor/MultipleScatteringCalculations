# Thesis §3.1 Energy-Normalised Interface R/T Notebook — Design Spec

**Status:** built & verified 2026-06-25.

## 1. Purpose

A **self-contained Mathematica notebook** implementing the PhD thesis §3.1
(`Thesis_Recompiled_2026/GRepresentations.tex`) energy-normalised displacement–traction
eigenbasis `D_z` and the canonical symplectic `J₆`, for **two half-spaces**, plus the
plane-interface R/T scaffold. The user drives the rederivation of the correct interface
energy-normalised R/T arrays from this notebook. This is foundational tooling for the
project's core aim: **validating (or refuting) the thesis**.

## 2. Formalism (faithful to §3.1)

- State vector `b = (u_z, u_x, u_y, t_z, t_x, t_y)` — 3 displacement, 3 traction-on-z-plane.
- Eigen-matrix `D_z(k_x,k_y) = [|+P⟩ |+S⟩ |+H⟩ |−P⟩ |−S⟩ |−H⟩]` (6×6), columns =
  down/up `P/S/H` eigenvectors (`Peigen`/`SVeigen`/`SHeigen`), each carrying the energy
  normalisation `ε_P, ε_S, ε_H` (`epsdef`).
- `K̂_c = √((ω/c)² − k_y²)`, `k_{z,c} = √(K̂_c² − k_x²)`; velocity `c`: P→α, S/H→β.
- Canonical symplectic `J₆ = [[0₃, I₃],[−I₃, 0₃]]`; symplectic inverse
  `D_z⁻¹(k) = −i J₆ D_zᵀ(−k) J₆` (`D1def`).

## 3. Conventions

- `+` = downgoing, `−` = upgoing; column order `(+P,+S,+H, −P,−S,−H)`.
- **Time `e^{−iωt}`** — the THESIS convention (`GRepresentations` l.29), *not* the project's
  `e^{+iωt}` (CartesianT0 / Phase-3 code). Stated in the notebook header so the two are not
  confused.
- Traction = physical stress on the +z plane `(σ_zz, σ_zx, σ_zy)`; `λ = ρ(α²−2β²)`, `μ = ρβ²`.
- General `(k_x, k_y)` (full 6×6 `J₆`; P-SV and SH couple when `k_y ≠ 0`, decoupling only at
  `k_y = 0`).

## 4. Self-checks (all PASS at MACHINE precision in seismic units)

Instantiated in **seismic units** (km/s, g/cm³ → moduli in GPa), where `cond(D_z) ~ ρωv ~ 1.9e4`
and every check holds at machine precision. (In SI the same `cond` is ~1.9e10 — a pure UNITS
artifact of stacking displacement (length) over traction (stress); in nondimensional units
`cond(D_z) = 1.66`. NOT a defect: the thesis never inverts `D_z` naively — it uses the symplectic
`D1def` — and the symplectic identity below holds at 1e-16 in every unit system.)

1. **Symplectic / energy-normalisation identity** `(J₆ D_z(−k))ᵀ D_z(k) == i J₆` (`dinv2`),
   both half-spaces — `3e-16`. The authoritative proof the eigenvectors + `ε` are correct.
2. **Inverse consistency** `D_z · D_z⁻¹ == I₆` — `2e-12` (machine precision, seismic units).
3. **Traction = Hooke(displacement)** — traction rows reproduced from displacement rows via
   Hooke's law (A-matrix-free physical check) — `= 0`.
4. **Interface energy balance** `Σ|R|² + Σ|T|² = 1` per incident column (P,S,H) — `max|·−1| = 0`.

## 5. Interface R/T scaffold (worked demonstration)

Continuity of `b` at `z=0`: `D₁·[a_inc; a_refl] = D₂·[a_trans; 0]`. Rather than a 6×6 solve, use
the **symplectic inverse** to reduce to a single 3×3 inverse: form `Q = D₁⁻¹·D₂` with
`D₁⁻¹ = −i J₆ D₁ᵀ(−k) J₆` (`D1def`, no elimination); partition `Q` into 3×3 blocks
`[[Q11,Q12],[Q21,Q22]]`; since `a₂` has no upgoing part, `a_inc = Q11·a_trans`,
`a_refl = Q21·a_trans`, so **`T = Q11⁻¹`** (the only inverse, 3×3) and **`R = Q21·T`**. Verified
identical to the 6×6 `LinearSolve` (`4e-16`). The energy-normalised
`R` exhibits the thesis symplectic structure (P-SV off-diagonal antisymmetry; P-SH purely
imaginary), and the full `T` carries the ballistic through-wave automatically from `b`-continuity
(`T ≈ I` at weak contrast).

## 6. Deliverables

- `Mathematica/ThesisInterfaceRT.wl` — executable source (self-contained, no `Get[]`).
- `Mathematica/ThesisInterfaceRT.nb` — notebook twin (sectioned cells) for interactive rederivation.
- `Mathematica/makeThesisNotebook.wl` — standalone `.nb` generator for thesis-validation scripts.

See [[thesis-energy-normalisation]], [[project-intraplane-layer-rt]].
