# IntraPlaneFoldyLax — Phase 2 Learnings

**Phase 2 COMPLETE (items a–f, closed 2026-06-23).** Planar (intra-plane) multiple scattering of
spherical voxels in the full vector-spherical (Hansen `L/M/N`) basis: assemble the single-site `T0`,
the lattice-summed coupling `G0(k_par)`, and solve the collective Foldy-Lax operator
`T_coll = T0 (I − G0 T0)⁻¹`. This note distils what we learned; the authoritative per-item record is
`IntraPlaneFoldyLax_Plan.md` §6.

---

## 1. What got built (item by item)

| Item | Deliverable | One-line result |
|---|---|---|
| **Phase 0** | `IntraPlaneTranslation.wl` | Elastic vector-spherical translation-addition operator (Cruzan/Stein); reconstructs the translated field, reciprocal, reduces to scalar Gegenbauer on the L (P) part. |
| **Phase 1** | `IntraPlaneLatticeSum.wl` | Planar Ewald lattice sum `G0(k_par)`; `η`-independent to 2e-16 (undamped scalar), matches damped direct sum 2.4e-11. |
| **TB1** | `IntraPlaneFoldyLax.wl` | Single-site `T0` in the `L/M/N` basis from `CartesianT0.wl`; `T_coll`, single-voxel limit exact, closed monopole collective. |
| **TB2** | `IntraPlaneVectorTranslation.wl` | Phase-0 translation as an explicit `L/M/N` matrix `W(d)` by sign-safe projection onto P/B/C vector harmonics. |
| **(a)** | `IntraPlaneTwoBody.wl` | Two-voxel direct Foldy-Lax; Neumann/Born series = matrix inverse to 6.9e-18; fixed point 7.7e-18. |
| **(b)** | `IntraPlaneVectorLattice.wl` | Lattice-summed multi-channel vector `G0(k_par)`; L-block closed form = direct 1.4e-15. |
| **(d)** | `IntraPlaneCollectiveReciprocity.wl` | Collective reciprocity via the **symplectic-J metric** (see §2). |
| **(f)** | `test_intraplane_collective.py`, `makeIntraPlaneNotebooks.wl` | Python cross-check + faithful `.nb` twins of every Phase-2 `.wl`. |
| **(c)** | `IntraPlaneConvergence.wl` | Convergence in multipole order + packing density (see §3). |
| **(e)** | `IntraPlaneDiscretisation.wl` | Sphere-packing discretisation error vs the cube slab (see §4). |

---

## 2. The reciprocity metric was the central conceptual trap (item d)

A naïve "far-field amplitude in an arbitrary direction is reciprocal" test is the **wrong statement for a
lattice at fixed `k_par`**. `T0` (from the clean `L/M/N` `CartesianT0`) and `G0` (Phase 0/1) are each
reciprocal but **in different metrics**, so the naïve collective reciprocity failed.

- **Fix:** a single **symplectic channel metric** `D = diag((kS/kP)^{3/2}, i·√(n(n+1)), √(n(n+1)))`
  makes *both* `T0` and `G0` σ-symmetric (`J0 (D A D⁻¹) J0 = (D A D⁻¹)ᵀ`), so `T_coll` is reciprocal
  (D-conj `T_coll` ≈ 4.4e-17). This is *not* a fit — each weight is pinned independently (`d_L` by `T0`'s
  L–N ratio, `d_N` by the Jspher weight, `d_M` by `G0`'s M–N).
- **Lesson:** when two operators are assembled in different normalisations, reconcile the *metric*, don't
  patch the test. See `[[project_intraplane_reciprocity_metric]]`.

---

## 3. Convergence is conditioning-gated; the lattice sum grows but the Mie spectrum wins (item c)

- The translation/lattice structure constants **grow super-exponentially** in `n` (`g0LL[n,0,n,0] ~ 1e10`
  by `n=8`) while the single-site Mie coefficients **decay super-exponentially** (`‖T0(n)‖ ~ 1e-11`).
  Convergence is the statement that the Mie decay wins in the product `G0·T0`.
- This holds for **non-overlapping spheres** (`aa/aL < 1/2`) and **degrades toward touching**: coupling,
  spectral radius `ρ(G0 T0)`, and `cond(I − G0 T0)` all grow as `aL → 2·aa` (e.g. `cond ~ 1` at `aL≥3`,
  `~7` at `aL=2.2`, `n_max=5`).
- **Operating rule:** **conditioning-gated `n_max`** — stop when `‖T0(n)‖ < tol` OR `cond` crosses a
  threshold. The huge `g0LL ~ 1e6` entries are an un-normalised clean-`L/M/N` basis artifact, not physics;
  escape hatches are extended precision (`~log10(cond)` guard digits) or an energy/flux-normalised basis.

---

## 4. Sphere-packing discretisation error: shape is small, the renorm has a hard floor (item e)

- **Shape factor is small** because the effective-contrast extraction normalises by volume
  (`C_P = V/4πρα²`): a diluted sphere and a space-filling cube of the *same* `Δ` scatter nearly identically
  per unit volume. Single-site error ≤0.4% (moderate) to ~3.5% (−60%).
- **`Δ→Δ/φ` is a layer-level correction** (`layer ~ φ·single_site(Δ/φ)`, `φ=π/6`): it collapses the raw
  **~48% dilution** error to **0.38% (weak) / 3.95% (moderate)** of the cube-layer `R_PP`.
- **Validity floor `|Δ| < φ·background ≈ 52%`.** `Δ/φ ≈ 1.91·Δ`, so a negative contrast past the floor
  drives the inner moduli/density negative (the **−60% case → −115%**, flagged unphysical, `cond ~ 57`).
  This is a genuine limit, not a bug: a strong-negative *space-filling* cube cannot be represented by a
  *diluted* sphere packing via contrast concentration. See `[[project_sphere_packing_discretisation]]`.

---

## 5. Cross-cutting: the collective is negligible at Rayleigh

Both item (c) and item (e) measure the inter-sphere multiple-scattering correction at Rayleigh and find it
**negligible** (`r_ms − 1 ≤ 2e-4` for moderate contrast even at touching; the renormalised layer error is
`aL`-independent). The discretisation error is **shape + dilution, not multiple scattering** — the same
verdict the Cartesian study reached (`[[t27-lattice-verdict]]`: inter-voxel coupling ≤0.07%). This is a
recurring, robust finding across both the cube and sphere paths.

---

## 6. Numerical / tooling lessons

- **Outgoing Hankel:** always `SphericalHankelH1[n, x]`, **never** `j_n + i·y_n` — in the damped far field
  (`Im(κ·R)` large) `j_n, y_n ~ e^{Im}` individually but `h_n ~ e^{−Im}`, so the sum cancels catastrophically.
- **Lattice damping is artificial loss.** Phase 2 uses a damped lattice (`Im κ = 0.25`) for convergence;
  this is fine for reciprocity/convergence but **forbids an energy-balance claim** (`|R|²+|T|² = 1`). The
  undamped vector `G0` (Ewald-accelerated Cruzan/Gaunt structure constants, full Kambe) was **deferred** —
  it is the prerequisite for Phase 3's flux/energy acceptance.
- **Deliverable pattern:** each phase is a self-verifying `.wl` (PASS/FAIL prints) + a `.nb` twin
  (`makeIntraPlaneNotebooks.wl`, sections split on `=`-banners) + an independent Python cross-check
  reading a dumped JSON reference. Keep complex numbers as `[re, im]` and reference JSONs in `Mathematica/`.
- **`CartesianT0.wl` self-containment (fixed in item e):** it had imported the bracket builders
  (`Tspheroidal/Ttoroidal/T0mono`) from a transient `/tmp/cell_sec3_builders.m` that did not survive across
  sessions, silently breaking every intra-plane `.wl` re-run. Now loads them from `ElasticMieTmatrix.nb`
  cell 6 directly.

---

## 7. What Phase 2 deliberately did NOT settle (→ Phase 3)

- **Energy balance `|R|²+|T|² = 1`** (lossless plane) — needs the **undamped** `G0`; the damping is
  artificial loss.
- **Kennett flux normalisation match** (`√(ηρ)` basis) — the projection of the collective onto up/down
  P-SV-SH plane waves at slowness `p`.
- **The layer R/T(p) operator itself** (`Rd, Ru, Td, Tu` 2×2 + SH scalar) — item (e)'s normal-incidence
  `R_PP` used the Rayleigh effective-contrast → `kennett_reference_rpp` shortcut, *not* the full Weyl
  projection. Phase 3 builds the real projection.
- **`n_max(p)` for specular R/T** — the order needed grows with slowness `p` and for P-SV-SH conversions;
  the hypothesis (per item c) is that specular up/down R/T converges at a low `n_max` even where the full
  near-field operator does not.
