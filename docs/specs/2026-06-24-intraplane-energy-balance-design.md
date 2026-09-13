# Intra-Plane Layer Energy Balance (Phase 3b cycle 3) — Design Spec

**Status:** approved 2026-06-24; ready for an implementation plan (`writing-plans`).

## 1. Goal & Boundary

Verify the **lossless energy balance** of the Phase-3a layer R/T(p), built on the cycle-2
**undamped** vector coupling `G0^vec` (Im κ = 0). Acceptance is **S-matrix unitarity**
`S_p† S_p = I` at every horizontal slowness `p` in the sweep, with `S_p` restricted to the
**propagating** channels at that `p`.

This closes Phase 3b. Phase 3a (`IntraPlaneRT.wl`) already established **symplectic reciprocity**
(`Rd, Ru` antisymmetric; `Tu = Σ·Td·Σ`) in the thesis §3.1 ε-energy normalization, but it builds
its coupling at the **damped** lattice wavenumbers `κ_P = kPo + 0.25 i`, `κ_S = kSo + 0.25 i`. That
artificial `Im κ` makes the conditionally-convergent planar sum converge and preserves reciprocity,
but it **bleeds energy**, so `|R|² + |T|² = 1` cannot hold. Cycle 3 swaps in the cycle-2 undamped
`G0^vec` (Ewald structure constants `D[q,s]`, η-independent, lossless) and tests unitarity.

**In scope (cycle 3):**
- a new self-contained script `IntraPlaneEnergyBalance.wl`;
- the cycle-2 undamped `G0^vec` builders **lifted to physical parameters** `(aL, k_x, k_y, κ_P, κ_S, Nmax, η)`;
- the Phase-3a thesis-ε R/T projection (`rtBlocksE`/`rtAmpE`/`projPW`/`incVec`), **reused unchanged**;
- assembly of the propagating-channel S-matrix and the unitarity gate `S_p† S_p = I`;
- the multi-gate suite (§5), an `Nmax` convergence sub-study, a Python cross-check, a `.nb` twin,
  and plan/memory closeout.

**Out of scope:**
- Phase 4 (inserting the layer R/T into `kennett_layers` over the stratified background);
- open-diffraction-order (super-wavelength) layers — explicitly **forbidden** by gate [1] (§4);
- re-deriving the R/T projection or the ε-normalization (Phase 3a, settled).

**Untouched:** `IntraPlaneRT.wl` (the damped Phase-3a reciprocity artifact) and
`IntraPlaneKambeVector.wl` (the cycle-2 fixed-parameter `G0^vec`) stay as they are.

## 2. Physical foundation — why `p → p` unitarity is the right statement

The voxel plane is **laterally periodic** (lattice pitch `aL`), **not laterally invariant**. The
two give different conservation laws, and the distinction is load-bearing:

- **Laterally invariant** (homogeneous): continuous translation symmetry → horizontal momentum
  conserved exactly → `p → p`, no diffraction.
- **Laterally periodic** (this lattice): only discrete translation symmetry → **Bloch/Floquet**:
  the horizontal wavevector is conserved **modulo a reciprocal-lattice vector `G`**. An incident
  `k_par` scatters into `k_par + G` for all `G`. The genuinely unitary S-matrix lives in one
  **Bloch sector** (fixed reduced `k_par`) and spans **all open channels** in it: every
  `(mode, ±, G)` with real `k_z(G) = √(κ² − |k_par + G|²)`.

We recover the `p → p` (specular) statement **only** because of the **sub-wavelength condition**.
At these parameters `κ_S·aL ≈ 0.5·2.5 ≈ 1.25 < 2π`, the smallest reciprocal vector
`|G| = 2π/aL ≈ 2.5 ≫ κ`, so every `G ≠ 0` has imaginary `k_z` → evanescent → zero vertical flux.
The only open channel in the Bloch sector is the specular `G = 0` order, whose horizontal
wavevector is `k_par + 0 = k_par` → same `p`. Hence:

> sub-wavelength periodic ⇒ only `G = 0` propagates ⇒ open channels collapse to specular
> up/down per mode ⇒ `p → p`, and the specular S (4×4 P-SV + 2×2 SH) is unitary.

This is exactly the regime in which the periodic plane **acts as an effective laterally-homogeneous
layer** for the propagating field — the entire premise of "intra-plane Foldy-Lax → effective layer
R/T(p)". Gate [1] (no open diffraction orders) is therefore the **physical license** for the whole
reduction, asserted not assumed (§4).

**Failure mode if gate [1] ever fails** (super-wavelength `κ·aL`): `p → p` specular unitarity is
**false**; evanescent orders become propagating flux channels; the specular `S_p† S_p` is
**sub-unitary**, with `I − S_p† S_p` equal to the energy diffracted into `k_par + G ≠ k_par`. The
script must then **refuse** (error out) rather than report a spurious energy defect.

## 3. Channel structure — two spaces (do not conflate)

- **(A) Internal multipole space** — the `L/M/N` basis indexed by `(n, m, channel)`; **25×25** at
  `Nmax = 2`. `G0^vec`, `T0`, `T_coll` are matrices here; the Foldy-Lax solve
  `T_coll = T0 (I − G0^vec T0)^{-1}` happens here.
- **(B) External plane-wave channel space** — the propagating up/down plane waves at fixed `(ω, p)`;
  the S-matrix lives here. The R/T projection (`projPW` + the Weyl prefactor
  `i/(2 η_out ω A_cell)` in `rtAmpE`) is the map **A → B**: it **sums over all 25 multipoles** to
  produce one plane-wave amplitude per external channel.

At fixed `(ω, p)` a propagating plane wave must satisfy `|k| = ω/c_mode`; with `ω, p, mode` fixed
the only freedom is the vertical-wavenumber sign `k_z = ± ω η_mode`, so **per mode there are exactly
two** plane waves (down `+η`, up `−η`). The multipole index `(n,m)` is **not** an external channel —
it is collapsed by the projection. P-SV decouple from SH:

| block | channels at fixed (ω,p) | S size |
|---|---|---|
| P-SV | P↓, SV↓, P↑, SV↑ | 4×4 |
| SH | SH↓, SH↑ | 2×2 |

The S-matrix dimension is `(modes) × (up/down) × (open diffraction orders)` = `2 × 2 × 1 = 4` for
P-SV here. Diffraction orders (not partial waves) would enlarge it if gate [1] failed.

## 4. The build (the actual cycle-3 work)

For each `p` in the sweep, in the new `IntraPlaneEnergyBalance.wl`:

1. **Bloch + wavenumbers (undamped):** `k_x = ω·p`, `k_y = 0`, `κ_P = kPo` (real), `κ_S = kSo`
   (real). Physical values: `aL = aLpitch = 2.5`, `kPo = kaTest = 0.3`, `kSo = 0.5`, `ω = ωOf`.
2. **Undamped `G0^vec`:** build the cycle-2 vector coupling, **lifted to parameters**. The cycle-2
   builders (`DstructU` Ewald field; `g0LLk`; `g0MNblock`; `coeffq`) are currently **hardcoded** to
   `aL = 2.0, k_x = 0.2, k_y = 0.1, κ = 0.9/1.5`. They must become functions of
   `(aL, k_x, k_y, κ_P, κ_S, Nmax, η)` and be driven at the physical values above. The L→L block
   uses `D(κ_P)`; the M/N blocks use `Σ_q coeff_q · D[q, m−μ](κ_S)`. `coeff_q` are
   `(k_x, k_y)`-independent (pure Wigner), so they are extracted **once** and reused across the
   sweep; only the `D[q,s]` re-evaluate per `p`.
3. **Collective solve:** `T_coll = T0 (I − G0^vec · T0)^{-1}` (25×25 at `Nmax = 2`).
4. **R/T projection (reused):** `rtBlocksE[T_coll, p, Nmax]` → `{Rd, Td, Ru, Tu}` (each 2×2 P-SV) +
   `{Rsh, Tsh}` scalars, in the thesis §3.1 ε-energy normalization (the incident **is** the
   ε-eigenvector; the scattered field is projected onto ε-eigenvectors; no slab `D`, no post-hoc
   factors). The down-incident SH gives `Rsh, Tsh`; the up-incident SH (`Rsh_u, Tsh_u`) is added to
   `rtBlocksE` for the full 2×2 SH S-matrix (small extension).
5. **Assemble & restrict S:** P-SV `S = [[Rd, Tu],[Td, Ru]]`; SH `S_SH = [[Rsh_d, Tsh_u],[Tsh_d,
   Rsh_u]]`. Restrict to `propagating(p) = {modes with Im η_mode = 0}` (rows **and** columns):
   normal/sub-critical → full P-SV (+SH); post-critical (`0.8·pcritS`, P evanescent) → SV (+SH)
   sub-block only.

**Sweep:** carry the Phase-3a `pList = {pNormal = 1e-6, 0.5·pcritP, 0.8·pcritS}` (normal /
sub-critical / post-critical), `pcritP = 1/α0`, `pcritS = 1/β0`.

## 5. Gates (multi-gate, cycle-1/2 style; each PASS/FAIL with residual)

- **[1] No open diffraction orders** (precondition for specular-only unitarity, §2): for every `p`,
  assert `k_z(G) = √(κ² − |k_par + G|²)` is imaginary for **all** reciprocal `G ≠ 0` at both
  `κ = κ_P` and `κ = κ_S`. If any order is open → **hard error / refuse**, not a soft fail.
- **[2] Undamped η-independence at physical params:** re-verify `DstructU[q,s,κ,η]` is η-independent
  (e.g. `η = 0.7` vs `1.15`) at `κ_P = 0.3, κ_S = 0.5, aL = 2.5`. Cycle 1/2 validated this at
  `κ = 0.9/1.5, aL = 2.0`; this is a fresh-parameter re-check that the Ewald split still holds.
- **[3] Energy / unitarity (the headline):** `max_p ‖S_p† S_p − I‖_∞` over the propagating
  restriction, across all `p`. PASS below the tolerance settled in §6 (R3).
- **[4] Reciprocity preserved:** the Phase-3a symplectic gates (`Rd, Ru` antisymmetric;
  `Tu = Σ·Td·Σ`, `Σ = diag(1,−1)`) must still hold on the undamped build — undamping must not break
  reciprocity. Carry the Phase-3a thresholds (~1e-6).
- **[5] `Nmax` convergence sub-study** (settles the plan's open question): residual of [3] at
  `Nmax = 1, 2, 3`; report whether the energy defect plateaus by `Nmax = 2` or needs higher. This
  is **reported**, not a binary gate (it informs the production `Nmax`).

## 6. Risks & open items

- **R1 — energy metric (primary).** Does `S_p† S_p = I` hold **directly** in the ε-normalization, or
  up to a diagonal SV-parity sign metric `Σ` (the normalization is symplectic; energy may be
  `Σ`-twisted)? Resolve by a short derivation from the thesis §3.1 epsdef `(J6 Dz(−k))^T Dz(k) = i J6`
  plus an empirical check. The gate adapts to whichever invariant is exact (plain `S†S = I`, or
  `S† M S = M` for a fixed diagonal `M`); record which in the dump and memory.
- **R2 — `Nmax` sufficiency.** Empirical (gate [5]). The specular up/down R/T is fed by the
  fastest-converging low-order channels (cf. item (c) convergence study), so `Nmax = 2` is the
  expectation, but the energy residual is the arbiter.
- **R3 — quadrature/Ewald tolerance floor.** The undamped energy residual will be **looser** than the
  1e-6 reciprocity floor (numerical Ewald `DstructU` + sphere-projection quadrature compound). Settle
  the gate-[3] tolerance from the **observed** residual + the `Nmax` trend, not by asserting 1e-6 up
  front. Document the achieved floor.
- **R4 — post-critical channel bookkeeping.** Confirm the propagating restriction is applied to both
  rows and columns of S (an evanescent incident channel is also not a valid input). At
  `0.8·pcritS`, `η_P` imaginary → P dropped, SV (+SH) retained.

## 7. Deliverables

- `Mathematica/IntraPlaneEnergyBalance.wl` (+ `.nb` twin via `makeIntraPlaneNotebooks.wl`).
- `Mathematica/IntraPlaneEnergyBalance_reference.json` — per-`p` S-matrices (reim), propagating
  channel sets, `‖S†S − I‖`, the chosen energy metric (R1), the `Nmax` sub-study, and reciprocity
  residuals.
- `cubic_scattering/tests/test_intraplane_energy.py` — Python cross-check: reload the dumped
  S-matrices, independently recompute `S_p† S_p` (or `S† M S`), assert the energy and reciprocity
  invariants on the restricted channels.
- Plan update (`IntraPlaneFoldyLax_Plan.md`: cycle 3 DONE, Phase 3b closed) and a memory file
  (`project-intraplane-energy-balance.md`), linked from `MEMORY.md`.

## 8. Conventions

Coordinate frame `z` (down, axis 0), `x` (axis 1), `y` (axis 2); horizontal slowness `p` along `x`;
project-frame `(z,x,y)` permuted into the bridge frame by `toSph` at every bridge call; time
`e^{−iωt}`, outgoing `h_n^(1)` (`SphericalHankelH1`); inner `T0` at **real** background wavenumbers;
conda env `seismic`; Mathematica via `/Applications/Wolfram.app/Contents/MacOS/wolframscript`.
Related: [[project-undamped-vector-g0]], [[project-undamped-kambe-structure-constants]],
[[project-intraplane-layer-rt]], [[thesis-energy-normalisation]],
[[kambe-validates-thesis-spectral]].
