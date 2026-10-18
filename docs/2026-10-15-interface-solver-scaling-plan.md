# Scaling the Interface Solver to Realistic Sizes — Plan

**15 October 2026.** Status: proposed. To start **after** the current GMRES gate and the TeX update of
`LatexPDFs/InterfaceCouplingIntegrals` are done (both done 16 October: paragraph "Larger scatterers"). It follows step 2 of
`docs/2026-10-10-stratified-reference-legendre-cells-3d.md` and leads into that plan's Stage A/B.

Conventions as in the parent plan. n × n cells laterally, L depth layers, N = n²L cells, R the lateral
radius of the corrections (in cells).

---

## 0. Where we are (measured, `scripts/gate_interface_gmres.py`, 4 workers)

| | 2 × 5 × 5 (50 cells) | 2 × 12 × 12 (288 cells) | ratio |
|---|---|---|---|
| set-up (A0, A1, lateral tables) | 364 s, 60/60 classes corrected | 1477 s, 128/312 corrected | 4.1× |
| kernels, per frequency | 92 s, 17 MB | 183 s, 110 MB | 2.0× time, 6.5× memory |
| GMRES (block-Jacobi, 10 % contrast) | 8 products, 0.1 s | 9 products, 0.7 s | |
| against the dense solve | 4.6e-12, PASS | — | |

**What does not scale.**
1. **Kernel storage**, O((2n−1)² L²) dense 36 × 90 blocks per frequency: ~52 kB per lateral point per pair
   of layers. Extrapolated to 100 × 100 × 20: **~0.8 TB per frequency.** This is the first wall.
2. **The set-up per class.** The radius bounds the lateral classes, but the classes are keyed by
   *absolute* depth pairs (zr, zs): O(R² L²). At ~10–45 CPU-s per corrected class, 100 × 100 × 20 would
   be ~17 h. And the 288-cell set-up grew 4.1× for 2.1× the corrected classes, so the classes *beyond*
   the radius (lateral tables only) are not as cheap as assumed, or the cost per class grows with
   offset; not yet known which.
3. **D per frequency for every class**, O(n² L²) radial quadratures, though at 288 cells a fixed cost
   (the near-series moments) still dominates.

**What does scale.** GMRES: 8 → 9 products for 5.8× the cells, at 10 % contrast. Not yet tested at
realistic contrast.

---

## 1. What the analytic integration buys, and what it does not

The analytic integration is the closed-form K0 (static image and transmission, 356 / 275 terms) and
K1 (the ω² term, 744 / 582 terms with Hadamard finite parts), integrated over cell pairs by master
integrals. It enters the solver only through A0 = K0 − ∫_{q<Q0} Y0 and A1.

**It buys:**
- **Correctness at touching pairs.** For a cell touching the interface, the mode integral does not
  converge once the Galerkin moments are taken (the moments grow as e^{κh} per side against e^{−2κh}
  of propagation; errors O(10²)). The image of such a cell touches it, so its block is singular.
  Something has to handle these pairs in space, and the closed forms do so exactly.
- **A short wavenumber integral.** Subtracting K0 turns the Q⁻³ tail into Q⁻⁵. Adding ω²K1 makes it Q⁻⁶.
  So D can stop at Q0h = 10 instead of Qh ≈ 400.
- **Frequency independence.** K0 and K1 are computed once and reused for every frequency and incidence.
  That is what makes the 3.5 s per further frequency possible.
- **An independent arbiter.** It shares no step with the wavenumber route. Every gate of the series
  rests on that.

**It does not buy:**
- **Solution accuracy beyond the discretisation.** The blocks are good to ~1e-10. The discretisation
  error is ~2e-4 (p = 1 against p = 2 at ωh/β = 1/6). That is 10⁶ more precision than the solution can
  use.
- **Anything that grows with the scatterer.** The analytic work is local: touching and near-interface
  pairs within the radius. It is a fixed cost per *class*. The scaling problems (§0) are the per-class
  bookkeeping over absolute depths, and the plane-wave part D and its storage, which are not analytic.

**The open question this raises** (decision D2): measured on 15 October, D's time per pair does not depend
on Q0 for Q0h = 3 … 10. If that still holds at Q0h = 40, K0 alone would give ~5e-8. That estimate comes
from the static-subtracted change of 4.8e-5 between Q0h 10 and 40, measured at p = 0. It is 4000× below
the discretisation error, so production would not need K1. K1 would stay in the paper and in the arbiter.

---

## 2. Principle: an error budget

Fix a block tolerance from the discretisation error, not from the rounding floor. **Proposed: 1e-6 of
the touching block**, two orders below the ~2e-4 discretisation error. Every truncation below
(lateral radius, depth radius, Q0, padding) is chosen against that one number and gated against the
dense or reference solve at 1e-10, so the reference remains available.

---

## 3. Steps

Each step has a gate. Profiling and measurement come before any rewrite.

### S1. Profile the set-up (≈ ½ day)
- Time A0, A1 and `lateral_harmonics` per class on the 288-cell box, against lateral offset and depth pair.
- **Gate:** a table that splits the 1477 s. Decide whether the classes beyond the radius need a cheaper
  table (Jacobi–Anger was checked on 12 October and needs no per-offset tables).
- **Done (16 October), `scripts/profile_interface_setup.py`** → `scripts/data/profile_interface_setup.json`.
  Wall 1443 s on 4 workers, 5733 CPU-s summed over the 312 jobs (the pieces sum to each job's time; the first
  job per worker pays 157 s of one-off caches in all).

  | CPU-s | K0 | Y0 < Q0 | K1 | FP Y1 | window | lateral | total |
  |---|---|---|---|---|---|---|---|
  | 128 classes within 16 h | 514 | 215 | 1173 | 717 | 648 | 323 | 3591 |
  | 184 classes beyond | — | — | — | — | — | 2142 | 2142 |

  - **The lateral tables are 43 % of the set-up (2465 s), and each is built four times.** `lateral_harmonics`
    depends on the lateral offset only, but `_parts_job` builds it per class (zr, zs, a, b). The four depth
    pairs of an offset agree in time to 5 %; 78 offsets, 312 tables.
  - **A table's cost grows as the square of the reach** (log-log slope 2.07; 0.4 s at ≤ 4 h, 4.3 s at 12–16 h,
    16.6 s at 24–32 h): panels ∝ reach (`_radial_width`) × azimuthal nodes ∝ reach (`_n_phi`). This answers §0's
    open point: the classes beyond the radius are not cheap, and their cost grows with offset. Extrapolated to
    100 × 100, one table per offset is **~690 CPU-h**, before the ×L² of the per-class keying. Tables, not
    corrections, are then the set-up's wall.
  - **The corrections cost 19–30 CPU-s per class** and grow only mildly with reach. The ω² part (K1 + FP Y1)
    is 58 % of them, K0 16 %, the window 20 % (which grows with reach, 2 → 7 s). Within the radius there are
    32 offsets whatever the grid, so 32 L² classes × ~25 s: 100 × 100 × 20 ≈ 90 CPU-h (≈ 22 h on 4) unless S2
    bounds the depth pairs.
  - **No depth asymmetry:** the four depth pairs cost the same (1367–1453 s).
  - **Decision for S3.** (a) One table per offset, not per class: behaviour-identical, saves ~1850 CPU-s
    (32 %) here, and makes the tables independent of L. (b) The reach² growth: Jacobi–Anger (checked
    12 October, no per-offset tables), or S4, which samples D on the FFT grid and needs no S_n at all. If S4
    goes ahead, Jacobi–Anger would only be interim.

### S2. Depth locality of the corrections (≈ 1 day)
- **Hypothesis.** A0 and A1 are the q > Q0 part of the reflected/transmitted kernel. They should decay
  like e^{−Q0(d_r + d_s)}, where d_r, d_s are the cells' distances from the interface. At Q0h = 10,
  that is ~1e-9 when the two cells are two cell-widths from the interface in total.
- **Measure** |A0| and ω²|A1| relative to the touching block, for receiver and source depths
  iz = −1 … −4, both media, reflected and transmitted, at the window C^∞ 0.25.
- **Gate:** a depth radius at 1e-6 / 1e-8 / 1e-10, like tab:locality's lateral one. If it holds, the
  set-up is O(R² · constant) independent of L. Only the layers near the interface carry analytic work,
  and the rest are plane waves.
- **Done (16 October), `scripts/measure_interface_depth_locality.py`** → `scripts/data/interface_depth_locality.json`.
  Lateral offset 0, z = ±1 … ±7 h, reflected in A and in B and transmitted, relative to the family's touching
  K0 (worst of the pairs at each total gap g = d_r + d_s):

  | g / h | 0 | 2 | 4 | 6 | 8 | 10 |
  |---|---|---|---|---|---|---|
  | \|A0 + ω²A1\| | 2.4e-2 | 2.6e-6 | 3.7e-9 | 3.7e-12 | 6.2e-15 | 1.0e-15 |
  | ω²\|A1\| | 1.2e-5 | 1.1e-8 | 2.8e-11 | 4.6e-14 | 1.0e-16 | 1.8e-17 |

  - **The hypothesis holds, at a different rate:** about 10⁻⁴ per 2 h of gap, e^{−4.6 g/h}, alike for
    reflection and transmission and on both sides. That is neither e^{−Q0 g} (the plan's guess, e^{−10 g/h})
    nor e^{−start·Q0 g} (e^{−2.5 g/h}, the window's first rise). The C^∞ step is negligible early on its ramp,
    so it effectively turns on near Q0/2. The floor is ~1e-15 (two cells at 7 h: 1.8e-14, rounding of D).
  - **Depth radius:** the corrections fall below 1e-6 and 1e-8 for g ≥ 4 h, and below 1e-10 for g ≥ 6 h. At
    1e-10, only pairs with g ≤ 4 h carry corrections: both cells in the three layers nearest the interface on
    their side. So the set-up within the radii is O(R² · 3² · 4) whatever L, and the rest of the blocks are
    D alone. With change 1 (one table per offset) the tables are independent of L as well.
  - **Not yet built:** the depth radius in `InterfaceCouplings` (it drops A0, A1 for pairs beyond it, as the
    lateral radius does). It is a decision for S3, together with D1.

### S3. Decide the budget (user, after S1–S2)
- D1: the block tolerance (1e-6 proposed) and with it the lateral radius (12 h at 1e-6) and the depth
  radius from S2.
- D2: K1 in production or not (§1). If not, measure D's cost and accuracy at Q0h = 20, 40 with K0 only.
- **Decided (user, 16 October).**
  - **D1: block tolerance 1e-8** (not the 1e-6 proposed). This gives a lateral radius of 16 h (tab:locality)
    and a depth radius of 4 h: corrections are dropped for a total gap g ≥ 4 h, the same as at 1e-6 (S2).
  - **D2: keep K1** (the ω² closed form) in production. With the set-up bounded by both radii, dropping it
    would save a few minutes once, and would need the unmeasured Q0h = 40 check.
  - **The depth radius is built:** `InterfaceCouplings(depth_radius=4.0)`. Gate
    `scripts/gate_interface_depth_radius.py`: six layers (iz = −3 … 2) × 2 cells, against the full set-up.
    At 4 h, 24 of 72 classes keep their corrections; blocks 1.8e-10 (relative to the largest block), solution
    2.7e-11, PASS. At 6 h: 48 classes, 1.8e-13, 1.9e-14. The real 4 h set-up gives parts identical to the full
    set-up's, in 175 s against 469 s.

### S4. D in the wavenumber domain (≈ 3–4 days)
- Today D is a radial quadrature per class, put onto the lateral grid and then FFT'd. Instead, **sample
  the windowed spectrum directly on the FFT grid** (k_x, k_y) of the padded lateral box. That spectrum is
  the receiver's lateral moments × R/T × the source's moments, and it is band-limited by χ to q < Q0.
  That gives the kernel's transform directly, with no per-offset work.
- **The risk is wrap-around.** D's propagating part decays only as ~1/r laterally. So the padding, or a
  smooth taper, has to be sized against D1's tolerance. Measure the error against the present kernels as
  the padding grows.
- **Gate:** the 288-cell kernels to 1e-10 (or to D1). Time per frequency O(L² · n² log n) with a small
  constant. Check n = 12, 24, 48.
- **Measured first (16 October): the wrap-around rules out padding alone.**
  `scripts/measure_interface_d_decay.py` → `scripts/data/interface_d_decay.json`. Max |D| / max |K0| for the
  touching pairs:

  | offset / h | 0 | 4 | 8 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|---|
  | reflected, ωh/β = 1/6 | 1.0 | 0.13 | 2.8e-2 | 1.1e-2 | 5.3e-3 | 5.6e-3 | 1.1e-2 |
  | reflected, 1/2 | 1.2 | 0.23 | 0.13 | 0.11 | 0.11 | 8.1e-2 | 3.3e-2 |
  | transmitted, 1/6 | 0.97 | 0.19 | 3.0e-2 | 8.6e-3 | 3.2e-3 | 1.5e-3 | 9.1e-3 |
  | transmitted, 1/2 | 0.95 | 0.26 | 0.10 | 4.3e-2 | 2.5e-2 | 9.5e-3 | 2.7e-3 |

  - **This is physics, not quadrature.** Against the independent polar mode sum (`_truncated_blocks_direct`,
    sharp cutoff), D agrees to 7e-12, 1.7e-12 and 1e-12 of K0 at 8, 32 and 64 h. No Stoneley pole: |R_d| is
    smooth beyond 1/β_A and peaks only at the branch points. D is the far field of the waves along the
    interface, ~h/r, and the P and S phases beat (hence the rise from 64 h to 128 h).
  - **So a periodic image at distance P costs ~h/P,** and D1 (1e-8) would need P ~ 10⁸ h. A smooth taper
    does not help either: it changes the kernel the solve needs.
  - **Proposed revision (to decide): split D's spectrum at q_p ≈ 1/h** with a smooth partition ψ(q),
    D = D_low + D_high.
    - **D_high**, the part above q_p (q_p above every branch point; ωh/c ≤ 1/2 here), is smooth in k, so its
      kernel decays faster than any power of r. It is sampled on the FFT grid (with the Brillouin-zone
      aliases q < Q0) with modest padding, gated to 1e-8.
    - **D_low**, the part below q_p, carries the branch points and the h/r tail. It stays a per-offset
      radial integral, over [0, q_p] instead of [0, Q0]: about 10× fewer panels. Its S_n come from
      Jacobi–Anger (|m| ≤ 3.5 qh + 20, so ~24 terms at qh ≤ 1), with no per-offset tables.
    - Cost per frequency: D_high O(L² n² log n); D_low O(L² n² × the low rule's nodes) with a small constant.
- **Built (16 October, user: "build S4 with the split").** `interface_spectral.high_band_blocks`,
  `truncated_blocks(low_pass=...)`, `InterfaceCouplings(split=True, split_pad=40)`. The band runs from twice the
  highest branch point (at least 0.35/h) to the window's start (2.5/h).
  - **D_high's decay** (`|D_high|` relative to K0, by cells along x):

    | band | 8 | 16 | 24 | 32 | 40 | 48 |
    |---|---|---|---|---|---|---|
    | [1, 2.5] (needed at ωh/β = 1/2) | 1.1e-4 | 2.3e-6 | 1.1e-7 | 1.7e-8 | 2.8e-9 | 7.5e-10 |
    | [0.35, 2.5] (ωh/β = 1/6) | 3.8e-5 | 4.3e-7 | 1.7e-8 | 1.2e-9 | 1.1e-10 | 1.4e-11 |

  - **Gate `scripts/gate_interface_split.py`: PASS.**
    - **Blocks:** every class of 2 × 4 × 4 cells, sources on both sides, against the unsplit D: 1.3e-10 of K0 at
      ωh/β = 1/6, 2.9e-9 at 1/2.
    - **Solution:** GMRES, split against unsplit, 7.4e-12.
    - **288 cells:** the set-up takes 931 s (S1: 1443 s), since the tables cover the low band only. **But one
      frequency's kernels take 321 s against 183 s**: D_high costs ~0.35 ms per grid point, and with the 40-cell
      padding the grid is 54² cells × ~93k wavenumbers per depth pair. Unsplit D grows as ~n³ per frequency
      (n² offsets × rule nodes ∝ reach), the split as L²(n + pad)², so the crossover should be near n ≈ 20.
      This is not yet measured.
  - **Measured in one process (16 October, `scripts/measure_interface_split_cost.py`).** The gate times the
    first frequency in a process, which also paid the whole-space blocks' per-process series coefficients
    (~246 s at 288 cells). Those coefficients are now computed in the set-up, in parallel with one BLAS thread
    per worker (`far_coeffs`). The high band uses the lateral symmetries (an eighth of the wavenumbers).

    | 288 cells | set-up | first frequency | each further frequency |
    |---|---|---|---|
    | unsplit | 839 s | 79 s | 67 s |
    | split | 823 s | 27 s | **25 s** |

    The run-to-run spread of the set-up on this container is ~±40 s. A pool of one task per pair of depths
    for each frequency was slower (77 s split, 144 s unsplit): the serial route already uses the CPUs through
    BLAS threads, so it was removed.
  - **Open.** (a) The high band's constant: the k-grid's 8-fold symmetry (signed permutations of kx, ky, as
    `_transform` maps the blocks), and the lateral moments and rotation shared by the depth pairs, for perhaps
    ~5–10×. (b) Measure the crossover at n = 24, 48 (the unsplit set-up at n = 24 alone is hours, because of its
    reach² tables). (c) Jacobi–Anger for the low band's tables, now cheap anyway.

### S5. Matrix-free in depth (≈ 1 week; this is Stage A's machinery)
- **Decided (user, 16 October).** D3: built as Stage A's machinery, in `stratified_march`. D_low is applied by
  non-uniform FFTs (finufft) over its polar quadrature: exact and non-periodic.
- **Revised, after S4.** A uniform k-grid cannot carry D_low: it decays only as ~h/r (S4), and the grid wraps
  that tail. So:
  - **D_low** (q h < 2.5): project each source layer onto the modes at its polar quadrature nodes, by a
    type-2 NUFFT of the layer's source coefficients (cells on the lateral grid, nodes non-uniform); apply
    R/T; reconstruct at each receiver layer by a type-1 NUFFT. 3 amplitudes per node and source side.
  - **D_high**: the same on the uniform grid of the periodic box, each Brillouin-zone alias κ + G its own
    node, so the product stays rank 3 per node (the alias sum of S4 has rank ~3 × 37).
  - **The window's static parts** (χ(Y0 + ω²Y1) in D_high) are separable in depth through their vertical
    moments: pseudo-modes, one per moment order.
  - **Whole-space blocks** by Paper 2's 3-D FFT per side; the corrections A0, A1 (within the lateral and
    depth radii) as sparse local blocks.
  - finufft at eps = 1e-14 reaches 1.1e-14 (type 2, 24² grid, 2·10⁴ nodes).
- **Built (16 October).** `stratified_apply` (`mode_factors`, `NodeProduct` / `TwoSidedProduct`,
  `apply_low_band`, `apply_high_band`) and `interface_solver.MatrixFreeOperator`.
  - **Pieces, against today's per-offset products:** the factors rebuild `integrand_points` (to the rounding
    of the modes' cancellation, ~1e-19 absolute); D_low 9.3e-14 / 1.7e-14 (ωh/β = 1/6, 1/2); D_high
    1.2e-11 / 2.1e-13. The terms at large q reach ~1e5 of the sum, which sets the high band's rounding.
  - **Gate `scripts/gate_interface_matrix_free.py`: PASS.** The products agree with `LateralOperator`'s to
    7.3e-13 (50 cells) and 2.1e-12 (288 cells), from the same split set-up with radius 16 h and depth radius 4 h.
  - **Cost:** the operator for a frequency takes 30–38 s (factors kept), but **each product 12–16 s against
    0.01–0.04 s** stored. The product is the NUFFTs: ~55 ms per 4096 nodes × 90 components, whatever the box.
    The high band's nodes number π(Q0 N h/π)², N = box + 41 cells, so at these sizes the padding sets the
    count (~93k per side at 288 cells).
  - **Memory:** the first 288-cell run was killed. The cached rotated static kernels cost ~83 kB per node;
    now only their q-polynomial coefficients are kept, and peak memory is 5.7 GB with both operators.
  - **The trade.** At these sizes the stored kernels are far faster per product. The matrix-free product's
    memory grows as nodes × layers, not L²(2n)², which is what the 0.8 TB wall needs; its time per product
    grows as nodes × layers × components. The rest of S5's gate, memory and time against n = 12 … 64 and
    L = 2 … 16, is still to measure.
  - **Item (a) done (16 October, user: "Item 2"): the whole-space coupling between planes two or more apart by
    a march in depth** (`stratified_apply.WholeSpaceMarch`). G2's factorisation, each cell's moments about its own
    centre, so the factors do not depend on the layer; the vertical phase steps layer by layer (|p| ≤ 1);
    nodes: the low band's polar quadrature and the box's lattice to q h = 16. Stored lateral kernels remain only
    for planes 0 and 1 apart: 3 (2n − 1)² blocks per side, whatever L.
    - **Against Paper 2's blocks** (5 layers × 3 × 2): 3.9e-10 (1/6), 3.6e-9 (1/2). This is the lattice's
      images 40 cells away, as for the split D: 5.5e-11 at 1/2 with 60 cells.
    - **Full product against `LateralOperator`:** 96 cells in six layers, 3.6e-11; 50 cells, 7.3e-13; PASS.
    - **Memory now sits in the cached factors:** ~23 kB per node for D with six layers, ~12 kB per node for the
      march; 12.8 GB peak at 96 cells with both operators. `MatrixFreeOperator(cache=False)` recomputes them
      in each product instead. At 64 × 64 × 8 the nodes number ~10⁶, so there the factors are not cached.
  - **Scaling (16 October, `scripts/measure_interface_matrix_free_scaling.py`** → `scripts/data/
    interface_matrix_free_scaling.jsonl`; a fresh process per box). The run stopped at the 2 h limit before n = 36,
    L = 8; cached factors at n = 12, L = 8 ran out of memory.

    | n × n, L | product, uncached | peak, uncached | product, cached | peak, cached | stored kernels |
    |---|---|---|---|---|---|
    | 12, 2 | 47 s | 0.8 GB | 16 s | 3.6 GB | 0.1 GB |
    | 12, 4 | 67 s | 1.2 GB | 26 s | 6.3 GB | 0.4 GB |
    | 12, 8 | 243 s | 1.2 GB | killed | — | 1.8 GB |
    | 24, 2 | 67 s | 1.7 GB | 24 s | 6.1 GB | 0.5 GB |
    | 24, 4 | 95 s | 2.8 GB | | | 1.8 GB |
    | 24, 8 | 377 s | 2.9 GB | | | 7.3 GB |
    | 36, 2 | 90 s | 3.1 GB | | | 1.0 GB |
    | 36, 4 | 138 s | 5.7 GB | | | 4.2 GB |

    - **Memory:** uncached, it is set by n, not L (L = 4 → 8 at n = 24: 2.8 → 2.9 GB); the stored kernels grow as
      L² n². Most of it is the whole-space series coefficients kept in the set-up (~1300 orbits × 2.5 MB = 3.3 GB at
      n = 36; 10 GB at 64), which are needed only to build the stored kernels of planes 0 and 1 apart
      (3 (2n − 1)² per side: 1.6 GB at 36, 5 GB at 64). Dropping the coefficients once the kernels are built
      would leave those kernels plus the grids.
    - **Time per product:** 47 s → 377 s, set by the wavenumber nodes (D: 0.26 M at n = 12 to 0.74 M at 36, the
      padded lattice) and, with the march (from L = 6), its lattice to q h = 16 (0.54 M at 12, 0.84 M at 24), times
      the layers. That is minutes per product, so a GMRES solve takes about an hour at n = 24, L = 8. **The
      product's cost is now the obstacle, not memory.**
    - **Set-up:** 130 s → 895 s (n = 36, L = 4), mostly the whole-space coefficients of the near planes (∝ n²).
  - **Done (16 October): items 1 and part of 2.** `release_far_coeffs()` drops the coefficients after the last
    operator. The factors are now kept factored (lateral moments, mode maps and layer phases apart), and the
    window's frame kernels are warmed when the operator is built. Re-measured:

    | n × n, L | uncached before | uncached after | cached before | cached after |
    |---|---|---|---|---|
    | 12, 4 | 67 s, 1.2 GB | 42 s, 1.2 GB | 26 s, 6.3 GB | 31 s, 2.5 GB |
    | 12, 8 | 243 s | | killed | 84 s, 3.8 GB |
    | 24, 8 | 377 s, 2.9 GB | | — | 139 s, 7.2 GB |
  - **Done (17 October): the rest of the speed work.** `MatrixFreeOperator(stored_planes=2)` (the default) stores the
    whole-space kernels of planes up to 2 apart (5 (2n − 1)² per side, 5/3 of before), so the march starts at
    gap 3, where the integrand falls as e^{−4qh}, and its lattice stops at q h = 8. That is a third of the nodes
    (58k against 176k on 5 layers × 3 × 2), and against Paper 2 it agrees to 3.4e-10 / 1.6e-9 (1/6, 1/2). The
    NUFFT tolerance is now 1e-11. Gate, 96 cells in six layers: products 7.0e-13 apart (was 3.6e-11); operator
    22 s (was 179 s), product 24.7 s (was 39.4 s), peak 4.2 GB (was 12.8 GB). 50 cells: 8.3e-13.
  - **Done (17 October): the lattice nodes by FFT.** The nodes of D_high and the march's nodes above the low band
    lie on the periodic box's lattice, k = 2πm/(N·2h), so they need no NUFFT: one zero-padded N × N FFT per layer
    gives all of them (aliases included), and the way back folds by m mod N and takes one inverse FFT per layer.
    Exact, so it adds no error (against the dense product it sits at the modes' cancellation, ~1e-11, as the NUFFT
    at 1e-13 does). Lattice share of the nodes: all of D_high, 65–70 % of the march's at n = 12. n = 24: D_high
    47 s → 11 s, march (6 layers) 14 s → 8 s. Gate: 50 cells 6.3e-13, product 11.4 s → 4.0 s; 96 cells in six
    layers 6.3e-13, product 24.7 s → 9.3 s, peak 4.2 → 2.7 GB. The polar low band is now the NUFFT's only use.
  - **Done (17 October): the NUFFT tolerance and the transforms' sizes (8.2, 8.3).** Only the polar low band still
    uses the NUFFT. On its nodes alone (one source and one receiver layer, n_trans 90 and 36): eps 1e-8 is 1.5–1.8×
    faster than 1e-11 at n = 12, 24, 48; upsampfac 1.25 is slower at every size; one large chunk instead of 4096-node
    chunks is no better (0.6–1.6×). Per product the NUFFT is ~47 % at 12 × 12 × 2 (contractions ~25 %). Product
    error against the stored kernels (96 cells): 6e-13 / 7e-12 / 2e-12 / 2e-11 at eps 1e-11 / 1e-9 / 1e-8 / 1e-7.
    Default now 1e-9: ~1e-11 from the stored product, below GMRES's 1e-10 (72 cells at 1/2, 50 %: 16 products
    with either operator). Layers are not stacked into n_trans: chunking makes no difference, so neither would that.
  - **Done (17 October): the low band's quadrature (8.4).** `stratified_apply.LowBandRule`: the matrix-free product's
    low band (D's and the march's) uses Gauss of order 8 on the plain radial panels (was 12) and an azimuthal margin
    of 16 beyond the phase's bandwidth (was 64); the stored kernels keep the reference rule. Scanned on 4 × 4 to
    24 × 24 boxes at 1/6 and 1/2: 33–56 % of the nodes, the low band's product within 1.3e-12 of the reference's.
    What does not survive: the branch panels' order below 24 (4e-8 at 20, the square-root map), and fewer ramp panels
    with order 8 on narrow boxes (6 panels: 8e-10 at n = 4, 5, where the width rule allows wide panels). Gate:
    1.7e-11 on both boxes (was ~6e-12), product 3.7 → 2.6 s (50 cells), 9.6 → 6.9 s (96 cells); GMRES on 72 cells
    at 1/2, 50 %: 16 products with either operator, solutions 1.5e-11 apart.
  - **Next limits.** (a) Paper 2's whole-space kernels: (2L − 1)(2n − 1)² blocks per side (~12 GB per side at
    64 × 64 × 8). Non-touching depth offsets could go through the same modes (G2), keeping FFT kernels only for
    touching planes. (b) The NUFFT cost per product: eps 1e-11, or fewer components per transform.
- **Storage.** Per lateral wavenumber, the interface part is rank ≤ 6 (the modes P, SV, SH, up and down):
  K^rev(zr, zs; k) = Σ_m r_m(zr, k) R_m(k) s_m(zs, k). So a product can be done in three steps, with
  memory O(n² · modes · moments) and no L²:
  1. project every source layer onto the mode amplitudes (sum over zs);
  2. apply R/T;
  3. reconstruct at every receiver layer.

  These are the projection/reconstruction maps already in Stage A (`sweep_modes`, gate G2).
- **Whole-space part.** It is invariant under translation in z within a side, so a 3-D FFT applies,
  as `fft.py` already does for K^ws.
- **Near field.** The touching whole-space blocks and the A0/A1 corrections within the radii are sparse
  local blocks, applied as such.
- **Gate:**
  - the product equals today's `LateralOperator` product to 1e-12 on 50 and 288 cells;
  - memory and time per product for n = 12 … 64 and L = 2 … 16, against O(N log N).

### S6. The iteration at realistic contrast (≈ 2 days)
- GMRES products against contrast (10, 30, 50 %), size and frequency (ωh/β = 1/6, 1/2), with the
  block-Jacobi preconditioner. Then with a layer-wise preconditioner: Stage B's two-way march as the
  preconditioner, as the parent plan already proposes.
- **Gate:** a table of products. Products should grow sublinearly with N at fixed contrast.
- **Done (17 October): block-Jacobi.** `scripts/measure_interface_gmres_iterations.py`, relative tolerance 1e-10,
  P wave at lateral slowness (0.3, 0.1)/α_A, every cell at the contrast of its own medium; products
  (`scripts/data/interface_gmres_iterations.jsonl`):

  | box (layers × n × n) | cells | ωh/β = 1/6: 10 / 30 / 50 % | ωh/β = 1/2: 10 / 30 / 50 % |
  |---|---|---|---|
  | 2 × 6 × 6 | 72 | 8 / 10 / 12 | 9 / 13 / 16 |
  | 2 × 12 × 12 | 288 | 8 / 11 / 13 | 10 / 15 / 19 |
  | 4 × 12 × 12 | 576 | 8 / 11 / 14 | 11 / 17 / 24 |
  | 2 × 24 × 24 | 1152 | 8 / 11 / 14 | 10 / 16 / 22 |

  The gate holds: at 1/6 the count is flat in N; at 1/2 it grows by 3 at 50 % over 16× the cells laterally
  (72 → 1152). Depth costs more than width: doubling the layers (288 → 576) adds more products at 1/2 (+5 at 50 %) than
  quadrupling the lateral area (+3). All converged; the 1152-cell run peaked at 3.1 GB (stored-kernel operator).
  This is the baseline. The layer recursion as the preconditioner is the point of the approach, so it is built
  and measured against it (S6b), not held in reserve.

### S6b. The layer recursion as the preconditioner (DownUp)

The thesis's two-way march (Eqs. 5.60-5.62, `phasescreen.f`'s `DownUp`) for the Legendre cells, as the
preconditioner of a flexible GMRES. Decided (17 October): split by direction; each plane solved by an inner GMRES.

- **The splitting.** K = K_0 + L + U + R, by depth plane:
  - K_0, a plane with itself: the whole-space self plane, the interface blocks (reflection) and corrections of
    the plane with itself;
  - L, from planes above: the whole-space coupling (stored kernels for touching planes, the march beyond) and
    the transmission A -> B (interface blocks of opposite sides, source in A);
  - U, from planes below: the mirror (whole-space from below, transmission B -> A);
  - R, the reflections between different planes of one side (interface blocks, D and corrections, same side,
    z_r != z_s): the backscatter.
- **One application** to b, from zero: a down sweep, plane by plane from the top, x'_i = A_i^-1 (b_i + sum_{j<i}
  L_ij x'_j), with A_i = M - K_0,ii E the plane's own system; then the backscatter r = R x' (lagged); then an
  up sweep from the bottom, x''_i = A_i^-1 (b_i + r_i + sum_{j<i} L_ij x'_j + sum_{j>i} U_ij x''_j). Forward
  scattering is exact within each sweep, backscatter is lagged once. Optionally several DownUp passes.
- **The planes' solves.** Inner GMRES on A_i (lateral FFT products of the plane, block-Jacobi), to a loose
  tolerance; the outer solve is FGMRES (right preconditioning, restarted), since the preconditioner varies.
- **Two implementations.** (1) On the stored kernels (`LateralOperator` keeping the whole-space and interface
  kernels apart): L, U, R per pair of planes, O(L^2) per application; it answers the question, how many outer
  iterations, and at what total cost. (2) On the recursions (`MatrixFreeOperator`): L and U through the
  march's amplitudes and the transmitted modes in one pass down and one up, R through the reflected modes from
  the down sweep's amplitudes, O(L) per application; gated against (1) to ~1e-10.
- **Gate:** against block-Jacobi on the S6 boxes and on boxes with more planes (2 ... 8 layers): outer
  iterations, in-plane products and time to 1e-10. The march preconditioner should make the outer count
  nearly independent of the number of layers, where block-Jacobi's grows (+5 from two to four layers at 1/2).
- **Measured (17 October), stored kernels, 6 x 6 cells per plane** (`scripts/measure_interface_downup.py`,
  `scripts/data/interface_downup.jsonl`; one pass, inner tolerance 1e-2). Outer products, block-Jacobi / DownUp,
  at 10 / 30 / 50 %:

  | layers | 1/6 | 1/2 |
  |---|---|---|
  | 2 | 8/10/12 / 4/6/6 | 9/13/16 / 5/7/7 |
  | 4 | 8/10/12 / 5/6/7 | 10/14/19 / 6/8/9 |
  | 6 | 8/11/12 / 5/6/7 | 10/15/20 / 6/9/11 |
  | 8 | 8/11/13 / 5/6/7 | 10/16/21 / 6/9/12 |

  At 1/6 one pass makes the count independent of the layers. At 1/2 it grows with them, from the backscatter
  lagged once: more passes remove the growth (8 layers, 50 %: 12 / 8 / 6 outer with 1 / 2 / 3 passes; 2 layers:
  7 / 6 / 6), a tighter inner solve does not (1e-4: 12). With three passes the outer count is 6 on 2 and on 8
  layers, against block-Jacobi's 16 and 21. Solutions agree with block-Jacobi's to the tolerance.
- **Recursion version** (`MatrixFreeDownUp`, `scripts/gate_interface_downup_recursion.py`): the same application
  to 1.5e-12 (1/6) and 1.9e-11 (1/2) on 8 x 2 x 2, where the march acts. Its cost is the planes' own systems:
  with a plane's own reflection D in K_0 each inner product runs the plane-wave D over every wavenumber
  (~3.5 s), so one application costs ~8 full products. The option `self_reflection="lagged"` moves that D into R,
  leaving K_0 the plane's stored whole-space kernel and corrections (lateral FFTs). Measured on 4 x 6 x 6: the
  same outer counts as the exact K_0 (5/6/7 at 1/6, 6/8/9 at 1/2). Several passes: gated to 1.9e-11 (2 passes).
- **Cost, matrix-free, 8 x 6 x 6 at 1/2** (one product 13.5 s). 30 %: block-Jacobi 16 outer, 221 s; DownUp
  1 / 2 / 3 passes 9 / 6 / 5 outer, 388 / 429 / 490 s. 50 %: block-Jacobi 21 outer, 264 s; DownUp 12 / 8 / 6
  outer, 476 / 565 / 563 s. A pass costs ~2 products: each plane's amplitudes (both sweeps), the transmitted and
  reflected fields and the march fields each run one layer at a time over every wavenumber chunk, where a full
  product shares one pass over the chunks among all layers. So the iteration count halves (and with three passes
  no longer grows with the layers) but the time does not yet: the pass must cost well under a product. Next:
  profile the pass, then a cheaper pass (coarser wavenumbers for the preconditioner, which need not be exact).
- **A cheaper pass (17 October).** Profile of one application (8 x 6 x 6, 38 s against 12.8 s per product): D's
  amplitudes 15 s and fields 8 s, summing amplitude lists 4 s (now kept running), the march 1 s, the planes'
  solves 0.4 s. So the preconditioner's D and march get nodes of their own (`MatrixFreeDownUp(coarse=...)`,
  `interface_downup.COARSE`): D's low band only (the high band is local), a coarse low-band rule (3.6k nodes
  against the product's 14.9k), NUFFT eps 1e-6, the march's lattice to q h = 4. None of it changes the outer
  count. Matrix-free, 8 x 6 x 6 at 1/2, one product 12 s:

  | | 30 % | 50 % |
  |---|---|---|
  | block-Jacobi | 16 outer, 191 s | 21 outer, 247 s |
  | DownUp, product's D (1 pass) | 9 outer, 342 s | 12 outer, 460 s |
  | DownUp, D's low band only | 9 outer, 222 s | 12 outer, 298 s |
  | DownUp, coarse D | 9 outer, 167 s | 12 outer, 220 s |
  | DownUp, coarse D and march (`COARSE`) | 9 outer, 138 s | 12 outer, 185 s |
  | the same, 3 passes | 8 outer, 181 s | 8 outer, 176 s |

  An application now costs ~1/4 of a product: DownUp is 25-28 % faster than block-Jacobi on eight layers, and
  the margin should grow with the layers, where block-Jacobi's count grows. With D's low band alone three passes
  give 8 outer, not the 6 of the full D: its high band matters to the later passes.
- **Sixteen layers** (16 x 6 x 6 at 1/2, one product 25.2 s): block-Jacobi 17 / 24 outer, 404 / 542 s (30 / 50 %);
  DownUp (`COARSE`) one pass 10 / 14 outer, 294 / 414 s; three passes 7 / 8 outer, 303 / 337 s. With three passes
  the outer count at 50 % is 8 on eight and on sixteen layers, block-Jacobi's 21 and 24: the saving in time grows
  with the layers (29 % on eight, 38 % on sixteen).
- **A larger lateral box** (8 x 12 x 12 at 1/2, one product 19.1 s, peak 3.2 GB): block-Jacobi 19 / 27 outer,
  343 / 486 s (30 / 50 %); DownUp (`COARSE`) one pass 10 / 14 outer, 295 / 414 s (-14 / -15 %); three passes 7 / 8
  outer, 374 / 433 s (+9 / -11 %). The counts hold (block-Jacobi grows with the area too, 21 -> 27 at 50 %, three
  passes stay at 8), but the time saving narrows: the coarse low band grows with the reach (9.0k nodes against
  3.6k; the azimuthal count is proportional to the reach), so a pass costs ~1/2 product. Next: bound the
  preconditioner's nodes independently of the reach (e.g. a fixed azimuthal count, accepting a less exact D far
  away), then S7.

### S6c. Algorithm 5.1 for the plane-averaged contrast (decided 18 October)

The thesis's exact up/down sweep (Alg. 5.1, A = UL, Eq. 5.49, pivots Kennett's net reflection operators,
5.50-5.51) for the medium in which each plane's contrast is its lateral mean: then M - K E is invariant under
lateral translations within each plane and separates by lateral wavenumber (on the stored kernels' padded grid,
the circulant approximation), and at each bin the depth system over the planes is solved exactly, so the
backscatter between planes of the averaged medium is exact and only the lateral fluctuations are left to the
outer iteration. `interface_downup.AveragedSweep`: on the stored kernels a dense LU per bin is the same inverse;
the Riccati form is needed for the matrix-free operator and many planes.

Future (noted by the thesis's author, 18 October):
- **The homogenised layer** in place of the plane's mean contrast (the next step of the PhD development, left
  out for time): each plane's effective layer by the layer law (Paper 3, gate G7, `kennett_layers`), its net
  R/T per wavenumber as the pivots.
- **The Dirichlet-to-Neumann version of Algorithm 5.1, solved partially.** The Schur complement S_z is the
  discrete DtN map of the half-domain above plane z; with the laterally varying contrast it is dense per plane,
  but FGMRES accepts a preconditioner that varies between applications, so an inexact (partial) DtN solve,
  e.g. an iterative solve of S_z or a low-rank or banded approximation, can stand in for the exact one.

### S7. A realistic size (≈ 1 day)
- For example 64 × 64 × 8 (32k cells, ~300k unknowns at p = 1) on 4 workers and < 16 GB: set-up, time per
  frequency and per incidence, and convergence in p on a subset.
- **Gate:** it runs, and a 12 × 12 × 2 sub-box reproduces S5's product.

---

## 4. Decisions to confirm

- **D1** (S3): the block tolerance, 1e-6 proposed, and the radii it implies.
- **D2** (S3): K1 in production, or K0 with a larger Q0.
- **D3** (S5): whether S5 is built as Stage A itself (the march's projection and reconstruction), or as
  an interim operator within the present `interface_solver.py`. The proposal is to build it as Stage A,
  so that there is one plane-wave machinery.

## 5. Out of scope

- Cells straddling an interface; anisotropic layers (as the parent plan).
- Fast multipole methods: in a layered medium the plane-wave/FFT route is the natural one, and the
  lateral grid is regular.
- Changes to the closed forms themselves.
