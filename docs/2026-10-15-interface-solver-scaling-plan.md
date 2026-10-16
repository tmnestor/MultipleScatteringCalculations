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

### S5. Matrix-free in depth (≈ 1 week; this is Stage A's machinery)
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
