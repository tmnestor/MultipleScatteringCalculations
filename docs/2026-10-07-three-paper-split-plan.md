# Re-scoping the voxel series into three papers: section-by-section move plan

**Status:** a plan for review. No `.tex` file has been changed.
**Line numbers** refer to commit `0cad309` on `claude/practical-tesla-a2hpru`, which is the state you will pull.

| File | Lines | PDF pages | Short name below |
|---|---|---|---|
| `LatexPDFs/ContinuumLimit/ContinuumLimit.tex` | 3192 | 72 | **CL** |
| `LatexPDFs/ExactCouplingIntegrals/ExactCouplingIntegrals.tex` | 1340 | 29 | **EC** |
| `LatexPDFs/AdaptiveOctree/AdaptiveOctree.tex` | 947 | 21 | **AO** |

## 1. The new scopes

| Paper | Subject | Working title (proposal) | Home directory |
|---|---|---|---|
| **P1** | The single-site T-matrix of a cubic voxel, by closed-form integration | *The Single-Site T-Matrix of an Elastic Cube in Closed Form: a Cartesian Multipole Hierarchy* | `ContinuumLimit/` (rename later if you want; renaming now breaks paths in scripts and the JCP builder) |
| **P2** | The coupling between two cubic voxels, by closed-form integration | Unchanged: *The Coupling Integrals of Polynomial Cubes in Closed Form* | `ExactCouplingIntegrals/` |
| **P3** | Error analysis: the convergence law, its constants, the projection-error law, 3D and the octree | *The Error of Polynomial Voxels for Elastic Waves: a Law Isolated on a Layer, Carried to Three Dimensions and Used to Refine an Octree* | `AdaptiveOctree/` |

**Rough sizes after the move.** These are estimates from the current page map, not measurements.

| Paper | Estimate | Comment |
|---|---|---|
| P1 | about 25–28 pp | Down from 72 |
| P2 | about 28–30 pp | Gains the hierarchy-as-a-voxel-scheme section, loses the single cell |
| P3 | about 60–65 pp | **This is the new long paper.** See decision D6 (§6) |

## 2. Paper 1, `ContinuumLimit.tex` (CL)

Each row gives where the text goes and why. **ALL** means each of the three papers needs its own version of it.

### Front matter and introduction

| CL lines | Content | To | Notes |
|---|---|---|---|
| 1–52 | Preamble, macros | ALL | Copy verbatim. P3 drops `\bA` and P1 drops `\sinc` only if unused (see §5). |
| 54–58 | Title, running head | P1 | Retitle |
| 63–86 | Abstract | P1 rewritten | 65–77 (two bases, the layer, order law, c_p, projection-error law) → P3's abstract. 78–83 (hierarchy, four constants, 300× against Mie) stay in P1. 83–84 ("as a voxel scheme it is fourth order") → P2. |
| 90–97 | The VIE / T-matrix / Foldy–Lax problem | ALL | Shared opening paragraph |
| 99–112 | The two bases; Chobanyan, Georgakis | P3 | |
| 114–120 | Why the error can't be measured | P3 | |
| 122–138 | The layer method; patch test | P3 | |
| 140–165 | "The answer" (1)–(6) | P3 | Item (6) mentions `sec:gradvoxel`; cite P2 there |
| 167–172 | (a) exact coupling, tiling | P3 | |
| 172–184 | (b) no polarisability, moments as distributions, Eshelby δ, cubic anisotropy; (c) nine-component state | **P1** | |
| 186–198 | Coupled-dipole history and the polarisability literature | **P1** | |
| 198–203 | Coupling literature (integrated/filtered Green tensor, periodic) | **P2** | |
| 203–216 | DDA convergence (Yurkin, Kahnert, Costabel); "what this paper adds" | P3 | |
| 218–228 | Elastic counterparts (Gubernatis, Nestor 1996, Kanaun, …) | P1 (copy the relevant ones to P3) | |
| 230–240 | Roadmap | Rewrite per paper | |

### §2 Setting (`sec:setting`, 242–378)

| CL lines | Content | To |
|---|---|---|
| 244–260 | Medium, state ψ, Foldy–Lax (`eq:foldylax`) | ALL |
| 262–296 | Body force, the 9×9 kernel, Δ (`eq:bodyforce`, `eq:byparts`, `eq:delta`); `fig:kernel` 298–322 | **P1** (P2 needs the kernel's block structure: one paragraph plus a citation) |
| 324–340 | Three kinds of averaging; collocation against receiver test | P3 (P2 needs the source-cell-average definition) |
| 342–352 | The layer as n planes, specular sums (`eq:specsum`); `fig:lattice` 354–372 | P3 |
| 374–378 | Implementations (`cubic_scattering`) | ALL |

### §§3–5: the continuous layer, the specular sums, the tiling identity

| CL lines | Content | To | Notes |
|---|---|---|---|
| 380–416 `sec:continuum` | Exact layer R/T, thin-layer series (`eq:thin`, `eq:thin1`) | P3 | P1's `sec:blocks` needs a two-line exact-layer reference (D1) |
| 418–476 `sec:point` | Specular sums; "converges to the wrong limit" (`eq:poisson`, `fig:poisson`) | P3 | Alternative: P2 (D3) |
| 478–540 `sec:tiling` | Tiling identity, exact same-plane kernel (`eq:between`, `eq:tiling`, `eq:exactkernel`, `fig:tiling`) | P3 | Alternative: P2 (D3) |
| 542–552 | "Cell integrals without quadrature": the Kelvin tensor by face/corner sums, the self term | **P1** | `sec:distributions` (L841) depends on it |

### §6 The single site (554–1203): the core of the new P1

| CL lines | Content | To | Notes |
|---|---|---|---|
| 554–675 `sec:single` | Literature; naming the hierarchy; multipole orders; `tab:multipoles`, `fig:multipole` | **P1** | Reword L556 ("every error … is made at the single site (§discussion)") as a citation of P3. **`fig:multipole`'s caption (L637) cites `eq:gradkernel`, which goes to P2**: cite P2 there, or redraw panel (a) for one cell. |
| 677–812 `sec:hierarchy` | The closed ℓ = 3 system, `eq:hierarchy`, grades, parity separation | **P1** | Self-contained |
| 814–881 `sec:distributions` | Moments as distributions; 4224 closed forms on four constants | **P1** | Uses the corner sums of 542–552 (kept in P1) |
| 883–978 `sec:parity` | O_h channels; closed-form concentrations; cube against sphere; Mie to 2e-10 | **P1** | L895–898 and L964–968 point to `sec:convergence`/`sec:fourth`: cite P3 instead. L952 takes ν = 7/32 from `sec:convergence`: state the background locally (copy CL 1209–1212). |
| 980–1046 `sec:blocks` | The hierarchy on the 1D layer: 36 orders, truncation factor `eq:taylorfactor` | **P1** (D1) | Needs the exact layer (restate 382–386 in two lines). The link to `eq:errorR` (L1045) becomes a P3 citation. |
| 1048–1123 `sec:singlesite` | Sphere (300×) and cube against references | **P1** | Its cube reference uses the Legendre voxels of `sec:fourth` (L1093): cite P3, or describe it as a converged subdivided-cube Galerkin solution. Contrast parameters from `sec:convergence` (L1057): state them locally. |
| 1127–1134 `sec:closure` (first part) | The uniform-field closure T-matrix (`eq:closure`) | **P1** | |
| 1135–1146 | Self-term cancellation: the closure as an exact collocation of the continuum (`eq:collocation`) | P3 | Uses the tiling results. P1 keeps one sentence citing P3. |
| 1148–1155 | Package cube T against the closure; missing even-power self terms | **P1** | The last sentence ("by `eq:collocation` that error cannot enter") → P3 |
| 1157–1203 `sec:whyclosed` | What a closed-form single site gives | **P1** | Trim 1196–1202 (lattice economy against Legendre) to a pointer to P2 and P3 |

### §§7–10: the error analysis, all to P3

| CL lines | Content | To | Notes |
|---|---|---|---|
| 1204–1273 `sec:convergence` | Canonical experiment; order 2.000; (kd)²/8 and −(kd)²/24; `eq:errorR` | P3 | **Copy** the medium parameters 1209–1212 into P1 as well |
| 1275–1426 `sec:fourth` | Legendre voxels; orders 4, 6, 8; c_p (`eq:cp`, `tab:ratio`, `tab:fourth1d`, `tab:higher`) | P3 | The lateral form-factor paragraph 1403–1416 is cited from P2 |
| 1430–1655 `sec:hetero` | Heterogeneous layer; min(2p+2, 2r+2); projection-error law (`eq:defectlaw`); stratified background | P3 | |
| 1658–1781 `sec:sphere` | Graded sphere in 3D; orders 2.0 and 4.0; `fig:sphere` | P3 | Joins the graded-sphere series study from EC (§3) |

### §11 The hierarchy as a voxel scheme (1784–1933)

| CL lines | Content | To | Notes |
|---|---|---|---|
| 1786–1825 | Scheme, coupling block `eq:gradkernel`, self block = moment, checks | **P2** | This is cell-to-cell coupling of hierarchy cells. CL's conclusions (2142–2144) already call P2 its worked application. |
| 1827–1856 | Subdivided homogeneous cube, `tab:hiersub` | **P2** | |
| 1858–1907 | Graded sphere: hierarchy against Legendre orders, `tab:hiergraded` | P3 (D2) | A convergence comparison; P2 is the alternative |
| 1909–1930 | Cost of the tables | **P2** | |

### Discussion and conclusions

| CL lines | Content | To |
|---|---|---|
| 1936–1974 | `fig:convergence` | P3 |
| 1976–2009 | What each basis is for; where the error lives | P3 |
| 2011–2017 | What the single site is | **P1** |
| 2019–2025 | Relation to other methods (pointer) | P3 |
| 2027–2045 | Two-error tension | P3 (2027–2039 also useful in P2) |
| 2047–2066 | Cost, fast solver, limitations, graded voxel | P3 |
| 2066–2071 | Contrast range, k_S h ≤ 0.3 | P3; the single-site validity sentence also goes to P1 |
| 2071–2076 | Single-site dynamic error in closed form only for a sphere | **P1** |
| 2077–2082 | Hierarchy as a voxel scheme: 3.8 not 4; fifth gradient not assembled | **P2** |
| 2082–2089 | Extensions (3D solver, stratified, finite body) | P3 |
| 2093–2106 | Conclusions: bases, order law, projection-error law, sphere | P3 |
| 2108–2116 | The cube's three properties; hierarchy; 300× against Mie | **P1** (the first and last clauses of 2108–2111 → P3) |
| 2117–2119 | "As a voxel scheme it is fourth order" | **P2** |
| 2122–2140 | Error known in advance; `NestorOctree2026` | P3 (now its own content) |
| 2141–2144 | Series roadmap | Rewrite for all three |

### Appendices

| CL appendix | Lines | To | Notes |
|---|---|---|---|
| A `app:engine` (moment gates, second route) | 2148–2173 | **P1** | |
| B `app:legendre` (oblique, incident S, degree 2) | 2174–2285 | P3 | |
| C `app:strat` (stratified background) | 2286–2487 | P3 | |
| D `app:sphere` (near field, impedance march, sharp sphere) | 2488–2671 | P3 | Fix: L2575 attributes `eq:riccati` to §hetero, but it is in `app:strat` |
| E `app:solver` (3D voxel solver) | 2672–2744 | **SPLIT** (D4) | The Galerkin block construction 2686–2713 → **P2**; the solver (FFT, GMRES, overlap, orders) → P3. AO L98 already points readers here. |
| F `app:closuremie` (closure against exact sphere) | 2745–2781 | **P1** | L2775 links to `sec:convergence`: cite P3 |
| G `app:borndefect` (scattering series, projection-error law) | 2782–2935 | P3 | |
| H `app:related` (coupled-dipole, spectral-element) | 2936–3013 | P3 | P1 borrows 2940–2948 (polarisability against a derived single site), condensed |
| I `app:cost` (cost of higher order, fast solver) | 3014–3074 | P3 | Items 3 and 5 (3052–3069) are also relevant to P2 |
| J `app:evidence` (`tab:evidence`) | 3075–3166 | **SPLIT by row** | P1: rows 3123–3128, 3130, 3155–3159 and the CubeMoment* / Mie / hierarchy scripts. P2: row 3160. P3: everything else. |
| Acknowledgments, AI declaration | 3167–3187 | ALL | The AI declaration refers to "the repository given under Data availability" (L3176), **but CL has no Data availability section**. Add one to P1 and P3 (P2 has one). |

## 3. Paper 2, `ExactCouplingIntegrals.tex` (EC)

Everything stays except the rows below.

| EC lines | Content | To | Notes |
|---|---|---|---|
| 885–914 `sec:site` | Single-cell Lippmann–Schwinger, T36 (`eq:sitels`, `eq:t36`) | **P1** | |
| 916–974 | Cube-group Schur reduction (`tab:irreps`); graded contrast in one cell | **P1** | |
| 976–983 | Born limit with a gradient, slopes 2.0001/2.0004 (cites the min(2p+2, 2r+2) rule) | P1 (D5) | A single-cell view of the convergence law; P3 is the alternative |
| 985–1038 | Static blocks in closed form on five constants (`eq:tblock`, `eq:fiveconstants`, `eq:uniformblocks`, `eq:c1`, `eq:c0`); averaged against collocated | **P1** | |
| 1040–1050 | Checks against `blocks.near_block`, T36 | **P1** | Keep one sentence of 1043–1045 (the 41 self moments against the stable evaluation, 1.8e-15) in EC `sec:results`, as a check of P2's method |
| 1052–1061 | Reciprocity of every touching block | **stays P2** | Move into `sec:blocks` or `sec:results` |
| 1082–1091 `sec:sphereseries` intro | | P2, rewritten | |
| 1092–1105 | Exact series of the sphere by a Cauchy contour (`GradedSphere_LowFrequency.wl`) | P3 | P2 keeps a pointer. **`eq:farseries` is defined twice (L805 and L1101): rename one.** |
| 1107–1112 | No quadrature error in the static term: closed and Gauss tables agree | **stays P2** | |
| 1112–1114 | Apparent order falls 4.2 → 3.2 | P3 | |
| 1116–1129 | Cell model as a power series in k (`eq:seriessolve`); agrees with direct solves to 6e-11 | **stays P2** | An application of the frequency-independent coefficients |
| 1130–1173 | `tab:sphereseries`, per-power orders; profile dependence | P3 | **Out of date:** the text says the quadratic contrast raises (ka)⁴ and (ka)⁷ to 3.4–3.5. The runs of 7 October give 3.99 at 20 cells for (ka)⁴ and 3.79 at 12 cells for (ka)⁷ (`scratch/graded_sphere_ka4/`). Rewrite with the new results when it moves. |
| 1264 | Data availability: the single-cell clause | P1 | |
| 1266–1333 `app:blocks` | Static blocks of the linear strain (`tab:blocks`) | **P1** | |
| 63–87 | Abstract | Rewrite | It is already stale: it does not mention `sec:twocentre`, `sec:distant`, `sec:sweep`, `sec:fmm` or `sec:why` |
| `data/site_blocks_exact.txt`, `data/site_moments_exact.txt` | | **P1** | Move with `sec:site` |

**Gains from CL:** §11 (1786–1856 and 1909–1930), the block construction of `app:solver` (2686–2713), intro 198–203, discussion 2077–2082, and conclusion 2117–2119.

**Gains from AO:** the unequal-leaf blocks as exact sums of equal-cell blocks, and the O_h orbit reduction (AO 174–187). This is coupling material and fits as a short subsection or remark; AO keeps a summary and cites P2.

## 4. Paper 3, `AdaptiveOctree.tex` (AO), as the error-analysis paper

### Proposed structure

| Part | Content | Source |
|---|---|---|
| **I. The law on a layer** | Setting (averaging, the layer as planes); continuous layer; specular sums and tiling (D3); convergence and c_p; Legendre voxels and orders 4, 6, 8; heterogeneous layer; projection-error law | CL 324–352, 380–540, 1135–1146, 1204–1655; apps B, C, G |
| **II. Three dimensions** | Graded sphere at fixed frequency; graded sphere power by power in ka (the matched-contrast result); hierarchy against Legendre orders | CL 1658–1781, app D, the solver part of app E, 1858–1907; EC 1092–1105, 1112–1114, 1130–1173 |
| **III. The octree** | The current AO body | AO 154–856 |
| Discussion and appendices | | CL discussion (P3 rows), apps H and I; AO `sec:status`, `app:experiments` |

### Citations of P1 that become internal references

Each of these currently cites `NestorContinuum2026` (or the uncited "companion paper"). Rewrite it as `\S\ref{…}` to the absorbed section.

| AO line | What it takes from P1 |
|---|---|
| 92–98 | Order law, projection error |
| 244 | The constants c₀, c₁, c₂ |
| 300 | The layer law |
| 310 | The T₂ error equals the projection error |
| 492 | Layer higher orders, 0.90–0.94 |
| 505–514 | "The companion paper" |
| 728 | Matched bases |
| 874 | Exact sphere reference |
| 935–942 | "The companion paper" |

**Retarget to P2:** AO 172 (the coupling blocks), and AO 98 (the 3D voxel blocks, which go to P2 under D4).
**Keep as P1 citations:** AO 100–104 (the gradient-hierarchy single site).

### Inconsistencies to fix

- AO L186 says "344 leaves of three sizes" and L805 says "four sizes".
- AO's abstract does not mention `sec:higher`.

## 5. Cross-references that break

The table below was generated by script from the line assignment above. Each row is a `\ref`/`\eqref` whose text and target label land in different papers. These must become citations, for example "(Ref. [P3], §x)", or the target must be restated locally. Shared (ALL) and the row-split evidence table are excluded.

### ContinuumLimit.tex

**P1 text → label now in P2 (3)**
- `eq:gradkernel` (L1802) ← L637 (`fig:multipole` caption)
- `sec:gradvoxel` (L1784) ← L1194, L1201

**P1 text → label now in P3 (17)**
- `sec:discussion` (L1934) ← L556
- `sec:tiling` (L478) ← L841. This one goes away if L841 is re-pointed at the corner sums 542–552, which stay in P1.
- `sec:convergence` (L1204) ← L896, L952, L967, L985, L1057, L2073, L2775
- `sec:fourth` (L1275) ← L897, L967, L1093, L1199, L2073
- `eq:errorR` (L1257) ← L1045, L2014
- `eq:collocation` (L1137) ← L1154

**P2 text (from CL §11, app E) → label now in P1 (7)**
- `sec:single` ← L1787
- `sec:blocks` ← L1788
- `eq:taylorfield` ← L1793
- `sec:singlesite` ← L1823, L1827, L1925
- `sec:distributions` ← L1912

**P2 text → label now in P3 (3)**
- `sec:fourth` ← L1786
- `sec:hetero` ← L1816
- `sec:setting` ← L2691

**P3 text → label now in P1 (10)**
- `sec:single` ← L233, L1317, L1945
- `sec:whyclosed` ← L235
- `sec:closure` ← L1671
- `sec:blocks` ← L1859, L1862, L1899
- `sec:parity` ← L2065
- `eq:delta` ← L2498

**P3 text → label now in P2 (5)**
- `sec:gradvoxel` ← L165, L237, L2061, L2674
- `tab:hiersub` ← L1906

### ExactCouplingIntegrals.tex

**P1 text (`sec:site`) → label staying in P2 (6)**
- `sec:blocks` (L163) ← L887
- `eq:series` (L213) ← L996, L1050
- `eq:umoment` (L222) ← L999
- `sec:failure` (L370) ← L1001
- `sec:stable` (L499) ← L1044

These six are the dependency behind decision D5.

### Citations of P1 in EC to retarget

- **To P3** (the convergence law, the sphere, the measured errors): L120, L139, L848, L983, L1084, L1162, and the phrases "that paper" at L1086, L1090, L1109, L1119.
- **Keep as P1:** L136, L169, L677, L799, L854, L914, L1032.

### Bibliography keys

The series is cited as `NestorContinuum2026` (P1), `NestorOctree2026` and `NestorGraded2026`. Under the new scopes, give each paper a key that names its subject, so the retargeting is visible in the source. For example: `NestorSingleSite2026`, `NestorCoupling2026`, `NestorError2026`. Each directory has its own `references.bib`, so the entries must be updated in all three.

## 6. Decisions for you

| | Decision | Recommendation | Alternative |
|---|---|---|---|
| **D1** | Where the hierarchy-on-the-layer validation (`sec:blocks`, `tab:hierlayer`, `eq:taylorfactor`) goes | **P1.** It is what validates the hierarchy's truncation order by order, and `tab:multipoles` and the P1 abstract depend on it. Cost: a two-line exact-layer reference in P1 and a citation of P3 for `eq:errorR`. | P3, which leaves P1 without its order-pair validation |
| **D2** | Hierarchy against Legendre orders on the graded sphere (CL 1858–1907) | **P3**, a convergence comparison | P2, as the worked application of the hierarchy's coupling |
| **D3** | The specular sums and the tiling identity (CL 418–540) | **P3.** They make the layer exact, which is the layer law's foundation. | P2. They concern lattice coupling, but P2 is about pairwise blocks. |
| **D4** | `app:solver` | **Split:** the Galerkin block construction → P2, the solver → P3 | Keep it whole in P3 and cite P2 for the blocks |
| **D5** | The ordering circularity between P1 and P2. `sec:site` is the single site of the **Galerkin Legendre voxel**. Its self block is the double integral `eq:block` at offset zero, evaluated with P2's machinery (`eq:series`, `eq:umoment`, the reductions of `sec:failure`, the stable evaluation). P1's own moment engine (`sec:distributions`) gives *single* integrals of the Green's tensor over the cube, which is what the hierarchy's single site needs. **It does not by itself give the Galerkin self block, so this needs checking before deciding.** | **Move `sec:site` to P1 and cite P2 for the self-block integrals, submitting P1 and P2 together.** The cube-group reduction, the five constants and the comparison with the hierarchy's collocated values (EC 1028–1038) are single-site results. | Keep `sec:site` in P2 as "the self block, applied to one cell", so that P1 is only the hierarchy's single site and needs nothing from P2 |
| **D6** | P3's length (estimated at 60–65 pages) | Keep it as one paper for now, in three parts (§4), and decide after the move whether Part III (the octree) stands alone | Split now: P3 = the error law (layer plus 3D), P4 = the octree |

## 7. Suggested order of work, once the plan is agreed

1. **Build P3 first.** Copy (do not yet delete) the CL error material and the EC sphere study into `AdaptiveOctree.tex` as Parts I–II. Then convert AO's citations of P1 into internal references.
2. **Move `sec:site` and `app:blocks` from EC to P1,** along with `data/site_*.txt`. Resolve D5 there.
3. **Move CL §11 and the app E blocks to EC.**
4. **Prune CL down to the P1 rows,** and write P1's title, abstract, introduction and conclusions.
5. **Work through §5's list,** then rename the bibliography keys.
6. **Split `tab:evidence` by row,** and add Data availability sections.
7. **Build all three PDFs** with LuaLaTeX and check:
   - no undefined references;
   - no duplicate labels (`eq:farseries`);
   - the JCP builder (`ContinuumLimit/JCP/make_jcp.py`) still runs.

## 8. The strain model, and collocation versus Galerkin

### 8.1 The organising principle: the strain model

The governing equation is written in **displacement**. CL `eq:bodyforce` and `eq:byparts` give

u = u⁰ + ∫ G (ω²Δρ u) + ∫ ∂G (δc : ε).

Strain enters only through the stiffness contrast, as the moment δc : ε. **How the strain field is represented is a modelling decision.** The original approximation, from the PhD work, the coupled-dipole method and Eshelby's inclusion, was to take the strain as **uniform in each cell**. The series relaxes that approximation in two different ways:

| Strain model | Scheme | Unknowns | How strain relates to displacement | Test that follows | Where |
|---|---|---|---|---|---|
| **Uniform strain** (the original approximation) | Uniform-strain closure; the hierarchy at ℓ = 1 (CL 762) | Displacement and strain at the centre | Uniform in the cell | At the centre | CL `sec:parity`, `sec:closure`; the "collocation" rows of `tab:ratio`, `fig:convergence` |
| **Derived strain**: the gradient of a Taylor displacement | The hierarchy of degree ℓ ≥ 2 | ∂^P u at the centre, \|P\| ≤ ℓ | Derived: strain to degree ℓ − 1, consistent with the displacement inside the cell | The equation and its derivatives at the centre (Hermite collocation) | CL `sec:hierarchy`, `sec:gradvoxel` (`eq:hierarchy`, `eq:gradlattice`) |
| **Independent strain** | The Legendre cell of degree p | Legendre moments of displacement *and* strain to degree p | Independent. CL 2679–2681 gives the reason: "a strain derived from a cell's projected displacement jumps across the faces between cells and leaves a spurious source there". | Against the cell's own polynomials (Galerkin) | CL `eq:galerkin`, `app:solver`; all of AO; EC `sec:site` |

**Collocation versus Galerkin is not a free choice in this series: it follows from the strain model.**
- Unknowns that are derivatives at a point are determined by imposing the equation and its derivatives at that point.
- Unknowns that are moments over a cell are determined by testing against the cell's own polynomials.

The single place where the test is the *only* difference is at the bottom. There, the uniform-strain closure (centre test) and the Legendre cell of degree 0 (cell-average test) share the uniform state and differ in the constant of their (kd)² error: (kd)²/8 and −(kd)²/24 against ±(kd)²/12 (`tab:ratio`).

**Recommendation for the text.** Each paper's Setting should:
1. start from the displacement equation;
2. state that the representation of strain is a modelling decision;
3. give the three rows above as a short table, identical in all three papers.

Each scheme is then introduced by its strain model, with its test as a consequence. The words "collocation" and "Galerkin" then describe only how a given scheme's equation is imposed, never which scheme it is.

### 8.2 How the text names things now

The current text names schemes mainly by their **test**, which hides the strain model.

**Senses of "collocation":**

| Sense | Where | What it means |
|---|---|---|
| **C1. Point test of the uniform-strain cell** | CL 328–330, `sec:closure` (`eq:collocation`), `tab:ratio`, `fig:convergence`; CL 178, 191, 2944–2946 (Lakhtakia) | The uniform-strain closure, imposed at the centre |
| **C2. Hermite collocation** | CL `sec:gradvoxel`; AO 102 ("a voxel scheme collocated at the centres") | The derived-strain hierarchy. CL 665 and 1854 call it the "Taylor form". |
| **C3. Centre value of the self-interaction** | EC 1028–1038 ("Averaged and collocated") | Not a scheme: the self-interaction at the cube's centre against its cell average |
| **C4. Centre-sampled contrast** | CL 1750–1761, 1956–1958, 1971–1973 | The **medium's** representation (sampled against projected), unrelated to the test |

"Galerkin" means the Legendre cell, at p = 0 (the "mean-only voxel", CL 330–332) or at degree p (`eq:galerkin`, AO throughout).

**One scheme, several names:**
- The uniform-strain closure is also called "collocation", "collocation closure", "the collocation voxel" and "uniform-field voxel".
- The Legendre cell of degree 0 is also "mean-only voxel", "Legendre, mean only" and "Galerkin of degree zero".
- At degree 1 it is "first-moment voxel", "Legendre voxel" and "mean and first moment".

**A claim to check.** CL 1851 says the uniform-strain closure and the mean-only voxel "agree on one cell, as they must, being the same approximation". In `tab:hiersub` they agree at n = 1 (5.4e-3), but they are not the same approximation:
- **They differ in the test.** EC 1028–1038 shows that the cube's self-interaction at the centre and its cell average differ (0.5929 / 0.4314 against 0.4535 / 0.5243).
- **They may differ in the trial too.** The hierarchy at ℓ = 1 carries 12 unknowns (displacement and its full gradient) against the Legendre cell's 9.

The agreement is likely only to the digits shown at this weak contrast. The sentence should be corrected once that is checked.

### 8.3 Proposed names

| Current name(s) | Proposed name | Strain model |
|---|---|---|
| uniform-strain closure, collocation, collocation closure, collocation voxel | **uniform-strain closure** (one name, kept) | Uniform |
| the hierarchy of degree ℓ, Taylor form | **the hierarchy of degree ℓ** (derived strain), defined once as a Hermite collocation | Derived |
| mean-only voxel; first-moment voxel; Legendre voxel | **Legendre cell of degree p** (p = 0: "mean only", p = 1: "first moment", as nicknames after the definition) | Independent |
| the collocation voxel with the contrast sampled at the centre | **uniform-strain closure with sampled contrast** | — |

In addition:
- The medium is "sampled" or "projected (degree r)", never "collocated".
- The self-interaction is the "centre value" or the "cell average", never "collocated".

### 8.4 Why both relaxations are presented

They start from the same original approximation and relax it for **different purposes**, which the new split separates.

- **Derived strain: the T-matrix of one scatterer (P1).**
  - For a single body there are no faces between cells. A Taylor displacement about the centre, with its strain derived from it, is consistent everywhere inside the body.
  - The Taylor expansion about the centre is the Cartesian form of the regular-wave expansion in which multiple-scattering theory defines a T-matrix. Imposing the equation and its derivatives there makes the single-site T-matrix a property of the cube alone. That is why it can be checked against Mie for a sphere and applies to any centrosymmetric body.
  - It is the equivalent-inclusion lineage (Eshelby, Moschovidis, Fu–Mura), and the coupled-dipole method's point dipole is its lowest member.
- **Independent strain: the discretisation of a medium (P3).**
  - When cells tile space, a strain derived from each cell's projected displacement jumps across the faces and leaves a spurious surface source (CL 2679–2681). Making strain an independent unknown removes it.
  - Testing against the cell's own polynomials makes the cell hold the L² projection of the field and of the medium. That is what makes the error law possible:
    - the first-order error is a projection error (CL `eq:ratio`, AO `sec:born`);
    - the second-order error is the relative projection error of the medium (`eq:defectlaw`);
    - functionals converge at twice the order of the field (CL 2978).
  - It also follows non-polynomial fields, edges and corners better than an expansion about a point (Brisard 2014; CL 665, 1852–1856).
- **The comparison is a result where both are present.**
  - At the bottom, the two tests of the uniform-strain cell differ only in their error constant (`tab:ratio`).
  - At fourth order, the hierarchy of degree 3 and the Legendre cell of degree 1 are both fourth order, with 60 against 36 unknowns (`tab:hiergraded`). On a body with corners, the Legendre cell is far ahead (`tab:hiersub`).

**A paragraph each paper could carry, adapted:**

> The governing equation is written in displacement; strain enters through the stiffness contrast, and how it is represented within a cell is a modelling decision. The original approximation takes it as uniform. This series relaxes that approximation in two ways. In the first, the displacement is expanded about the cell's centre and its strain derived from it; the equation and its derivatives are imposed at the centre, and the cell's single-site T-matrix is then a property of the cell alone (Paper 1). In the second, the strain is an independent unknown with its own Legendre expansion, because a strain derived from a cell's projected displacement would jump across the faces between cells. The equation is tested against the cell's own polynomials, so the cell holds the projection of the field and of the medium, and its error is a projection error known before the solve (Paper 3). Paper 2 supplies the coupling integrals both need.

### 8.5 Consequences for the move plan

- **D5 (where EC `sec:site` goes).** `sec:site` is the single cell of the **independent-strain** Legendre cell (T36), whose self block is P2's double integral.
  - If P1 is the T-matrix of one scatterer in the derived-strain form, `sec:site` fits P1 only as its independent-strain counterpart. In that case, EC 1028–1038 (centre value against cell average) is the bridge, and P1 depends on P2.
  - If `sec:site` stays in P2, P1 is self-contained.

  Decide D5 together with this section.
- **D2.** The comparison tables (`tab:hiersub`, `tab:hiergraded`) are where the two relaxations meet. Whichever paper carries them should carry the paragraph above in full.
- **Renaming pass.** CL has about 45 uses of "collocat" and 38 of "mean-only"; EC 1028–1038 and AO 102–105 need the same treatment. Do this during the move (step 4 of §7).
- **Settings.** Rewrite each paper's Setting to open from the displacement equation and the strain-model table (§8.1).

## 9. Decision D7: the formulation adopted (stated 7 October)

**Decision (author):** adopt the T-matrix formulation with the most accurate cell-to-cell representation.

### 9.1 Which formulation that is

The evidence already in the papers identifies the **Legendre cell** (independent strain, Galerkin).

| Comparison | Legendre cell, p = 1 (36 unknowns) | Hierarchy, ℓ = 3 (60 unknowns) | Source |
|---|---|---|---|
| How the coupling is represented | The receiver is a volume. The block is the exact double integral over both cells, to round-off (about 1e-15) for touching cells. | The receiver is a point. The source cube is integrated exactly; the receiver's field is a Taylor expansion about its centre. | EC `sec:results`; CL `sec:gradvoxel` |
| Body with corners (cube cut into n³) | 4.8e-5 at n = 1, 6.0e-6 at n = 2 | 3.4e-4 and 7.0e-5: 7–12× worse, with more unknowns | CL `tab:hiersub` |
| Smooth body (graded sphere, shell 9 m) | Order 3.92 and 4.02 | Order 3.80 and 3.84. Error a fifth to a quarter smaller per grid, but larger per unknown. | CL `tab:hiergraded` |
| Error known in advance | Yes: projection error | No | CL `sec:hetero`; AO |

The hierarchy keeps three advantages, none of which is cell-to-cell accuracy:
- cheaper tables (4.5 s against 23.3 s at 14 cells);
- a self block in closed form on four constants;
- a single-site T-matrix of a *scatterer* that can be checked against Mie.

### 9.2 Consequences, to confirm

1. **P1's subject becomes the single-site T-matrix of the Legendre cell.** That is EC `sec:site` and `app:blocks`: Lippmann–Schwinger of one cell, T36, the cube-group reduction (15 even and 21 odd unknowns), and the static blocks on five constants.
   - **This resolves D5:** they move to P1.
   - The hierarchy material of CL §6 (about 13 pages) shrinks to a section or appendix of P1: the uniform-strain lineage, the Mie check, and the comparison of centre value against cell average (EC 1028–1038). How much of it to keep is open.
2. **P3's graded-sphere series study** (`tab:sphereseries`, `tab:sphereseriesquad`, 7 October) was computed with the **hierarchy** cell model, because `scripts/measure_graded_sphere_frequency_series.py` is built on it. For consistency with D7 it should be recomputed with Legendre cells.
   - EC `sec:sweep` already has frequency-independent coefficients for the Legendre cell.
   - The series solve (the analogue of EC `eq:seriessolve`) would need writing for it.
3. **§8 simplifies.**
   - The independent-strain Legendre cell is the series' formulation.
   - The uniform-strain closure is the original approximation it relaxes.
   - The derived-strain hierarchy is the comparison: what a Taylor closure about the centre gives, and why the series does not use it (corners; per-unknown accuracy; no error law).
4. **D2:** `tab:hiergraded` and `tab:hiersub` become the evidence for D7. They belong wherever D7 is justified, most naturally in P1 when it introduces the formulation, or in P3.
