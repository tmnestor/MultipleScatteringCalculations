# Disorder-Resolved 3-D Marine Scattering — Design

- **Date:** 2026-06-26
- **Status:** Draft for review
- **Author:** Tod Nestor (with Claude Code)
- **Topic:** A full 3-D, deterministic forward solver for surface shot gathers over a marine ocean / heterogeneous-crust / half-space model, built as the 3-D extension of Nestor (1996) thesis Chapter 5.

---

## 1. Objective

Forward-model **one specific, realistic, deterministic Earth** and produce **surface marine shot gathers**. This is *not* a statistical/ensemble random-medium theory (no von Kármán autocorrelation, no ensemble averaging, no radiative transfer). The dominant material variation is along depth; each depth plane is laterally inhomogeneous.

The thesis (Nestor 1996) is a **2½-D** formulation (heterogeneity in $(z,x)$, invariant in $y$, field synthesised over $k_y$). This work is the **full 3-D extension** (heterogeneity in $(x,y,z)$; $k_y$ is no longer a passive parameter), built to scale for eventual **cloud (GPU) execution**.

This is "direction (B)": solve the deterministic, laterally-inhomogeneous, depth-layered medium **directly** by a Krylov (GMRES) iteration whose matrix–vector product is a pair of forward sweeps — never a per-iterate matrix inversion.

## 2. Anchor: thesis Chapter 5

The build is the 3-D realisation of thesis Chapter 5 ("Multiple scattering formulations for a stratified half-space with superimposed heterogeneity", `GstratRep.tex`), which generalises Chin, Hedstrom & Thigpen (1984) from interface *displacement–traction* balances to volumetric **wavefield balances**.

Master result (thesis eq. `LSstrat`):
$$\mathcal{G} = \mathcal{G}_{ref} + \mathbf{P}^{z,R}_s\,\Delta\mathcal{C}_{eff}\,\big(\mathbf{I} - [\mathbf{P}^z_{r,s}+\mathbf{P}^x_{r,s}]\,\Delta\mathcal{C}_{eff}\big)^{-1}\,\mathcal{G}^z_r .$$

The quantity solved iteratively is the matrix Lippmann–Schwinger equation $\mathcal{V}_{int} = \mathcal{V}_{inc} + \mathbf{Q}^\partial\,\Delta\mathcal{S}\,\mathcal{V}_{int}$, with $\mathbf{Q}^\partial=(\mathbf{I}-\mathcal{S}_{int}\mathbf{E})^{-1}\mathcal{S}_{int}$ the stratified propagator. The **GMRES matvec** is

$$\big(\mathbf{I} - [\mathbf{P}^z_{r,s}+\mathbf{P}^x_{r,s}]\,\Delta\mathcal{C}_{eff}\big)\,\hat{\mathcal{B}},$$

with three pieces:

| Symbol | Meaning | Our realisation |
|---|---|---|
| $\Delta\mathcal{C}_{eff}$ | block-diagonal per-voxel scattering screens | `cubic_scattering` 9×9 $(u,\varepsilon)$ effective-contrast $T_0$ |
| $\mathbf{P}^x_{r,s}$ | **horizontal** intra-plane coupling (thesis Alg 5.2: right/left sweep + horizontal phase $\mathbf{E}_x$, no inversion) | spectral sweep / `horizontal_greens.py`, **extended to $(x,y)$** |
| $\mathbf{P}^z_{r,s}$ | **vertical** inter-plane up/down sweep (thesis Alg 5.3: project to modal up/down P-S-H, Riccati up-sweep + down-sweep) | `GlobalMatrix` block-Riccati sweep, **extended to lateral coupling** |

**Key efficiency property (the reason this is not a Kennett recursion per iterate):** $\mathbf{Q}^\partial$ depends only on the *reference* layered medium, so the Riccati net-reflection operators are built **once**; each GMRES iterate is then only the cheap up-sweep/down-sweep (forward/back substitution) plus the local $\Delta\mathcal{C}_{eff}$ apply. GMRES resums the full multiple-scattering (lateral and inter-plane) series. (Born ≡ one Jacobi iterate; Wu 1994 complex screen ≡ one Gauss–Seidel iterate; full ≡ GMRES to convergence.)

The thesis offers two splittings, both matvec-ready and both in scope to support:
- **Standard** (recursive Riccati, Alg 5.1) — for weak heterogeneity on a possibly rough reference.
- **Forward-scattering "two-way marching"** $\mathbf{P}_{\downarrow\downarrow}/\mathbf{P}_{\uparrow\uparrow}$ (§`fscat`, inversion-free) — for smooth reference media; generalises Wu (1994).

## 3. Physical model

A marine 3-region stack, depth-dominant:

1. **Ocean** — fluid top layer ($\beta=0$, water). Free surface above (pressure-release $\approx-1$). Fluid–solid interface at the seabed (`psv_fluid_solid`, with P→SV→P conversion).
2. **Ocean crust (heterogeneous)** — discretised into $N_z$ depth planes, each an $N_x\times N_y$ grid of voxels. **The crust reference is a single homogeneous block (the crust mean).** The internal crust sub-layer interfaces are **transparent**: reflection $S^i_{\uparrow\downarrow}=S^i_{\downarrow\uparrow}=\mathbf{0}$, transmission $S^i_{\uparrow\uparrow}=S^i_{\downarrow\downarrow}=\mathbf{I}$, carrying only the propagation phase $\mathbf{E}_i$. The $z$-subdivision is a numerical grid, not physical layering. **All** crust structure — lateral *and* vertical — is carried by the per-voxel $T_0$ screens $\Delta\mathcal{C}_{eff}$ (contrast of $(\Delta\lambda,\Delta\mu,\Delta\rho)$ relative to the single crust mean), resolved by GMRES. This is the thesis "uniform host" subclass: the background supports no reverberation inside the crust.
3. **Half-space** — uniform, **stiffer and denser** than the crust. Bottom termination $S^{[L,\infty)}$.

**Only three reference-reflecting interfaces:** free surface, seabed (fluid–solid), crust-base → half-space.

Contrasts are referenced to the crust mean so they stay small — keeping GMRES well-conditioned and respecting the renormalisation validity floor $|\Delta|<\varphi\cdot\text{background}$ established in the sphere-packing study.

## 4. Architecture — port/extend `GlobalMatrix`

The vertical operator and Foldy–Lax/ocean machinery already exist in `~/Desktop/SeismicInversion/GlobalMatrix/` (shares conda env `seismic`). The new 3-D solver **lives in `GlobalMatrix`** and imports `cubic_scattering` for $T_0$ and the horizontal sweep.

| Piece | Source | Action |
|---|---|---|
| $\mathbf{P}^z$ up/down Riccati sweep (numpy + differentiable torch) | `GlobalMatrix/riccati_solver.py` | reuse |
| modal eigenvectors (P-SV-SH, ocean acoustic), batched over $k_H$ | `GlobalMatrix/layer_matrix.py` | reuse |
| differentiable global-matrix solve, Foldy–Lax, ocean extraction | `gmm_torch.py`, `interlayer_ms.py` (`ScattererSlab9x9` already uses 9×9 $(u,\varepsilon)$ T-matrices) | reuse |
| $\Delta\mathcal{C}_{eff}$ screens (9×9 $(u,\varepsilon)$ $T_0$) | `cubic_scattering` effective contrasts | port in |
| $\mathbf{P}^x$ intra-plane $(x,y)$ horizontal coupling | `horizontal_greens.py` / spectral sweep | port + extend to $(x,y)$ |
| 3-D matvec $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})$ + GMRES driver | `torch_gmres` | new (thin) |
| earth-model builder; marine shot-gather driver | `ocean_bottom.py`, `seismic_survey.py` | reuse + new builder |

New focused modules (each one clear purpose, testable in isolation):
- `earth_model.py` — build the 3-region marine model: ocean (fluid), crust (homogeneous reference + per-voxel $T_0$ screens over $N_z\times N_x\times N_y$, transparent internal interfaces), stiff half-space; emit reference interface operators (only the 3 real ones non-trivial) and the screen array.
- `interlayer_ms_3d.py` — extend `ScattererSlab9x9` from interface scatterers to laterally-heterogeneous planes; add the intra-plane $\mathbf{P}^x$ coupling (currently "interlayer-only" / side-scattering neglected → add side scattering).
- `matvec_3d.py` — assemble $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})$; the GMRES driver wrapping `torch_gmres`.
- `marine_survey_3d.py` — loop over frequency and source conditions; assemble $\mathcal{G}$ (eq. `LSstrat`) at receivers; hand to `ocean_bottom`/`seismic_survey` for ghosts, wavelet, IFFT → shot gather.

## 5. The 3-D extension (what actually changes vs the thesis)

The extension is localised to **two** operators; everything else transfers unchanged:
- $\mathbf{P}^x$: from x-only sideways sweeps (Alg 5.2) to an $(x,y)$ horizontal propagator — drop the $k_y$ passive-parameter assumption. Realised by the 3-D horizontal Green's tensor (`horizontal_greens.py` is already 3-D) / the intra-plane spectral sweep generalised to the $(x,y)$ planar lattice (xy-FFT or 2-D directional sweep).
- $\mathbf{P}^z$ screens: carry $(x,y)$ lateral coupling at each plane (the per-voxel grid), not a single scatterer per interface.

The $\mathbf{Q}^\partial$ vertical machinery (modal up/down P-S-H, Riccati sweep), the ocean/half-space terminations, GMRES, and the marine dressing are unchanged.

## 6. Acquisition & observable

Marine towed-streamer survey (reuse `SurveyConfig` / `seismic_survey.py`): P source in the water column near the surface, hydrophone/geophone receivers at streamer depth, offsets, Ricker wavelet, **source + receiver ghosts**, free-surface (water-column) multiples. Source/receiver layout is in $(x,y)$ for the 3-D case.

Per frequency $\omega$ and source: solve the matvec system → receiver-layer response (eq. `LSstrat`) → $R(\omega,\text{offset})$ → IFFT → shot gather $(n_r, n_t)$.

## 7. Validation — reduce-to-known-limits ladder

1. **Heterogeneity off** ($\Delta\mathcal{C}_{eff}=\mathbf{0}$) → exact pure 3-interface marine reflectivity (`ocean_bottom.py`); no internal crust reflections (transparent interfaces).
2. **2½-D limit** ($y$-invariant heterogeneity) → matches the thesis 2½-D result / existing slab solver.
3. **Born** = one GMRES iterate → matches single-scatter.
4. **Weak contrast** → linearised agreement; reproduce the thesis Ch 6 convergence (≈8 iterations at 5% heterogeneity, ≈28 at 10%) as a target.
5. **Energy & reciprocity** — flux conservation (undamped), source–receiver reciprocity.
6. **GMM ≡ Kennett** (`<1e-12`, existing `GlobalMatrix` tests) preserved; **GPU ≡ CPU** parity; cloud scaling.

Tests are local-only (`tests/` gitignored), `pytest`, ≥80% coverage, run in conda env `seismic`.

## 8. Configuration & fail-fast

- **YAML is the single source of truth.** All model/acquisition/solver parameters live in YAML; Python reads YAML and never shadows with hardcoded defaults. Extend `GlobalMatrix/config.py` + `configs/default_ocean_crust.yaml`.
- **Fail-fast at startup** with the 4-element diagnostic on every config error: what is wrong, where to fix it (absolute path + dotted key), what a valid value looks like, how to recover.
- `B904` exception chaining; `ruff` + `ruff format` + `mypy`; line length ≤ 108.

## 9. Non-goals / out of scope

- No statistical/ensemble random-medium modelling, no radiative transfer.
- No inversion (the differentiable torch path is preserved for *future* inversion but FWI is not in scope here).
- No anisotropic background; the reference layers are isotropic (heterogeneity may induce effective anisotropy via $T_0$, which is fine).
- No topographic/irregular interfaces; the three real interfaces are planar.

## 10. Risks & open questions

- **Intra-plane $\mathbf{P}^x$ in 3-D:** choice between xy-FFT convolution vs 2-D directional spectral sweep for the planar coupling — to be settled in the implementation plan against cost/accuracy and GPU friendliness.
- **GMRES conditioning** at larger crust contrasts; the crust-mean referencing keeps $|\Delta|$ small but the floor $|\Delta|<\varphi\cdot\text{background}$ must be enforced/validated.
- **Cross-repo packaging:** `GlobalMatrix` importing `cubic_scattering` — ensure both are importable in env `seismic` (path/packaging).
- **Splitting choice** (standard Riccati vs forward-scattering marching) per regime — both supported; default to standard, expose forward-scattering for smooth-reference cases.

## 11. References

- Nestor (1996), PhD thesis, ANU — **Chapter 5** (`GstratRep.tex`, multiple-scattering formulations; eqs. `LSstrat`, `LSmat`, Algs 5.1–5.3) and **Chapter 6** (`VariationalSum.tex`, GMRES iterative solution).
- Chin, R. C. Y., Hedstrom, G. W. & Thigpen, L. (1984). Matrix methods in synthetic seismograms. *Geophys. J. R. astr. Soc.* 77, 483–502.
- Schmidt, H. & Tango, G. (1986). Efficient global matrix approach to the computation of synthetic seismograms. *Geophys. J. R. astr. Soc.* 84, 331.
- Wu, R.-S. (1994). Elastic wave complex-screen method.
- Companion notes: `EwaldIntraPlanePropagator` (I), `ContrastSourceVIE` (II), `DirectionalSweepSolver` (III), `IntraPlaneSpectralSweep`.
- Existing code: `~/Desktop/SeismicInversion/GlobalMatrix/` (GMM, block-Riccati, interlayer MS, torch); `cubic_scattering/` ($T_0$, `horizontal_greens.py`, `ocean_bottom.py`, `seismic_survey.py`, `kennett_layers.py`, `slab_scattering_gpu.py`).
