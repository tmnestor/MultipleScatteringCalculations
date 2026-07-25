# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Elastic multiple scattering from cubic heterogeneities — T-matrix approach for seismic waves. Originates from Nestor (1996) PhD thesis (ANU). Three pillars: single-site T-matrix `T₀`, inter-site Green's tensor `G₀`, and Foldy-Lax multiple-scattering solve `T = T₀(I − G₀T₀)⁻¹`.

## Research Methodology (READ FIRST)

**The deliverable is the LaTeX, not the code.** The principal product of this research is the set of `.tex` documents (`docs/`, `LatexPDFs/`) stating the physics and its mathematical formulation. Python and Mathematica exist to **validate** that formulation — they are the test harness, not the product. Every formulation anchors to a specific thesis §/equation.

### Formulations are developed test-first

TDD applies to the mathematics, not just to the software:

1. State the claim as a check with a knowable answer — a limit (Born, Rayleigh, static), a symmetry (O_h irrep, reciprocity), a conservation law (energy balance, optical theorem), or an independent derivation.
2. Run it and **watch it fail**.
3. Derive/implement until it passes in **both** Wolfram Mathematica (symbolic, high-precision) *and* Python (numerical), each cross-checking the other.
4. **Only then** write the result into the `.tex`.

Two independent implementations agreeing is the evidence standard. One implementation agreeing with itself is not. A formula that appears only in a `.tex` with no Mathematica or Python check behind it is **unvalidated** — say so explicitly rather than presenting it as established.

### ⚠ The `.tex` files lag the development

This is the standing failure mode of this repo. **When a document and a validated script disagree, the validated script wins** — assume the `.tex` is stale, not the code. Consequences:

- Never quote a `.tex` equation as ground truth without re-running the check that produced it.
- Read the Mathematica `.wl` and Python source **before** proposing work based on a document.
- When a validation changes a result, updating the affected `.tex` is **part of the same task**, not a follow-up. Leaving the document behind is how the drift accumulates.
- Auto-generated fragments (`cube_galerkin27_results.tex`, `cube_galerkin27_closedforms.tex`) are regenerated from their `.wl` scripts — never hand-edited.
- If you cannot update the `.tex` in the same pass, state plainly which document is now out of date and in what respect.

### Division of labour

| Layer | Role |
|-------|------|
| Wolfram Mathematica (`Mathematica/*.wl`) | Symbolic derivation, exact/high-precision closed forms, auto-generated LaTeX fragments |
| Python (`cubic_scattering/`, `scripts/`) | Numerical implementation, regression tests, independent cross-check of the symbolic result |
| LaTeX (`docs/`, `LatexPDFs/`) | **The product** — physics narrative plus the validated formulation |

## Environment & Commands

```bash
# Conda environment
conda activate seismic          # or: conda run -n seismic <cmd>
conda env create -f envs/seismic.yml

# Tests (all)
conda run -n seismic python -m pytest cubic_scattering/tests/ -v

# Single test
conda run -n seismic python -m pytest cubic_scattering/tests/test_cubic_tmatrix.py::test_born_limit -v

# Lint & format (run after every Python file change)
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports

# Other test suites
conda run -n seismic pytest FFTProp.py/test_package.py -v
conda run -n seismic pytest PhD_fortran_code/Kennett_Reflectivity/test_package.py -v

# LaTeX (MUST use lualatex — fontspec requires it, pdflatex will fail)
# Compile in-place in docs/ directory; run twice for cross-references
cd docs && /usr/local/bin/lualatex -interaction=nonstopmode cube_galerkin27.tex

# Mathematica scripts
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/<script>.wl
```

## Architecture

### Coordinate System (CRITICAL)

z = axis 0 (down), x = axis 1 (right), y = axis 2 (out) — right-handed. All Green's tensor formulas, Voigt conventions, and Fourier transforms assume this ordering.

### Module Dependency Flow

```
effective_contrasts.py  →  voigt_tmatrix.py  →  kennett_layers.py  →  seismic_survey.py
     (T₀ Rayleigh)          (6×6 Voigt)         (layered reflectivity)   (shot gathers)

resonance_tmatrix.py    →  sphere_scattering.py
     (T₀ full-wave)         (Mie validation)

lattice_greens.py + horizontal_greens.py  →  slab_scattering.py / slab_scattering_gpu.py
     (G₀ inter-site)                           (Foldy-Lax solver)

cpa_iteration.py        (self-consistent effective medium via ⟨T⟩ = 0)
cube_eshelby.py         (static Eshelby tensor — MUST include delta-function correction at r=0)
```

### T-matrix Regimes

| ka range | Module | Notes |
|----------|--------|-------|
| ka < 0.3 | `effective_contrasts.py` | Analytical, fast. 27×27 Galerkin closure with O_h irrep decomposition |
| 0.3–1.0 | `resonance_tmatrix.py` | Subdivide cube into n³ Rayleigh sub-cells, internal Foldy-Lax |
| ka ≥ 1.0 | `resonance_tmatrix.py` | Full resonance, n ≥ 4 |

### 27-Component Galerkin T-matrix (`effective_contrasts.py`)

The T₂₇ uses a basis of 27 trial functions (3 displacement + 6 strain + 18 quadratic) and decomposes under O_h symmetry into 7 irreducible representations:

- **Ungerade (21D)**: 4×T₁ᵤ (12D) + 2×T₂ᵤ (6D) + A₂ᵤ (1D) + Eᵤ (2D)
- **Gerade (6D)**: A₁g (1D) + Eg (2D) + T₂g (3D)

The Lippmann-Schwinger bilinear forms decompose into 4 channels: `a₀·Δλ`, `a₀·Δμ`, `b₀·Δλ`, `b₀·Δμ` where `a₀`, `b₀` are isotropic and cubic parts of the Eshelby decomposition.

All body and surface integrals reduce analytically to 3D master integrals:
- A-channel: `M⁺[p,q,r] = ∫₀¹ ∫₀¹ ∫₀¹ xᵖyᵍzʳ/√(x²+y²+z²) dx dy dz`
- B-channel: `M^{B+}[p,q,r]` with 1/ρ³ kernel

### Key Dataclasses

- `ReferenceMedium(alpha, beta, rho)` — isotropic background
- `MaterialContrast(Dlambda, Dmu, Drho)` — perturbation from reference
- `CubeTMatrixResult` — full T-matrix output with all intermediates

### Mathematica (`Mathematica/`)

Symbolic derivation and validation. `.wl` scripts output auto-generated LaTeX fragments included by `docs/*.tex`. Key scripts: `CubeGalerkin27.wl` (body bilinear), `CubeT27Stiffness_LS.wl` (surface stiffness), `CubeT6Block.wl` (quad-quad block).

### LaTeX Documentation (`docs/`)

- `cube_galerkin27.tex` — main document (inputs the others via `\input{}`)
- `cube_tmatrix_closedform.tex` — T-matrix physics and closed-form derivations
- `cube_galerkin27_results.tex` — master integral tables (auto-generated by Mathematica)
- `cube_galerkin27_closedforms.tex` — high-precision scalar values

TinyTeX distribution. Install missing packages: `tlmgr install <pkg>`.

## Known Pitfalls

- **Eshelby delta function**: The Green's tensor ∂²G/∂xₖ∂xₗ integral has a delta at r=0 from the 1/r singularity. Numerical quadrature misses this, causing wrong-sign amplification factors. The analytical Eshelby correction is essential.
- **NIntegrate precision**: Mathematica's default `MachinePrecision` gives only 6-8 digits on singular integrands. Use `PrecisionGoal → 12, AccuracyGoal → 12, MaxRecursion → 25` for reliable results.
- **Mp canonicalization**: Master integrals must sort indices: `Mp[p,q,r] → Mp@@Sort[{p,q,r}]`. Without this, symbolic simplification fails silently.

## Test Parameters (validated reference values)

- **Background**: α=5 km/s, β=3 km/s, ρ=2.5 g/cm³
- **Moderate contrast**: Δλ=+2 GPa, Δμ=+1 GPa, Δρ=+0.1 g/cm³
- **ka targets**: 0.05 (Rayleigh), 0.1, 0.3 (transition), 1.5 (resonance)
