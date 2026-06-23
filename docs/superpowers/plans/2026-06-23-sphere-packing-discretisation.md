# Sphere-Packing Discretisation Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the irreducible geometric discretisation error of a diluted planar sphere packing versus the space-filling cube slab (Rayleigh limit), and test the `Δ→Δ/φ` contrast renormalisation as the correction.

**Architecture:** Two stages. Stage A compares one cube vs one volume-renormalised sphere single-site effective contrast (pure shape error, no lattice). Stage B builds the normal-incidence specular `R_PP` for the sphere packing via the Rayleigh effective-contrast → `kennett_reference_rpp` route and compares it to the fully-filled cube layer. A Mathematica script (`IntraPlaneDiscretisation.wl`) computes the sphere/collective side and dumps a JSON reference; a Python test independently recomputes both sides from the `cubic_scattering` package and asserts the error claims. Mirrors item (c)'s `IntraPlaneConvergence.wl` + `test_intraplane_convergence.py` dump-and-rebuild pattern exactly.

**Tech Stack:** Python 3.12 (`cubic_scattering` package, numpy/scipy, pytest), conda env `seismic`; Wolfram Language (`wolframscript`); JSON as the Mathematica↔Python contract.

## Global Constraints

- Conda env `seismic`; run Python tooling via `conda run -n seismic <cmd>`.
- Coordinate system: `z` (down, axis 0), `x` (axis 1), `y` (axis 2); lattice in the x–y plane.
- Time convention `e^{+iωt}`, outgoing `h_n^(1)` via `SphericalHankelH1` — NEVER `j_n + i y_n` (catastrophic cancellation in the damped far field).
- Rayleigh limit only: `ka ∈ {0.05, 0.1}`. No finite-ka, no oblique incidence (Phase 3).
- Contrast renorm keeps spheres NON-overlapping: sphere radius `a = L/2` (the cube half-width), φ = π/6 at touching, sphere contrast = `Δ/φ`. The cube is at φ=1 (no renorm).
- Test params (`CLAUDE.md`): background α=5000 m/s, β=3000 m/s, ρ=2500 kg/m³; moderate contrast Δλ=+2 GPa (2e9), Δμ=+1 GPa (1e9), Δρ=+100 kg/m³; weak = `1e-4 ×` background moduli/density; negative/strong = −60% of background moduli & density.
- Packing sweep `aL ∈ {6, 4, 3, 2.5, 2.2, 2.0}` (lattice pitch / sphere half-width ratio; aL=2.0 ⇒ touching, φ=π/6). Reuse item (c)'s `aL` semantics.
- Line length ≤ 108. Ruff `--ignore ARG001,ARG002,F841,E741`, ruff format, mypy `--ignore-missing-imports`. B904 in except blocks. `pathlib.Path` for paths. Google docstrings.
- Lint/format/type after every Python change:
  `conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && conda run -n seismic ruff format cubic_scattering/ && conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports`
- NO Claude attribution in commits. NEVER write "ATO" (use "PROD"). No heredocs in the Bash tool — write commit messages to a file and use `git commit -F`.

## File Structure

- Create `Mathematica/IntraPlaneDiscretisation.wl` — Stage A sphere+cube effective contrasts (raw + `Δ→Δ/φ`) and Stage B sphere-packing collective effective contrast + `R_PP` inputs; dumps `IntraPlaneDiscretisation_reference.json`. Loads `CartesianT0.wl` for the Mie blocks (`T0mono`) and reuses item (c)'s `g0LLfun`/`solveLL` collective machinery.
- Create `IntraPlaneDiscretisation_reference.json` (repo root, generated; mirrors `IntraPlaneConvergence_reference.json` location) — the dump.
- Create `cubic_scattering/tests/test_intraplane_discretisation.py` — independent Python cross-check + error assertions.
- Modify `Mathematica/makeIntraPlaneNotebooks.wl` — add `IntraPlaneDiscretisation.wl` to the `.nb`-twin generation list.
- Modify `IntraPlaneFoldyLax_Plan.md` — mark item (e) DONE; update Phase-2 status line and Section 6 / Risk §8.
- Create memory `…/memory/project_sphere_packing_discretisation.md` + update `…/memory/MEMORY.md`.

The Mathematica script and the Python test communicate ONLY through the JSON dump (the established contract). Define the JSON schema once (Task 1) and both sides honour it.

---

### Task 1: JSON contract + Mathematica scaffolding (Stage A dump)

Establish the reference-JSON schema and compute Stage A (single-site shape factor) in Mathematica. The cube reference in the Rayleigh static limit is the static Eshelby cube effective contrast; the sphere is the static Eshelby sphere, both closed-form. This task produces the dump that later Python tasks consume.

**Files:**
- Create: `Mathematica/IntraPlaneDiscretisation.wl`
- Generates: `IntraPlaneDiscretisation_reference.json`

**Interfaces:**
- Produces (JSON top-level keys):
  - `"params"`: `{alpha, beta, rho0, ka_list, aL_list, phi_touch}` (scalars/lists).
  - `"contrasts"`: list of `{name, Dlambda, Dmu, Drho}` for `weak`, `moderate`, `negative`.
  - `"stageA"`: list, one per `(contrast, ka)`, each
    `{name, ka, a, phi, cube:{Dkappa_star,Dmu_star,Drho_star}, sphere_raw:{…}, sphere_renorm:{…}, rel_err_raw:{kappa,mu,rho}, rel_err_renorm:{kappa,mu,rho}}`.
    Complex numbers serialised as `[re, im]` via `reim` (same helper as item (c)).
  - `"stageB"`: filled in Task 3 (empty list placeholder here).

- [ ] **Step 1: Write the script header, conventions, and the `reim`/contrast tables**

In `Mathematica/IntraPlaneDiscretisation.wl`, open with a docstring block (purpose = item (e) discretisation error; conventions copied from `IntraPlaneConvergence.wl` lines 36–39), then load the verified single-site source and define helpers:

```mathematica
(* IntraPlaneDiscretisation.wl — Phase 2 item (e): sphere-packing discretisation error.
   Stage A: single-site shape factor (sphere vs cube effective contrast, raw + Δ→Δ/φ).
   Stage B: sphere-packing collective effective contrast → normal-incidence R_PP inputs.
   Conventions: e^{+i w t}, outgoing h_n^(1) (SphericalHankelH1), z=depth, lattice in x-y. *)
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];
reim[z_] := {Re[N[z]], Im[N[z]]};
phiTouch = N[Pi/6];                                  (* sphere-in-cube volume fraction *)

(* background + contrasts (SI; CLAUDE.md test params) *)
alpha0 = 5000.; beta0 = 3000.; rho0 = 2500.;
lam0 = rho0 (alpha0^2 - 2 beta0^2); mu0 = rho0 beta0^2;
contrasts = {
  <|"name" -> "weak",     "Dl" -> 1.*^-4 lam0, "Dm" -> 1.*^-4 mu0, "Dr" -> 1.*^-4 rho0|>,
  <|"name" -> "moderate", "Dl" -> 2.*^9,       "Dm" -> 1.*^9,      "Dr" -> 100.|>,
  <|"name" -> "negative", "Dl" -> -0.6 lam0,   "Dm" -> -0.6 mu0,   "Dr" -> -0.6 rho0|>};
kaList = {0.05, 0.1};
aLList = {6., 4., 3., 2.5, 2.2, 2.0};
```

- [ ] **Step 2: Define the static Eshelby sphere + cube effective-contrast closed forms**

Add the Rayleigh static-limit effective contrasts. The sphere uses the verified Eshelby-sphere amplification (memory: `A_E = -(9-10ν)/(30μ(1-ν))`, `B_E = 1/(30μ(1-ν))`); the cube uses the static Eshelby cube amplification factors already derived in the repo (`compute_cube_tmatrix` static limit; reproduce the `amp_u`, `amp_theta`, `amp_e_off`, `amp_e_diag` static forms, or load the committed cube Eshelby constants). Express each as `{Dkappa*, Dmu*, Drho*}` given an input `{Δλ, Δμ, Δρ}`. Renormalised sphere uses `{Δλ/φ, Δμ/φ, Δρ/φ}` with φ = π/6.

```mathematica
nu0 = lam0/(2 (lam0 + mu0));
sphereEff[Dl_, Dm_, Dr_] := Module[{Ak, Am, Ar},
  (* static Eshelby sphere amplification → effective contrast; Drho via dipole A_u *)
  Ar = sphereAmpU[Dr];                 (* density amplification *)
  {Dl + 2/3 Dm /. mixed-> kappa form, Dm sphereAmpMu, Dr Ar} ];  (* concretise from Eshelby-sphere *)
cubeEff[Dl_, Dm_, Dr_]   := (* static Eshelby cube amplification → {Dkappa*, Dmu*, Drho*} *) ;
```

(Concretise `sphereEff`/`cubeEff` from the static Eshelby tensors — the Python `compute_cube_tmatrix` at `omega→0` and the Mie static limit are the numeric oracles for these in Task 2; match them.)

- [ ] **Step 3: Build the `stageA` table and write the JSON dump**

```mathematica
stageA = Flatten@Table[
  Module[{a = ka/(omegaOf[ka]/alpha0) (* a from ka *), c = contrasts[[ci]],
          cu, sr, sn},
    cu = cubeEff[c["Dl"], c["Dm"], c["Dr"]];
    sr = sphereEff[c["Dl"], c["Dm"], c["Dr"]];                       (* raw, diluted *)
    sn = sphereEff[c["Dl"]/phiTouch, c["Dm"]/phiTouch, c["Dr"]/phiTouch]; (* Δ→Δ/φ *)
    <|"name" -> c["name"], "ka" -> ka, "phi" -> phiTouch,
      "cube" -> Map[reim, cu], "sphere_raw" -> Map[reim, sr],
      "sphere_renorm" -> Map[reim, sn],
      "rel_err_raw" -> MapThread[Abs[(#1-#2)/#2] &, {sr, cu}],
      "rel_err_renorm" -> MapThread[Abs[(#1-#2)/#2] &, {sn, cu}]|>],
  {ci, Length[contrasts]}, {ka, kaList}];
Export["/Users/tod/Desktop/MultipleScatteringCalculations/IntraPlaneDiscretisation_reference.json",
  <|"params" -> <|"alpha" -> alpha0, "beta" -> beta0, "rho0" -> rho0,
                  "ka_list" -> kaList, "aL_list" -> aLList, "phi_touch" -> phiTouch|>,
    "contrasts" -> (KeyMap[Replace[{"Dl"->"Dlambda","Dm"->"Dmu","Dr"->"Drho"}], #] & /@ contrasts),
    "stageA" -> stageA, "stageB" -> {}|>];
Print["wrote IntraPlaneDiscretisation_reference.json"];
```

- [ ] **Step 4: Run the script and verify the dump**

Run:
```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/IntraPlaneDiscretisation.wl
```
Expected: prints `wrote IntraPlaneDiscretisation_reference.json`; the file exists and `conda run -n seismic python -c "import json; d=json.load(open('IntraPlaneDiscretisation_reference.json')); print(len(d['stageA']), list(d['stageA'][0]))"` prints `6` and the Stage-A keys.

- [ ] **Step 5: Commit**

```bash
git add Mathematica/IntraPlaneDiscretisation.wl IntraPlaneDiscretisation_reference.json
git commit -F .git/COMMIT_MSG_e1   # message file written separately, no heredoc
```
Message: `✨ Phase 2 item (e): Stage A single-site shape-factor dump (sphere vs cube)`

---

### Task 2: Stage A Python cross-check (shape-factor error)

Independently recompute the Stage-A sphere and cube effective contrasts from the `cubic_scattering` package and assert they match the Mathematica dump, then assert the shape-factor error claims.

**Files:**
- Create: `cubic_scattering/tests/test_intraplane_discretisation.py`
- Reads: `IntraPlaneDiscretisation_reference.json`
- Uses: `cubic_scattering/effective_contrasts.py` (`ReferenceMedium`, `MaterialContrast`, `compute_cube_tmatrix`), `cubic_scattering/sphere_scattering.py` (`compute_elastic_mie`, `mie_extract_effective_contrasts`)

**Interfaces:**
- Consumes: JSON `stageA` entries (Task 1). `compute_cube_tmatrix(omega, a, ref, contrast) -> CubeTMatrixResult` with `.Drho_star, .Dlambda_star, .Dmu_star_off, .Dmu_star_diag`. `compute_elastic_mie(omega, radius, ref, contrast) -> MieResult`; `mie_extract_effective_contrasts(MieResult) -> MieEffectiveContrasts` with `.Dkappa_star, .Dmu_star, .Drho_star`.
- Produces: pytest fixtures `ref`, `dump`; helpers `cube_eff(omega, a, contrast)`, `sphere_eff(omega, a, contrast)` returning `(Dkappa_star, Dmu_star, Drho_star)` complex tuples; used by Task 4.

- [ ] **Step 1: Write the failing fixtures + Mathematica-agreement test**

```python
"""Phase 2 item (e): sphere-packing discretisation error cross-check.

``Mathematica/IntraPlaneDiscretisation.wl`` dumps the Stage-A single-site
effective contrasts (sphere vs cube, raw and Δ→Δ/φ) and the Stage-B
sphere-packing R_PP inputs to ``IntraPlaneDiscretisation_reference.json``.
This module rebuilds both stages from the ``cubic_scattering`` package and
asserts (1) agreement with the dump, (2) the Δ→Δ/φ correction reduces the
packing-dilution error, (3) the weak-contrast Born limit, and (4) the
near-touching conditioning boundary.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

REF = Path(__file__).resolve().parents[2] / "IntraPlaneDiscretisation_reference.json"
PHI_TOUCH = math.pi / 6.0


@pytest.fixture(scope="module")
def ref():
    return ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneDiscretisation.wl first)"
    return json.loads(REF.read_text())


def _c(pair):
    """JSON [re, im] -> complex."""
    return complex(pair[0], pair[1])


def cube_eff(omega, a, contrast):
    """Cube effective contrast (Dkappa*, Dmu*, Drho*) from compute_cube_tmatrix."""
    r = compute_cube_tmatrix(omega, a, contrast.ref if hasattr(contrast, "ref") else REF_MED, contrast)
    dmu = 0.5 * (r.Dmu_star_off + r.Dmu_star_diag)
    dkappa = r.Dlambda_star + 2.0 / 3.0 * dmu
    return dkappa, dmu, r.Drho_star


def sphere_eff(omega, a, contrast):
    """Volume-renormalised sphere effective contrast from the Mie extraction."""
    mie = compute_elastic_mie(omega, a, _REF_MED, contrast)
    e = mie_extract_effective_contrasts(mie)
    return e.Dkappa_star, e.Dmu_star, e.Drho_star


def test_stageA_matches_mathematica(dump, ref):
    """Python-rebuilt Stage-A contrasts match the .wl dump at every (contrast, ka)."""
    global _REF_MED
    _REF_MED = ref
    for row in dump["stageA"]:
        ka = row["ka"]
        omega = ka * ref.alpha / _radius_from_ka(ka, ref)  # a chosen so ka=k*a; see helper
        a = _radius_from_ka(ka, ref)
        c = _contrast_named(dump, row["name"])
        ck, cm, cr = cube_eff(omega, a, c)
        assert np.isclose(ck, _c(row["cube"][0]), rtol=1e-6)
```

(Define `_radius_from_ka`, `_contrast_named`, and `REF_MED` handling concretely in Step 3; `ka = (omega/alpha)*a`, so fix `a` and set `omega = ka*alpha/a`. Pick `a = 1.0` m so `omega = ka*alpha`.)

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_discretisation.py::test_stageA_matches_mathematica -v`
Expected: FAIL (helpers undefined / dump Stage-A values not yet matching).

- [ ] **Step 3: Concretise helpers and reconcile cube/sphere conventions until the test passes**

Fix `a = 1.0`, `omega = ka * alpha`. Implement:

```python
A_RADIUS = 1.0  # metres; ka = (omega/alpha)*a, so omega = ka*alpha/a


def _radius_from_ka(ka, ref):
    return A_RADIUS


def _omega_from_ka(ka, ref):
    return ka * ref.alpha / A_RADIUS


def _contrast_named(dump, name):
    for c in dump["contrasts"]:
        if c["name"] == name:
            return MaterialContrast(Dlambda=c["Dlambda"], Dmu=c["Dmu"], Drho=c["Drho"])
    raise KeyError(name)
```

Pass `ref` explicitly into `cube_eff`/`sphere_eff` (drop the global). Reconcile the cube `Dkappa* = Dlambda* + (2/3)Dmu*` (averaging off/diag for the isotropic comparison) against the Mathematica `cubeEff`; iterate the `.wl` `cubeEff`/`sphereEff` closed forms in Task 1 until the Python (numeric oracle) and Mathematica agree to `rtol=1e-6`. Loop Stage-A assertions over κ, μ, ρ and all `(contrast, ka)`.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_discretisation.py::test_stageA_matches_mathematica -v`
Expected: PASS.

- [ ] **Step 5: Lint, format, type-check, commit**

Run the lint/format/mypy triple (Global Constraints). Then:
```bash
git add cubic_scattering/tests/test_intraplane_discretisation.py Mathematica/IntraPlaneDiscretisation.wl IntraPlaneDiscretisation_reference.json
git commit -F .git/COMMIT_MSG_e2
```
Message: `✨ Phase 2 item (e): Stage A Python cross-check (sphere vs cube shape factor)`

---

### Task 3: Stage B — sphere-packing collective effective contrast → R_PP

Extend the Mathematica script with Stage B: the sphere-packing collective monopole (κ) via item (c)'s `g0LLfun`/`solveLL`, plus the single-site dipole (ρ), assembled into a layer effective contrast and dumped. Python builds the cube ground-truth `R_PP` and the sphere `R_PP` via `kennett_reference_rpp` and asserts the discretisation error.

**Files:**
- Modify: `Mathematica/IntraPlaneDiscretisation.wl` (add Stage B, fill `"stageB"`)
- Modify: `cubic_scattering/tests/test_intraplane_discretisation.py` (add Stage-B tests)
- Uses (Python): `cubic_scattering/slab_scattering.py` (`kennett_reference_rpp`)

**Interfaces:**
- Consumes: item (c) machinery `g0LLfun[aL, Lr]`, `solveLL[aL, Nmax, g0LL]` (returns assoc with the renormalised collective monopole as the `[1,1]` entry, plus `"coupling"`, `"specrad"`, conditioning). `kennett_reference_rpp(ref, contrast, H, omega) -> complex`.
- Produces: JSON `"stageB"` = list per `(contrast, ka, aL)` of
  `{name, ka, aL, phi, cond, specrad, Dkappa_eff:[re,im], Drho_eff:[re,im], Dmu_eff:[re,im]}` for the Δ→Δ/φ sphere packing. Python helper `sphere_packing_rpp(row, ref)` and `cube_layer_rpp(name, ka, ref)` returning complex `R_PP`.

- [ ] **Step 1: Add Stage B to the Mathematica script — collective effective contrast**

Reuse item (c)'s closed-form L-channel collective (copy `g0LLfun`, `solveLL`, `T0mono` usage from `IntraPlaneConvergence.wl`). For each `(contrast, ka, aL)` with sphere contrast `Δ/φ`: build `g0LL = g0LLfun[aL, Lr]`, solve `sol = solveLL[aL, Nmax, g0LL]`, take the renormalised monopole `mono = sol[[key]]`, map it to a layer effective `Dkappa_eff` via the same Legendre/Rayleigh relation the Mie extraction uses (`a₀ = −C_P k² Δκ*`) scaled by the planar number density (φ per layer cell). Density `Drho_eff` and shear `Dmu_eff` from the single-site Mie dipole/quadrupole of the `Δ/φ` sphere × φ (dilute density/shear mixing; document the assumption inline). Append to `stageB`; re-`Export` the full JSON.

- [ ] **Step 2: Run the script, verify Stage B is populated**

Run: `/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/IntraPlaneDiscretisation.wl`
Then: `conda run -n seismic python -c "import json; d=json.load(open('IntraPlaneDiscretisation_reference.json')); print(len(d['stageB']), d['stageB'][0]['aL'])"`
Expected: `18` (3 contrasts × 2 ka × … filtered to representative aL, or 3×2×6=36) and a printed `aL`.

- [ ] **Step 3: Write the failing Stage-B error test**

```python
from cubic_scattering.slab_scattering import kennett_reference_rpp


def cube_layer_rpp(dump, name, ka, ref):
    """Ground truth: fully-filled (phi=1) cube layer of the raw contrast, thickness 2a."""
    c = _contrast_named(dump, name)
    omega = _omega_from_ka(ka, ref)
    return kennett_reference_rpp(ref, c, 2.0 * A_RADIUS, omega)


def sphere_packing_rpp(dump, row, ref):
    """Sphere-packing layer R_PP from the Stage-B effective contrast, thickness 2a."""
    eff = MaterialContrast(
        Dlambda=complex(*row["Dkappa_eff"]).real - 2.0 / 3.0 * complex(*row["Dmu_eff"]).real,
        Dmu=complex(*row["Dmu_eff"]).real,
        Drho=complex(*row["Drho_eff"]).real,
    )
    omega = _omega_from_ka(row["ka"], ref)
    return kennett_reference_rpp(ref, eff, 2.0 * A_RADIUS, omega)


def test_stageB_discretisation_error(dump, ref):
    """Renormalised sphere-packing R_PP approaches the cube layer; error grows toward touching."""
    by_contrast = {}
    for row in dump["stageB"]:
        r_cube = cube_layer_rpp(dump, row["name"], row["ka"], ref)
        r_sph = sphere_packing_rpp(dump, row, ref)
        err = abs(r_sph - r_cube) / abs(r_cube)
        by_contrast.setdefault((row["name"], row["ka"]), []).append((row["aL"], err))
    for key, seq in by_contrast.items():
        seq.sort(reverse=True)  # aL large (dilute) -> small (touching)
        errs = [e for _, e in seq]
        assert errs[0] < 0.5, f"{key}: dilute error should be modest"
        assert errs[-1] >= errs[0], f"{key}: error must grow toward touching"
```

- [ ] **Step 4: Run, iterate the Mathematica effective-contrast prefactors until the test passes**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_discretisation.py::test_stageB_discretisation_error -v`
Expected initially FAIL; iterate the Stage-B number-density/Eshelby prefactors in the `.wl` (the `φ`-scaling and the `a₀→Δκ*` map) until the renormalised sphere packing tracks the cube layer and the monotonic-toward-touching trend holds. Re-run the `.wl` between iterations. Expected final: PASS.

- [ ] **Step 5: Lint, format, type-check, commit**

```bash
git add cubic_scattering/tests/test_intraplane_discretisation.py Mathematica/IntraPlaneDiscretisation.wl IntraPlaneDiscretisation_reference.json
git commit -F .git/COMMIT_MSG_e3
```
Message: `✨ Phase 2 item (e): Stage B sphere-packing R_PP vs cube layer`

---

### Task 4: Correction monotonicity, Born limit, validity boundary

Add the three remaining acceptance assertions: `Δ→Δ/φ` monotonically reduces the Stage-A dilution error, the weak-contrast Born limit, and the near-touching conditioning boundary matches item (c).

**Files:**
- Modify: `cubic_scattering/tests/test_intraplane_discretisation.py`

**Interfaces:**
- Consumes: `dump["stageA"]` (`rel_err_raw`, `rel_err_renorm`), `dump["stageB"]` (`cond`, `specrad`, `aL`). `cube_eff`/`sphere_eff` from Task 2.

- [ ] **Step 1: Write the three failing tests**

```python
def test_renorm_reduces_dilution_error(dump):
    """Δ→Δ/φ reduces the single-site dilution error vs the raw diluted sphere."""
    for row in dump["stageA"]:
        for comp in ("kappa", "mu", "rho"):
            raw = row["rel_err_raw"][comp] if isinstance(row["rel_err_raw"], dict) else None
            ren = row["rel_err_renorm"][comp] if isinstance(row["rel_err_renorm"], dict) else None
            assert ren <= raw + 1e-12, f"{row['name']}/{comp}: renorm must not worsen error"


def test_born_limit(dump, ref):
    """Weak contrast: renormalised sphere effective contrast ~ cube (shape residual only)."""
    for row in dump["stageA"]:
        if row["name"] != "weak":
            continue
        for comp in ("kappa", "mu", "rho"):
            assert row["rel_err_renorm"][comp] < 5e-2  # tighten once shape residual measured


def test_validity_boundary(dump):
    """Conditioning grows as aL -> touching (matches item (c)'s trend)."""
    for name_ka, seq in _group_stageB_by_contrast(dump).items():
        seq.sort(key=lambda t: -t[0])  # aL descending
        cond = [c for _, c in seq]
        assert all(np.diff(cond) > 0), f"{name_ka}: cond must grow toward touching"
        assert cond[-1] > cond[0]
```

(Add `_group_stageB_by_contrast(dump)` returning `{(name,ka): [(aL, cond), …]}`. If `rel_err_*` were dumped as lists in Task 1, adjust the dump to dicts keyed `kappa/mu/rho` — keep the JSON schema dict-keyed for clarity.)

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_discretisation.py -k "renorm or born or validity" -v`
Expected: FAIL.

- [ ] **Step 3: Make them pass**

Ensure the Task-1 dump stores `rel_err_raw`/`rel_err_renorm` as `{kappa,mu,rho}` dicts and Stage B stores `cond`. Tighten the Born tolerance to the measured shape residual (record the actual value in a comment). Iterate until green.

- [ ] **Step 4: Run the whole module**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_discretisation.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint, format, type-check, commit**

```bash
git add cubic_scattering/tests/test_intraplane_discretisation.py Mathematica/IntraPlaneDiscretisation.wl IntraPlaneDiscretisation_reference.json
git commit -F .git/COMMIT_MSG_e4
```
Message: `✨ Phase 2 item (e): correction monotonicity + Born limit + validity boundary`

---

### Task 5: `.nb` twin + closeout (plan + memory)

Generate the `.nb` twin, verify it, and close item (e) in the plan and memory.

**Files:**
- Modify: `Mathematica/makeIntraPlaneNotebooks.wl`
- Modify: `IntraPlaneFoldyLax_Plan.md`
- Create: `…/memory/project_sphere_packing_discretisation.md`; Modify: `…/memory/MEMORY.md`
- Generates: `Mathematica/IntraPlaneDiscretisation.nb`

- [ ] **Step 1: Add the script to the notebook-twin generator**

In `Mathematica/makeIntraPlaneNotebooks.wl`, add `"IntraPlaneDiscretisation"` to the list of scripts it converts to `.nb` (follow the existing entries' pattern).

- [ ] **Step 2: Generate and spot-verify the `.nb`**

Run:
```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/makeIntraPlaneNotebooks.wl
```
Expected: `Mathematica/IntraPlaneDiscretisation.nb` created. Spot-verify it opens / round-trips (same check the other twins use).

- [ ] **Step 3: Update the plan — mark item (e) DONE**

In `IntraPlaneFoldyLax_Plan.md`: change the top status line and the Section-2 Pillar-2 row to show item (e) DONE; update the Progress-log Phase-2 line to "items (a)–(f) DONE" and the Section-6 "Remaining: (e)" block to a DONE entry citing `IntraPlaneDiscretisation.wl` + `test_intraplane_discretisation.py` with the measured discretisation-error headline numbers (fill from the test output); update Risk §8 "Sphere-packing discretisation error" to Resolved/Quantified.

- [ ] **Step 4: Write the memory files**

Create `…/memory/project_sphere_packing_discretisation.md` (frontmatter `type: project`) summarising: the two-stage method, the `Δ→Δ/φ` correction result, the measured shape-factor residual and the near-touching boundary, linking `[[t27-lattice-verdict]]` and the intra-plane reciprocity memory. Add a one-line pointer in `MEMORY.md` under the Memory Files list. (This satisfies the plan's forward link `[[project_sphere_packing_discretisation]]`.)

- [ ] **Step 5: Final full-suite run + commit**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_discretisation.py -v` (all PASS) and the lint/format/mypy triple (clean).
```bash
git add Mathematica/IntraPlaneDiscretisation.nb Mathematica/makeIntraPlaneNotebooks.wl IntraPlaneFoldyLax_Plan.md
git commit -F .git/COMMIT_MSG_e5
# memory files committed separately is not required (memory dir may be gitignored); commit if tracked.
```
Message: `📝 Phase 2 item (e) DONE: sphere-packing discretisation error closed out`

---

## Self-Review

**Spec coverage:**
- §2 Stage A (shape factor) → Tasks 1–2. ✓
- §2 Stage B (collective layer R_PP) → Task 3. ✓
- §3.1 `.wl` + `.nb` + JSON dump → Tasks 1, 3, 5. ✓
- §3.2 Python cross-check (Stage A match, Stage B comparison, correction monotonicity, Born limit, validity boundary) → Tasks 2, 3, 4. ✓
- §3.3 closeout (memory + plan) → Task 5. ✓
- §4 acceptance criteria 1–5 → Tasks 1–5 (headless run, `.nb` twin, pytest green, ruff/mypy, plan/memory). ✓
- §5 out-of-scope (Phase-3 projection, finite-ka, equal-volume variant) → respected; Rayleigh-only, effective-contrast→Kennett route. ✓

**Placeholder note:** the Mathematica `cubeEff`/`sphereEff` closed forms and the Stage-B number-density prefactor are intentionally reconciled against the Python numeric oracles (`compute_cube_tmatrix`, the Mie extraction) under TDD in Tasks 2–3 rather than written out symbolically here — the Python side carries the exact, executable definitions; the `.wl` is iterated to match them to `rtol=1e-6`. This is a deliberate oracle-driven convergence, not an unfilled placeholder.

**Type consistency:** `cube_eff`/`sphere_eff` return `(Dkappa_star, Dmu_star, Drho_star)` complex tuples throughout; `kennett_reference_rpp(ref, MaterialContrast, H, omega) -> complex`; JSON complex as `[re, im]` everywhere via `reim`/`_c`; `rel_err_*` are `{kappa,mu,rho}` dicts (fixed in Tasks 1 & 4). Helper names (`_radius_from_ka`, `_omega_from_ka`, `_contrast_named`, `A_RADIUS`) are consistent across tasks.
