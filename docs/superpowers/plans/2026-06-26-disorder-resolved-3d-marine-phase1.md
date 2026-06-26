# Disorder-Resolved 3-D Marine Scattering — Implementation Plan (Phase 0 + Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new **self-contained** repo `~/Desktop/Marine3D` for the deterministic 3-D marine scattering solver — first by **cherry-picking only the required modules** from the three source repos (Phase 0), then standing up the reference marine model + shot gather (Phase 1).

**Architecture:** No cross-repo mixing. Marine3D imports nothing from the source repos. Phase 0 traces the minimal import closure from the entry points and copies only those modules, rewriting imports repo-local; the duplicate Kennett is resolved by **keeping `cubic_scattering`'s `kennett_layers.py`** and adapting the cherry-picked GMM to consume its `LayerStack`/`IsotropicLayer`/`FluidLayer`. Parity is proven by running each source module's own tests in the new repo before any extension.

**Tech Stack:** Python 3.12, NumPy, PyTorch, conda env `seismic`; pytest.

**Source repos (reference / extraction source only):**
- `~/Desktop/SeismicInversion/GlobalMatrix/` — GMM, block-Riccati $\mathbf{P}^z$, `interlayer_ms`, `layer_matrix`, `gmm_torch`, `config`.
- `~/Desktop/MultipleScatteringCalculations/cubic_scattering/` — `kennett_layers`, `ocean_bottom`, `seismic_survey`, `effective_contrasts`, `voigt_tmatrix`, `horizontal_greens`, `torch_gmres`.
- `~/Desktop/SeismicInversion/Kennett_Reflectivity/` — reference only; its Kennett is **dropped** in favour of `cubic_scattering`'s.

## Global Constraints

- **Self-contained:** Marine3D has zero imports from the source repos after Phase 0. Verify with `grep -rE "GlobalMatrix|Kennett_Reflectivity|cubic_scattering" marine3d/` → no hits.
- **Keep `cubic_scattering`'s Kennett** (`kennett_layers.py`); drop `Kennett_Reflectivity`'s. Adapt GMM to `LayerStack`/`IsotropicLayer`/`FluidLayer`.
- **Seismic units:** km/s, g/cm³, km, GPa. Time convention $e^{-i\omega t}$.
- **YAML single source of truth**; fail-fast with the 4-element diagnostic (what / where: abs path + dotted key / valid example / recovery).
- **B904** exception chaining; `ruff check --fix --ignore ARG001,ARG002,F841,E741`, `ruff format`, `mypy --ignore-missing-imports`; line length ≤ 108.
- **Tests** are local-only (`Marine3D/tests/` gitignored, mirror source), run via `conda run -n seismic`.
- **No Claude attribution** in commits; gitmoji conventional style.
- Run everything from the Marine3D repo root unless noted.

---

## Phase 0 — Bootstrap Marine3D and cherry-pick the required modules

### Task 0.1: Bootstrap the empty self-contained repo

**Files:**
- Create: `~/Desktop/Marine3D/pyproject.toml`, `~/Desktop/Marine3D/environment.yml`, `~/Desktop/Marine3D/.gitignore`, `~/Desktop/Marine3D/marine3d/__init__.py`, `~/Desktop/Marine3D/README.md`

**Interfaces:** Produces an importable empty package `marine3d` and a git repo.

- [ ] **Step 1: Create the repo skeleton**

```bash
mkdir -p ~/Desktop/Marine3D/marine3d ~/Desktop/Marine3D/tests
cd ~/Desktop/Marine3D
git init
printf '%s\n' "tests/" "__pycache__/" "*.pyc" ".pytest_cache/" "*.egg-info/" > .gitignore
printf '%s\n' '"""Marine3D — deterministic 3-D marine scattering solver."""' '__version__ = "0.0.1"' > marine3d/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml` and `environment.yml`**

```toml
# pyproject.toml
[project]
name = "marine3d"
version = "0.0.1"
requires-python = ">=3.12"
dependencies = ["numpy", "scipy", "torch", "pyyaml"]

[tool.ruff]
line-length = 108

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```yaml
# environment.yml
name: seismic
channels: [conda-forge]
dependencies: [python=3.12, numpy, scipy, pytorch, pyyaml, pytest, ruff, mypy]
```

- [ ] **Step 3: Verify the package imports**

Run: `cd ~/Desktop/Marine3D && conda run -n seismic python -c "import marine3d; print(marine3d.__version__)"`
Expected: prints `0.0.1`.

- [ ] **Step 4: Commit**

```bash
cd ~/Desktop/Marine3D
git add pyproject.toml environment.yml .gitignore marine3d/__init__.py README.md
git commit -m "🎉 bootstrap Marine3D self-contained repo"
```

### Task 0.2: Record the cherry-pick manifest (dependency closure)

**Files:**
- Create: `~/Desktop/Marine3D/PORT_MANIFEST.md`

**Interfaces:** Produces the authoritative list of source files to copy and the import rewrites, so later tasks copy a known-complete set.

- [ ] **Step 1: Trace the closure from each entry point**

Run (record output into the manifest):
```bash
# GMM vertical operator entry points
cd ~/Desktop/SeismicInversion/GlobalMatrix
grep -rEn "^(from|import) " global_matrix.py riccati_solver.py layer_matrix.py \
    layered_greens.py interlayer_ms.py gmm_torch.py config.py
# cubic_scattering entry points
cd ~/Desktop/MultipleScatteringCalculations
grep -rEn "^(from|import) " cubic_scattering/kennett_layers.py cubic_scattering/ocean_bottom.py \
    cubic_scattering/seismic_survey.py cubic_scattering/effective_contrasts.py \
    cubic_scattering/voigt_tmatrix.py cubic_scattering/horizontal_greens.py cubic_scattering/torch_gmres.py
```

- [ ] **Step 2: Write `PORT_MANIFEST.md`**

Record, as a checklist, the transitive closure (follow any intra-package import surfaced above until no new files appear). Seed set (extend per trace):
- **GMM cluster →** `marine3d/gmm/`: `global_matrix.py`, `riccati_solver.py`, `layer_matrix.py`, `layered_greens.py`, `interlayer_ms.py`, `gmm_torch.py`, `config.py` (+ closure). **Adapt:** replace `Kennett_Reflectivity.layer_model.LayerModel` usage with an adapter over `cubic_scattering` `LayerStack` (Task 0.4).
- **Kennett + marine cluster →** `marine3d/kennett/`: `kennett_layers.py`, `ocean_bottom.py`, `seismic_survey.py` (+ closure).
- **T-matrix + horizontal cluster →** `marine3d/tmatrix/`: `effective_contrasts.py`, `voigt_tmatrix.py`, `horizontal_greens.py`, `torch_gmres.py` (+ closure).
- **Drop:** `Kennett_Reflectivity` entirely (its Kennett replaced by `cubic_scattering/kennett_layers.py`).

- [ ] **Step 3: Commit**

```bash
cd ~/Desktop/Marine3D && git add PORT_MANIFEST.md
git commit -m "📝 PORT_MANIFEST: dependency closure for cherry-pick"
```

### Task 0.3: Cherry-pick the cubic_scattering clusters (Kennett/marine + T-matrix/horizontal)

**Files:**
- Create: `~/Desktop/Marine3D/marine3d/kennett/*.py`, `~/Desktop/Marine3D/marine3d/tmatrix/*.py` (the manifest's cubic_scattering closure)
- Create (local, gitignored): `~/Desktop/Marine3D/tests/` mirrored tests for the copied modules

**Interfaces:** Produces repo-local `marine3d.kennett.kennett_layers` (`LayerStack`, `IsotropicLayer`, `FluidLayer`, `kennett_layers`), `marine3d.kennett.ocean_bottom`, `marine3d.kennett.seismic_survey`, `marine3d.tmatrix.effective_contrasts`, `marine3d.tmatrix.voigt_tmatrix`, `marine3d.tmatrix.horizontal_greens`, `marine3d.tmatrix.torch_gmres`.

- [ ] **Step 1: Copy the files per the manifest**

```bash
cd ~/Desktop/Marine3D
mkdir -p marine3d/kennett marine3d/tmatrix
SRC=~/Desktop/MultipleScatteringCalculations/cubic_scattering
cp $SRC/kennett_layers.py $SRC/ocean_bottom.py $SRC/seismic_survey.py marine3d/kennett/
cp $SRC/effective_contrasts.py $SRC/voigt_tmatrix.py $SRC/horizontal_greens.py $SRC/torch_gmres.py marine3d/tmatrix/
# copy any additional closure files the manifest lists
touch marine3d/kennett/__init__.py marine3d/tmatrix/__init__.py
# bring their tests (local only)
cp $SRC/tests/test_*kennett*.py $SRC/tests/test_*ocean*.py $SRC/tests/test_*survey*.py tests/ 2>/dev/null || true
cp $SRC/tests/test_*contrast*.py $SRC/tests/test_*voigt*.py $SRC/tests/test_*horizontal*.py tests/ 2>/dev/null || true
```

- [ ] **Step 2: Rewrite imports repo-local**

In every copied file and test, rewrite `from cubic_scattering.X import …` / `from .X import …` to the new package paths (`from marine3d.kennett.X import …` or `from marine3d.tmatrix.X import …`). Verify no stragglers:

Run: `cd ~/Desktop/Marine3D && grep -rE "cubic_scattering|from \.\." marine3d/ tests/`
Expected: no hits (all imports repo-local).

- [ ] **Step 3: Run the relocated tests (parity)**

Run: `cd ~/Desktop/Marine3D && conda run -n seismic python -m pytest tests/ -v`
Expected: PASS — the copied physics behaves identically in isolation.

- [ ] **Step 4: Lint, type-check, commit (source only — tests are gitignored)**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic ruff check --fix --ignore ARG001,ARG002,F841,E741 marine3d/
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/kennett marine3d/tmatrix
git commit -m "✨ cherry-pick cubic_scattering Kennett/marine + T-matrix/horizontal clusters"
```

### Task 0.4: Cherry-pick the GMM cluster and adapt it to `cubic_scattering`'s `LayerStack`

**Files:**
- Create: `~/Desktop/Marine3D/marine3d/gmm/*.py` (the manifest's GlobalMatrix closure)
- Create: `~/Desktop/Marine3D/marine3d/gmm/layerstack_adapter.py`
- Create (local): `~/Desktop/Marine3D/tests/test_gmm_parity.py`, plus relocated `GlobalMatrix` tests

**Interfaces:**
- Consumes: `marine3d.kennett.kennett_layers.LayerStack` (Task 0.3); `marine3d.kennett.kennett_layers.kennett_layers(stack, p, omega) -> KennettResult` (the repo's Kennett reflectivity).
- Produces: `marine3d.gmm.gmm_reflectivity(stack: LayerStack, p: float, omega: np.ndarray, free_surface: bool = False, solver: str = "riccati") -> np.ndarray` — the GMM reflectivity, now consuming a `LayerStack`. The adapter `layerstack_to_gmm(stack) -> _GmmModel` supplies `alpha/beta/rho/thickness/Q_alpha/Q_beta` arrays + `complex_slowness_p/_s`, `complex_velocity_s` that the ported GMM expects.

- [ ] **Step 1: Copy the GMM files and write the failing parity test**

```bash
cd ~/Desktop/Marine3D && mkdir -p marine3d/gmm
SRC=~/Desktop/SeismicInversion/GlobalMatrix
cp $SRC/global_matrix.py $SRC/riccati_solver.py $SRC/layer_matrix.py \
   $SRC/layered_greens.py $SRC/interlayer_ms.py $SRC/gmm_torch.py $SRC/config.py marine3d/gmm/
touch marine3d/gmm/__init__.py
cp $SRC/test_gmm.py $SRC/test_riccati.py tests/ 2>/dev/null || true
```

```python
# tests/test_gmm_parity.py
import numpy as np
from marine3d.kennett.kennett_layers import IsotropicLayer, FluidLayer, LayerStack, kennett_layers
from marine3d.gmm import gmm_reflectivity

def test_gmm_equals_cubic_kennett():
    """Ported GMM, fed a cubic_scattering LayerStack, must equal that repo's
    own Kennett reflectivity to <1e-10."""
    stack = LayerStack([
        FluidLayer(alpha=1.5, rho=1.0, thickness=2.0),
        IsotropicLayer(alpha=3.0, beta=1.5, rho=2.6, thickness=1.0),
        IsotropicLayer(alpha=5.0, beta=3.0, rho=3.2, thickness=np.inf),
    ])
    omega = np.linspace(0.5, 25.0, 128)
    p = 0.12
    r_gmm = gmm_reflectivity(stack, p=p, omega=omega)
    r_ken = kennett_layers(stack, p=p, omega=omega).RPP
    rel = np.max(np.abs(r_gmm - r_ken)) / np.max(np.abs(r_ken))
    assert rel < 1e-10, f"GMM != cubic Kennett: rel={rel:.2e}"
```

(Adjust the `LayerStack`/`FluidLayer`/`IsotropicLayer` constructor calls and `KennettResult.RPP` accessor to match the relocated `kennett_layers.py` API — read it during Step 3.)

- [ ] **Step 2: Run parity test to verify it fails**

Run: `cd ~/Desktop/Marine3D && conda run -n seismic python -m pytest tests/test_gmm_parity.py -v`
Expected: FAIL — `gmm_reflectivity` not importable / still expects `Kennett_Reflectivity.LayerModel`.

- [ ] **Step 3: Rewrite GMM imports repo-local + write the adapter**

In the copied `marine3d/gmm/*.py`, replace every `from Kennett_Reflectivity.layer_model import LayerModel` (and `model_to_tensors`, etc.) with the adapter. Write `marine3d/gmm/layerstack_adapter.py` exposing a small `_GmmModel` dataclass built from a `LayerStack` providing exactly the attributes/methods the ported GMM reads (`alpha`, `beta`, `rho`, `thickness`, `Q_alpha`, `Q_beta`, `n_layers`, `complex_slowness_p()`, `complex_slowness_s()`, `complex_velocity_s()`), and a `layerstack_to_gmm(stack) -> _GmmModel`. Have `gmm_reflectivity(stack, ...)` call `layerstack_to_gmm(stack)` then the original GMM core.

Verify no source-repo imports remain:
Run: `cd ~/Desktop/Marine3D && grep -rE "Kennett_Reflectivity|GlobalMatrix" marine3d/`
Expected: no hits.

- [ ] **Step 4: Run parity + relocated GMM tests to verify they pass**

Run: `cd ~/Desktop/Marine3D && conda run -n seismic python -m pytest tests/test_gmm_parity.py tests/test_gmm.py tests/test_riccati.py -v`
Expected: PASS (parity <1e-10; relocated GMM/Riccati tests green).

- [ ] **Step 5: Lint, type-check, commit**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic ruff check --fix --ignore ARG001,ARG002,F841,E741 marine3d/gmm/
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/gmm
git commit -m "✨ cherry-pick GMM cluster; adapt to cubic_scattering LayerStack (parity <1e-10)"
```

### Task 0.5: Self-contained smoke test

**Files:**
- Create (local): `~/Desktop/Marine3D/tests/test_selfcontained.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_selfcontained.py
import subprocess, sys, pathlib

def test_no_source_repo_imports():
    root = pathlib.Path(__file__).resolve().parent.parent / "marine3d"
    out = subprocess.run(
        ["grep", "-rE", "GlobalMatrix|Kennett_Reflectivity|cubic_scattering", str(root)],
        capture_output=True, text=True,
    )
    assert out.stdout == "", f"source-repo imports leaked:\n{out.stdout}"

def test_end_to_end_imports():
    from marine3d.gmm import gmm_reflectivity            # noqa: F401
    from marine3d.kennett.kennett_layers import LayerStack  # noqa: F401
    from marine3d.tmatrix import horizontal_greens       # noqa: F401
```

- [ ] **Step 2: Run, then commit (test is gitignored; nothing to add unless source changed)**

Run: `cd ~/Desktop/Marine3D && conda run -n seismic python -m pytest tests/ -v`
Expected: PASS (full relocated suite + smoke). Phase 0 complete: Marine3D is self-contained and parity-verified.

---

## Phase 1 — Reference marine model + shot gather (heterogeneity off)

Builds on the repo-local APIs from Phase 0 (`marine3d.kennett.kennett_layers`, `marine3d.gmm.gmm_reflectivity`, `marine3d.kennett.seismic_survey`/`ocean_bottom`). Establishes the $\Delta\mathcal{C}_{eff}=0$ baseline and the transparent-crust property.

### Task 1.1: `build_marine_reference_model` + transparent-crust subdivision invariance

**Files:**
- Create: `~/Desktop/Marine3D/marine3d/earth_model.py`
- Test (local): `~/Desktop/Marine3D/tests/test_earth_model.py`

**Interfaces:**
- Consumes: `marine3d.kennett.kennett_layers.{LayerStack, IsotropicLayer, FluidLayer}`; `marine3d.gmm.gmm_reflectivity(stack, p, omega)`.
- Produces: `build_marine_reference_model(ocean: FluidLayer, crust: IsotropicLayer, crust_thickness: float, halfspace: IsotropicLayer, n_crust_planes: int) -> LayerStack` — ocean / `n_crust_planes` identical crust sub-layers (each `crust_thickness/n_crust_planes` thick) / half-space (`thickness=np.inf`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_earth_model.py
import numpy as np
from marine3d.kennett.kennett_layers import IsotropicLayer, FluidLayer
from marine3d.gmm import gmm_reflectivity
from marine3d.earth_model import build_marine_reference_model

OCEAN = FluidLayer(alpha=1.5, rho=1.0, thickness=2.0)
CRUST = IsotropicLayer(alpha=3.0, beta=1.5, rho=2.6, thickness=1.0)
HALF = IsotropicLayer(alpha=5.0, beta=3.0, rho=3.2, thickness=np.inf)

def test_crust_subdivision_is_transparent():
    omega = np.linspace(0.5, 25.0, 128)
    p = 0.12
    m1 = build_marine_reference_model(OCEAN, CRUST, 1.0, HALF, n_crust_planes=1)
    m8 = build_marine_reference_model(OCEAN, CRUST, 1.0, HALF, n_crust_planes=8)
    rel = np.max(np.abs(gmm_reflectivity(m8, p, omega) - gmm_reflectivity(m1, p, omega))) \
        / np.max(np.abs(gmm_reflectivity(m1, p, omega)))
    assert rel < 1e-10, f"crust subdivision not transparent: rel={rel:.2e}"
```

(Match the actual `kennett_layers` constructor signatures read in Phase 0.)

- [ ] **Step 2: Run to verify it fails** — `conda run -n seismic python -m pytest tests/test_earth_model.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# marine3d/earth_model.py
"""Reference layered marine model: ocean / homogeneous crust (N transparent
sub-layers) / stiff half-space."""
import copy
import numpy as np
from marine3d.kennett.kennett_layers import LayerStack, IsotropicLayer, FluidLayer

def build_marine_reference_model(ocean: FluidLayer, crust: IsotropicLayer,
                                 crust_thickness: float, halfspace: IsotropicLayer,
                                 n_crust_planes: int) -> LayerStack:
    if n_crust_planes < 1:
        msg = f"n_crust_planes must be >= 1, got {n_crust_planes}"
        raise ValueError(msg)
    dz = crust_thickness / n_crust_planes
    sub = []
    for _ in range(n_crust_planes):
        c = copy.deepcopy(crust)
        c.thickness = dz
        sub.append(c)
    return LayerStack([ocean, *sub, halfspace])
```

(Adapt the per-layer thickness mutation to the actual `IsotropicLayer` API — e.g. `IsotropicLayer(alpha=crust.alpha, beta=crust.beta, rho=crust.rho, thickness=dz)`.)

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic ruff check --fix --ignore ARG001,ARG002,F841,E741 marine3d/earth_model.py
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/earth_model.py
git commit -m "✨ marine3d: reference earth model with transparent-crust subdivision"
```

### Task 1.2: Reference shot gather (heterogeneity off)

**Files:**
- Create: `~/Desktop/Marine3D/marine3d/marine_survey_3d.py`
- Test (local): `~/Desktop/Marine3D/tests/test_marine_survey.py`

**Interfaces:**
- Consumes: `build_marine_reference_model` (1.1); the relocated `marine3d.kennett.seismic_survey` / `ocean_bottom` gather entry point (exact signature read in Phase 0 — record it in `PORT_MANIFEST.md`).
- Produces: `marine_reference_gather(stack: LayerStack, survey_cfg) -> ShotGatherResult` (thin wrapper over the relocated gather machinery), returning the gather array `(nr, nt)` and `time`/`offsets`.

- [ ] **Step 1: Write the failing test** — assert gather shape `(nr, nt)`, finite, nonzero; and **subdivision invariance** (the reference gather is identical for `n_crust_planes=1` vs `6` to <1e-8). (Use the relocated survey config types.)

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the thin wrapper calling the relocated `seismic_survey`/`ocean_bottom` gather with the reference `LayerStack`.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Lint, type-check, commit** (`✨ marine3d: reference shot gather (heterogeneity off)`).

### Task 1.3: YAML config + fail-fast loader

**Files:**
- Create: `~/Desktop/Marine3D/marine3d/config.py`, `~/Desktop/Marine3D/configs/marine_reference.yaml`
- Test (local): `~/Desktop/Marine3D/tests/test_config.py`

**Interfaces:** Produces `load_marine_reference(path: Path) -> tuple[LayerStack, dict]`; raises `ConfigError` with the 4-element diagnostic on missing/invalid keys.

- [ ] **Step 1: Write the failing test** — valid YAML builds the `LayerStack`; a missing key (`crust_thickness`) raises `ConfigError` whose message contains the key, the file path, the dotted key `model.crust_thickness`, a valid example, and a recovery step.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the YAML loader (single source of truth; no hardcoded defaults) with a `_require(d, key, dotted, path, example)` helper producing the 4-element diagnostic, building the model via `build_marine_reference_model`.

- [ ] **Step 4: Run full Phase-1 suite to verify it passes.**

- [ ] **Step 5: Lint, type-check, commit** (`✨ marine3d: YAML config + fail-fast loader`).

---

## Phase roadmap (subsequent plans — each its own spec→plan cycle)

- **Phase 2 — per-voxel $T_0$ screens + 2½-D coupling.** Wire `marine3d.tmatrix` 9×9 $(u,\varepsilon)$ $T_0$ into per-voxel $\Delta\mathcal{C}_{eff}$ screens at each crust plane; add the standard-splitting GMRES matvec with the cherry-picked $\mathbf{P}^z$ Riccati sweep and an x-only $\mathbf{P}^x$ (`interlayer_ms` + `horizontal_greens`); reproduce the thesis 2½-D limit and Born-as-one-iterate.
- **Phase 3 — full 3-D $\mathbf{P}^x(x,y)$ + multiple scattering.** Generalise $\mathbf{P}^x$ to the $(x,y)$ planar lattice (resolve xy-FFT vs 2-D directional sweep), full $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})$ matvec, GMRES convergence study, energy/reciprocity.
- **Phase 4 — GPU/cloud scaling + differentiable path.** Torch matvec end-to-end, GPU≡CPU parity, cloud scaling; preserve `gmm_torch` for future inversion.

---

## Self-Review

**1. Spec coverage:** Spec §4 (new self-contained repo, cherry-pick, keep cubic Kennett, adapter) → Tasks 0.1–0.5. §3 marine model + transparent crust → Task 1.1 (+ 1.3 config). §6 shot gather → Task 1.2. §7 validation step 1 (heterogeneity-off reference + subdivision invariance) → 1.1/1.2. §8 YAML/fail-fast → 1.3. Later spec items (screens, $\mathbf{P}^x/\mathbf{P}^z$ matvec, 3-D, GPU) → Phase roadmap.

**2. Placeholder scan:** Port tasks (0.3/0.4) are relocation+rewire+verify with exact `cp`/`grep`/`pytest` commands; the few "adapt to the actual API read in Phase 0" notes are explicit contingencies on a prior task's deliverable (`PORT_MANIFEST.md` / the relocated `kennett_layers`), not unfilled requirements. Phase 1 Tasks 1.2/1.3 give full interfaces + step intents with concrete commands; 1.1/0.4 show full code.

**3. Type consistency:** `build_marine_reference_model(ocean, crust, crust_thickness, halfspace, n_crust_planes) -> LayerStack` is used identically in 1.1/1.2/1.3. `gmm_reflectivity(stack, p, omega, free_surface, solver) -> np.ndarray` consistent in 0.4/1.1. `load_marine_reference(path) -> (LayerStack, dict)` matches its test.
