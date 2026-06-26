# Disorder-Resolved 3-D Marine Scattering — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `marine3d` package in `GlobalMatrix` and produce an end-to-end **reference** marine shot gather (heterogeneity OFF) for an ocean / homogeneous-crust / stiff-half-space model, with the transparent-internal-crust-interface property validated.

**Architecture:** Build the layered reference as a `LayerModel` in which the homogeneous crust is replicated into `N_z` identical sub-layers (transparent internal interfaces: identical adjacent media ⇒ reflection 0, transmission I). Compute reflectivity via the existing `GlobalMatrix.gmm_reflectivity` (block-Riccati sweep) and gathers via the existing `Kennett_Reflectivity.compute_gather`. This is the thinnest end-to-end tracer bullet; later phases add the per-voxel $T_0$ screens and the horizontal/vertical scattering operators.

**Tech Stack:** Python 3.12, NumPy, conda env `seismic`; `GlobalMatrix` (GMM, block-Riccati), `Kennett_Reflectivity` (`LayerModel`, `compute_gather`), pytest.

## Global Constraints

(Copied from the spec; every task implicitly includes these.)

- **YAML is the single source of truth.** Python reads YAML; no hardcoded config defaults shadowing YAML. Missing key ⇒ fail fast.
- **Fail-fast with the 4-element diagnostic** on every config error: what is wrong; where to fix it (absolute path + dotted key); a concrete valid YAML example / allowed values; one-line recovery step.
- **Seismic units throughout:** velocities km/s, density g/cm³, thickness km, moduli GPa. Time convention $e^{-i\omega t}$.
- **B904** exception chaining (`raise ... from err` / `from None`) in every except block.
- **Lint/type:** `ruff check --fix --ignore ARG001,ARG002,F841,E741`, `ruff format`, `mypy --ignore-missing-imports`. Line length ≤ 108.
- **Env:** run all Python via `conda run -n seismic`.
- **Home repo:** code lands in `~/Desktop/SeismicInversion/GlobalMatrix/` (the `marine3d` subpackage). Tests are committed alongside source as `test_*.py` (existing `GlobalMatrix` convention — it uses no `tests/` directory).
- **No Claude attribution in commits.** Gitmoji conventional-commit style.
- **Working directory for tests:** `~/Desktop/SeismicInversion/` (so `GlobalMatrix` and `Kennett_Reflectivity` are importable). Confirm with `conda run -n seismic python -c "import GlobalMatrix, Kennett_Reflectivity"`.

---

### Task 1: `build_marine_reference_model` + transparent-crust subdivision invariance

**Files:**
- Create: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/__init__.py`
- Create: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/earth_model.py`
- Test: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/test_earth_model.py`

**Interfaces:**
- Consumes: `GlobalMatrix.gmm_reflectivity(model, p, omega, free_surface=False, solver="riccati") -> np.ndarray`; `Kennett_Reflectivity.layer_model.LayerModel.from_arrays(alpha, beta, rho, thickness, Q_alpha, Q_beta)`.
- Produces:
  - `@dataclass LayerSpec(alpha: float, beta: float, rho: float, Q_alpha: float, Q_beta: float)` — one homogeneous region.
  - `build_marine_reference_model(ocean: LayerSpec, ocean_thickness: float, crust: LayerSpec, crust_thickness: float, halfspace: LayerSpec, n_crust_planes: int) -> LayerModel`. Builds an `(2 + n_crust_planes)`-layer `LayerModel`: ocean (β=0) of `ocean_thickness`; `n_crust_planes` identical crust sub-layers each `crust_thickness / n_crust_planes` thick; half-space (`thickness = np.inf`).

- [ ] **Step 1: Write the failing test**

```python
# GlobalMatrix/marine3d/test_earth_model.py
import numpy as np
from GlobalMatrix import gmm_reflectivity
from GlobalMatrix.marine3d.earth_model import LayerSpec, build_marine_reference_model

OCEAN = LayerSpec(alpha=1.5, beta=0.0, rho=1.0, Q_alpha=20000.0, Q_beta=1e10)
CRUST = LayerSpec(alpha=3.0, beta=1.5, rho=2.6, Q_alpha=100.0, Q_beta=100.0)
HALF = LayerSpec(alpha=5.0, beta=3.0, rho=3.2, Q_alpha=100.0, Q_beta=100.0)


def test_crust_subdivision_is_transparent():
    """Splitting the homogeneous crust into N identical sub-layers must not
    change the reflectivity: identical adjacent media => R=0, T=I at the
    internal interfaces."""
    omega = np.linspace(0.5, 25.0, 128)
    p = 0.12
    m1 = build_marine_reference_model(OCEAN, 2.0, CRUST, 1.0, HALF, n_crust_planes=1)
    m8 = build_marine_reference_model(OCEAN, 2.0, CRUST, 1.0, HALF, n_crust_planes=8)
    r1 = gmm_reflectivity(m1, p=p, omega=omega)
    r8 = gmm_reflectivity(m8, p=p, omega=omega)
    rel = np.max(np.abs(r8 - r1)) / np.max(np.abs(r1))
    assert rel < 1e-10, f"crust subdivision not transparent: rel={rel:.2e}"


def test_layer_count_and_halfspace():
    m = build_marine_reference_model(OCEAN, 2.0, CRUST, 1.0, HALF, n_crust_planes=4)
    assert m.n_layers == 2 + 4
    assert np.isinf(m.thickness[-1])
    assert m.beta[0] == 0.0  # ocean is acoustic
    # crust sub-layers identical and summing to crust_thickness
    assert np.allclose(m.alpha[1:5], CRUST.alpha)
    assert np.isclose(m.thickness[1:5].sum(), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Desktop/SeismicInversion && conda run -n seismic python -m pytest GlobalMatrix/marine3d/test_earth_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'GlobalMatrix.marine3d'`.

- [ ] **Step 3: Write minimal implementation**

```python
# GlobalMatrix/marine3d/__init__.py
"""3-D deterministic marine scattering solver (thesis Ch5 extension)."""
from .earth_model import LayerSpec, build_marine_reference_model

__all__ = ["LayerSpec", "build_marine_reference_model"]
```

```python
# GlobalMatrix/marine3d/earth_model.py
"""Build the layered reference model for the marine ocean/crust/half-space."""

from dataclasses import dataclass

import numpy as np
from Kennett_Reflectivity.layer_model import LayerModel


@dataclass
class LayerSpec:
    """One homogeneous region (seismic units: km/s, km/s, g/cm^3)."""

    alpha: float
    beta: float
    rho: float
    Q_alpha: float
    Q_beta: float


def build_marine_reference_model(
    ocean: LayerSpec,
    ocean_thickness: float,
    crust: LayerSpec,
    crust_thickness: float,
    halfspace: LayerSpec,
    n_crust_planes: int,
) -> LayerModel:
    """Ocean (acoustic) / homogeneous crust (n_crust_planes transparent
    sub-layers) / stiff half-space, as a LayerModel.

    The crust is one homogeneous reference replicated into n_crust_planes
    identical sub-layers; their internal interfaces are transparent (R=0,
    T=I). All heterogeneity is added later as per-voxel T0 screens.
    """
    if n_crust_planes < 1:
        msg = f"n_crust_planes must be >= 1, got {n_crust_planes}"
        raise ValueError(msg)
    dz = crust_thickness / n_crust_planes
    specs = [ocean] + [crust] * n_crust_planes + [halfspace]
    thick = [ocean_thickness] + [dz] * n_crust_planes + [np.inf]
    return LayerModel.from_arrays(
        alpha=[s.alpha for s in specs],
        beta=[s.beta for s in specs],
        rho=[s.rho for s in specs],
        thickness=thick,
        Q_alpha=[s.Q_alpha for s in specs],
        Q_beta=[s.Q_beta for s in specs],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Desktop/SeismicInversion && conda run -n seismic python -m pytest GlobalMatrix/marine3d/test_earth_model.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
cd ~/Desktop/SeismicInversion
conda run -n seismic ruff check --fix --ignore ARG001,ARG002,F841,E741 GlobalMatrix/marine3d/*.py
conda run -n seismic ruff format GlobalMatrix/marine3d/
conda run -n seismic mypy GlobalMatrix/marine3d/ --ignore-missing-imports
git add GlobalMatrix/marine3d/__init__.py GlobalMatrix/marine3d/earth_model.py GlobalMatrix/marine3d/test_earth_model.py
git commit -m "✨ marine3d: reference earth model with transparent-crust subdivision"
```

---

### Task 2: `marine_reference_gather` — end-to-end reference shot gather (heterogeneity off)

**Files:**
- Create: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/survey.py`
- Test: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/test_survey.py`

**Interfaces:**
- Consumes: `build_marine_reference_model` (Task 1); `Kennett_Reflectivity.kennett_gather.compute_gather(model, offsets, T=64.0, nw=2048, np_slow=2048, p_max=1.0, gamma=None, source_func=None, n_workers=None, free_surface=False) -> (time, offsets, gather)` where `gather` has shape `(nr, nt)`.
- Produces: `marine_reference_gather(model: LayerModel, offsets: np.ndarray, *, T: float = 64.0, nw: int = 1024, np_slow: int = 1024, p_max: float = 0.4, free_surface: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(time, offsets, gather)`.

- [ ] **Step 1: Write the failing test**

```python
# GlobalMatrix/marine3d/test_survey.py
import numpy as np
from GlobalMatrix.marine3d.earth_model import LayerSpec, build_marine_reference_model
from GlobalMatrix.marine3d.survey import marine_reference_gather

OCEAN = LayerSpec(1.5, 0.0, 1.0, 20000.0, 1e10)
CRUST = LayerSpec(3.0, 1.5, 2.6, 100.0, 100.0)
HALF = LayerSpec(5.0, 3.0, 3.2, 100.0, 100.0)


def test_gather_shape_and_finiteness():
    offsets = np.linspace(0.1, 3.0, 16)
    m = build_marine_reference_model(OCEAN, 2.0, CRUST, 1.0, HALF, n_crust_planes=4)
    time, off, gather = marine_reference_gather(
        m, offsets, T=8.0, nw=256, np_slow=256, p_max=0.4
    )
    assert gather.shape == (offsets.size, time.size)
    assert np.all(np.isfinite(gather))
    assert np.max(np.abs(gather)) > 0.0


def test_gather_invariant_under_crust_subdivision():
    """The reference gather must not depend on the numerical crust subdivision."""
    offsets = np.linspace(0.1, 3.0, 8)
    kw = dict(T=8.0, nw=256, np_slow=256, p_max=0.4)
    m1 = build_marine_reference_model(OCEAN, 2.0, CRUST, 1.0, HALF, n_crust_planes=1)
    m6 = build_marine_reference_model(OCEAN, 2.0, CRUST, 1.0, HALF, n_crust_planes=6)
    _, _, g1 = marine_reference_gather(m1, offsets, **kw)
    _, _, g6 = marine_reference_gather(m6, offsets, **kw)
    rel = np.max(np.abs(g6 - g1)) / np.max(np.abs(g1))
    assert rel < 1e-8, f"gather not subdivision-invariant: rel={rel:.2e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Desktop/SeismicInversion && conda run -n seismic python -m pytest GlobalMatrix/marine3d/test_survey.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'GlobalMatrix.marine3d.survey'`.

- [ ] **Step 3: Write minimal implementation**

```python
# GlobalMatrix/marine3d/survey.py
"""Marine shot-gather drivers for the marine3d solver."""

import numpy as np
from Kennett_Reflectivity.kennett_gather import compute_gather
from Kennett_Reflectivity.layer_model import LayerModel


def marine_reference_gather(
    model: LayerModel,
    offsets: np.ndarray,
    *,
    T: float = 64.0,
    nw: int = 1024,
    np_slow: int = 1024,
    p_max: float = 0.4,
    free_surface: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference (heterogeneity-off) marine shot gather via slowness-integrated
    layered reflectivity. Returns (time, offsets, gather) with gather (nr, nt)."""
    return compute_gather(
        model,
        np.asarray(offsets, dtype=np.float64),
        T=T,
        nw=nw,
        np_slow=np_slow,
        p_max=p_max,
        free_surface=free_surface,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Desktop/SeismicInversion && conda run -n seismic python -m pytest GlobalMatrix/marine3d/test_survey.py -v`
Expected: PASS (2 passed). (If `compute_gather` returns `gather` transposed, adapt the wrapper to return `(nr, nt)` — assert the shape in the test is the contract.)

- [ ] **Step 5: Lint, type-check, commit**

```bash
cd ~/Desktop/SeismicInversion
conda run -n seismic ruff check --fix --ignore ARG001,ARG002,F841,E741 GlobalMatrix/marine3d/survey.py GlobalMatrix/marine3d/test_survey.py
conda run -n seismic ruff format GlobalMatrix/marine3d/
conda run -n seismic mypy GlobalMatrix/marine3d/ --ignore-missing-imports
git add GlobalMatrix/marine3d/survey.py GlobalMatrix/marine3d/test_survey.py
git commit -m "✨ marine3d: end-to-end reference shot gather (heterogeneity off)"
```

---

### Task 3: YAML config + fail-fast loader for the marine reference model

**Files:**
- Create: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/config.py`
- Create: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/configs/marine_reference.yaml`
- Test: `~/Desktop/SeismicInversion/GlobalMatrix/marine3d/test_config.py`

**Interfaces:**
- Consumes: `LayerSpec`, `build_marine_reference_model` (Task 1).
- Produces: `load_marine_reference_model(path: Path) -> tuple[LayerModel, dict]` returning the built `LayerModel` and the validated survey-parameters dict (`offsets`, `T`, `nw`, `np_slow`, `p_max`, `free_surface`). Raises `ConfigError` with the 4-element diagnostic on any missing/invalid key.

- [ ] **Step 1: Write the failing test**

```python
# GlobalMatrix/marine3d/test_config.py
from pathlib import Path

import pytest
from GlobalMatrix.marine3d.config import ConfigError, load_marine_reference_model

VALID = """
model:
  ocean:     {alpha: 1.5, beta: 0.0, rho: 1.0, Q_alpha: 20000, Q_beta: 1.0e10}
  ocean_thickness: 2.0
  crust:     {alpha: 3.0, beta: 1.5, rho: 2.6, Q_alpha: 100, Q_beta: 100}
  crust_thickness: 1.0
  n_crust_planes: 4
  halfspace: {alpha: 5.0, beta: 3.0, rho: 3.2, Q_alpha: 100, Q_beta: 100}
survey:
  offsets: {start: 0.1, stop: 3.0, num: 16}
  T: 8.0
  nw: 256
  np_slow: 256
  p_max: 0.4
  free_surface: false
"""


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_valid_config_builds_model(tmp_path):
    model, survey = load_marine_reference_model(_write(tmp_path, VALID))
    assert model.n_layers == 2 + 4
    assert survey["offsets"].size == 16
    assert survey["T"] == 8.0


def test_missing_key_fails_with_full_diagnostic(tmp_path):
    bad = VALID.replace("  crust_thickness: 1.0\n", "")
    p = _write(tmp_path, bad)
    with pytest.raises(ConfigError) as exc:
        load_marine_reference_model(p)
    msg = str(exc.value)
    assert "crust_thickness" in msg          # what
    assert str(p) in msg                      # where (file)
    assert "model.crust_thickness" in msg     # where (dotted key)
    assert "example" in msg.lower()           # valid example
    assert "add" in msg.lower()               # recovery step
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Desktop/SeismicInversion && conda run -n seismic python -m pytest GlobalMatrix/marine3d/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'GlobalMatrix.marine3d.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# GlobalMatrix/marine3d/config.py
"""YAML loader (single source of truth) + fail-fast validation for marine3d."""

from pathlib import Path

import numpy as np
import yaml
from Kennett_Reflectivity.layer_model import LayerModel

from .earth_model import LayerSpec, build_marine_reference_model


class ConfigError(ValueError):
    """Raised on any invalid/missing marine3d configuration key."""


def _require(d: dict, key: str, dotted: str, path: Path, example: str):
    if key not in d:
        msg = (
            f"Missing required config key '{key}'.\n"
            f"  Where: {path}  (key: {dotted})\n"
            f"  Expected: e.g. `{example}`\n"
            f"  Recovery: add `{example}` under `{dotted.rsplit('.', 1)[0]}`."
        )
        raise ConfigError(msg)
    return d[key]


def _spec(d: dict, dotted: str, path: Path) -> LayerSpec:
    ex = "{alpha: 3.0, beta: 1.5, rho: 2.6, Q_alpha: 100, Q_beta: 100}"
    for k in ("alpha", "beta", "rho", "Q_alpha", "Q_beta"):
        _require(d, k, f"{dotted}.{k}", path, ex)
    return LayerSpec(d["alpha"], d["beta"], d["rho"], d["Q_alpha"], d["Q_beta"])


def load_marine_reference_model(path: Path) -> tuple[LayerModel, dict]:
    path = Path(path)
    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise ConfigError(msg)
    raw = yaml.safe_load(path.read_text())
    m = _require(raw, "model", "model", path, "model: {ocean: ..., crust: ...}")
    s = _require(raw, "survey", "survey", path, "survey: {offsets: ..., T: 8.0}")

    ocean = _spec(_require(m, "ocean", "model.ocean", path, "ocean: {...}"), "model.ocean", path)
    crust = _spec(_require(m, "crust", "model.crust", path, "crust: {...}"), "model.crust", path)
    half = _spec(
        _require(m, "halfspace", "model.halfspace", path, "halfspace: {...}"),
        "model.halfspace",
        path,
    )
    ocean_th = _require(m, "ocean_thickness", "model.ocean_thickness", path, "ocean_thickness: 2.0")
    crust_th = _require(m, "crust_thickness", "model.crust_thickness", path, "crust_thickness: 1.0")
    n_planes = _require(m, "n_crust_planes", "model.n_crust_planes", path, "n_crust_planes: 4")

    model = build_marine_reference_model(ocean, ocean_th, crust, crust_th, half, n_planes)

    o = _require(s, "offsets", "survey.offsets", path, "offsets: {start: 0.1, stop: 3.0, num: 16}")
    survey = {
        "offsets": np.linspace(
            _require(o, "start", "survey.offsets.start", path, "start: 0.1"),
            _require(o, "stop", "survey.offsets.stop", path, "stop: 3.0"),
            _require(o, "num", "survey.offsets.num", path, "num: 16"),
        ),
        "T": _require(s, "T", "survey.T", path, "T: 8.0"),
        "nw": _require(s, "nw", "survey.nw", path, "nw: 256"),
        "np_slow": _require(s, "np_slow", "survey.np_slow", path, "np_slow: 256"),
        "p_max": _require(s, "p_max", "survey.p_max", path, "p_max: 0.4"),
        "free_surface": _require(s, "free_surface", "survey.free_surface", path, "free_surface: false"),
    }
    return model, survey
```

```yaml
# GlobalMatrix/marine3d/configs/marine_reference.yaml
model:
  ocean:     {alpha: 1.5, beta: 0.0, rho: 1.0, Q_alpha: 20000, Q_beta: 1.0e10}
  ocean_thickness: 2.0          # km
  crust:     {alpha: 3.0, beta: 1.5, rho: 2.6, Q_alpha: 100, Q_beta: 100}
  crust_thickness: 1.0          # km (subdivided into n_crust_planes transparent planes)
  n_crust_planes: 4
  halfspace: {alpha: 5.0, beta: 3.0, rho: 3.2, Q_alpha: 100, Q_beta: 100}
survey:
  offsets: {start: 0.1, stop: 3.0, num: 16}   # km
  T: 8.0            # s
  nw: 256
  np_slow: 256
  p_max: 0.4        # s/km
  free_surface: false
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Desktop/SeismicInversion && conda run -n seismic python -m pytest GlobalMatrix/marine3d/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, type-check, run full Phase-1 suite, commit**

```bash
cd ~/Desktop/SeismicInversion
conda run -n seismic ruff check --fix --ignore ARG001,ARG002,F841,E741 GlobalMatrix/marine3d/*.py
conda run -n seismic ruff format GlobalMatrix/marine3d/
conda run -n seismic mypy GlobalMatrix/marine3d/ --ignore-missing-imports
conda run -n seismic python -m pytest GlobalMatrix/marine3d/ -v
git add GlobalMatrix/marine3d/config.py GlobalMatrix/marine3d/configs/marine_reference.yaml GlobalMatrix/marine3d/test_config.py
git commit -m "✨ marine3d: YAML config + fail-fast loader for the reference model"
```

---

## Phase roadmap (subsequent plans — each its own spec→plan cycle)

Phase 1 (this plan) delivers the package + reference model + reference gather + config, validating the transparent-crust property and establishing the $\Delta\mathcal{C}_{eff}=0$ baseline (spec validation step 1). The remaining phases:

- **Phase 2 — per-voxel $T_0$ screens + 2½-D coupling.** Port `cubic_scattering` 9×9 $(u,\varepsilon)$ effective-contrast $T_0$ into per-voxel $\Delta\mathcal{C}_{eff}$ screens at each crust plane; add the **standard-splitting** GMRES matvec with the existing $\mathbf{P}^z$ Riccati sweep and an x-only $\mathbf{P}^x$ (`interlayer_ms` + `horizontal_greens`); reproduce the **thesis 2½-D limit** and Born-as-one-iterate (spec validation steps 2–3). *Vertical slice: a 2½-D heterogeneous gather.*
- **Phase 3 — full 3-D $\mathbf{P}^x(x,y)$ + multiple scattering.** Generalise $\mathbf{P}^x$ to the $(x,y)$ planar lattice (resolve xy-FFT vs 2-D directional sweep), full $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})$ matvec, GMRES convergence study (target thesis Ch 6: ≈8 iters @5%, ≈28 @10%); energy/reciprocity checks (spec validation steps 4–5). *Vertical slice: a true 3-D heterogeneous gather.*
- **Phase 4 — GPU/cloud scaling + differentiable path.** Torch matvec end-to-end, GPU≡CPU parity, cloud scaling; preserve the differentiable `gmm_torch` route for future inversion (spec validation step 6).

---

## Self-Review

**1. Spec coverage (Phase 1 scope):** §3 marine model + transparent crust → Task 1 (built + invariance test) & Task 3 (config). §6 observable (shot gather) → Task 2. §7 validation step 1 (heterogeneity-off = reference) → Tasks 1–2 invariance + baseline gather. §8 YAML SoT + fail-fast → Task 3. Spec items deferred to later phases ($\Delta\mathcal{C}_{eff}$ screens, $\mathbf{P}^x/\mathbf{P}^z$ matvec, 3-D, GPU) are explicitly in the Phase roadmap, not dropped.

**2. Placeholder scan:** No TBD/TODO; every code/test step shows complete code; every run step shows the exact command and expected outcome. The one runtime contingency (gather array orientation in Task 2 Step 4) is stated as an explicit contract (the test asserts `(nr, nt)`), not a placeholder.

**3. Type consistency:** `LayerSpec(alpha, beta, rho, Q_alpha, Q_beta)` and `build_marine_reference_model(ocean, ocean_thickness, crust, crust_thickness, halfspace, n_crust_planes)` are used identically in Tasks 1, 2, 3. `marine_reference_gather(model, offsets, *, T, nw, np_slow, p_max, free_surface) -> (time, offsets, gather)` matches its test. `load_marine_reference_model(path) -> (LayerModel, dict)` matches its test (survey keys `offsets/T/nw/np_slow/p_max/free_surface`).
