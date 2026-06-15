# Point-Source vs Body-Force Galerkin T-Matrix Paper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Also invoke `superpowers:verification-before-completion` before any "done"/PASS claim — read the actual command output before checking a box.

**Goal:** Build a self-contained lualatex research paper in a new `paper/` directory that derives, from a reproducible re-run evidence base, whether the body-force Galerkin T₉ closure beats a local point-source representation of an elastic cubic scatterer — with T₂₇ excluded entirely.

**Architecture:** Three phases. (A) A reproducible **evidence pipeline** under `paper/evidence/` that re-runs existing validation scripts/tests, captures their logs, and emits LaTeX-ready CSV/table fragments — plus one new consolidation sweep against the exact Mie/Kennett oracles. (B) **Figures** (TikZ schematics + one pgfplots figure fed by the evidence CSVs). (C) **Prose** harvested from existing `docs/*.tex`, structurally reduced to T₉, organised as an ascending hierarchy-of-approximations argument whose conclusion follows from the evidence table.

**Tech Stack:** Python 3 (numpy, scipy), conda env `seismic`, pytest, ruff, mypy; lualatex (TinyTeX) with fontspec/tikz/pgfplots; existing `cubic_scattering` public API.

**Spec:** `docs/superpowers/specs/2026-06-16-tmatrix-point-vs-galerkin-paper-design.md`

**Conventions (CLAUDE.md):** conda env `seismic` (`conda run -n seismic <cmd>`); coordinate system z=axis 0 (down), x=axis 1, y=axis 2; Voigt order (ε11, ε22, ε33, 2ε23, 2ε13, 2ε12); line length ≤108; commits via gitmoji, **no Claude attribution**, never `--no-verify`; lualatex compiled **in-place** in `paper/`. **T₂₇ is excluded** from all paper content (see spec §1 hard rule).

**Verified public API (for the sweep in Task 9):**
```python
from cubic_scattering import (
    ReferenceMedium, MaterialContrast,
    compute_cube_tmatrix,        # (omega, a, ref, contrast, n_gauss=…, n_taylor=…, k_hat=None) -> CubeTMatrixResult
    compute_cube_tmatrix_galerkin,
    compute_elastic_mie,         # (omega, radius, ref, contrast, n_max=None) -> MieResult
    mie_far_field,               # (mie_result, theta_arr, incident_type="P") -> (f_P, f_SV, f_SH)
    cube_far_field,              # (c_inc, c_sc, theta, ref, galerkin, contrast, omega, a, k_vec=None, pol=None) -> (f_P, f_SV, f_SH)
)
```
`ReferenceMedium(alpha, beta, rho)`, `MaterialContrast(Dlambda, Dmu, Drho)`. Test params: REF=(5000,3000,2500); moderate CONTRAST=(2e9,1e9,100.0).

---

## File Structure

| Path | Created/Modified | Responsibility |
|------|------------------|----------------|
| `paper/tmatrix_point_vs_galerkin.tex` | Create | Self-contained main document (preamble, all sections, bibliography hookup). |
| `paper/references.bib` | Create | Bibliography, split standard vs this-work. |
| `paper/evidence/run_evidence.py` | Create | Orchestrates re-runs of existing scripts/tests; captures logs to `logs/`; parses headline numbers to per-item CSVs. |
| `paper/evidence/cost_accuracy_sweep.py` | Create | NEW far-field accuracy sweep (representations × contrast × ka × polarization) vs Mie; writes `cost_accuracy.csv`. |
| `paper/evidence/make_tables.py` | Create | Reads CSVs, emits `\input`-able LaTeX `tab_*.tex` fragments into `paper/evidence/tables/`. |
| `paper/evidence/test_evidence.py` | Create | pytest checks: sweep monotonicity/known anchors; parser smoke tests. |
| `paper/figures/fig_ladder.tex` … `fig_error_vs_ka.tex` | Create | One TikZ/pgfplots figure each (F1–F6). |

Originals in `docs/` and `LatexPDFs/` are **not modified** (acceptance criterion 6).

---

## Phase A — Evidence pipeline

### Task 1: Branch + scaffold `paper/` tree

**Files:** Create `paper/` directory tree.

- [ ] **Step 1: Create the feature branch off main**
```bash
git checkout main && git pull --ff-only 2>/dev/null; git checkout -b feature/tmatrix-point-vs-galerkin-paper
git branch --show-current   # expect: feature/tmatrix-point-vs-galerkin-paper
```

- [ ] **Step 2: Create the directory skeleton**
```bash
mkdir -p paper/evidence/logs paper/evidence/tables paper/figures
```

- [ ] **Step 3: Add a `.gitkeep` to the empty log/table dirs so they commit**
```bash
touch paper/evidence/logs/.gitkeep paper/evidence/tables/.gitkeep
```

- [ ] **Step 4: Commit**
```bash
git add paper/.gitkeep paper/evidence paper/figures 2>/dev/null; git add paper
git commit -m "🏗️ scaffold paper/ directory tree"
```

---

### Task 2: Evidence runner skeleton + log capture

**Files:** Create `paper/evidence/run_evidence.py`, `paper/evidence/test_evidence.py`.

- [ ] **Step 1: Write the failing test for the log-capture helper**
```python
# paper/evidence/test_evidence.py
import subprocess
from pathlib import Path

from run_evidence import capture  # noqa: E402

def test_capture_writes_log(tmp_path):
    log = tmp_path / "echo.log"
    rc, out = capture(["python", "-c", "print('hello-evidence')"], log)
    assert rc == 0
    assert "hello-evidence" in out
    assert log.read_text().strip() == "hello-evidence"
```

- [ ] **Step 2: Run it to verify failure**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_capture_writes_log -v
```
Expected: FAIL (`ModuleNotFoundError: run_evidence` or `ImportError: capture`).

- [ ] **Step 3: Implement `capture` in `run_evidence.py`**
```python
#!/usr/bin/env python3
"""Reproducible evidence runner for the point-vs-Galerkin paper.

Re-runs existing cubic_scattering validation scripts/tests in conda env
`seismic`, captures stdout to paper/evidence/logs/, and parses headline
numbers into per-item CSVs.  Run from the repository root:

    conda run -n seismic python paper/evidence/run_evidence.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path(__file__).resolve().parent / "logs"


def capture(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Run `cmd` from the repo root, tee stdout+stderr to `log_path`, return (rc, text)."""
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out)
    return proc.returncode, out
```

- [ ] **Step 4: Run the test to verify it passes**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_capture_writes_log -v
```
Expected: PASS.

- [ ] **Step 5: Lint/format and commit**
```bash
conda run -n seismic ruff check paper/evidence/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format paper/evidence/
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py
git commit -m "🔬 evidence: log-capture runner skeleton"
```

---

### Task 3: Capture the form-factor / Kennett convergence evidence

**Files:** Modify `paper/evidence/run_evidence.py` (add `evidence_formfactor`).

- [ ] **Step 1: Add a parser+runner that re-runs the existing slab convergence study**

The study `scripts/slab_convergence_study.py` prints a table whose rows end with a `Rel Err` column (single-layer, varying `a` → ka). Add:
```python
import csv
import re

EV = Path(__file__).resolve().parent

def _floats(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)]

def evidence_formfactor() -> Path:
    """Re-run the single-layer slab convergence study; tabulate ka vs rel-err vs Kennett."""
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "scripts/slab_convergence_study.py"],
        LOGS / "formfactor_kennett.log",
    )
    assert rc == 0, f"slab_convergence_study failed:\n{out[-2000:]}"
    rows = []
    for line in out.splitlines():
        nums = _floats(line)
        # data rows carry: a, ka_S, DOF, |R_FL|, |R_K|, Rel Err, ... (>=6 floats)
        if len(nums) >= 6 and line.strip()[0:1].isdigit():
            rows.append({"a": nums[0], "ka_S": nums[1], "rel_err": nums[5]})
    assert rows, "no data rows parsed from slab convergence study"
    csv_path = EV / "formfactor_kennett.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "ka_S", "rel_err"])
        w.writeheader()
        w.writerows(rows)
    return csv_path
```

- [ ] **Step 2: Write a smoke test asserting the known anchor**

From validated memory: single-layer rel-err is ~1% and decreases as `a` shrinks (1.1%→0.31%).
```python
# append to paper/evidence/test_evidence.py
import csv as _csv
from pathlib import Path as _Path
from run_evidence import evidence_formfactor

def test_formfactor_evidence_anchor():
    csv_path = evidence_formfactor()
    rows = list(_csv.DictReader(csv_path.open()))
    errs = [float(r["rel_err"]) for r in rows]
    assert all(e < 0.05 for e in errs), f"rel-err exceeded 5%: {errs}"
    assert min(errs) < 0.01, f"best rel-err not sub-1%: {min(errs)}"
```

- [ ] **Step 3: Run it (this exercises the real study; allow minutes)**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_formfactor_evidence_anchor -v
```
Expected: PASS, `paper/evidence/formfactor_kennett.csv` written. If the parsed numbers diverge from memory, **the measured number wins** (spec §10) — update the smoke-test bounds to bracket the measured values and note the divergence in the eventual paper §5.1.

- [ ] **Step 4: Commit**
```bash
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py paper/evidence/formfactor_kennett.csv paper/evidence/logs/formfactor_kennett.log
git commit -m "🔬 evidence: form-factor convergence vs Kennett"
```

---

### Task 4: Capture the radiation-reaction (LDOS) evidence

**Files:** Modify `paper/evidence/run_evidence.py` (add `evidence_radiation_reaction`).

- [ ] **Step 1: Add a runner that re-runs the two radiation-reaction gate tests and records pass + key values**

These tests assert the Mie a₀/a₂ gate (modulus) and the density Im[Γ₀] LDOS. Capture pass/fail and any printed ratios.
```python
def evidence_radiation_reaction() -> Path:
    """Re-run the density Im[Γ₀] and modulus Im[Δc*] (strain LDOS) Mie gates."""
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "-m", "pytest",
         "cubic_scattering/tests/test_gamma0_radiation_reaction.py",
         "cubic_scattering/tests/test_modulus_radiation_reaction.py", "-v"],
        LOGS / "radiation_reaction.log",
    )
    passed = "failed" not in out.lower() and rc == 0
    csv_path = EV / "radiation_reaction.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gate", "passed"])
        w.writerow(["density_Gamma0_LDOS", passed])
        w.writerow(["modulus_strain_LDOS_mie_a0_a2", passed])
    return csv_path
```

- [ ] **Step 2: Write the smoke test (gates must pass)**
```python
from run_evidence import evidence_radiation_reaction

def test_radiation_reaction_gates_pass():
    csv_path = evidence_radiation_reaction()
    rows = list(_csv.DictReader(csv_path.open()))
    assert all(r["passed"] == "True" for r in rows), "a radiation-reaction Mie gate failed"
```

- [ ] **Step 3: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_radiation_reaction_gates_pass -v
```
Expected: PASS.

- [ ] **Step 4: Commit**
```bash
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py paper/evidence/radiation_reaction.csv paper/evidence/logs/radiation_reaction.log
git commit -m "🔬 evidence: density + modulus radiation-reaction LDOS gates"
```

---

### Task 5: Capture the optical-theorem closure evidence

**Files:** Modify `paper/evidence/run_evidence.py` (add `evidence_optical_theorem`).

- [ ] **Step 1: Add a runner re-running the optical-theorem gate test**
```python
def evidence_optical_theorem() -> Path:
    """Re-run the energy-conservation optical-theorem gate (exact Mie σ_ext/σ_sc → 1.0)."""
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "-m", "pytest",
         "cubic_scattering/tests/test_cube_far_field_mie_correction.py", "-v", "-k", "optical"],
        LOGS / "optical_theorem.log",
    )
    csv_path = EV / "optical_theorem.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "passed"])
        w.writerow(["mie_sigma_ext_over_sigma_sc_eq_1", "failed" not in out.lower() and rc == 0])
    return csv_path
```

- [ ] **Step 2: Smoke test**
```python
from run_evidence import evidence_optical_theorem

def test_optical_theorem_gate_passes():
    csv_path = evidence_optical_theorem()
    rows = list(_csv.DictReader(csv_path.open()))
    assert all(r["passed"] == "True" for r in rows), "optical-theorem gate failed"
```

- [ ] **Step 3: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_optical_theorem_gate_passes -v
```
Expected: PASS. If the `-k optical` selector matches nothing, list the test names with `conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_far_field_mie_correction.py --collect-only -q` and update the selector to the actual optical-theorem test name.

- [ ] **Step 4: Commit**
```bash
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py paper/evidence/optical_theorem.csv paper/evidence/logs/optical_theorem.log
git commit -m "🔬 evidence: optical-theorem energy-conservation closure"
```

---

### Task 6: Capture the T₂₇ inter-voxel ≤0.07% evidence (excluded-rung)

**Files:** Modify `paper/evidence/run_evidence.py` (add `evidence_t27_intervoxel`).

- [ ] **Step 1: Add a runner re-running the T₂₇ coupling study + calibration test**
```python
def evidence_t27_intervoxel() -> Path:
    """Re-run the T₂₇ inter-voxel coupling study (the excluded-rung evidence: ≤0.07%)."""
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "scripts/t27_coupling_study.py"],
        LOGS / "t27_intervoxel.log",
    )
    assert rc == 0, f"t27_coupling_study failed:\n{out[-2000:]}"
    # Parse the maximum reported coupling fraction (percent).
    pcts = [float(x) for x in re.findall(r"([-+]?\d+\.\d+)\s*%", out)]
    max_pct = max(pcts) if pcts else float("nan")
    csv_path = EV / "t27_intervoxel.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value_percent"])
        w.writerow(["max_intervoxel_coupling", max_pct])
    return csv_path
```

- [ ] **Step 2: Smoke test the ≤0.07% verdict**
```python
from run_evidence import evidence_t27_intervoxel

def test_t27_intervoxel_negligible():
    csv_path = evidence_t27_intervoxel()
    rows = list(_csv.DictReader(csv_path.open()))
    val = float(rows[0]["value_percent"])
    assert val <= 0.1, f"inter-voxel coupling not negligible: {val}%"
```

- [ ] **Step 3: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_t27_intervoxel_negligible -v
```
Expected: PASS (value ≤ 0.07%). If the percent regex fails to find the headline number, inspect `logs/t27_intervoxel.log` and tighten the regex to the study's decision-table row.

- [ ] **Step 4: Commit**
```bash
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py paper/evidence/t27_intervoxel.csv paper/evidence/logs/t27_intervoxel.log
git commit -m "🔬 evidence: T27 inter-voxel coupling (excluded-rung, ≤0.07%)"
```

---

### Task 7: Capture the near-field radiation-part-need evidence

**Files:** Modify `paper/evidence/run_evidence.py` (add `evidence_radiation_need`).

- [ ] **Step 1: Add a runner re-running the radiation-part-need script**
```python
def evidence_radiation_need() -> Path:
    """Re-run the near-field radiation-part-need probe (~0.5% at normal incidence; O(1) of G by ka≈0.5)."""
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "scripts/test_radiation_part_need.py"],
        LOGS / "radiation_need.log",
    )
    assert rc == 0, f"test_radiation_part_need failed:\n{out[-2000:]}"
    csv_path = EV / "radiation_need.csv"
    # Persist the raw log reference; numbers are extracted into the paper §6 table by hand-checked parse.
    rows = []
    for line in out.splitlines():
        nums = _floats(line)
        if len(nums) >= 2 and ("ka" in line.lower() or "%" in line):
            rows.append({"raw": line.strip()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw"])
        w.writeheader()
        w.writerows(rows)
    return csv_path
```

- [ ] **Step 2: Smoke test (script runs, produces rows)**
```python
from run_evidence import evidence_radiation_need

def test_radiation_need_runs():
    csv_path = evidence_radiation_need()
    assert csv_path.exists() and csv_path.stat().st_size > 0
```

- [ ] **Step 3: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_radiation_need_runs -v
```
Expected: PASS. Read `logs/radiation_need.log` and record the ~0.5% (normal) and O(1)-by-ka≈0.5 figures for the paper §6 prose.

- [ ] **Step 4: Commit**
```bash
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py paper/evidence/radiation_need.csv paper/evidence/logs/radiation_need.log
git commit -m "🔬 evidence: near-field radiation-part need"
```

---

### Task 8: Pin the three single-site representations (discovery)

**Files:** Modify `paper/evidence/run_evidence.py` (add a documented `REPRESENTATIONS` table + smoke test).

The cost-accuracy sweep (Task 9) compares three single-site far-field representations against Mie. The full T₉ path is verified (`compute_cube_tmatrix` → `cube_far_field`). This task pins how the **point/Born** and **Eshelby-point** baselines are constructed, by reading the existing derivation, so Task 9 references only defined entry points.

- [ ] **Step 1: Read the existing baseline derivation and code it references**
```bash
sed -n '/Bare-Born versus Eshelby/,/Cube 27-Component/p' docs/mie_finite_contrast_validation.tex
grep -rn "born\|Born\|bare" cubic_scattering/*.py | grep -i "def \|point\|born"
```
Record, in a module-level docstring table in `run_evidence.py`, the exact callable (or construction) for:
- **rep="born"**: bare point scatterer (static contrasts, amplification A_u = 1, point monopole+dipole).
- **rep="eshelby"**: Eshelby-corrected point (self-consistent Δc\*, A_u amplification, point source — no finite-size form factor).
- **rep="t9"**: `compute_cube_tmatrix` → `cube_far_field` (finite-size form factor + radiation reaction).

- [ ] **Step 2: Implement a `far_field_amplitudes(rep, omega, a, ref, contrast, theta, incident="P")` dispatcher**

For `rep="t9"`, build `c_inc`, `c_sc` via the existing galerkin path and call `cube_far_field` (signature above). For `rep="born"` / `rep="eshelby"`, construct the point P-amplitude from the closed-form `f_P = -[r̂·F - ik_P V (r̂·Δσ·r̂)]/(4πρα²)` (the formula in `cube_far_field`'s docstring) with the rep-specific `F` and `Δσ`:
- born: `F = ω²Δρ V u_inc`, `Δσ = Δc : ε_inc` (static contrasts, no A_u, no form factor).
- eshelby: `F = ω²Δρ A_u V u_inc`, `Δσ = Δc\* : ε_inc` (self-consistent contrasts + A_u, still a point).
Use `compute_cube_tmatrix(...)` to obtain the self-consistent `Δλ*, Δμ*, Δρ*` and `A_u` for the eshelby/t9 reps (read the exact `CubeTMatrixResult` attribute names via `grep -n "class CubeTMatrixResult" -A40 cubic_scattering/effective_contrasts.py` in this step and use them verbatim).

- [ ] **Step 3: Smoke test — t9 reproduces the validated density-dipole accuracy**
```python
import numpy as np
from cubic_scattering import ReferenceMedium, MaterialContrast, compute_elastic_mie, mie_far_field
from run_evidence import far_field_amplitudes

def test_t9_matches_mie_moderate_low_ka():
    ref = ReferenceMedium(5000.0, 3000.0, 2500.0)
    con = MaterialContrast(2e9, 1e9, 100.0)
    a = 10.0; ka = 0.05; omega = ka * ref.beta / a
    theta = np.linspace(0.0, np.pi, 64)
    fP, _, _ = far_field_amplitudes("t9", omega, a, ref, con, theta)
    R = a * (6.0 / np.pi) ** (1.0 / 3.0)             # equal-volume sphere radius
    mie = compute_elastic_mie(omega, R, ref, con)
    mP, _, _ = mie_far_field(mie, theta, "P")
    l2 = np.linalg.norm(fP - mP) / np.linalg.norm(mP)
    assert l2 < 0.05, f"t9 P-channel L2 vs Mie too large at ka=0.05: {l2}"
```

- [ ] **Step 4: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_t9_matches_mie_moderate_low_ka -v
```
Expected: PASS (L2 ≈ 0.005 per validated memory). If it fails, fix the dispatcher (not the bound) until the validated t9 path matches.

- [ ] **Step 5: Lint/format and commit**
```bash
conda run -n seismic ruff check paper/evidence/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format paper/evidence/
git add paper/evidence/run_evidence.py paper/evidence/test_evidence.py
git commit -m "🔬 evidence: pin born/eshelby/t9 far-field representations"
```

---

### Task 9: Cost-accuracy sweep vs Mie

**Files:** Create `paper/evidence/cost_accuracy_sweep.py`; modify `paper/evidence/test_evidence.py`.

- [ ] **Step 1: Write the sweep using the pinned dispatcher**
```python
#!/usr/bin/env python3
"""Far-field cost-accuracy sweep: representations × contrast × ka × polarization vs Mie."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from cubic_scattering import ReferenceMedium, MaterialContrast, compute_elastic_mie, mie_far_field
from run_evidence import far_field_amplitudes

EV = Path(__file__).resolve().parent
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CONTRASTS = {
    "weak": MaterialContrast(REF.mu * 1e-4, REF.mu * 1e-4, REF.rho * 1e-4),
    "moderate": MaterialContrast(2e9, 1e9, 100.0),
    "strong_neg": MaterialContrast(-1.2e9, -0.6e9, -1500.0),  # ~-60% density
}
KA = [0.05, 0.1, 0.3, 0.5]
REPS = ["born", "eshelby", "t9"]
POLS = ["P", "SV", "SH"]
A = 10.0


def _l2_linf(f, m):
    denom = np.linalg.norm(m) or 1.0
    return float(np.linalg.norm(f - m) / denom), float(np.max(np.abs(f - m)))


def run() -> Path:
    theta = np.linspace(0.0, np.pi, 128)
    R = A * (6.0 / np.pi) ** (1.0 / 3.0)
    rows = []
    for cname, con in CONTRASTS.items():
        for ka in KA:
            omega = ka * REF.beta / A
            mie = compute_elastic_mie(omega, R, REF, con)
            for pol in POLS:
                mP, mSV, mSH = mie_far_field(mie, theta, pol)
                m = {"P": mP, "SV": mSV, "SH": mSH}[pol]
                for rep in REPS:
                    t0 = time.perf_counter()
                    fP, fSV, fSH = far_field_amplitudes(rep, omega, A, REF, con, theta, incident=pol)
                    dt = time.perf_counter() - t0
                    f = {"P": fP, "SV": fSV, "SH": fSH}[pol]
                    l2, linf = _l2_linf(f, m)
                    rows.append({"contrast": cname, "ka": ka, "pol": pol, "rep": rep,
                                 "l2": l2, "linf": linf, "cost_s": dt})
    out = EV / "cost_accuracy.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["contrast", "ka", "pol", "rep", "l2", "linf", "cost_s"])
        w.writeheader()
        w.writerows(rows)
    return out


if __name__ == "__main__":
    print(f"wrote {run()}")
```

- [ ] **Step 2: Write a test asserting the headline ladder ordering at moderate/low-ka P**
```python
def test_ladder_orders_at_moderate_lowka():
    import csv as c
    from cost_accuracy_sweep import run
    rows = list(c.DictReader(run().open()))
    sel = {r["rep"]: float(r["l2"]) for r in rows
           if r["contrast"] == "moderate" and r["pol"] == "P" and abs(float(r["ka"]) - 0.05) < 1e-9}
    # The hierarchy must improve monotonically born ≥ eshelby ≥ t9 (single-site).
    assert sel["born"] >= sel["eshelby"] >= sel["t9"], sel
```

- [ ] **Step 3: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_ladder_orders_at_moderate_lowka -v
```
Expected: PASS. The verdict in the paper is *derived* from this CSV; if the ordering is non-monotone for some (contrast, ka, pol), that is itself a reportable result for §7 (do **not** force it — record it).

- [ ] **Step 4: Generate the full CSV**
```bash
cd paper/evidence && conda run -n seismic python cost_accuracy_sweep.py
head -5 cost_accuracy.csv
```

- [ ] **Step 5: Lint/format and commit**
```bash
conda run -n seismic ruff check paper/evidence/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format paper/evidence/
git add paper/evidence/cost_accuracy_sweep.py paper/evidence/test_evidence.py paper/evidence/cost_accuracy.csv
git commit -m "🔬 evidence: cost-accuracy sweep vs Mie (born/eshelby/t9)"
```

---

### Task 10: LaTeX table generator

**Files:** Create `paper/evidence/make_tables.py`; modify `paper/evidence/test_evidence.py`.

- [ ] **Step 1: Implement `make_tables.py` that emits booktabs fragments from the CSVs**
```python
#!/usr/bin/env python3
"""Emit \\input-able LaTeX booktabs fragments from the evidence CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

EV = Path(__file__).resolve().parent
TAB = EV / "tables"


def _table(csv_name: str, cols: list[str], headers: list[str], caption: str, label: str) -> Path:
    rows = list(csv.DictReader((EV / csv_name).open()))
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / f"{label}.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    _table("cost_accuracy.csv", ["contrast", "ka", "pol", "rep", "l2", "linf"],
           ["Contrast", "$ka$", "Pol.", "Rep.", "$L^2$", "$L^\\infty$"],
           "Far-field error vs Mie across the hierarchy.", "tab_cost_accuracy")
    _table("formfactor_kennett.csv", ["a", "ka_S", "rel_err"],
           ["$a$ (m)", "$ka_S$", "Rel.\\ err"],
           "Volume-averaged T9 slab vs Kennett.", "tab_formfactor")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test that fragments are produced and contain booktabs rules**
```python
def test_make_tables_emits_fragments():
    from make_tables import main
    main()
    t = (Path(__file__).resolve().parent / "tables" / "tab_cost_accuracy.tex").read_text()
    assert r"\toprule" in t and r"\bottomrule" in t
```

- [ ] **Step 3: Run it**
```bash
cd paper/evidence && conda run -n seismic python -m pytest test_evidence.py::test_make_tables_emits_fragments -v && conda run -n seismic python make_tables.py
```
Expected: PASS; `paper/evidence/tables/tab_cost_accuracy.tex`, `tab_formfactor.tex` written.

- [ ] **Step 4: Lint/format and commit**
```bash
conda run -n seismic ruff check paper/evidence/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format paper/evidence/
git add paper/evidence/make_tables.py paper/evidence/test_evidence.py paper/evidence/tables/
git commit -m "🔬 evidence: LaTeX booktabs table generator"
```

---

## Phase B — Document skeleton & figures

### Task 11: Main document preamble + section skeleton + bib stub

**Files:** Create `paper/tmatrix_point_vs_galerkin.tex`, `paper/references.bib`.

- [ ] **Step 1: Write the preamble + empty sections + bib hookup**

Reuse the `docs/cube_galerkin27.tex` preamble (fontspec Latin Modern, amsmath/bm, booktabs, geometry margin 2.5cm, hyperref colorlinks, tikz with libraries `arrows.meta,calc,decorations.pathmorphing,shapes.geometric,positioning`, pgfplots compat=1.18, the `\D \br \be \bu \bphi` macros). Add `\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}` and `\addbibresource{references.bib}`. Body:
```latex
\title{Point-Source versus Body-Force Galerkin Representations of an
Elastic Cubic Scatterer:\\ a Mie- and Kennett-Validated Hierarchy}
\author{T.\,M.\ Nestor}
\date{June 2026}
\begin{document}\maketitle
\begin{abstract}\end{abstract}            % §1 (written last, Task 18)
\section{Introduction}\label{sec:intro}   % §2 (Task 18)
\section{Background theory}\label{sec:background}        % §3 (Task 14)
\section{The body-force Galerkin closure (\texorpdfstring{$T_9$}{T9})}\label{sec:t9}  % §4 (Task 15)
\section{Single-site self-energy corrections}\label{sec:single}  % §5 (Task 16)
\section{Inter-site coupling}\label{sec:inter}           % §6 (Task 17)
\section{Evidence synthesis and cost--accuracy}\label{sec:evidence}  % §7 (Task 18)
\section{Discussion and conclusion}\label{sec:conclusion}           % §8 (Task 18)
\appendix
\section{Master integrals (\texorpdfstring{$T_9$}{T9} subset)}\label{app:masters}  % (Task 19)
\printbibliography
\end{document}
```

- [ ] **Step 2: Seed `references.bib` with the standard-literature entries (verified)**

Add, with a comment banner `% === STANDARD LITERATURE ===`: `eshelby1957`, `mura1987`, `foldy1945`, `lax1951`, `waterman1971`, `kennett1983`, `akirichards2002`, `yingtruell1956`, `korneev1993`, `draineflatau1994`, `hudson1980`, `crampin1981`. Each as a complete `@book`/`@article` with author, title, year, publisher/journal. **Verify each exists** (do not fabricate) before committing; if a citation cannot be confirmed, drop it rather than invent details. Add a second banner `% === THIS WORK ===` (entries filled when cited).

- [ ] **Step 3: Compile (expect a near-empty but clean build)**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && \
  /usr/local/bin/biber tmatrix_point_vs_galerkin && \
  /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
echo "exit: $?"; ls tmatrix_point_vs_galerkin.pdf
```
Expected: PDF produced, no error in the log's final lines. If `biblatex`/`biber` is unavailable in TinyTeX, `tlmgr install biblatex biber` first.

- [ ] **Step 4: Commit**
```bash
git add paper/tmatrix_point_vs_galerkin.tex paper/references.bib
git commit -m "📝 paper: preamble, section skeleton, standard-literature bib"
```

---

### Task 12: Schematic TikZ figures F1–F5

**Files:** Create `paper/figures/fig_ladder.tex`, `fig_single_vs_inter.tex`, `fig_ldos.tex`, `fig_cm_relative.tex`, `fig_intersite_phase.tex`.

- [ ] **Step 1: Write each figure as a standalone `\input`-able `tikzpicture`**

One file per figure (no preamble; assumes the main doc's tikz libraries). Concrete content:
- `fig_ladder.tex`: a vertical ladder with rungs *Born*, *Eshelby-point*, *$T_9$ Galerkin*, and a greyed/crossed-out top rung *$T_{27}$ (excluded, $\le 0.07\%$)*, each annotated with an error arrow that shrinks down the rungs.
- `fig_single_vs_inter.tex`: a cube with a self-loop (self-energy) beside two cubes with an inter-site arrow.
- `fig_ldos.tex`: a point force radiating into 1 P-lobe + 2 S-lobes; a moment-tensor dipole radiating its P-monopole+quadrupole / S-quadrupole pattern — illustrating "modes available to radiate into".
- `fig_cm_relative.tex`: two overlapping cubes with the centre-of-mass $\bxi$ / relative $\bs$ vectors, and the 1-D tent $\Lambda_V$.
- `fig_intersite_phase.tex`: two stacked cells with a vertical coupling arrow (corrected) and a horizontal ~1 rad phase ramp (uncorrected).

- [ ] **Step 2: Add `\input` lines into the right sections of the main doc** (F1→§3.3/§7, F2→§3, F3→§5.3, F4→§4, F5→§6), each wrapped in a `figure` environment with `\caption` and `\label`.

- [ ] **Step 3: Compile and verify each figure renders**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
grep -iE "overfull|undefined|error" tmatrix_point_vs_galerkin.log | grep -vi "underfull" | head
ls tmatrix_point_vs_galerkin.pdf
```
Expected: PDF builds; no `! ` LaTeX errors. Inspect the PDF pages for the five figures.

- [ ] **Step 4: Commit**
```bash
git add paper/figures/fig_ladder.tex paper/figures/fig_single_vs_inter.tex paper/figures/fig_ldos.tex paper/figures/fig_cm_relative.tex paper/figures/fig_intersite_phase.tex paper/tmatrix_point_vs_galerkin.tex
git commit -m "📝 paper: schematic TikZ figures F1–F5"
```

---

### Task 13: pgfplots evidence figure F6

**Files:** Create `paper/figures/fig_error_vs_ka.tex`; modify main doc.

- [ ] **Step 1: Generate a pgfplots-friendly long CSV**

Add to `make_tables.py` a `dump_pgfplots()` that writes `paper/evidence/tables/error_vs_ka.dat` (whitespace-separated: `ka rep l2` filtered to contrast=moderate, pol=P), and re-run `make_tables.py`.

- [ ] **Step 2: Write `fig_error_vs_ka.tex` reading that data**
```latex
\begin{tikzpicture}
\begin{axis}[xlabel={$ka$},ylabel={$L^2$ error vs Mie},ymode=log,legend pos=north west,width=0.8\linewidth]
  \addplot table[x=ka,y=l2,col sep=space]{evidence/tables/error_vs_ka_born.dat};   \addlegendentry{Born}
  \addplot table[x=ka,y=l2,col sep=space]{evidence/tables/error_vs_ka_eshelby.dat}; \addlegendentry{Eshelby-point}
  \addplot table[x=ka,y=l2,col sep=space]{evidence/tables/error_vs_ka_t9.dat};      \addlegendentry{$T_9$ Galerkin}
\end{axis}
\end{tikzpicture}
```
(Adjust `dump_pgfplots()` to emit one `.dat` per rep so each `\addplot` has a clean series. Use a relative path the in-place build resolves; if needed, set `\pgfplotsset{table/search path={evidence/tables}}` in the preamble.)

- [ ] **Step 3: `\input` F6 into §7 inside a `figure`; compile**
```bash
cd paper && conda run -n seismic python evidence/make_tables.py && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
ls tmatrix_point_vs_galerkin.pdf
```
Expected: PDF builds with the log-scale error-vs-ka plot showing the three rungs.

- [ ] **Step 4: Commit**
```bash
git add paper/figures/fig_error_vs_ka.tex paper/evidence/make_tables.py paper/evidence/tables/ paper/tmatrix_point_vs_galerkin.tex
git commit -m "📝 paper: pgfplots error-vs-ka evidence figure (F6)"
```

---

## Phase C — Prose (harvest + de-T₂₇ + write)

> For each prose task: harvest from the named source, **reduce any T₂₇ content to the T₉ subspace**, write the section, then compile and run the T₂₇-exclusion grep gate (Task 20) locally before committing.

### Task 14: §3 Background theory

**Files:** Modify `paper/tmatrix_point_vs_galerkin.tex` (§3); add bib entries.

- [ ] **Step 1: Write §3.1 VIE** — harvest the Lippmann–Schwinger equation and singular-structure framing from `docs/cube_galerkin27.tex §1` and `docs/point_vs_volume_tmatrix_notes.tex §"direct elastodynamic match"`. Keep the LS equation \eqref form; cite `mura1987`, `eshelby1957`.
- [ ] **Step 2: Write §3.2 oracles** — the finite-radius finite-contrast elastic Mie sphere (harvest from `docs/mie_multipole_derivation.tex`; cite `yingtruell1956`, `korneev1993`) and Kennett layered reflectivity (cite `kennett1983`). State both as the exact references the paper validates against.
- [ ] **Step 3: Write §3.3 hierarchy** — define the three rungs (point/Born → Eshelby-point → $T_9$ Galerkin); add `\input{figures/fig_ladder.tex}` here; one sentence scoping out $T_{27}$ with `\ref` to the §7 ≤0.07% evidence.
- [ ] **Step 4: Compile + commit**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && grep -ci "error" tmatrix_point_vs_galerkin.log
git add paper/tmatrix_point_vs_galerkin.tex paper/references.bib && git commit -m "📝 paper: §3 background theory (VIE, oracles, hierarchy)"
```

### Task 15: §4 Body-force Galerkin closure (T₉)

**Files:** Modify main doc (§4 + Appendix A).

- [ ] **Step 1: Harvest the Galerkin LS, CM/relative reduction, cube autocorrelation tent, and master integrals from `docs/cube_galerkin27.tex §2–5` and `docs/cube_tmatrix_closedform.tex`** — **reduced to the $T_9$ subspace** (constants + symmetric-linear polynomials; dimensions $3+6=9$). Drop the 18 quadratic modes and the degree-4/5 master-integral tables entirely (those are $T_{27}$-only). Add `\input{figures/fig_cm_relative.tex}`.
- [ ] **Step 2: State the closed-form effective contrasts $\Delta\lambda^*,\Delta\mu^*,\Delta\rho^*$** as the output of the $T_9$ closure (real/static parts), citing `eshelby1957`/`mura1987` as standard and the closed form as this-work.
- [ ] **Step 3: Move the $T_9$-relevant master-integral values into Appendix A** (harvest the relevant rows from `docs/cube_galerkin27_results.tex`, `cube_galerkin27_closedforms.tex`).
- [ ] **Step 4: Compile + commit**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
git add paper/tmatrix_point_vs_galerkin.tex paper/references.bib && git commit -m "📝 paper: §4 body-force Galerkin T9 closure + appendix masters"
```

### Task 16: §5 Single-site self-energy corrections (incl. LDOS)

**Files:** Modify main doc (§5); add bib `akirichards2002`.

- [ ] **Step 1: §5.1 form factor** — write the cube $-1/3$ finite-size form factor and its Kennett validation; `\input{evidence/tables/tab_formfactor.tex}`; cite the convergence numbers from `formfactor_kennett.csv`. Flag the form factor as this-work.
- [ ] **Step 2: §5.2 amplification** — harvest the self-consistent $A_u = 1/(1-\omega^2\Delta\rho\,\Gamma_0)$ derivation from `docs/mie_finite_contrast_validation.tex §self-consistent`.
- [ ] **Step 3: §5.3 radiation reaction = LDOS** — the requested content. Write: density $\mathrm{Im}[\Gamma_0]=\omega/(12\pi\rho)(1/\alpha^3+2/\beta^3)=\tfrac13 S_\alpha+\tfrac23 S_\beta$; the **strain LDOS** for the modulus channel — moment dipole $M=V\,\Delta c{:}\varepsilon$ radiating Aki–Richards power $P_{\mathrm{rad}}=\tfrac{\omega^4}{8\pi\rho\alpha^5}\langle(\hat r{\cdot}M{\cdot}\hat r)^2\rangle+\tfrac{\omega^4}{8\pi\rho\beta^5}\langle|(M\hat r)_\perp|^2\rangle$ with $\langle(\hat r{\cdot}M{\cdot}\hat r)^2\rangle=\tfrac1{15}(\mathrm{tr}M^2+2M{:}M)$, $\langle|(M\hat r)_\perp|^2\rangle=\tfrac1{15}(3M{:}M-\mathrm{tr}M^2)$, giving the two closed forms for $\mathrm{Im}[\Delta\lambda^*]$, $\mathrm{Im}[\Delta\mu^*]$ (verbatim from spec §5 / `effective_contrasts.py::_modulus_radiation_reaction`). Emphasise it is derived from the background Green's LDOS, **not** by imposing $\sigma_{\mathrm{ext}}=\sigma_{\mathrm{sc}}$; the optical theorem closing to 1.0 (cite `optical_theorem.csv`) is a *consequence*; Mie $a_0/a_2$ gate passed (cite `radiation_reaction.csv`). `\input{figures/fig_ldos.tex}`. Cite `akirichards2002` (moment-tensor radiation) as standard; flag the strain-LDOS $\mathrm{Im}[\Delta c^*]$ and density $\mathrm{Im}[\Gamma_0]$ correction as this-work.
- [ ] **Step 4: Compile + commit**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
git add paper/tmatrix_point_vs_galerkin.tex paper/references.bib && git commit -m "📝 paper: §5 single-site corrections (form factor, amplification, LDOS)"
```

### Task 17: §6 Inter-site coupling

**Files:** Modify main doc (§6).

- [ ] **Step 1: Harvest the point vs volume-averaged propagator and the vertical-vs-transverse-phase argument** from `docs/inter_voxel_propagator.tex` and `docs/slab_scattering_explanation.tex`, de-T₂₇'d. State that volume averaging corrects vertical coupling but not the ~1 rad/cell transverse phase; cite the `radiation_need.csv` figures (~0.5% normal → O(1) of G by ka≈0.5).
- [ ] **Step 2: Present the $T_{27}$ inter-voxel ≤0.07% result** as the excluded-rung evidence (`\input{evidence/tables/...}` or inline from `t27_intervoxel.csv`); `\input{figures/fig_intersite_phase.tex}`. Cite the slab→Kennett ~1% floor.
- [ ] **Step 3: Compile + commit**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
git add paper/tmatrix_point_vs_galerkin.tex && git commit -m "📝 paper: §6 inter-site coupling (point suffices; T27 excluded-rung)"
```

### Task 18: §7 evidence synthesis, §8 conclusion, §1 abstract, §2 intro

**Files:** Modify main doc (§7, §8, §1, §2).

- [ ] **Step 1: §7** — `\input{evidence/tables/tab_cost_accuracy.tex}` and `\input{figures/fig_error_vs_ka.tex}`; write the synthesis: read the ladder ordering and the ka-dependence directly off the table; state where the error stops dropping faster than complexity rises. The verdict text must follow the numbers (if any (contrast,ka,pol) breaks monotonicity, report it honestly).
- [ ] **Step 2: §8 conclusion** — state the derived verdict: single-site self-energy rewards the volume/Eshelby/LDOS treatment; inter-site coupling does not reward more than a local point propagator; $T_{27}$ is never repaid.
- [ ] **Step 3: §1 abstract + §2 introduction** — written last so they match the body; contributions list flagged standard vs novel; the one $T_{27}$ scoping sentence.
- [ ] **Step 4: Compile (full ref pass) + commit**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && /usr/local/bin/biber tmatrix_point_vs_galerkin && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
git add paper/tmatrix_point_vs_galerkin.tex paper/references.bib && git commit -m "📝 paper: §7 synthesis, §8 conclusion, abstract + intro"
```

### Task 19: Appendix + this-work bib finalisation

**Files:** Modify main doc (Appendix A), `references.bib`.

- [ ] **Step 1: Finalise Appendix A** with the $T_9$-subset master integrals/closed forms harvested in Task 15 Step 3 (ensure no degree-4/5 $T_{27}$ rows leaked in).
- [ ] **Step 2: Fill the `% === THIS WORK ===` bib entries** for every novel claim actually cited (form factor, strain-LDOS $\mathrm{Im}[\Delta c^*]$, density $\mathrm{Im}[\Gamma_0]$, $T_{27}$ no-gain verdict, cost-accuracy result), as `@unpublished`/`@misc{..., note={this work}}`.
- [ ] **Step 3: Compile + commit**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && /usr/local/bin/biber tmatrix_point_vs_galerkin && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
git add paper/tmatrix_point_vs_galerkin.tex paper/references.bib && git commit -m "📝 paper: appendix masters + this-work bibliography"
```

---

## Phase D — Final verification

### Task 20: Acceptance gates

**Files:** none (verification only).

- [ ] **Step 1: T₂₇-exclusion grep gate** — must return only the scoping/excluded-rung mentions, no derivation.
```bash
grep -niE "T_?\{?27\}?|27-component|27 component|quadratic mode" paper/tmatrix_point_vs_galerkin.tex paper/figures/*.tex
```
Expected: only lines in §3.3/§6/§7 that *exclude* T₂₇ (scoping sentence + excluded ladder rung + ≤0.07% pointer). Any derivation/table/figure hit is a failure → remove it.

- [ ] **Step 2: Clean compile gate (zero unresolved refs/cites)**
```bash
cd paper && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && /usr/local/bin/biber tmatrix_point_vs_galerkin && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex && /usr/local/bin/lualatex -interaction=nonstopmode tmatrix_point_vs_galerkin.tex
grep -iE "undefined (reference|citation)|Citation .* undefined|LaTeX Warning: Reference" tmatrix_point_vs_galerkin.log
```
Expected: the grep returns nothing; `tmatrix_point_vs_galerkin.pdf` exists.

- [ ] **Step 3: Originals-untouched gate**
```bash
git diff --name-only main -- docs/ LatexPDFs/ | grep -vE "docs/superpowers/(specs|plans)/" || echo "OK: no original docs modified"
```
Expected: `OK: no original docs modified`.

- [ ] **Step 4: Evidence reproducibility gate (full pipeline from clean)**
```bash
conda run -n seismic python paper/evidence/run_evidence.py
conda run -n seismic python paper/evidence/cost_accuracy_sweep.py
conda run -n seismic python paper/evidence/make_tables.py
conda run -n seismic python -m pytest paper/evidence/test_evidence.py -v
```
Expected: all evidence CSVs regenerate; all `test_evidence.py` tests PASS.

- [ ] **Step 5: Lint/type the evidence package**
```bash
conda run -n seismic ruff check paper/evidence/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format paper/evidence/
conda run -n seismic mypy paper/evidence/ --ignore-missing-imports
```
Expected: ruff clean, mypy Success (or only pre-existing unrelated notes).

- [ ] **Step 6: Final commit + branch summary**
```bash
git add -A paper/
git commit -m "✅ paper: pass acceptance gates (T27-excluded, clean compile, reproducible evidence)"
git log --oneline main..HEAD
```

---

## Self-Review (author checklist — already applied)

- **Spec coverage:** §3 evidence items → Tasks 3–9; paper §1–§10 → Tasks 11–19; figures F1–F6 → Tasks 12–13; references split → Tasks 11/19; acceptance criteria 1–6 → Task 20. All covered.
- **Placeholders:** evidence reuses *existing* scripts/tests by name; the one new sweep uses only the verified API; the single genuine unknown (born/eshelby entry points) is an explicit discovery task (Task 8) with a smoke-test gate, not a placeholder.
- **Type consistency:** `far_field_amplitudes(rep, omega, a, ref, contrast, theta, incident=...)` defined in Task 8, used identically in Task 9; `capture()`, `evidence_*()`, `make_tables.main()` names consistent across tasks; CSV column names (`l2`, `linf`, `cost_s`, `rel_err`, `ka_S`) consistent between sweep, tables, and figures.
- **Risk tie-breaker:** "measured number wins" stated in Tasks 3 and 9 — bounds adjust to measurement, never the reverse.
