#!/usr/bin/env python3
"""Reproducible evidence runner for the point-vs-Galerkin paper.

Re-runs existing cubic_scattering validation scripts/tests in conda env
`seismic`, captures stdout to paper/evidence/logs/, and parses headline
numbers into per-item CSVs.  Run from the repository root:

    conda run -n seismic python paper/evidence/run_evidence.py
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = Path(__file__).resolve().parent
LOGS = EV / "logs"


def capture(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Run `cmd` from the repo root, tee stdout+stderr to `log_path`, return (rc, text)."""
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out)
    return proc.returncode, out


def _floats(line: str) -> list[float]:
    """Extract all floating-point numbers from a line of text."""
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", line)]


def evidence_radiation_reaction() -> Path:
    """Re-run the density Im[Γ₀] and modulus Im[Δc*] (strain LDOS) Mie gates.

    Executes both radiation-reaction pytest suites:
    - test_gamma0_radiation_reaction.py: density Im[Γ₀] LDOS gate
    - test_modulus_radiation_reaction.py: modulus strain-LDOS Mie a₀/a₂ gate

    Returns:
        Path to the written CSV file.
    """
    rc, out = capture(
        [
            "conda",
            "run",
            "-n",
            "seismic",
            "python",
            "-m",
            "pytest",
            "cubic_scattering/tests/test_gamma0_radiation_reaction.py",
            "cubic_scattering/tests/test_modulus_radiation_reaction.py",
            "-v",
        ],
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


def evidence_formfactor() -> Path:
    """Re-run the single-layer slab convergence study; tabulate ka vs rel-err vs Kennett.

    The study script prints a table with columns:
        a  M  ka_S  DOF  |R_FL|  |R_K|  Rel Err  GMRES  Resid  Time
    Only values with a decimal point are captured by _floats(); the integer
    columns M, DOF, and GMRES are skipped.  Time ends in 's' so only its
    numeric part (e.g. "0.0") is captured.  The resulting float sequence is:
        index 0: a
        index 1: ka_S
        index 2: |R_FL|
        index 3: |R_K|
        index 4: Rel Err   <- extracted here
        index 5: Resid
        index 6: Time (numeric part)

    Returns:
        Path to the written CSV file.
    """
    rc, out = capture(
        [
            "conda",
            "run",
            "-n",
            "seismic",
            "python",
            "scripts/slab_convergence_study.py",
        ],
        LOGS / "formfactor_kennett.log",
    )
    assert rc == 0, f"slab_convergence_study failed:\n{out[-2000:]}"
    rows = []
    for line in out.splitlines():
        nums = _floats(line)
        # data rows carry >= 5 floats and begin with a digit (the 'a' value).
        # Integer columns (M, DOF, GMRES) are not captured by _floats(), so:
        #   index 0=a, 1=ka_S, 2=|R_FL|, 3=|R_K|, 4=Rel Err, 5=Resid, 6=Time
        if len(nums) >= 5 and line.strip()[0:1].isdigit():
            rows.append({"a": nums[0], "ka_S": nums[1], "rel_err": nums[4]})
    assert rows, "no data rows parsed from slab convergence study"
    csv_path = EV / "formfactor_kennett.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "ka_S", "rel_err"])
        w.writeheader()
        w.writerows(rows)
    return csv_path
