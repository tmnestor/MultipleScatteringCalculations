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


def evidence_optical_theorem() -> Path:
    """Re-run the energy-conservation optical-theorem gate (exact Mie σ_ext/σ_sc → 1.0).

    Runs the two optical-theorem tests in test_scattered_field.py:
    - test_optical_theorem_mie_gate: exact-Mie sphere σ_ext/σ_sc = 1.0 to <2% across ka
    - test_optical_theorem_cube: cube optical-theorem ratio (structural 2nd-order deviation)

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
            "cubic_scattering/tests/test_scattered_field.py",
            "-v",
            "-k",
            "optical",
        ],
        LOGS / "optical_theorem.log",
    )
    csv_path = EV / "optical_theorem.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "passed"])
        w.writerow(
            [
                "mie_sigma_ext_over_sigma_sc_eq_1",
                "failed" not in out.lower() and rc == 0,
            ]
        )
    return csv_path


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


def evidence_t27_intervoxel() -> Path:
    """Re-run the T₂₇ inter-voxel coupling study (the excluded-rung evidence: ≤0.07%).

    The script prints a decision table whose PRIMARY column ``d(i,ii)`` is the
    fractional change in the two-voxel observable when the 18 quadratic
    inter-voxel coupling channels are zeroed — i.e., how much T₂₇ extra
    coupling matters beyond the validated 9-component chain.  Values are
    dimensionless ratios; the maximum across all separations, ka, and wave
    types is recorded here as a percentage.

    Table row format (space-separated):
        wave  sep  ka  d(i,ii)  d(i,ii_raw)  d(i,ii_K)  d(i,iii)
    Only rows whose first token is "P" or "SV" are data rows.

    Returns:
        Path to the written CSV file.
    """
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "scripts/t27_coupling_study.py"],
        LOGS / "t27_intervoxel.log",
    )
    assert rc == 0, f"t27_coupling_study failed:\n{out[-2000:]}"
    # Parse the d(i,ii) column (index 3, 0-based) from decision-table data rows.
    # Rows start with "P" or "SV" and contain 4 scientific-notation values.
    dii_values: list[float] = []
    for line in out.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] not in ("P", "SV"):
            continue
        # tokens: [wave, sep, ka, d(i,ii), d(i,ii_raw), d(i,ii_K), d(i,iii)]
        if len(tokens) >= 4:
            dii_values.append(float(tokens[3]))
    max_pct = max(dii_values) * 100.0 if dii_values else float("nan")
    csv_path = EV / "t27_intervoxel.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value_percent"])
        w.writerow(["max_intervoxel_coupling", max_pct])
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


def evidence_radiation_need() -> Path:
    """Re-run the near-field radiation-part-need probe.

    Quantifies how large the imaginary (radiation) part of the inter-voxel
    propagator is (~0.5% slab-R_PP effect at normal incidence; |Im/Re| G grows
    to O(1) by ka≈0.5 for face/edge/corner separations).

    Measurement A rows: separation × ka → |Im/Re| G, C, S blocks.
    Measurement B rows: ka → |R_K|, err pt, err VA n=2, err VA n=3,
        |VAn2-pt|/|R_K| (the propagator's own slab-solve contribution).

    The CSV captures raw text of every data line (>=3 floats) so the paper
    §6 table can be assembled by hand-checked parse of the log.  The log is
    the authoritative record; the CSV is a quick-scan summary.

    Returns:
        Path to the written CSV file.
    """
    rc, out = capture(
        ["conda", "run", "-n", "seismic", "python", "scripts/test_radiation_part_need.py"],
        LOGS / "radiation_need.log",
    )
    assert rc == 0, f"test_radiation_part_need failed:\n{out[-2000:]}"
    csv_path = EV / "radiation_need.csv"
    # Capture every line that carries at least 3 floats — this selects all
    # Measurement-A separation rows (12) and Measurement-B ka rows (5) plus
    # the background parameter line, while skipping headers and prose.
    rows = []
    for line in out.splitlines():
        nums = _floats(line)
        if len(nums) >= 3:
            rows.append({"raw": line.strip()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw"])
        w.writeheader()
        w.writerows(rows)
    return csv_path
