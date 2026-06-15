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
