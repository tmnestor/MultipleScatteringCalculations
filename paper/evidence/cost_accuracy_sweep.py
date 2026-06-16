#!/usr/bin/env python3
"""Far-field cost-accuracy sweep: representations × contrast × ka × polarization vs Mie.

For each (contrast, ka, pol, rep) cell this compares the single-site far-field of a
representation ("born", "eshelby", "t9") against the exact elastic Mie sphere of equal
volume (radius R = a·(6/π)^(1/3)) on the channel matching the incident polarization,
recording per-cell relative L²/L∞ error and a wall-time cost proxy.

The born/eshelby point representations are only derived for axial P-incidence (Task-8
limitation): SV/SH cells for them — and any cell the dispatcher rejects — are recorded
with empty metrics and ``status="undefined_for_incidence"`` rather than dropped, so the
master §7 table carries an honest coverage map.  Cells where the Mie reference channel
is ~0 (weak contrast / cross-channel zeros) are flagged ``status="ref_near_zero"``.

Run from the repository root or from paper/evidence/:

    conda run -n seismic python paper/evidence/cost_accuracy_sweep.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

# The sweep imports the cubic_scattering package and the sibling run_evidence module.
# Insert both the repo root (package) and the evidence dir (run_evidence) onto the
# path so `python cost_accuracy_sweep.py` works from repo root AND from paper/evidence/.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EV) not in sys.path:
    sys.path.insert(0, str(EV))

from run_evidence import far_field_amplitudes  # noqa: E402

from cubic_scattering import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_elastic_mie,
    mie_far_field,
)

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

# Channel index in the (f_P, f_SV, f_SH) tuple selected by the incident polarization.
_POL_INDEX = {"P": 0, "SV": 1, "SH": 2}

CSV_PATH = EV / "cost_accuracy.csv"
_FIELDS = ["contrast", "ka", "pol", "rep", "l2", "linf", "cost_s", "status"]


def run() -> Path:
    """Execute the full far-field cost-accuracy sweep and write ``cost_accuracy.csv``.

    Returns:
        Path to the written CSV file.
    """
    theta = np.linspace(0.0, np.pi, 128)
    rows: list[dict[str, object]] = []
    n_ok = 0
    n_undefined = 0
    n_ref_zero = 0

    for cname, contrast in CONTRASTS.items():
        for ka in KA:
            omega = ka * REF.beta / A
            radius = A * (6.0 / np.pi) ** (1.0 / 3.0)  # equal-volume sphere radius
            mie = compute_elastic_mie(omega, radius, REF, contrast)
            for pol in POLS:
                idx = _POL_INDEX[pol]
                mref = mie_far_field(mie, theta, pol)[idx]
                ref_norm = float(np.linalg.norm(mref))
                for rep in REPS:
                    row: dict[str, object] = {
                        "contrast": cname,
                        "ka": ka,
                        "pol": pol,
                        "rep": rep,
                        "l2": "",
                        "linf": "",
                        "cost_s": "",
                        "status": "",
                    }
                    t0 = time.perf_counter()
                    try:
                        f_tuple = far_field_amplitudes(
                            rep, omega, A, REF, contrast, theta, incident=pol
                        )
                    except (ValueError, NotImplementedError):
                        # Documented incidence limitation (born/eshelby P-only; the
                        # dispatcher may also reject t9 SV/SH). Log it, do not crash.
                        row["status"] = "undefined_for_incidence"
                        n_undefined += 1
                        rows.append(row)
                        continue
                    cost_s = time.perf_counter() - t0
                    row["cost_s"] = f"{cost_s:.6f}"

                    if ref_norm <= 1e-300 or not np.isfinite(ref_norm):
                        # Mie reference channel is ~0 (e.g. weak contrast / cross zero):
                        # a relative error is undefined. Flag it instead of dividing.
                        row["status"] = "ref_near_zero"
                        n_ref_zero += 1
                        rows.append(row)
                        continue

                    f = f_tuple[idx]
                    diff = f - mref
                    l2 = float(np.linalg.norm(diff) / ref_norm)
                    linf = float(np.max(np.abs(diff)))
                    row["l2"] = f"{l2:.8e}"
                    row["linf"] = f"{linf:.8e}"
                    row["status"] = "ok"
                    n_ok += 1
                    rows.append(row)

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"cells: ok={n_ok}  undefined_for_incidence={n_undefined}  "
        f"ref_near_zero={n_ref_zero}  total={len(rows)}"
    )
    return CSV_PATH


if __name__ == "__main__":
    out = run()
    print(out)
