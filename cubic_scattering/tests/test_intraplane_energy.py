"""Phase 3b cycle 3: layer energy-balance cross-check.

``Mathematica/IntraPlaneEnergyBalance.wl`` builds the undamped vector coupling G0^vec
at physical parameters, projects it to the layer R/T(p), assembles the propagating
scattering matrix S, and dumps ``IntraPlaneEnergyBalance_reference.json``. This module
independently reloads the dumped S-matrices and re-verifies unitarity (S^dag S = I, or
the Sigma-twisted invariant), the SH 2x2 energy, the symplectic reciprocity residuals,
and the sub-wavelength no-open-diffraction-order margin.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

REF = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneEnergyBalance_reference.json"
)


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneEnergyBalance.wl first)"
    return json.loads(REF.read_text())


def _cplx(reim_mat):
    """Map a nested [re, im] dump array to a complex numpy array."""
    a = np.asarray(reim_mat, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def _sig_metric(modes):
    diag = [(-1.0 if m == "SV" else 1.0) for m in modes] * 2
    return np.diag(diag).astype(complex)


def test_energy_unitarity_each_p(dump):
    """S^dag M S == M on the propagating sub-block at every p (M per energyMetric)."""
    metric = dump["params"]["energyMetric"]
    tol = dump["params"]["enTol"]
    worst = 0.0
    for st in dump["stageEB"]:
        s = _cplx(st["S_psv"])
        m = (
            _sig_metric(st["propModes"])
            if metric == "sigma"
            else np.eye(s.shape[0], dtype=complex)
        )
        resid = np.max(np.abs(s.conj().T @ m @ s - m))
        worst = max(worst, resid)
    assert worst < tol, f"energy ({metric}) residual {worst:.3e} >= tol {tol:.3e}"


def test_sh_energy_each_p(dump):
    """|Rsh|^2 + |Tsh|^2 == 1 (SH down-incident) at every p."""
    tol = dump["params"]["enTol"]
    for st in dump["stageEB"]:
        ssh = _cplx(st["S_sh"])  # [[Rsh_d, Tsh_u], [Tsh_d, Rsh_u]]
        rsh_d, tsh_d = ssh[0, 0], ssh[1, 0]
        assert abs(abs(rsh_d) ** 2 + abs(tsh_d) ** 2 - 1.0) < tol


def test_reciprocity_preserved(dump):
    """Undamped build keeps Phase-3a symplectic reciprocity (Rd,Ru antisym; reported residuals)."""
    for st in dump["stageEB"]:
        assert st["recip_Rd_anti"] < 1e-6
        assert st["recip_Ru_anti"] < 1e-6
        assert st["recip_T_parity"] < 1e-6


def test_no_open_diffraction_orders(dump):
    """Independently recompute the sub-wavelength margin: min_{G!=0}|k_par+G| - kSo > 0."""
    pr = dump["params"]
    omega = pr["kPo"] * pr["alpha"] / pr["aa"]
    recip_b = 2 * math.pi / pr["aLpitch"]
    shells = [
        (m, n) for m in range(-3, 4) for n in range(-3, 4) if not (m == 0 and n == 0)
    ]
    worst_margin = math.inf
    for st in dump["stageEB"]:
        kpar = np.array([omega * st["p"], 0.0])
        margin = (
            min(np.linalg.norm(kpar + recip_b * np.array([m, n])) for m, n in shells)
            - pr["kSo"]
        )
        worst_margin = min(worst_margin, margin)
    assert worst_margin > 0.0, f"open diffraction order: margin {worst_margin:.3e}"
    assert abs(worst_margin - dump["diffMargin"]) < 1e-6


def test_nmax_does_not_diverge(dump):
    """Energy residual must not blow up with Nmax (convergence, not divergence)."""
    study = {int(nmx): r for nmx, r in dump["nmaxStudy"]}
    assert study[3] <= 10.0 * study[2], "energy residual diverges with Nmax"
