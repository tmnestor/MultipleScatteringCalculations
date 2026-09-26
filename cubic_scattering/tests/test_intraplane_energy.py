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

REF = Path(__file__).resolve().parents[2] / "Mathematica" / "IntraPlaneEnergyBalance_reference.json"


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
        m = _sig_metric(st["propModes"]) if metric == "sigma" else np.eye(s.shape[0], dtype=complex)
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


def test_reciprocity_independent(dump):
    """Recompute symplectic reciprocity from the dumped S_psv (2-mode stages) and cross-check.

    Rd, Ru off-diagonal antisymmetry and the SV parity Tu = Σ·Td·Σ (Σ=diag(1,−1)) are
    recomputed in Python from S_psv = [[Rd, Tu], [Td, Ru]] rather than trusting the dumped
    scalars; the ballistic identity on Td,Tu cancels in the parity difference.
    """
    sig = np.diag([1.0, -1.0]).astype(complex)
    for st in dump["stageEB"]:
        if len(st["propModes"]) != 2:
            # post-critical SV-only: reciprocity is single-mode trivial; dump records 0
            assert st["recip_Rd_anti"] == 0.0
            continue
        s = _cplx(st["S_psv"])  # 4x4
        rd, tu = s[0:2, 0:2], s[0:2, 2:4]
        td, ru = s[2:4, 0:2], s[2:4, 2:4]
        rd_anti = abs(rd[0, 1] + rd[1, 0])
        ru_anti = abs(ru[0, 1] + ru[1, 0])
        t_parity = float(np.max(np.abs(tu - sig @ td @ sig)))
        assert rd_anti < 1e-6, f"Rd not antisymmetric: {rd_anti:.3e}"
        assert ru_anti < 1e-6, f"Ru not antisymmetric: {ru_anti:.3e}"
        assert t_parity < 1e-6, f"T parity violated: {t_parity:.3e}"
        # cross-check the independent recompute matches the dumped residuals
        assert abs(rd_anti - st["recip_Rd_anti"]) < 1e-6
        assert abs(ru_anti - st["recip_Ru_anti"]) < 1e-6
        assert abs(t_parity - st["recip_T_parity"]) < 1e-6


def test_no_open_diffraction_orders(dump):
    """Independently recompute the sub-wavelength margin: min_{G!=0}|k_par+G| - kSo > 0."""
    pr = dump["params"]
    omega = pr["kPo"] * pr["alpha"] / pr["aa"]
    recip_b = 2 * math.pi / pr["aLpitch"]
    shells = [(m, n) for m in range(-3, 4) for n in range(-3, 4) if not (m == 0 and n == 0)]
    worst_margin = math.inf
    for st in dump["stageEB"]:
        kpar = np.array([omega * st["p"], 0.0])
        margin = min(np.linalg.norm(kpar + recip_b * np.array([m, n])) for m, n in shells) - pr["kSo"]
        worst_margin = min(worst_margin, margin)
    assert worst_margin > 0.0, f"open diffraction order: margin {worst_margin:.3e}"
    assert abs(worst_margin - dump["diffMargin"]) < 1e-6


#: Below this the residual is round-off, whose ratio between two N_max is noise
#: (6.5e-15 -> 7.6e-15 at N_max 2 -> 3), not a trend.
ROUNDOFF_FLOOR = 1e-12


def _nmax_diverges(r2: float, r3: float) -> bool:
    """True if the N_max = 3 residual has grown past both 1.5x N_max = 2 and round-off."""
    return r3 > max(1.5 * r2, ROUNDOFF_FLOOR)


def test_nmax_does_not_diverge(dump):
    """Energy residual must not blow up with Nmax (convergence, not divergence)."""
    study = {int(nmx): r for nmx, r in dump["nmaxStudy"]}
    assert not _nmax_diverges(study[2], study[3]), (
        f"energy residual diverges with Nmax: {study[2]:.3e} -> {study[3]:.3e} "
        f"(ratio > 1.5 and above the {ROUNDOFF_FLOOR:.0e} round-off floor)"
    )


@pytest.mark.parametrize(
    ("r2", "r3", "diverges"),
    [
        (6.5e-15, 7.6e-15, False),  # round-off jitter, ratio 1.17
        (6.5e-15, 5e-13, False),  # ratio 77, still round-off
        (1e-14, 1e-9, True),  # a real blow-up from round-off
        (1e-6, 2e-6, True),  # above the floor, ratio 2
        (1e-6, 1.2e-6, False),  # above the floor, ratio 1.2
    ],
)
def test_nmax_divergence_rule(r2, r3, diverges):
    """The floor forgives round-off jitter but still catches growth above it."""
    assert _nmax_diverges(r2, r3) is diverges
