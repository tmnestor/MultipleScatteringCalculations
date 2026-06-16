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
        [
            "conda",
            "run",
            "-n",
            "seismic",
            "python",
            "scripts/test_radiation_part_need.py",
        ],
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


# =====================================================================
# Single-site far-field representations (the paper's central hierarchy)
# =====================================================================
#
# Three increasingly faithful single-scatterer far-field models, all
# compared against the exact elastic Mie sphere in the Task-9 cost-accuracy
# sweep.  Each builds the radiated P-amplitude from the SAME closed-form
# Lippmann-Schwinger far field used inside ``cube_far_field``:
#
#     f_P = -[ r̂·F - i k_P V (r̂·Δσ·r̂) ] / (4π ρ α²)
#     f_S = -[ (I-r̂r̂)·F - i k_S (I-r̂r̂)·V Δσ·r̂ ] / (4π ρ β²)
#
# with V = (2a)³ (equal-volume cube) and a rep-specific equivalent source
# (force monopole F, stress dipole Δσ).  ε_inc is the plane-wave Voigt
# strain ε_ij = ½(i k_i p_j + i k_j p_i); u_inc = pol (unit amplitude).
#
# ─────────────────────────────────────────────────────────────────────
# rep        force monopole F            stress dipole Δσ          A_u
# ─────────────────────────────────────────────────────────────────────
# "born"     ω²·Δρ·V·u_inc               (Δc_bare : ε_inc)·V       1
#            bare static density          static contrasts:
#            point source, NO amp         Δλ, Δμ, Δμ (isotropic),
#                                         no Eshelby concentration,
#                                         no finite-size form factor
#
# "eshelby"  ω²·Δρ·A_u·V·u_inc           (Δc* : ε_inc)·V           1/(1-ω²Δρ Γ₀)
#            density amplified by         self-consistent contrasts
#            the Eshelby self-            Δλ*, Δμ*_diag, Δμ*_off
#            interaction A_u; still a     (res.Dlambda_star,
#            POINT source — NO finite-     .Dmu_star_diag,
#            size cube form factor        .Dmu_star_off);
#                                         POINT source (no form factor)
#
# "t9"       ω²·Δρ·A_u·c_inc[:3]         T_Voigt @ ε_inc           result.amp_u
#            (cube_far_field): finite-     (cube_far_field): finite- (internal)
#            size form factor enters       size form factor +
#            via c_inc overlap integrals   radiation reaction; full
#            + radiation reaction          T₉ Galerkin path
# ─────────────────────────────────────────────────────────────────────
#
# Self-consistent contrast / amplification attribute names (read VERBATIM
# from the codebase, do NOT guess):
#   * CubeTMatrixResult.amp_u            — density displacement amplification A_u
#   * CubeTMatrixResult.Drho_star        — effective density contrast Δρ*
#   * CubeTMatrixResult.Dlambda_star     — effective Δλ*
#   * CubeTMatrixResult.Dmu_star_diag    — effective diagonal Δμ*
#   * CubeTMatrixResult.Dmu_star_off     — effective off-diagonal Δμ*
#   * GalerkinTMatrixResult.Dlambda_star / .Dmu_star_diag / .Dmu_star_off
#     — same physical effective stiffness (used by the t9 ``cube_far_field`` path)
#
# A_u for "eshelby" is taken from ``compute_cube_tmatrix(...).amp_u`` (the
# self-consistent density Eshelby self-interaction); the point stress dipole
# uses the *real* part of the effective stiffness (matching ``cube_far_field``,
# whose imaginary radiation-reaction part is a second-order self-energy not a
# first-order radiating source).
#
# INCIDENCE / CHANNEL SUPPORT.  The born/eshelby POINT closed forms below are
# implemented for P-incidence along a cube axis (k̂ = x̂, the canonical
# validation geometry); they return all three channels (f_P, f_SV, f_SH) from
# the dyadic far field, which is well-defined for the point source.  For
# incident="SV"/"SH" the born/eshelby point reps are not derived here and
# raise an error — Task 9 should compare those reps on the P channel only.
# rep="t9" supports P-incidence via the validated path.

REPRESENTATIONS = ("born", "eshelby", "t9")


def far_field_amplitudes(
    rep: str,
    omega: float,
    a: float,
    ref,
    contrast,
    theta,
    incident: str = "P",
):
    """Single-site far-field amplitudes for one of the three representations.

    See the ``REPRESENTATIONS`` table above for the exact equivalent-source
    construction of each rep.  All three share the closed-form
    Lippmann-Schwinger far field used inside ``cube_far_field``.

    Args:
        rep: One of ``"born"``, ``"eshelby"``, ``"t9"``.
        omega: Angular frequency (rad/s).
        a: Cube half-width (m); equal-volume V = (2a)³.
        ref: ``ReferenceMedium`` background.
        contrast: ``MaterialContrast`` perturbation (bare/static contrasts).
        theta: Scattering angle grid (rad) measured from the forward direction.
        incident: Incident wave type. Only ``"P"`` is supported; born/eshelby
            additionally require P-incidence along a cube axis.

    Returns:
        Tuple ``(f_P, f_SV, f_SH)`` of complex numpy arrays over ``theta``.

    Raises:
        ValueError: If ``rep`` is not in ``REPRESENTATIONS`` or ``incident``
            is unsupported.
    """
    import sys

    import numpy as np

    # Ensure the repo root (containing the cubic_scattering package) is
    # importable regardless of the pytest cwd (Task 9 runs from paper/evidence).
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from cubic_scattering import (
        compute_cube_tmatrix,
        compute_cube_tmatrix_galerkin,
        cube_far_field,
    )
    from cubic_scattering.incident_field import cube_overlap_integrals
    from cubic_scattering.scattered_field import (
        _incident_voigt_strain,
        _voigt_to_tensor,
    )
    from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27
    from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt

    if rep not in REPRESENTATIONS:
        raise ValueError(f"rep must be one of {REPRESENTATIONS!r}, got {rep!r}")
    if incident != "P":
        raise ValueError(f"only incident='P' is supported, got {incident!r}")

    theta = np.atleast_1d(np.asarray(theta, dtype=float))

    # Canonical P-incidence along a cube axis (k̂ = x̂): the validation geometry.
    kP = omega / ref.alpha
    kS = omega / ref.beta
    k_hat = np.array([1.0, 0.0, 0.0])
    pol = k_hat.copy()
    k_vec = kP * k_hat

    if rep == "t9":
        g = compute_cube_tmatrix_galerkin(omega, a, ref, contrast)
        T27 = assemble_tmatrix_27(g)
        c_inc = cube_overlap_integrals(k_vec, pol, a)
        c_sc = T27 @ c_inc
        return cube_far_field(
            c_inc, c_sc, theta, ref, g, contrast, omega, a, k_vec, pol
        )

    # ── born / eshelby: POINT source (no finite-size cube form factor) ──
    rho = ref.rho
    V = (2.0 * a) ** 3
    u_inc = pol.astype(complex)
    eps_inc_V = _incident_voigt_strain(k_vec, pol)

    if rep == "born":
        # Bare static contrasts, isotropic (Δμ_diag = Δμ_off = Δμ), A_u = 1.
        amp_u: complex = 1.0
        Dc = effective_stiffness_voigt(contrast.Dlambda, contrast.Dmu, contrast.Dmu)
    else:  # rep == "eshelby"
        # Self-consistent Eshelby-corrected contrasts + density amplification.
        res = compute_cube_tmatrix(omega, a, ref, contrast)
        amp_u = res.amp_u
        Dc = effective_stiffness_voigt(
            res.Dlambda_star.real,
            res.Dmu_star_diag.real,
            res.Dmu_star_off.real,
        )

    # Force monopole (density) and stress dipole (stiffness), as in cube_far_field.
    F = omega**2 * contrast.Drho * amp_u * V * u_inc
    dsigma = _voigt_to_tensor(Dc @ eps_inc_V)

    # Scattering-plane basis (same construction as cube_far_field).
    ref_vec = np.array([0.0, 1.0, 0.0])  # k̂ = x̂ ⇒ |k̂[0]| ≥ 0.9
    perp1 = ref_vec - np.dot(ref_vec, k_hat) * k_hat
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(k_hat, perp1)

    f_P = np.zeros_like(theta, dtype=complex)
    f_SV = np.zeros_like(theta, dtype=complex)
    f_SH = np.zeros_like(theta, dtype=complex)
    for idx, th in enumerate(theta):
        r_hat = np.sin(th) * perp1 + np.cos(th) * k_hat
        rF = np.dot(r_hat, F)
        Sr = (V * dsigma) @ r_hat
        rSr = r_hat @ Sr

        Q_P = rF - 1j * kP * rSr
        f_P[idx] = -Q_P / (4.0 * np.pi * rho * ref.alpha**2)

        F_perp = F - rF * r_hat
        S_perp = Sr - rSr * r_hat
        Q_S = F_perp - 1j * kS * S_perp
        u_S = -Q_S / (4.0 * np.pi * rho * ref.beta**2)
        f_SV[idx] = np.dot(np.cos(th) * perp1 - np.sin(th) * k_hat, u_S)
        f_SH[idx] = np.dot(perp2, u_S)

    return f_P, f_SV, f_SH
