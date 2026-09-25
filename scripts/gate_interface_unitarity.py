#!/usr/bin/env python3
"""Interface S-matrix unitarity: kennett_layers against the thesis eigenbasis.

ANCHOR: Nestor (1996) Section 3.1, GRepresentations.tex -- (Peigen), (SVeigen),
(SHeigen), (epsdef); the Mathematica side is ``Mathematica/InterfaceUnitarity.wl``.

TWO INDEPENDENT ROUTES
----------------------
``psv_solid_solid`` / ``sh_solid_solid`` evaluate the Aki & Richards closed-form
coefficients with sqrt(eta rho) (P-SV) and sqrt(Z) (SH) flux normalisation.
``InterfaceUnitarity.wl`` solves displacement-traction continuity across z = 0
in the thesis energy-normalised eigenbasis, at 50 digits.  They share no code.

WHAT IS COMPARED
----------------
Both are flux-normalised, so they may differ only by a unit-modulus phase per
channel (the thesis and Kennett sign/phase conventions are not the same).  The
comparison is therefore restricted to quantities invariant under that
rephasing: the element magnitudes |S_ij| and the eigenvalues of S^dagger S.
Reciprocity is not compared: its form depends on the phase convention
(S = S^T in the Kennett basis, symplectic in the thesis basis).

NEGATIVE CONTROL
----------------
The error this gate exists to catch is a missing or wrong flux normalisation.
Each case is re-run with the Kennett S converted back to displacement-basis
coefficients, S_disp = W^{-1/2} S W^{1/2} with the vertical flux weights
W = rho v^2 eta per channel.  That comparison must FAIL, which shows the
tolerance discriminates rather than passing vacuously.

Run:  conda run -n seismic python scripts/gate_interface_unitarity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cubic_scattering.kennett_layers import (  # noqa: E402
    _complex_slowness,
    _vertical_slowness,
    psv_solid_solid,
    sh_solid_solid,
)

REFERENCE = Path(__file__).resolve().parents[1] / "Mathematica" / "InterfaceUnitarity_reference.json"
TOL = 1e-12


def _q(value: str) -> float:
    """Parse a Mathematica Q string ('Infinity' or an integer)."""
    return np.inf if value == "Infinity" else float(value)


def kennett_s(
    med1: list[float], med2: list[float], p: float, qp: float, qs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided interface S-matrices from kennett_layers, plus flux weights.

    Rows (up in medium 1; down in medium 2), columns (down in 1; up in 2),
    each block ordered (P, S), matching InterfaceUnitarity.wl.

    Returns:
        (S_psv 4x4, S_sh 2x2, W_psv 4, W_sh 2), where W = rho v^2 eta is the
        vertical flux weight of a unit-displacement wave in each channel.
    """
    (a1, b1, r1), (a2, b2, r2) = med1, med2
    sa1, sb1 = _complex_slowness(a1, qp), _complex_slowness(b1, qs)
    sa2, sb2 = _complex_slowness(a2, qp), _complex_slowness(b2, qs)
    e1, n1 = _vertical_slowness(sa1, p), _vertical_slowness(sb1, p)
    e2, n2 = _vertical_slowness(sa2, p), _vertical_slowness(sb2, p)
    c = psv_solid_solid(p, e1, n1, r1, 1 / sb1, e2, n2, r2, 1 / sb2)
    s = sh_solid_solid(n1, r1, 1 / sb1, n2, r2, 1 / sb2)
    s_psv = np.block([[c.Rd, c.Tu], [c.Td, c.Ru]])
    s_sh = np.array([[s.Rd, s.Tu], [s.Td, s.Ru]])
    wp1, ws1 = r1 * e1 / sa1**2, r1 * n1 / sb1**2
    wp2, ws2 = r2 * e2 / sa2**2, r2 * n2 / sb2**2
    return s_psv, s_sh, np.array([wp1, ws1, wp2, ws2]), np.array([ws1, ws2])


def gram_eig(s: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues of S^dagger S."""
    return np.sort(np.linalg.eigvalsh(s.conj().T @ s))


def to_displacement(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Undo the flux normalisation: S_disp = W^{-1/2} S W^{1/2}."""
    rw = np.sqrt(w.astype(complex))
    return s * rw[None, :] / rw[:, None]


def deviation(s_psv: np.ndarray, s_sh: np.ndarray, ref: dict) -> float:
    """Largest deviation in |S_ij| and eig(S^dagger S) from one reference case."""
    return max(
        np.abs(np.abs(s_psv) - np.array(ref["absSpsv"])).max(),
        np.abs(np.abs(s_sh) - np.array(ref["absSsh"])).max(),
        np.abs(gram_eig(s_psv) - np.array(ref["eigPSV"])).max(),
        np.abs(gram_eig(s_sh) - np.array(ref["eigSH"])).max(),
    )


def main() -> int:
    """Run the cross-check and the negative control; return the exit status."""
    if not REFERENCE.exists():
        print(f"FAIL: {REFERENCE} not found. Run: wolframscript -file Mathematica/InterfaceUnitarity.wl")
        return 1
    data = json.loads(REFERENCE.read_text())
    med1, med2 = data["medium1"], data["medium2"]
    print(f"==== interface unitarity: kennett_layers vs {data['source']} ====")
    print(f"  medium1 {med1}  medium2 {med2}  tol {TOL:.0e}")

    ok = True
    for ref in data["cases"]:
        s_psv, s_sh, w_psv, w_sh = kennett_s(med1, med2, ref["p"], _q(ref["QP"]), _q(ref["QS"]))
        dev = deviation(s_psv, s_sh, ref)
        ctrl = deviation(to_displacement(s_psv, w_psv), to_displacement(s_sh, w_sh), ref)
        passed, control = dev < TOL, ctrl > 1e3 * TOL
        ok &= passed and control
        print(
            f"  {ref['name']:<20s} |d| = {dev:.2e} -> {'PASS' if passed else 'FAIL'}"
            f"   displacement-basis control |d| = {ctrl:.2e} -> {'PASS' if control else 'FAIL'}"
        )
    print(f"  overall -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
