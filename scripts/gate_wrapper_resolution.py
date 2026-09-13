#!/usr/bin/env python3
"""Resolution of the 9x9 wrapper problem: three defects, and GATE F passing.

Companion to ``gate_displacement_only_calibration.py``, which established the
absolute reference.  This one states the complete correction and verifies it.

THE THREE DEFECTS
-----------------
D1  ``assemble_greens_6x6`` rescales the stress SOURCE columns on the
    displacement rows only::

        G_psv_perm[..., 2:, :]  *= miw     # stress rows
        G_psv_perm[..., :2, 2:] *= miw     # stress columns, disp rows ONLY

    so the stress-row x stress-column block carries ONE factor of ``(-i w)``
    where a consistent basis change ``D_row . G . D_col`` requires two.
    Fix: ``G6[3:6, 3:6] *= (-i w)``.

D2  The traction half of BOTH indices carries an eigenvector normalisation

        K3 = 1  (+)  ( -P_par + eta_S P_perp )

    on ``(sigma_zz, sigma_xz, sigma_yz)``, where ``P_par = khat khat^T`` and
    ``P_perp = I - P_par`` with ``khat = (kx,ky)/|k|``, and ``eta_S`` is the
    vertical S slowness.  It appears on the rows as ``K3`` and on the source
    columns as ``(-i w)^2 K3``; the displacement source columns carry ``-1``.
    K3 is NOT diagonal in (z,x,y) when kx ky != 0 -- which is why no diagonal
    source correction was ever going to exist.

D3  The source operator B must carry the force and the stress glut with
    OPPOSITE relative sign, matching the derived jump vector
    (``[T] = -F`` but ``[u] = +M_zz/(lam+2mu)``) against the source convention
    of ``exact_propagator_9x9``.  Fix: ``B -> B . diag(-I3, I6)``.

RESULT
------
GATE F, ``|| W M - (W M)^T || / || W M ||``, goes from 0.40-0.81 to ~1e-16,
both for the corrected 6x6 and for the exact whole-space reference.

Run:  conda run -n seismic python scripts/gate_wrapper_resolution.py
"""

import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.layered_correction import k_operator  # isort: skip
from scripts.gate_displacement_only_calibration import (  # isort: skip
    _vertical,
    homogeneous_model,
)
from GlobalMatrix.layered_greens import (  # isort: skip
    _interface_elastic_properties,
    layered_greens_6x6,
    strain_from_displacement_traction,
)

W9 = np.diag(np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float))
J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)
FLIP = np.diag(np.array([-1, -1, -1, 1, 1, 1, 1, 1, 1], dtype=float))
TOL = 1e-10

MODEL = homogeneous_model(q=2.0)
SP = MODEL.complex_slowness_p()[1]
SS = MODEL.complex_slowness_s()[1]
RHO = MODEL.rho[1]
MU = RHO / SS**2
LAM = RHO / SP**2 - 2 * MU


def _traction(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """T_iz for one plane-wave mode, index order (z, x, y); z is index 0."""
    div = 1j * np.dot(q, u)
    return np.array(
        [LAM * (1.0 if i == 0 else 0.0) * div + MU * 1j * (q[i] * u[0] + q[0] * u[i]) for i in range(3)],
        dtype=complex,
    )


def whole_space_jump_response(kx: float, ky: float, dz: float, w: float) -> np.ndarray:
    """Exact whole-space response to a unit state jump, basis (u, T).

    A jump S maps to a definite point source: the moment part reproduces the
    displacement jump while contributing nothing to the traction rows (because
    gam*(lam+2mu) = lam), and the force part carries the traction jump.

    Args:
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        dz: Receiver depth minus source depth (km), non-zero.
        w: Angular frequency (rad/s).

    Returns:
        6x6 in basis (u_z, u_x, u_y, T_zz, T_xz, T_yz).
    """
    nu_p = complex(_vertical((w * SP) ** 2, np.asarray(kx), np.asarray(ky)))
    nu_s = complex(_vertical((w * SS) ** 2, np.asarray(kx), np.asarray(ky)))
    sgn = 1.0 if dz > 0 else -1.0
    g_p = 1j * np.exp(1j * nu_p * abs(dz)) / (2 * nu_p)
    g_s = 1j * np.exp(1j * nu_s * abs(dz)) / (2 * nu_s)
    q_p = np.array([sgn * nu_p, kx, ky], dtype=complex)
    q_s = np.array([sgn * nu_s, kx, ky], dtype=complex)
    gp = np.outer(q_p, q_p) * g_p / (RHO * w**2)
    gs = (-np.outer(q_s, q_s) + np.eye(3) * (w * SS) ** 2) * g_s / (RHO * w**2)

    out = np.zeros((6, 6), dtype=complex)
    for col in range(6):
        s = np.zeros(6)
        s[col] = 1.0
        mom = np.zeros((3, 3), dtype=complex)
        mom[0, 0] = (LAM + 2 * MU) * s[0]
        mom[1, 1] = mom[2, 2] = LAM * s[0]
        mom[0, 1] = mom[1, 0] = MU * s[1]
        mom[0, 2] = mom[2, 0] = MU * s[2]
        force = -s[3:6].astype(complex)
        u_p = gp @ force - 1j * np.einsum("jk,k,ij->i", mom, q_p, gp)
        u_s = gs @ force - 1j * np.einsum("jk,k,ij->i", mom, q_s, gs)
        out[0:3, col] = u_p + u_s
        out[3:6, col] = _traction(q_p, u_p) + _traction(q_s, u_s)
    return out


def k3(kx: float, ky: float, w: float) -> np.ndarray:
    """The traction-half normalisation operator, in (z, x, y).

    Thin wrapper over the package implementation, kept so this script reads as a
    self-contained statement of the result.

    Args:
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        w: Angular frequency (rad/s).

    Returns:
        3x3 acting on (sigma_zz, sigma_xz, sigma_yz).
    """
    return k_operator(SS, kx, ky, w)


def corrected_6x6(w: float, kx: float, ky: float, j: int = 9, i: int = 8) -> np.ndarray:
    """G6 with D1 and D2 applied; equals whole_space_jump_response in the limit.

    Args:
        w: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        j: Source interface index.
        i: Receiver interface index.

    Returns:
        Corrected 6x6.
    """
    miw = -1j * w
    g = layered_greens_6x6(MODEL, w, np.array([kx]), np.array([ky]), source_iface=j, receiver_iface=i)[
        0
    ].copy()
    g[3:, 3:] *= miw  # D1
    kk = k3(kx, ky, w)
    lrow = np.eye(6, dtype=complex)
    lrow[3:, 3:] = kk
    scol = np.eye(6, dtype=complex)
    scol[0:3, 0:3] = -np.eye(3)
    scol[3:, 3:] = -(miw**2) * kk
    return lrow @ g @ la.inv(scol)  # D2


def wrap9(g6: np.ndarray, w: float, kx: float, ky: float, j: int, i: int, flip: bool) -> np.ndarray:
    """A . G . B, with D3 optionally applied to B.

    Args:
        g6: 6x6 in basis (u, T).
        w: Angular frequency (rad/s).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        j: Source interface index.
        i: Receiver interface index.
        flip: Apply the D3 force/glut relative sign.

    Returns:
        9x9 in basis (u, eps_Voigt).
    """
    rho_r, al_r, be_r = _interface_elastic_properties(MODEL, i)
    a = strain_from_displacement_traction(np.array([kx]), np.array([ky]), rho_r, al_r, be_r)[0]
    rho_s, al_s, be_s = _interface_elastic_properties(MODEL, j)
    am = strain_from_displacement_traction(np.array([-kx]), np.array([-ky]), rho_s, al_s, be_s)[0]
    b = -J6 @ am.T @ W9
    if flip:
        b = b @ FLIP
    return a @ g6 @ b


def gate_f(m: np.ndarray) -> float:
    """GATE F residual for a 9x9.

    Args:
        m: The 9x9.

    Returns:
        ||W M - (W M)^T|| / ||W M||.
    """
    wm = W9 @ m
    return float(la.norm(wm - wm.T) / la.norm(wm))


def gate_d_homogeneous() -> float:
    """GATE D on the corrected 6x6, in the limit the correction was calibrated in.

    Prediction (doc section 3): the weight must lose its (-i w)^{+-1} factors,
    leaving Wapenaar's  G(x_A,x_B) = N G^T(x_B,x_A) N  with N = J6.  Measured:
    it loses the parity too, so the law is the bare one with NO weight at all.

    Returns:
        Worst relative residual.
    """
    print("=" * 78)
    print("PART 3 - GATE D on the corrected 6x6 (homogeneous limit)")
    print("  predicted:  G(i<-j)(+k) = J6 [ G(j<-i)(-k) ]^T J6   with NO weight")
    print("=" * 78)
    worst = 0.0
    for f in (12.0, 48.0):
        for p, c, s in ((0.12, 0.6, 0.8), (0.20, 0.6, 0.8)):
            w = 2 * np.pi * f
            kx, ky = w * p * c, w * p * s
            for tag, g1, g2 in (
                (
                    "exact ref",
                    whole_space_jump_response(kx, ky, -1.0, w),
                    whole_space_jump_response(-kx, -ky, +1.0, w),
                ),
                (
                    "corrected G6",
                    corrected_6x6(w, kx, ky, j=9, i=8),
                    corrected_6x6(w, -kx, -ky, j=8, i=9),
                ),
            ):
                rel = float(la.norm(g1 - J6 @ g2.T @ J6) / la.norm(g1))
                worst = max(worst, rel)
                print(f"  f={f:5.1f} p={p:5.2f} khat=({c},{s})  {tag:13s} rel = {rel:.3e}")
    print(f"\n  GATE D (homogeneous): {'PASS' if worst < TOL else 'FAIL'}  (worst {worst:.3e})\n")
    return worst


def gate_e_swap() -> float:
    """GATE E - the source<->receiver swap relation, on the corrected objects.

    ``M(i<-j)(+k) = W^-1 [ M(j<-i)(-k) ]^T W``.  The shipped construction passed
    this at 9.7e-16 while failing GATE F, because the relation is homogeneous of
    degree one and so blind to the overall scale and to within-matrix structure.
    It must still hold after the fix: a correction that closed GATE F by
    breaking GATE E would have traded one defect for another.

    Returns:
        Worst relative residual.
    """
    print("=" * 78)
    print("PART 4 - GATE E   M(i<-j)(+k) = W^-1 [M(j<-i)(-k)]^T W")
    print("  REPORTED, NOT GATED: this is a FALSE gate.  See the note below.")
    print("=" * 78)

    def resid(flip: bool, exact: bool, w: float, kx: float, ky: float) -> float:
        if exact:
            g1 = whole_space_jump_response(kx, ky, -1.0, w)
            g2 = whole_space_jump_response(-kx, -ky, +1.0, w)
        else:
            g1 = corrected_6x6(w, kx, ky, j=9, i=8)
            g2 = corrected_6x6(w, -kx, -ky, j=8, i=9)
        m1 = wrap9(g1, w, kx, ky, 9, 8, flip=flip)
        m2 = wrap9(g2, w, -kx, -ky, 8, 9, flip=flip)
        return float(la.norm(m1 - la.inv(W9) @ m2.T @ W9) / la.norm(la.inv(W9) @ m2.T @ W9))

    print(f"{'f':>6} {'p':>6}   {'corrected+D3':>13} {'exact ref+D3':>13} {'exact, no D3':>13}")
    print("-" * 78)
    worst = 0.0
    for f in (12.0, 48.0):
        for p, c, s in ((0.05, 0.6, 0.8), (0.12, 0.6, 0.8), (0.20, 1.0, 0.0)):
            w = 2 * np.pi * f
            kx, ky = w * p * c, w * p * s
            r_corr = resid(True, False, w, kx, ky)
            worst = max(worst, r_corr)
            print(
                f"{f:6.1f} {p:6.2f}   {r_corr:13.4e} "
                f"{resid(True, True, w, kx, ky):13.4e} "
                f"{resid(False, True, w, kx, ky):13.4e}"
            )
    print()
    print("  READ THE OTHER TWO COLUMNS BEFORE CONCLUDING FROM THE FIRST.")
    print("  The EXACT reference fails GATE E by the same amount, and passes it")
    print("  once D3 is removed.  GATE E and GATE F are MUTUALLY EXCLUSIVE:")
    print("  GATE F gives M1 = W^-1 M1^T W and GATE E gives M1 = W^-1 M2^T W, so")
    print("  both at once force M1 = M2, i.e. P(r) = P(-r).  But C and H are FIRST")
    print("  derivatives of the Green's tensor and so are ODD in r.  The two")
    print("  relations differ by exactly a sign on the mixed blocks -- which IS")
    print("  defect D3.  GATE A, measured on the validated closed form at twelve")
    print("  random separations, selects the GATE F convention; GATE E as written")
    print("  is the mis-stated one.  This also explains negative result N1.\n")
    return worst


def main() -> int:
    """Verify the correction and re-run GATE F, GATE D and GATE E.

    Returns:
        0 if every gate passes, 1 otherwise.
    """
    ok = True
    print("=" * 78)
    print("PART 1 - corrected 6x6 vs the exact whole-space jump response")
    print("=" * 78)
    for f in (48.0,):
        for p, c, s in ((0.12, 1.0, 0.0), (0.12, 0.6, 0.8), (0.20, 0.6, 0.8)):
            w = 2 * np.pi * f
            kx, ky = w * p * c, w * p * s
            gc = corrected_6x6(w, kx, ky)
            gt = whole_space_jump_response(kx, ky, -1.0, w)
            rel = float(la.norm(gc - gt) / la.norm(gt))
            ok = ok and rel < 1e-6
            print(f"  f={f:5.1f}  p={p:5.2f}  khat=({c},{s})   rel = {rel:.3e}")

    print()
    print("=" * 78)
    print("PART 2 - GATE F   || W M - (W M)^T || / || W M ||   (shipped: 0.40-0.81)")
    print("=" * 78)
    print(f"{'f (Hz)':>7} {'p':>6}   {'no D3 (sign)':>14}  {'all three fixes':>16}  {'exact ref + D3':>16}")
    print("-" * 78)
    worst = 0.0
    for f in (12.0, 48.0, 192.0):
        for p, c, s in ((0.05, 0.6, 0.8), (0.12, 0.6, 0.8), (0.20, 1.0, 0.0)):
            w = 2 * np.pi * f
            kx, ky = w * p * c, w * p * s
            gc = corrected_6x6(w, kx, ky)
            gt = whole_space_jump_response(kx, ky, -1.0, w)
            r0 = gate_f(wrap9(gc, w, kx, ky, 9, 8, flip=False))
            r1 = gate_f(wrap9(gc, w, kx, ky, 9, 8, flip=True))
            r2 = gate_f(wrap9(gt, w, kx, ky, 9, 8, flip=True))
            worst = max(worst, r1, r2)
            print(f"{f:7.1f} {p:6.2f}   {r0:14.4e}  {r1:16.4e}  {r2:16.4e}")
    ok = ok and worst < TOL
    print(f"\n  GATE F: {'PASS' if worst < TOL else 'FAIL'}  (worst {worst:.3e})\n")

    ok = ok and gate_d_homogeneous() < TOL
    # GATE E is reported but NOT gated: it is incompatible with GATE F, and the
    # exact reference fails it identically.  See gate_e_swap.
    gate_e_swap()

    print("=" * 78)
    print("SCOPE")
    print("=" * 78)
    print("  The correction carries over to STRATIFIED media unchanged -- see")
    print("  scripts/gate_stratified_correction.py, which holds the clean law to")
    print("  1.7e-15 with a fast slab crossed twice between the planes.  The one")
    print("  restriction is that a source or receiver plane must not coincide with")
    print("  a material discontinuity, where eta_S is two-valued and K is simply")
    print("  not defined.  cubic_scattering.layered_correction enforces that.")
    print("  Scattering voxels live in layer interiors, so it costs nothing.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
