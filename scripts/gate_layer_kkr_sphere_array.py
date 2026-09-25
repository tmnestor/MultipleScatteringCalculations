#!/usr/bin/env python3
"""GATE: an exact multiple-scattering solution for the periodic ARRAY of spheres.

WHAT THIS IS FOR.  The impedance march solves a square array of spheres; exact
Mie solves ONE.  Until now the march could only be scored against the isolated
sphere, and its residual attributed to the array's coupling by argument (and at
small contrast by second-order Born).  Layer-KKR solves the ARRAY exactly -- at
any contrast, to a truncation order N_max that is measured -- so the march can
be scored against the problem it actually solves.

THE SOLVE (``cubic_scattering/layer_kkr.py``).  Per sphere, outgoing
coefficients b = T (I - G0 T)^-1 a_inc, with T the Mie T-matrix of each order
(``mie_tmatrix_psv/sh``), G0 the lattice coupling built from the planar
structure constants (``planar_kambe``), and a_inc the incident plane wave's
regular multipoles.  Each diffraction order then carries (2 pi)^2/A times the
plane-wave spectrum of the b-multipoles, with the same phase conventions as the
march gate's Mie predictions.

THE CHECKS, in order of how much each assumes:
  [1] G0 = 0 reproduces the march gate's isolated-sphere predictions in complex
      value at every order, for P and for SV incidence -- the T-matrix mapping,
      the incident expansion, the spectrum and the phases, all at once;
  [2] convergence in N_max at eps = 0.1;
  [3] at eps -> 0 the array's departure from the isolated sphere is the
      second-order Born prediction, 0.2880 - 0.0556i per unit eps;
  [4] against the MARCH: its residual from this solution must fall under
      lateral refinement, where its residual from Mie does not.

Run:  conda run -n seismic python scripts/gate_layer_kkr_sphere_array.py
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering import layer_kkr as lk  # noqa: E402
from cubic_scattering.effective_contrasts import MaterialContrast  # noqa: E402
from cubic_scattering.planar_kambe import structure_constants  # noqa: E402
from scripts import gate_sphere_vs_impedance_march as march  # noqa: E402
from scripts.gate_sphere_plane_wave_spectrum import kz_of  # noqa: E402

REF, OMEGA, RADIUS = march.REF, march.OMEGA, march.RADIUS
KP, KS = OMEGA / REF.alpha, OMEGA / REF.beta
BORN2 = 0.288013 - 0.055596j  # gate_born2_array_coupling.py, 2.5 diameters

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: What is checked.
        ok: Whether it passed.
    """
    _PASS.append((label, ok))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def contrast(eps: float) -> MaterialContrast:
    """alpha, beta, rho all scaled by 1 + eps inside the sphere.

    Args:
        eps: Fractional perturbation.

    Returns:
        The contrast.
    """
    s = 1.0 + eps
    return MaterialContrast(
        Dlambda=(s**3 - 1.0) * REF.lam, Dmu=(s**3 - 1.0) * REF.mu, Drho=(s - 1.0) * REF.rho
    )


@functools.cache
def coupling(period: float, nmax: int, omega: float = OMEGA) -> tuple:
    """The lattice coupling for a square lattice at normal incidence.

    The projection radius is 0.25 L, but no more than 2/k_S: the M, N
    projection divides by j_nu(k_S rho0), and past k_S rho0 = 4.49 (the first
    zero of j_1) that division would be by a near-zero.

    Args:
        period: Lattice period.
        nmax: Truncation order.
        omega: Angular frequency.

    Returns:
        (G0, sources) from ``vector_coupling``.
    """
    k_p, k_s = omega / REF.alpha, omega / REF.beta
    extra = lk.DEFAULT_N_EVAL_EXTRA
    qmax = 2 * nmax + extra
    kpar = np.zeros(2)
    d_p = structure_constants(k_p, qmax, period, kpar)
    d_s = structure_constants(k_s, qmax, period, kpar)
    rho0 = min(0.25 * period, 2.0 / k_s)
    return lk.vector_coupling(d_p, d_s, k_p, k_s, nmax, rho0=rho0, n_eval_extra=extra)


def kkr_columns(
    n_grid: int, period: float, eps: float, incident: str, nmax: int, coupled: bool, omega: float = OMEGA
) -> tuple:
    """R and T columns of the array, in the march gate's (mode, order) layout.

    Args:
        n_grid: Lateral grid size (square).
        period: Period (square).
        eps: Contrast.
        incident: "P" or "SV" (x-polarised S) at normal incidence.
        nmax: Truncation order.
        coupled: False switches the lattice coupling off (the isolated sphere).
        omega: Angular frequency.

    Returns:
        (R column, T column), each shape (3 n_grid^2,).
    """
    KP, KS = omega / REF.alpha, omega / REF.beta  # noqa: N806
    tm = lk.sphere_tmatrix(omega, RADIUS, REF, contrast(eps), nmax)
    down = np.array([0.0, 0.0, 1.0])
    if incident == "P":
        a_inc = lk.incident_coefficients("P", down, KP, KS, nmax)
        k_in, slot = KP, 0
    else:
        a_inc = lk.incident_coefficients("S", down, KP, KS, nmax, pol=np.array([1.0, 0.0, 0.0]))
        k_in, slot = KS, 1
    g0 = coupling(period, nmax, omega)[0] if coupled else {}
    b = lk.solve_array(g0, tm, a_inc, nmax)

    n = n_grid * n_grid
    kx = np.repeat(march.grid_wavenumbers(n_grid, period), n_grid)
    ky = np.tile(march.grid_wavenumbers(n_grid, period), n_grid)
    q = np.hypot(kx, ky)
    kz = {"P": kz_of(q, KP), "S": kz_of(q, KS)}
    pref = (2.0 * np.pi) ** 2 / period**2 * march.centre_phase(n_grid, n_grid, period, period)
    cols = []
    for upward in (True, False):
        c = np.zeros((3, n), dtype=complex)
        for (fam, nn, m), coef in b.items():
            if coef == 0.0:
                continue
            modes = lk.plane_wave_modes(fam, nn, m, kx, ky, KP, KS, upward)
            for i in range(3):
                c[i] += coef * modes[i]
        col = np.zeros(3 * n, dtype=complex)
        col[:n] = pref * c[0] * np.exp(1j * (k_in + kz["P"]) * RADIUS)
        col[n : 2 * n] = pref * c[1] * np.exp(1j * (k_in + kz["S"]) * RADIUS)
        col[2 * n :] = pref * c[2] * np.exp(1j * (k_in + kz["S"]) * RADIUS)
        cols.append(col)
    cols[1][slot * n] += np.exp(2j * k_in * RADIUS)  # the unscattered wave
    return cols[0], cols[1]


def _keep(n_grid: int) -> NDArray:
    return ~np.tile(march.nyquist_orders(n_grid, n_grid), 3)


def main() -> int:
    """Run the checks.

    Returns:
        0 if all pass.
    """
    print("=" * 78)
    print("LAYER-KKR: THE PERIODIC ARRAY OF SPHERES, SOLVED EXACTLY")
    print(f"  k_P a = {KP * RADIUS:.2f}   k_S a = {KS * RADIUS:.2f}   contrast 0.1")
    print("=" * 78)

    n_grid, period = 6, 600.0
    # The same truncation as compute_elastic_mie's own (Wiscombe) cutoff, so
    # that [1] compares like with like: at N_max = 8 the omitted n = 9, 10
    # terms alone leave 1e-7 to 5e-6.
    nmax = march.compute_elastic_mie(OMEGA, RADIUS, REF, march.MIE_CONTRAST).n_max
    keep = _keep(n_grid)

    print(f"\n[1] with the coupling OFF (N_max = {nmax}): against the march gate's Mie columns")
    r_p, t_p = kkr_columns(n_grid, period, 0.1, "P", nmax, coupled=False)
    want_r, want_t = (
        march.mie_prediction(n_grid, n_grid, period, period),
        march.mie_transmission(n_grid, n_grid, period, period),
    )
    r_s, t_s = kkr_columns(n_grid, period, 0.1, "SV", nmax, coupled=False)
    want_rs, want_ts = march.mie_columns_s(n_grid, n_grid, period, period)
    errs = {
        "P  R": march.l2_rel(r_p[keep], want_r[keep]),
        "P  T": march.l2_rel(t_p[keep], want_t[keep]),
        "SV R": march.l2_rel(r_s[keep], want_rs[keep]),
        "SV T": march.l2_rel(t_s[keep], want_ts[keep]),
    }
    for k, v in errs.items():
        print(f"      {k}: {v:.2e}")
    report("G0 = 0 reproduces the isolated sphere at every order", max(errs.values()) < 1e-8)

    print("\n[2] with the coupling ON: convergence in N_max (eps = 0.1, P incidence)")
    prev = None
    cols = {}
    for nm in (4, 6, 8, 10):
        cols[nm] = kkr_columns(n_grid, period, 0.1, "P", nm, coupled=True)
        if prev is not None:
            d = march.l2_rel(
                np.concatenate(cols[nm])[np.tile(keep, 2)], np.concatenate(prev)[np.tile(keep, 2)]
            )
            print(f"      N_max {nm}: change from the previous {d:.2e}")
        prev = cols[nm]
    report("the array response converges in N_max", d < 1e-6)

    print("\n[3] eps -> 0: the array's departure from the isolated sphere, against second-order Born")
    eps = 1e-3
    rk, _ = kkr_columns(n_grid, period, eps, "P", nmax, coupled=True)
    rm, _ = kkr_columns(n_grid, period, eps, "P", nmax, coupled=False)
    dep = (rk[0] - rm[0]) / rm[0] / eps
    print(f"      KKR departure/eps {dep:.5f}   second-order Born {BORN2:.5f}")
    report(
        "the array coupling at small contrast is the Born-2 prediction",
        abs(dep - BORN2) < 0.01 * abs(BORN2),
    )

    print("\n[4] against the MARCH, which solves this same periodic problem (eps = 0.1, 128 depth steps)")
    print("      complex residual over the P and SV columns, R and T, scoreable orders")
    print(f"      {'N':>4}{'march vs KKR':>16}{'march vs Mie':>16}")
    vs_kkr, vs_mie = [], []
    for ng in (6, 9, 12):
        nn = ng * ng
        r_mat, t_mat = march.reflection_transmission(ng, ng, period, period, 128)
        got = np.concatenate([r_mat[:, 0], t_mat[:, 0], r_mat[:, nn], t_mat[:, nn]])
        kkr = np.concatenate(
            [
                *kkr_columns(ng, period, 0.1, "P", nmax, True),
                *kkr_columns(ng, period, 0.1, "SV", nmax, True),
            ]
        )
        mie = np.concatenate(
            [
                *kkr_columns(ng, period, 0.1, "P", nmax, False),
                *kkr_columns(ng, period, 0.1, "SV", nmax, False),
            ]
        )
        # The unscattered wave is identical on all three and would dilute T.
        direct = np.zeros_like(got)
        direct[3 * nn] = np.exp(2j * KP * RADIUS)
        direct[9 * nn + nn] = np.exp(2j * KS * RADIUS)
        k4 = np.tile(_keep(ng), 4)
        vs_kkr.append(march.l2_rel((got - direct)[k4], (kkr - direct)[k4]))
        vs_mie.append(march.l2_rel((got - direct)[k4], (mie - direct)[k4]))
        print(f"      {ng:4d}{vs_kkr[-1]:16.4e}{vs_mie[-1]:16.4e}")
    report("the march converges towards the exact array solution", vs_kkr[-1] < vs_kkr[0])
    report("and ends closer to it than to the isolated sphere", vs_kkr[-1] < 0.5 * vs_mie[-1])

    npass = sum(ok for _, ok in _PASS)
    print("\n" + "=" * 78)
    print(f"{npass}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if npass == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
