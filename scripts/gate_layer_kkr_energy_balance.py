#!/usr/bin/env python3
"""GATE: the layer-KKR sphere array conserves energy over EVERY open diffraction order.

WHAT THIS IS FOR.  ``gate_layer_kkr_sphere_array.py`` scores two columns of the
array's response -- P and SV at normal incidence -- against arbiters.  Energy
balance scores all of it at once and needs no arbiter: every incident channel
(P, SV, SH; every open order; from above and from below), every outgoing one.
The June chain (``Mathematica/IntraPlaneEnergyBalance.wl``) could only check
the specular S-matrix, because it forbade open orders; here the lattice is
super-wavelength and the diffracted orders carry flux.

WHY ONE COUPLING SERVES EVERY INCIDENCE.  An open order has lateral wavevector
G, a reciprocal-lattice vector, and e^{i G.R} = 1 on the lattice: every
incidence in the scattering matrix lies in the same Bloch sector as normal
incidence, so G0 at k_par = 0 couples them all.

WHY THE RESIDUAL DOES NOT MEASURE N_max.  For a lossless sphere each order's
Mie T is unitary, and the truncated Foldy-Lax problem conserves energy exactly
IF the anti-Hermitian part of G0 equals the far-field sum over open orders.
The residual therefore measures the lattice sums -- the reciprocal-space part
of the Ewald sum, the only place the open orders enter G0 -- and should sit at
their precision for every N_max.

THE S-MATRIX.  Reference plane z = 0 (the sphere centres) for incoming and
outgoing waves alike.  Channel (mode, order G, direction); unit displacement
along e_P = k^, e_SV = (u cos phi, u sin phi, -s), e_SH = (-sin phi, cos phi, 0)
at direction k^ = (s cos phi, s sin phi, u) -- the polarisations of
``layer_kkr.plane_wave_amplitude``.  Scattered amplitude into channel c is
(2 pi)^2/A times ``plane_wave_modes``; the direct wave adds the identity on the
transmitted channel.  Flux basis: w_c = sqrt(c^2 k_z/omega), two-sided, as in
the march's ``energy_residual`` (measured there at 8e-8).

THE CHECKS:
  [1] S^H S = I at eps = 0.1, at N_max = 2, 4, 6 and the Wiscombe order;
  [2] CONTROL: with the lattice coupling switched off (independent spheres) the
      same S is not unitary -- the check sees the coupling;
  [3] the diffracted orders carry a flux the check cannot pass without: the
      balance restricted to the specular channels fails;
  [4] a stronger contrast and a second period, with more open orders;
  [5] the SECOND IMPLEMENTATION: ``Mathematica/IntraPlaneEnergyBalanceOpenOrders.wl``
      builds the same S-matrix sharing no code with this one (CartesianT0 Mie,
      its own Ewald sums and projection, numerically extracted vector
      translations); the two must agree ENTRY BY ENTRY, not only in modulus.

Run:  conda run -n seismic python scripts/gate_layer_kkr_energy_balance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering import layer_kkr as lk  # noqa: E402
from scripts import gate_layer_kkr_sphere_array as arr  # noqa: E402

REF, OMEGA, RADIUS = arr.REF, arr.OMEGA, arr.RADIUS
KP, KS = OMEGA / REF.alpha, OMEGA / REF.beta

#: An order is open when k_z/k exceeds this; closer to grazing the flux weight
#: vanishes and the channel carries no energy either way.
GRAZING = 1e-3

MATHEMATICA_REF = ROOT / "Mathematica" / "IntraPlaneEnergyBalanceOpenOrders_reference.json"

_PASS: list[tuple[str, bool]] = []


def report(label: str, ok: bool) -> None:
    """Record and print one check.

    Args:
        label: What is checked.
        ok: Whether it passed.
    """
    _PASS.append((label, ok))
    print(f"  {'PASS' if ok else '****FAIL****'}  {label}")


def open_channels(period: float) -> list[tuple[str, float, float, bool]]:
    """Every propagating (mode, G_x, G_y, upward) of the square lattice's k_par = 0 sector.

    Args:
        period: Lattice period.

    Returns:
        Channels as (mode, gx, gy, upward), mode in "P", "SV", "SH".
    """
    b = 2.0 * np.pi / period
    jmax = int(KS / b) + 1
    out = []
    for upward in (False, True):
        for mode, k in (("P", KP), ("SV", KS), ("SH", KS)):
            for i in range(-jmax, jmax + 1):
                for j in range(-jmax, jmax + 1):
                    q = b * np.hypot(i, j)
                    if q < k and np.sqrt(k**2 - q**2) / k > GRAZING:
                        out.append((mode, b * i, b * j, upward))
    return out


def direction(mode: str, gx: float, gy: float, upward: bool) -> tuple[NDArray, NDArray | None]:
    """Unit propagation direction and (for S) polarisation of one channel.

    Args:
        mode: "P", "SV" or "SH".
        gx: Lateral wavenumber, x.
        gy: Lateral wavenumber, y.
        upward: Travelling towards z < 0.

    Returns:
        (k^, polarisation or None for P).
    """
    k = KP if mode == "P" else KS
    q = float(np.hypot(gx, gy))
    cph, sph = (gx / q, gy / q) if q > 0 else (1.0, 0.0)
    s = q / k
    u = np.sqrt(1.0 - s * s) * (-1.0 if upward else 1.0)
    khat = np.array([s * cph, s * sph, u])
    if mode == "P":
        return khat, None
    if mode == "SV":
        return khat, np.array([u * cph, u * sph, -s])
    return khat, np.array([-sph, cph, 0.0])


def flux_weight(mode: str, gx: float, gy: float) -> float:
    """sqrt(c^2 k_z / omega) -- the root of a unit-displacement wave's vertical flux.

    Args:
        mode: "P", "SV" or "SH".
        gx: Lateral wavenumber, x.
        gy: Lateral wavenumber, y.

    Returns:
        The weight.
    """
    c, k = (REF.alpha, KP) if mode == "P" else (REF.beta, KS)
    return float(np.sqrt(c**2 * np.sqrt(k**2 - gx**2 - gy**2) / OMEGA))


def s_matrix(period: float, eps: float, nmax: int, coupled: bool) -> NDArray:
    """The flux-basis scattering matrix over every open channel, reference plane z = 0.

    Args:
        period: Lattice period (square).
        eps: Contrast (``gate_layer_kkr_sphere_array.contrast``).
        nmax: Truncation order.
        coupled: False switches the lattice coupling off.

    Returns:
        S, shape (n_open, n_open); column = incident channel.
    """
    chans = open_channels(period)
    tm = lk.sphere_tmatrix(OMEGA, RADIUS, REF, arr.contrast(eps), nmax)
    g0 = arr.coupling(period, nmax)[0] if coupled else {}
    pref = (2.0 * np.pi) ** 2 / period**2
    gx = np.array([c[1] for c in chans])
    gy = np.array([c[2] for c in chans])
    w = np.array([flux_weight(m, x, y) for m, x, y, _ in chans])
    s = np.zeros((len(chans), len(chans)), dtype=complex)
    for j, (mode, x, y, up) in enumerate(chans):
        khat, pol = direction(mode, x, y, up)
        a_inc = lk.incident_coefficients("P" if mode == "P" else "S", khat, KP, KS, nmax, pol=pol)
        b = lk.solve_array(g0, tm, a_inc, nmax)
        for upward in (False, True):
            rows = [i for i, c in enumerate(chans) if c[3] == upward]
            amp = np.zeros((3, len(rows)), dtype=complex)
            for (fam, nn, m), coef in b.items():
                if coef == 0.0:
                    continue
                modes = lk.plane_wave_modes(fam, nn, m, gx[rows], gy[rows], KP, KS, upward)
                for c in range(3):
                    amp[c] += coef * modes[c]
            for r, i in enumerate(rows):
                s[i, j] = pref * amp[("P", "SV", "SH").index(chans[i][0]), r]
        s[j, j] += 1.0  # the unscattered wave, on its own channel
    return (w[:, None] * s) / w[None, :]


def residual(s: NDArray) -> float:
    """||S^H S - I||_F / sqrt(n).

    Args:
        s: Scattering matrix.

    Returns:
        The residual.
    """
    k = s.shape[0]
    return float(np.linalg.norm(s.conj().T @ s - np.eye(k)) / np.sqrt(k))


def census(period: float) -> str:
    """Open-channel count by mode, one direction.

    Args:
        period: Lattice period.

    Returns:
        Printable summary.
    """
    ch = [c for c in open_channels(period) if not c[3]]
    return ", ".join(f"{m} {sum(c[0] == m for c in ch)}" for m in ("P", "SV", "SH"))


def mathematica_s() -> tuple[NDArray, list[tuple[str, int, int, bool]], dict]:
    """The Mathematica chain's flux-basis S-matrix, its channels and parameters.

    Returns:
        (S, channels as (mode, i, j, upward) with G = 2 pi (i, j)/L, params).
    """
    if not MATHEMATICA_REF.exists():
        raise FileNotFoundError(
            f"missing the second implementation's dump {MATHEMATICA_REF}.\n"
            "  It is written by Mathematica/IntraPlaneEnergyBalanceOpenOrders.wl.  Recover with:\n"
            "  /Applications/Wolfram.app/Contents/MacOS/wolframscript -file "
            "Mathematica/IntraPlaneEnergyBalanceOpenOrders.wl 3"
        )
    d = json.loads(MATHEMATICA_REF.read_text())
    s = np.asarray(d["S"], dtype=float)
    chans = [(c["mode"], int(c["i"]), int(c["j"]), bool(c["upward"])) for c in d["channels"]]
    return s[..., 0] + 1j * s[..., 1], chans, d["params"]


def main() -> int:
    """Run the checks.

    Returns:
        0 if all pass.
    """
    period = 600.0
    nw = arr.march.compute_elastic_mie(OMEGA, RADIUS, REF, arr.march.MIE_CONTRAST).n_max
    print("=" * 78)
    print("LAYER-KKR ENERGY BALANCE OVER EVERY OPEN DIFFRACTION ORDER")
    print(f"  k_P a = {KP * RADIUS:.2f}   k_S a = {KS * RADIUS:.2f}   L = {period:.0f} m")
    print(f"  open orders per direction: {census(period)}  ({len(open_channels(period))} channels)")
    print("=" * 78)

    print("\n[1] S^H S = I, eps = 0.1")
    res = {}
    for nm in sorted({2, 4, 6, nw}):
        res[nm] = residual(s_matrix(period, 0.1, nm, coupled=True))
        print(f"      N_max {nm:2d}: {res[nm]:.2e}")
    report("the array conserves energy at every N_max", max(res.values()) < 1e-8)

    print("\n[2] CONTROL: the same spheres, coupling OFF")
    s_off = s_matrix(period, 0.1, nw, coupled=False)
    r_off = residual(s_off)
    print(f"      {r_off:.2e}")
    report("independent spheres do not conserve energy", r_off > 1e3 * res[nw])

    print("\n[3] CONTROL: the balance over the specular channels alone")
    s_on = s_matrix(period, 0.1, nw, coupled=True)
    spec = [i for i, c in enumerate(open_channels(period)) if c[1] == 0.0 and c[2] == 0.0]
    r_spec = residual(s_on[np.ix_(spec, spec)])
    diffracted = 1.0 - np.sum(np.abs(s_on[np.ix_(spec, spec)]) ** 2, axis=0)
    print(f"      specular-only residual {r_spec:.2e}; flux diffracted per specular incidence:")
    print("      " + "  ".join(f"{d:.2e}" for d in diffracted))
    report("the diffracted orders carry the missing flux", r_spec > 1e3 * res[nw])

    print("\n[4] a stronger contrast, and a longer period with more open orders")
    for per, eps in ((600.0, 0.3), (900.0, 0.1)):
        nm = arr.march.compute_elastic_mie(OMEGA, RADIUS, REF, arr.contrast(eps)).n_max
        r = residual(s_matrix(per, eps, nm, coupled=True))
        print(f"      L = {per:.0f}, eps = {eps}: {census(per)}; N_max {nm}: {r:.2e}")
        report(f"energy conserved at L = {per:.0f}, eps = {eps}", r < 1e-8)

    print("\n[5] against the independent Mathematica chain, entry by entry")
    s_m, ch_m, pm = mathematica_s()
    same = (
        abs(pm["aL"] / pm["radius"] - period / RADIUS) < 1e-12
        and abs(pm["kPa"] - KP * RADIUS) < 1e-12
        and abs(pm["eps"] - 0.1) < 1e-12
    )
    report("the Mathematica dump is of this same array", same)
    b = 2.0 * np.pi / period
    ch_py = [(m, round(x / b), round(y / b), up) for m, x, y, up in open_channels(period)]
    report("both enumerate the same open channels", sorted(ch_py) == sorted(ch_m))
    perm = [ch_py.index(c) for c in ch_m]
    s_py = s_matrix(period, 0.1, int(pm["Nmax"]), coupled=True)[np.ix_(perm, perm)]
    scat = s_py - np.eye(len(perm))
    diff = np.max(np.abs(s_m - s_py)) / np.max(np.abs(scat))
    print(f"      N_max {pm['Nmax']}: Mathematica S^H S - I {residual(s_m):.2e}")
    print(f"      max |S_Mathematica - S_Python| / max |S - I| = {diff:.2e}")
    report("the two implementations agree in complex value", diff < 1e-6)

    npass = sum(ok for _, ok in _PASS)
    print("\n" + "=" * 78)
    print(f"{npass}/{len(_PASS)} checks passed")
    print("=" * 78)
    return 0 if npass == len(_PASS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
