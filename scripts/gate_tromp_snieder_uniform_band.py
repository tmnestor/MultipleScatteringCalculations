#!/usr/bin/env python3
"""Tromp & Snieder (1989) invariant-imbedding Riccati equations, scored against
the Kennett recursion on a uniform band.

THE CLAIM UNDER TEST
--------------------
Tromp & Snieder, Geophys. J. 96, 447-456 (1989), build R/T Riccati equations
from the FIRST-BORN response of a thin homogeneous layer (their eqs 2.12a-f,
2.15a-f), arguing (below their Fig. 2) that the n-th Born term of a layer of
width dz is O(dz^n), so that first Born is exact at O(dz) and the Riccati
equations are exact.  Their coefficients are LINEAR in the density and
stiffness perturbations.

The exact generator is not linear in stiffness: across a thin layer the
normal traction is continuous, so the vertical strain inside it is rescaled by
M0/M1 at O(1), not O(dz).  The prediction is therefore:

  * density-only band      -> exact (Riccati == Kennett to integrator tolerance)
  * stiffness contrast     -> correct at first order, wrong at O(eps^2)
  * p = 0, any contrast    -> EXACTLY Kennett for a band whose moduli are
                              replaced by the linearised compliance
                              M_eff = M0^2 / (M0 - Ms)   (M = lambda+2mu, and mu)

WHAT IS COMPARED
----------------
The Riccati solution uses displacement-normalised amplitudes (their eqs 2.3,
2.4); `kennett_layers` returns the flux-normalised ("modified") convention.
The two are related by a diagonal similarity, so only similarity invariants
are compared: RPP, RSS, RSH and the product RPS*RSP.

Run:
    conda run -n seismic python scripts/gate_tromp_snieder_uniform_band.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubic_scattering.kennett_layers import IsotropicLayer, LayerStack, kennett_layers  # noqa: E402

# Background: the project's validated test parameters (SI).
ALPHA0, BETA0, RHO0 = 5000.0, 3000.0, 2500.0
MU0 = RHO0 * BETA0**2
LAM0 = RHO0 * ALPHA0**2 - 2.0 * MU0
M0 = LAM0 + 2.0 * MU0

OMEGA = 2.0 * np.pi * 10.0
H_BAND = 100.0
P_LIST = (0.0, 0.8e-4, 1.5e-4)  # s/m; all sub-critical for both P and S


@dataclass(frozen=True)
class Contrast:
    """Perturbation of the band from the background (SI units)."""

    dlam: float
    dmu: float
    drho: float
    label: str


def _stiffness(dlam: float, dmu: float) -> NDArray:
    """Isotropic c_ijkl for (dlam, dmu), indices 0,1,2 = x,y,z (the paper's 1,2,3)."""
    d = np.eye(3)
    return dlam * np.einsum("ij,kl->ijkl", d, d) + dmu * (
        np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)
    )


def _modes(k: float, down: bool) -> list[tuple[NDArray, NDArray, float, float]]:
    """(polarisation, wavevector, nu, modulus) for P, SV, SH; eqs (2.3)-(2.4)."""
    nu_a = np.sqrt((OMEGA / ALPHA0) ** 2 - k**2)
    nu_b = np.sqrt((OMEGA / BETA0) ** 2 - k**2)
    s = 1.0 if down else -1.0
    pol_p = (ALPHA0 / OMEGA) * np.array([k, 0.0, s * nu_a])
    pol_sv = (BETA0 / OMEGA) * np.array([-s * nu_b, 0.0, k])
    pol_sh = np.array([0.0, 1.0, 0.0])
    return [
        (pol_p, np.array([k, 0.0, s * nu_a]), nu_a, M0),
        (pol_sv, np.array([k, 0.0, s * nu_b]), nu_b, MU0),
        (pol_sh, np.array([k, 0.0, s * nu_b]), nu_b, MU0),
    ]


def _thin_layer(k: float, c: Contrast, in_down: bool, out_down: bool) -> NDArray:
    """Born thin-layer matrix per unit width, [out, in], WITHOUT the z0 phase.

    Eqs (2.12a-f) / (2.15a-f):
        p_out_i [rho_s w^2 d_ij - c_iljm kout_l kin_m] p_in_j * i / (2 M_out nu_out)
    The phase exp(i(kin_z - kout_z) z0) is returned separately by `_phase_rates`.
    """
    cs = _stiffness(c.dlam, c.dmu)
    ins = _modes(k, in_down)
    outs = _modes(k, out_down)
    m = np.zeros((3, 3), dtype=complex)
    for a, (po, ko, nuo, mod_o) in enumerate(outs):
        for b, (pi, ki, _, _) in enumerate(ins):
            kernel = c.drho * OMEGA**2 * np.eye(3) - np.einsum("iljm,l,m->ij", cs, ko, ki)
            m[a, b] = po @ kernel @ pi * 1j / (2.0 * mod_o * nuo)
    return m


def _phase_rates(k: float, in_down: bool, out_down: bool) -> NDArray:
    """kin_z - kout_z for each [out, in] pair."""
    kin = np.array([w[1][2] for w in _modes(k, in_down)])
    kout = np.array([w[1][2] for w in _modes(k, out_down)])
    return kin[np.newaxis, :] - kout[:, np.newaxis]


def tromp_snieder_band(c: Contrast, p: float, h: float = H_BAND) -> NDArray:
    """Integrate the invariant-imbedding Riccati system across a uniform band.

    Thin layer added at the bottom (depth z) of the band [0, z]; with
    R_B = rU dz, T_B = I + tD dz, R'_B = rD dz, T'_B = I + tU dz the addition
    rule gives
        Td' = tD Td + Ru rU Td        Rd' = Tu rU Td
        Tu' = Tu tU + Tu rU Ru        Ru' = rD + tD Ru + Ru tU + Ru rU Ru
    (Rd, Ru: reflection for incidence from above / below.)  Rd' = Tu rU Td is
    their eq (3.5).  Returns Rd, [out, in], order (P, SV, SH).
    """
    k = OMEGA * p
    blocks = {
        "tD": (_thin_layer(k, c, True, True), _phase_rates(k, True, True)),
        "rU": (_thin_layer(k, c, True, False), _phase_rates(k, True, False)),
        "tU": (_thin_layer(k, c, False, False), _phase_rates(k, False, False)),
        "rD": (_thin_layer(k, c, False, True), _phase_rates(k, False, True)),
    }

    def at(name: str, z: float) -> NDArray:
        base, rate = blocks[name]
        return base * np.exp(1j * rate * z)

    def rhs(z: float, y: NDArray) -> NDArray:
        td, rd, tu, ru = y.reshape(4, 3, 3)
        t_d, r_u, t_u, r_d = at("tD", z), at("rU", z), at("tU", z), at("rD", z)
        d_td = t_d @ td + ru @ r_u @ td
        d_rd = tu @ r_u @ td
        d_tu = tu @ t_u + tu @ r_u @ ru
        d_ru = r_d + t_d @ ru + ru @ t_u + ru @ r_u @ ru
        return np.stack([d_td, d_rd, d_tu, d_ru]).ravel()

    eye = np.eye(3, dtype=complex)
    zero = np.zeros((3, 3), dtype=complex)
    y0 = np.stack([eye, zero, eye, zero]).ravel()
    sol = solve_ivp(rhs, (0.0, h), y0, method="DOP853", rtol=1e-12, atol=1e-14)
    if not sol.success:
        msg = f"Riccati integration failed: {sol.message}"
        raise RuntimeError(msg)
    return sol.y[:, -1].reshape(4, 3, 3)[1]


def kennett_band(dlam: float, dmu: float, drho: float, p: float, h: float = H_BAND) -> NDArray:
    """Kennett RD for background / uniform band / background.  (2x2 P-SV, SH)."""
    rho1 = RHO0 + drho
    mu1 = MU0 + dmu
    m1 = M0 + dlam + 2.0 * dmu
    band = IsotropicLayer(alpha=np.sqrt(m1 / rho1), beta=np.sqrt(mu1 / rho1), rho=rho1, thickness=h)
    top = IsotropicLayer(alpha=ALPHA0, beta=BETA0, rho=RHO0, thickness=1.0)
    half = IsotropicLayer(alpha=ALPHA0, beta=BETA0, rho=RHO0, thickness=np.inf)
    res = kennett_layers(LayerStack(layers=[top, band, half]), p, np.array([OMEGA]))
    out = np.zeros((3, 3), dtype=complex)
    out[:2, :2] = res.RD_psv[0]
    out[2, 2] = res.RD_sh[0]
    return out


def invariants(r: NDArray, p: float) -> dict[str, complex]:
    """Similarity invariants; the P-SV off-diagonals vanish at p = 0."""
    inv = {"RPP": r[0, 0], "RSS": r[1, 1], "RSH": r[2, 2]}
    if p > 0:
        inv["RPS*RSP"] = r[0, 1] * r[1, 0]
    return inv


def rel_err(ts: NDArray, ke: NDArray, p: float) -> dict[str, float]:
    """Relative error of each invariant."""
    a, b = invariants(ts, p), invariants(ke, p)
    return {key: abs(a[key] - b[key]) / abs(b[key]) for key in b}


def worst(errs: dict[str, float]) -> float:
    return max(errs.values())


def main() -> int:
    passed = 0
    total = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed, total
        total += 1
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"Background alpha={ALPHA0} beta={BETA0} rho={RHO0}; band H={H_BAND} m; f=10 Hz")

    # 1. Density-only: the harness.  Their construction must be exact here.
    print("\n1. Density-only band (prediction: exact)")
    for drho in (0.1 * RHO0, 0.4 * RHO0):
        c = Contrast(0.0, 0.0, drho, f"drho={drho / RHO0:.0%}")
        for p in P_LIST:
            e = rel_err(tromp_snieder_band(c, p), kennett_band(0.0, 0.0, drho, p), p)
            check(f"{c.label} p={p:.1e}", worst(e) < 1e-7, f"worst rel err {worst(e):.2e}")

    # 2. Stiffness-only, eps -> 0: correct at first order means rel err ~ eps.
    print("\n2. Stiffness-only, eps -> 0 (prediction: rel err proportional to eps)")
    eps_list = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
    for p in P_LIST:
        errs = []
        for eps in eps_list:
            c = Contrast(eps * LAM0, eps * MU0, 0.0, "")
            ke = kennett_band(eps * LAM0, eps * MU0, 0.0, p)
            errs.append(worst(rel_err(tromp_snieder_band(c, p), ke, p)))
        slope = np.polyfit(np.log(eps_list), np.log(errs), 1)[0]
        check(
            f"p={p:.1e} log-log slope",
            abs(slope - 1.0) < 0.1,
            f"slope {slope:.3f}; errs " + " ".join(f"{x:.1e}" for x in errs),
        )

    # 3. Finite stiffness contrast: the claim itself.
    print("\n3. Finite contrast (their claim: exact.  Prediction: O(eps^2) error)")
    cases = [
        Contrast(2.0e9, 1.0e9, 100.0, "project moderate (dlam 2 GPa, dmu 1 GPa, drho 100)"),
        Contrast(0.2 * LAM0, 0.2 * MU0, 0.0, "moduli +20%"),
        Contrast(-0.2 * LAM0, -0.2 * MU0, 0.0, "moduli -20%"),
    ]
    for c in cases:
        for p in P_LIST:
            e = rel_err(tromp_snieder_band(c, p), kennett_band(c.dlam, c.dmu, c.drho, p), p)
            detail = ", ".join(f"{key} {val:.2e}" for key, val in e.items())
            check(f"{c.label} p={p:.1e} NOT exact", worst(e) > 1e-4, detail)

    # 4. Mechanism: at p=0 the Born generator is the exact generator of a band
    #    with linearised compliances, 1/M_eff = 1/M0 - Ms/M0^2.
    print("\n4. p = 0: TS == Kennett(band with M_eff = M0^2/(M0 - Ms)) (prediction: exact)")
    for c in cases:
        ms = c.dlam + 2.0 * c.dmu
        m_eff = M0**2 / (M0 - ms)
        mu_eff = MU0**2 / (MU0 - c.dmu)
        dmu_eff = mu_eff - MU0
        dlam_eff = (m_eff - 2.0 * mu_eff) - LAM0
        ts = tromp_snieder_band(c, 0.0)
        e_eff = rel_err(ts, kennett_band(dlam_eff, dmu_eff, c.drho, 0.0), 0.0)
        e_true = rel_err(ts, kennett_band(c.dlam, c.dmu, c.drho, 0.0), 0.0)
        check(
            c.label,
            worst(e_eff) < 1e-7,
            f"vs linearised-compliance band {worst(e_eff):.2e}; vs true band {worst(e_true):.2e}",
        )

    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
