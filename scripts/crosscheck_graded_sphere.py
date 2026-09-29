#!/usr/bin/env python3
"""CROSS-CHECK and REFERENCE: elastic scattering by a sphere whose contrast falls smoothly to zero.

``Mathematica/ContinuumLimit_GradedSphere.wl`` (notebook 16) derives the radial equations symbolically and
integrates them in 90-digit arithmetic. This script shares only the DEFINITION of the body and computes
every ingredient another way:

  * the radial equations as published (Takeuchi & Saito 1972, in the notation of Dahlen & Tromp 1998),
    typed in, not derived: y = (U, V, R, S) with u_r = U P_n, u_theta = V dP_n/dtheta, sigma_rr = R P_n,
    sigma_rtheta = S dP_n/dtheta, and the toroidal (W, T);
  * the fields of the potentials from the package's hand-simplified Bessel formulas
    (``_mie_pwave_fields``, ``_mie_swave_fields``), not from symbolic differentiation;
  * the integration by SciPy's DOP853 in double precision.

THE BODY: a homogeneous core r < b carrying the contrast; for b < r < a the contrast is multiplied by
s(x) = 10x^3 - 15x^4 + 6x^5, x = (a - r)/(a - b), which vanishes like (a - r)^3 at the surface.

CHECKS: [1] with a uniform shell the construction reproduces the package's homogeneous Mie T-matrix;
[2] the graded T-matrices agree with notebook 16's (Mathematica/ContinuumLimit_graded_sphere_tmatrix.json),
order by order, at k_S a = 0.5 and 1; [3] the graded sphere's S-matrix is unitary per order.

``graded_mie_result`` returns the graded sphere as a ``MieResult``, so the package's far-field machinery
(``mie_scattered_displacement``) evaluates its exact scattered field.

Run:  conda run -n seismic python scripts/crosscheck_graded_sphere.py
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import (  # noqa: E402
    MieResult,
    _mie_pwave_fields,
    _mie_swave_fields,
    _spherical_h1_complex,
    _spherical_h1_deriv,
    _spherical_jn_complex,
    _spherical_jn_deriv,
    mie_tmatrix_psv,
    mie_tmatrix_sh,
)

REFJSON = ROOT / "Mathematica" / "ContinuumLimit_graded_sphere_tmatrix.json"


def smoothstep(x: float) -> float:
    return x**3 * (10 - 15 * x + 6 * x**2)


def spheroidal_matrix(r: float, n: int, omega: float, lam: float, mu: float, rho: float) -> np.ndarray:
    """Takeuchi & Saito (1972) in the order (U, V, R, S); L = n(n+1)."""
    ell = n * (n + 1)
    m = lam + 2 * mu
    xi = mu * (3 * lam + 2 * mu) / m
    return np.array(
        [
            [-2 * lam / (r * m), ell * lam / (r * m), 1 / m, 0],
            [-1 / r, 1 / r, 0, 1 / mu],
            [-(omega**2) * rho + 4 * xi / r**2, -2 * ell * xi / r**2, -4 * mu / (r * m), ell / r],
            [
                -2 * xi / r**2,
                -(omega**2) * rho + (2 * mu / r**2) * (2 * ell * (lam + mu) / m - 1),
                -lam / (r * m),
                -3 / r,
            ],
        ]
    )


def toroidal_matrix(r: float, n: int, omega: float, mu: float, rho: float) -> np.ndarray:
    """(W, T), T = mu (W' - W/r)."""
    ell = n * (n + 1)
    return np.array([[1 / r, 1 / mu], [(ell - 2) * mu / r**2 - omega**2 * rho, -3 / r]])


def _propagate(y0: np.ndarray, rhs: Callable, b: float, a: float) -> np.ndarray:
    """Integrate a regular solution from b to a. It starts at unit size: for large n the core solution is
    j_n(k b) ~ 1e-20, far below any absolute tolerance, and T does not depend on its normalisation."""
    if a == b:
        return y0
    size = float(np.max(np.abs(y0)))
    sol = solve_ivp(rhs, (b, a), y0 / size, method="DOP853", rtol=1e-13, atol=1e-15)
    return sol.y[:, -1] * size


def graded_tmatrix(
    n: int,
    omega: float,
    radius: float,
    core: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    profile: Callable[[float], float] | None = None,
) -> tuple[np.ndarray, complex]:
    """Per-order P-SV T (2 x 2; only [0, 0] for n = 0) and SH scalar of the graded sphere.

    The core r < ``core`` carries the contrast; for core < r < radius it is multiplied by ``profile(r)``
    (default: the smoothstep falling to zero at the surface). Convention of ``mie_tmatrix_psv``.
    """
    a, b = radius, core
    f = profile if profile is not None else (lambda r: smoothstep((a - r) / (a - b)))
    lam0, mu0, rho0 = ref.lam, ref.mu, ref.rho
    lam_c, mu_c, rho_c = lam0 + contrast.Dlambda, mu0 + contrast.Dmu, rho0 + contrast.Drho
    kp, ks = omega / ref.alpha, omega / ref.beta
    kp_c, ks_c = omega * np.sqrt(rho_c / (lam_c + 2 * mu_c)), omega * np.sqrt(rho_c / mu_c)
    scale = np.array([1.0, 1.0, a / mu0, a / mu0])  # tractions to O(1) for the integrator

    def mat_at(r: float) -> tuple[float, float, float]:
        g = f(r)
        return lam0 + contrast.Dlambda * g, mu0 + contrast.Dmu * g, rho0 + contrast.Drho * g

    def rhs_sph(r: float, y: np.ndarray) -> np.ndarray:
        lam, mu, rho = mat_at(r)
        a_mat = spheroidal_matrix(r, n, omega, lam, mu, rho)
        return scale * (a_mat @ (y / scale))

    tm = np.zeros((2, 2), dtype=complex)
    if n == 0:
        sel = [0, 2]

        def rhs0(r: float, y: np.ndarray) -> np.ndarray:
            lam, mu, rho = mat_at(r)
            a_mat = spheroidal_matrix(r, 0, omega, lam, mu, rho)[np.ix_(sel, sel)]
            return scale[sel] * (a_mat @ (y / scale[sel]))

        ur, _, srr, _ = _mie_pwave_fields(0, kp_c, b, lam_c, mu_c, "j")
        yc = np.real(np.array([ur, srr])) * scale[sel]
        ya = _propagate(yc, rhs0, b, a) / scale[sel]
        us, _, ss, _ = _mie_pwave_fields(0, kp, a, lam0, mu0, "h1")
        uj, _, sj, _ = _mie_pwave_fields(0, kp, a, lam0, mu0, "j")
        m = np.array([[us, -ya[0]], [ss, -ya[1]]], dtype=complex)
        tm[0, 0] = np.linalg.solve(m, -np.array([uj, sj], dtype=complex))[0]
        return tm, 0j
    y_in = []
    for vals in (_mie_pwave_fields(n, kp_c, b, lam_c, mu_c, "j"), _mie_swave_fields(n, ks_c, b, mu_c, "j")):
        y0 = np.real(np.array(vals)) * scale
        y_in.append(_propagate(y0, rhs_sph, b, a) / scale)
    out_p = np.array(_mie_pwave_fields(n, kp, a, lam0, mu0, "h1"))
    out_s = np.array(_mie_swave_fields(n, ks, a, mu0, "h1"))
    inc_p = np.array(_mie_pwave_fields(n, kp, a, lam0, mu0, "j"))
    inc_s = np.array(_mie_swave_fields(n, ks, a, mu0, "j"))
    m = np.column_stack([out_p, out_s, -y_in[0], -y_in[1]])
    tm = np.linalg.solve(m, -np.column_stack([inc_p, inc_s]))[:2]
    # toroidal
    tscale = np.array([1.0, a / mu0])

    def rhs_tor(r: float, z: np.ndarray) -> np.ndarray:
        lam, mu, rho = mat_at(r)
        return tscale * (toroidal_matrix(r, n, omega, mu, rho) @ (z / tscale))

    jc = _spherical_jn_complex(n, ks_c * b)
    z0 = np.real(np.array([jc, mu_c * (ks_c * _spherical_jn_deriv(n, ks_c * b) - jc / b)])) * tscale
    z_in = _propagate(z0, rhs_tor, b, a) / tscale
    h = _spherical_h1_complex(n, ks * a)
    j = _spherical_jn_complex(n, ks * a)
    out = np.array([h, mu0 * (ks * _spherical_h1_deriv(n, ks * a) - h / a)])
    inc = np.array([j, mu0 * (ks * _spherical_jn_deriv(n, ks * a) - j / a)])
    tsh = np.linalg.solve(np.column_stack([out, -z_in]), -inc)[0]
    return tm, complex(tsh)


def graded_mie_result(
    omega: float,
    radius: float,
    core: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    n_max: int,
) -> MieResult:
    """The graded sphere as a MieResult: coefficients = T x the plane wave's (2n+1) i^n / (i k)."""
    kp, ks = omega / ref.alpha, omega / ref.beta
    a_n, b_n, c_n, a_sv, b_sv = (np.zeros(n_max + 1, dtype=complex) for _ in range(5))
    for n in range(n_max + 1):
        cp = (2 * n + 1) * (1j) ** n / (1j * kp)
        cs = (2 * n + 1) * (1j) ** n / (1j * ks)
        tm, tsh = graded_tmatrix(n, omega, radius, core, ref, contrast)
        a_n[n] = tm[0, 0] * cp
        if n == 0:
            continue
        b_n[n], a_sv[n], b_sv[n] = tm[1, 0] * cp, tm[0, 1] * cs, tm[1, 1] * cs
        c_n[n] = tsh * cs
    return MieResult(
        a_n=a_n,
        b_n=b_n,
        c_n=c_n,
        a_n_sv=a_sv,
        b_n_sv=b_sv,
        n_max=n_max,
        omega=omega,
        radius=radius,
        ref=ref,
        contrast=contrast,
        ka_P=omega * radius / ref.alpha,
        ka_S=omega * radius / ref.beta,
    )


def main() -> int:
    ref_json = json.loads(REFJSON.read_text())
    ref = ReferenceMedium(alpha=ref_json["alpha"], beta=ref_json["beta"], rho=ref_json["rho"])
    c = ref_json["contrast"]
    contrast = MaterialContrast(Dlambda=c["Dlambda"], Dmu=c["Dmu"], Drho=c["Drho"])
    a, b = ref_json["radius"], ref_json["core"]
    oks = []
    print("==== crosscheck_graded_sphere :: a smoothly graded sphere, independently ====")

    worst = 0.0
    for ksa in (0.5, 1.0):
        omega = ksa * ref.beta / a
        for n in range(0, 13):
            tm, tsh = graded_tmatrix(n, omega, a, b, ref, contrast, profile=lambda _r: 1.0)
            want = mie_tmatrix_psv(n, omega, a, ref, contrast)
            worst = max(worst, float(np.max(np.abs(tm - want)) / np.max(np.abs(want))))
            if n >= 1:
                ws = mie_tmatrix_sh(n, omega, a, ref, contrast)
                worst = max(worst, abs(tsh - ws) / abs(ws))
    oks.append(worst < 1e-10)
    print(
        f"  [1] uniform shell vs the package's homogeneous Mie, k_S a = 0.5, 1, n = 0..12: {worst:.1e}: "
        f"{'PASS' if oks[-1] else 'FAIL'}"
    )

    worst, worst_u = 0.0, 0.0
    for key, st in ref_json["sets"].items():
        omega = st["omega"]
        for row in st["tmatrices"]:
            n = row["n"]
            want = np.array([[complex(*v) for v in rr] for rr in row["Tpsv"]])
            tm, tsh = graded_tmatrix(n, omega, a, b, ref, contrast)
            dev = float(np.max(np.abs(tm - want)) / np.max(np.abs(want)))
            if n >= 1:
                dev = max(dev, abs(tsh - complex(*row["Tsh"])) / abs(complex(*row["Tsh"])))
                w = np.diag([np.sqrt(ref.alpha), np.sqrt(ref.beta * n * (n + 1))])
                s_mat = w @ (np.eye(2) + 2 * tm) @ np.linalg.inv(w)
                worst_u = max(
                    worst_u,
                    float(np.linalg.norm(s_mat.conj().T @ s_mat - np.eye(2))),
                    abs(abs(1 + 2 * tsh) - 1),
                )
            else:
                worst_u = max(worst_u, abs(abs(1 + 2 * tm[0, 0]) - 1))
            worst = max(worst, dev)
        print(f"      k_S a = {st['kSa']}: n = 0..{ref_json['sets'][key]['tmatrices'][-1]['n']} compared")
    oks.append(worst < 1e-9)
    print(
        f"  [2] graded sphere vs notebook 16, per order: worst {worst:.1e}: {'PASS' if oks[-1] else 'FAIL'}"
    )
    oks.append(worst_u < 1e-10)
    print(f"  [3] unitarity per order: worst defect {worst_u:.1e}: {'PASS' if oks[-1] else 'FAIL'}")

    summary = f"ALL {len(oks)} CHECKS PASS" if all(oks) else "CHECKS FAILED"
    print(f"==== crosscheck_graded_sphere: {summary} ====")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
