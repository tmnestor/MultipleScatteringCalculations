#!/usr/bin/env python3
"""The graded sphere at low frequency: the cell model's leading (static) scattering against the exact
series.

The body is the continuum-limit paper's graded sphere (radius a = 10 m, core a/10, smoothstep shell, the
gate contrast), discretised by the third-gradient Cartesian multipole hierarchy and solved by GMRES with the
FFT product (``measure_graded_sphere_gradient_hierarchy_fft.solve_fft``). The far field at nine angles is
compared with the EXACT graded sphere, whose per-order T-matrices come from their low-frequency series
(``Mathematica/GradedSphere_LowFrequency.json``, Taylor coefficients in w = k_P a to w^24, checked there to
4e-33 against the direct T-matrix): a double-precision radial solve loses digits as the frequency falls, the
series does not. At k_S a = 0.01 and 0.005 the far field is its leading (static) term to O((ka)^2), so the
relative error measures the cell model's error in the static scattering coefficients.

Two routes for the cell-to-cell tables, the operator otherwise identical:
  gauss    the production tables (``moment_table``, Gauss rules calibrated to 1e-10);
  closed   the touching and near orbits (Chebyshev distance <= 2) from the series in k with CLOSED-FORM
           coefficients (``derivatives.kseries_coefficients_closed``, certified to 3.7 eps M against a
           40-digit reference); farther orbits as in gauss. The coefficients are universal
           (independent of frequency, cell size and medium), so they are computed once and cached on disk
           for every grid and frequency.

Run:  python -u scripts/measure_graded_sphere_low_frequency.py [gauss|closed] n1 n2 ...
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
import measure_graded_sphere_gradient_hierarchy_fft as gfft  # noqa: E402
from cubic_scattering.graded_voxel import derivatives as gd  # noqa: E402
from cubic_scattering.sphere_scattering import MieResult, mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

SERIES = ROOT / "Mathematica" / "GradedSphere_LowFrequency.json"
CACHE = ROOT / "scripts" / "data" / "closed_hierarchy_coefficients"
NEAR = 2  # orbits within this Chebyshev distance take the closed coefficients
D4 = tuple(gd.multi_indices(4))


def series_path() -> Path:
    """The exact series of the body with the current profile (gs.PROFILE)."""
    return (
        SERIES
        if gs.PROFILE == "smoothstep"
        else SERIES.with_name(f"GradedSphere_LowFrequency_{gs.PROFILE}.json")
    )


def series_tmatrices(omega: float) -> list[tuple[np.ndarray, complex]]:
    """The exact per-order T-matrices of the graded sphere at omega, from their series in w = k_P a."""
    data = json.loads(series_path().read_text())
    body = data["body"]
    assert (body["a"], body["core"]) == (gs.RADIUS, gs.CORE), (body, gs.RADIUS, gs.CORE)
    w = omega * gs.RADIUS / hier.REF.alpha
    out = []
    for o in data["orders"]:
        tp = sum(
            np.array([[c[0] + 1j * c[1] for c in row] for row in o["Tpsv"][k]]) * w**k
            for k in range(len(o["Tpsv"]))
        )
        tsh = sum((o["Tsh"][k][0] + 1j * o["Tsh"][k][1]) * w**k for k in range(len(o["Tsh"])))
        out.append((tp, tsh))
    return out


def exact_result(omega: float) -> MieResult:
    """As ``crosscheck_graded_sphere.graded_mie_result``, with the T-matrices from the exact series."""
    ref = hier.REF
    kp, ks = omega / ref.alpha, omega / ref.beta
    tms = series_tmatrices(omega)
    n_max = len(tms) - 1
    a_n, b_n, c_n, a_sv, b_sv = (np.zeros(n_max + 1, dtype=complex) for _ in range(5))
    for n, (tm, tsh) in enumerate(tms):
        cp, cs = (2 * n + 1) * 1j**n / (1j * kp), (2 * n + 1) * 1j**n / (1j * ks)
        a_n[n] = tm[0, 0] * cp
        if n:
            b_n[n], a_sv[n], b_sv[n], c_n[n] = tm[1, 0] * cp, tm[0, 1] * cs, tm[1, 1] * cs, tsh * cs
    return MieResult(
        a_n=a_n, b_n=b_n, c_n=c_n, a_n_sv=a_sv, b_n_sv=b_sv, n_max=n_max, omega=omega, radius=gs.RADIUS,
        ref=ref, contrast=hier.CONTRAST, ka_P=omega * gs.RADIUS / ref.alpha,
        ka_S=omega * gs.RADIUS / ref.beta,
    )  # fmt: skip


def closed_coefficients(offset: tuple[int, int, int]) -> tuple[np.ndarray, tuple]:
    """The closed coefficients of one canonical offset, from the disk cache or computed and stored."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"U_{offset[0]}_{offset[1]}_{offset[2]}.npy"
    a_list = gd.multi_indices(max(sum(d) for d in D4) + 2)
    if path.exists():
        return np.load(path), a_list
    t0 = time.perf_counter()
    u, a_list = gd.kseries_coefficients_closed(offset, D4, D4)
    np.save(path, u)
    print(f"    closed coefficients of {offset}: {time.perf_counter() - t0:.0f} s (cached)", flush=True)
    return u, a_list


def install_closed_tables() -> None:
    """Route the near orbits' tables through the series in k with closed coefficients."""
    original = gs.coupling_array

    def coupling_array(offset, side, omega, d_list, w_list, n_gauss):
        units = tuple(int(round(v)) for v in np.asarray(offset, float) / side)
        if max(abs(u) for u in units) > NEAR:
            return original(offset, side, omega, d_list, w_list, n_gauss)
        d_exp = [gd.as_exponents(d) for d in d_list]
        w_exp = [gd.as_exponents(w) for w in w_list]
        return gd.moment_table_kseries(
            units, side, omega, hier.REF, d_exp, w_exp, tol=1e-15,
            coefficients=lambda o, _dl, _wl: closed_coefficients(o),
        )  # fmt: skip

    gs.coupling_array = coupling_array


def main() -> int:
    route = sys.argv[1] if len(sys.argv) > 1 else "gauss"
    ladder = [int(v) for v in sys.argv[2:]] or [4, 6, 8]
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile("smoothstep")
    if route == "closed":
        install_closed_tables()
    obs = obs_points(gs.R_FAR, gs.THETA)
    print(
        f"graded sphere at low frequency, tables by {route}, core {gs.CORE} m, smoothstep shell", flush=True
    )
    for ka_s in (0.01, 0.005):
        omega = ka_s * hier.REF.beta / gs.RADIUS
        exact = mie_scattered_displacement(exact_result(omega), obs)
        peak = float(np.max(np.abs(exact)))
        rows = []
        for n in ladder:
            side, centres, coefs, asm, sol, info = gfft.solve_fft(n, 3, omega, hier.CONTRAST)
            got = gs.far_field(side, centres, coefs, asm, sol, omega, hier.CONTRAST, obs)
            err = float(np.max(np.abs(got - exact)) / peak)
            rows.append((n, err))
            print(
                f"  k_S a = {ka_s}  n_sub {n:2d}  error {err:.3e}  (tables {info['tables_s']:.0f} s, "
                f"{info['iterations']} iterations)",
                flush=True,
            )
        for (n1, e1), (n2, e2) in zip(rows, rows[1:], strict=False):
            order = math.log(e1 / e2) / math.log(n2 / n1)
            print(f"  k_S a = {ka_s}  apparent order {n1} -> {n2}: {order:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
