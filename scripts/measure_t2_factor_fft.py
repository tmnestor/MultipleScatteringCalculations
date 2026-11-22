#!/usr/bin/env python3
"""The factor F of the second term of the three-dimensional error law, on grids the dense solver cannot reach.

THE LAW. The error of the far field beyond first order in the contrast is (nonlinear fraction) x F x E, with
E the relative projection error of the medium on the grid and F a factor of order one. For the term of
second order in the contrast, T2, the statement is exact in form (Error paper, Eq. (t2factor)):

    (T2 - T2_scheme) / T2  =  E F .

For constant cells and a body with an isotropic spectrum, F tends to 2 - m_ax / m_bar as the cells shrink
(m the static strain Green operator contracted with the incident and outgoing strains, m_bar its average
over all directions, m_ax its average over the three cube axes). The limit depends on the scattering angle
and on whether the outgoing wave is P or SV. ``measure_t2_law_3d.py`` measured F with the scheme at up to 8
cells across, where it is about three quarters of the way to the limit. This script measures it with the
matrix-free FFT solver on much larger grids, to see the approach to the limit.

WHAT IS MEASURED, for each grid of n cells across and each of the nine scattering angles:

  T2 of the scheme   (u(+d) + u(-d)) / (2 d^2), u(s) the scheme's far field with the contrast multiplied
                     by s: two solves. The terms of odd order cancel; what is left out is d^2 T4, smaller
                     than T2 by about (nonlinear fraction x d)^2.
  T2 exact           the same combination of the exact graded sphere at +-d.
  F measured         Re[(T2 - T2_scheme) . conj(T2)] / (|T2|^2 E), for the P (radial) part and the S
                     (transverse) part of the far field separately.
  F limit            2 - m_ax / m_bar, for constant cells and a stiffness contrast.

The body is a sphere of radius a whose contrast is uniform within ``core_frac`` a and falls smoothly to
zero at the surface. With ``--density`` the density contrast is included (the derivation of F is for a
stiffness contrast in the long-wave limit; the density channel is of higher order in k a).

Results are written after every grid to
``scripts/measure_t2_factor_fft_ka<ka>_core<c>_p<p>[_density].json``.

Run small first:
    conda run -n seismic python -u scripts/measure_t2_factor_fft.py 0.125 0.5 0 --uniform=4,8
    ... <k_S a> <core_frac> <p> [--uniform=4,8,16,32] [--step=0.3] [--density]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from crosscheck_graded_sphere import graded_mie_result, smoothstep  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST as GATE  # noqa: E402
from gate_sphere_cell_average_vs_mie import REF, obs_points  # noqa: E402
from measure_t2_law_3d import predicted_factor, split  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft  # noqa: E402
from cubic_scattering.graded_voxel.octree import leaf_energies, uniform_leaves  # noqa: E402
from cubic_scattering.sphere_scattering import mie_scattered_displacement  # noqa: E402

K = np.array([1.0, 0.0, 0.0])


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ka, core_frac, p = float(args[0]), float(args[1]), int(args[2])
    ns = [int(v) for v in opts.get("uniform", "4,8").split(",") if v]
    d = float(opts.get("step", "0.3"))
    density = "density" in flags
    core = RADIUS * core_frac
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    tangents = np.stack([-dirs[:, 1], dirs[:, 0], np.zeros(len(dirs))], 1)

    def contrast(scale: float) -> MaterialContrast:
        return MaterialContrast(
            scale * GATE.Dlambda, scale * GATE.Dmu, scale * GATE.Drho if density else 0.0
        )

    def prof(pos: np.ndarray) -> float:
        r = float(np.linalg.norm(pos))
        if r <= core:
            return 1.0
        return 0.0 if r >= RADIUS else smoothstep((RADIUS - r) / (RADIUS - core))

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        x = np.clip((RADIUS - np.linalg.norm(pos, axis=-1)) / (RADIUS - core), 0.0, 1.0)
        return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    def exact(scale: float) -> np.ndarray:
        return mie_scattered_displacement(
            graded_mie_result(omega, RADIUS, core, REF, contrast(scale), 12), pts
        )

    t2p_ex, t2s_ex = split((exact(d) + exact(-d)) / (2 * d * d), dirs)
    nan = [float("nan")] * len(dirs)
    lim_p = [predicted_factor(np.outer(n, n)) for n in dirs] if p == 0 else nan
    lim_s = (
        [
            predicted_factor(0.5 * (np.outer(n, t) + np.outer(t, n)))
            for n, t in zip(dirs, tangents, strict=True)
        ]
        if p == 0
        else nan
    )
    tag = f"ka{ka:g}_core{core_frac:g}_p{p}" + ("_density" if density else "")
    out = ROOT / f"scripts/measure_t2_factor_fft_{tag}.json"
    print(
        f"factor of the second term: k_S a = {ka}, core = {core_frac} a, p = r = {p}, "
        f"{'stiffness and density' if density else 'stiffness'} contrast, step d = {d}",
        flush=True,
    )
    rows = []
    for n in ns:
        t0 = time.perf_counter()
        centres, hs = uniform_leaves(RADIUS, n)
        defect, _, norm = leaf_energies(prof_vec, centres, hs, p)
        e_p = float(defect.sum() / norm.sum())
        t2p = np.zeros_like(t2p_ex)
        t2s = np.zeros_like(t2s_ex)
        for sign in (1.0, -1.0):
            res = solve_graded_sphere_fft(
                omega, RADIUS, REF, contrast(sign * d), n, prof, K, K, "P", p=p, r=p
            )
            up, us = graded_far_field(res, dirs, rf, K, K, "P")
            t2p += up / (2 * d * d)
            t2s += us / (2 * d * d)
        f_p = [
            float((((t2p_ex[j] - t2p[j]) @ np.conj(t2p_ex[j])) / np.vdot(t2p_ex[j], t2p_ex[j])).real / e_p)
            for j in range(len(dirs))
        ]
        f_s = [
            float((((t2s_ex[j] - t2s[j]) @ np.conj(t2s_ex[j])) / np.vdot(t2s_ex[j], t2s_ex[j])).real / e_p)
            for j in range(len(dirs))
        ]
        rows.append(
            {
                "n": n,
                "cells": int((norm > 0).sum()),
                "E": e_p,
                "F_P": f_p,
                "F_S": f_s,
                "seconds": time.perf_counter() - t0,
            }
        )
        print(f"\n n = {n}: {rows[-1]['cells']} cells, E = {e_p:.4e}   [{rows[-1]['seconds']:.0f} s]")
        print("  angle    F for outgoing P   limit      F for outgoing S   limit")
        for j, th in enumerate(THETA):
            print(
                f"  {th:5.2f}      {f_p[j]: .4f}        {lim_p[j]: .4f}        {f_s[j]: .4f}        {lim_s[j]: .4f}"
            )
        sys.stdout.flush()
        out.write_text(
            json.dumps(
                {
                    "ka_s": ka,
                    "core_frac": core_frac,
                    "p": p,
                    "density": density,
                    "step": d,
                    "theta": [float(t) for t in THETA],
                    "limit_P": lim_p,
                    "limit_S": lim_s,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"\n wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
