#!/usr/bin/env python3
"""Table cost over a sweep of 256 frequencies: gradient hierarchy (l = 3, linear contrast) against the
first-moment Legendre voxel (p = r = 1), on the graded-sphere grid of Table tab:hiergraded.

A synthetic needs the coupling tables at every frequency of the sweep, so what matters is the cost that is
paid once (whatever depends only on the geometry) and the cost paid again at each frequency. Routes:

  hierarchy, Gauss      ``derivatives.moment_table`` per orbit of the cube group, at each frequency, with
                        the calibrated Gauss order (``gauss_points``); nothing reused.
  hierarchy, k-series   ``derivatives.moment_table_kseries``: the unit-cube coefficients U(t; a, W) once
                        per orbit (independent of the frequency and the cell size), then at each frequency
                        only the sum over t.
  Legendre, quadrature  ``blocks.coupling_block``: the touching blocks by ``near_block`` (Duffy and Gauss,
                        n_q = 12), the rest by the s-form Gauss rule, at each frequency.
  Legendre, series      the touching blocks by ``near_block_series`` (universal moments once, the
                        assembly at each frequency), the rest by ``far_block_series`` (the s-form
                        coefficients of each power of k once per orbit, then only the sum).

Both schemes then form the self table at each frequency (the hierarchy's closed-form distributional
moments; the Legendre self block among the touching ones). The per-frequency cost is timed at four
frequencies of the sweep, k_S a = 0.5 j / 256 for j = 1, 64, 160, 256, and multiplied by 256; the two
hierarchy routes are compared entry by entry at each of them.

Run:  OMP_NUM_THREADS=8 python -u scripts/measure_frequency_sweep_cost.py [n_sub ...]   (default 6 10)
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
from cubic_scattering.graded_voxel import blocks  # noqa: E402
from cubic_scattering.graded_voxel import derivatives as gd  # noqa: E402

REF, CONTRAST = hier.REF, hier.CONTRAST
N_FREQ = 256
KA_MAX = 0.5
SAMPLES = (1, 64, 160, 256)
KSERIES_TOL = 1e-13


def omega_of(j: int) -> float:
    return KA_MAX * j / N_FREQ * REF.beta / gs.RADIUS


def orbits(n_sub: int) -> list[tuple[int, int, int]]:
    """One offset per orbit of the cube group within an n_sub grid, the self cell excluded."""
    reps = {gd.canonical_offset(k) for k in itertools.product(range(n_sub), repeat=3)}
    reps.discard((0, 0, 0))
    return sorted(reps)


def hierarchy_lists() -> tuple[list, list]:
    asm = gs.Assembler(3, 1, omega_of(1), CONTRAST)
    return [gd.as_exponents(d) for d in asm.d_list], [gd.as_exponents(w) for w in asm.w_list], asm


def hierarchy_gauss(reps, side, omega, d_list, w_list) -> dict:
    ks_side = omega / REF.beta * side
    return {
        rep: gd.moment_table(
            side * np.array(rep, float),
            side,
            omega,
            REF,
            d_list,
            w_list,
            gs.gauss_points(max(rep), gs.TABLE_TOL, ks_side),
        )
        for rep in reps
    }


def hierarchy_kseries(reps, side, omega, d_list, w_list) -> dict:
    return {
        rep: gd.moment_table_kseries(rep, side, omega, REF, d_list, w_list, KSERIES_TOL) for rep in reps
    }


def coefficients(reps, d_list, w_list) -> None:
    for r in reps:
        gd.kseries_coefficients(r, tuple(d_list), tuple(w_list))


def legendre(n_sub: int, h: float, omega: float, series: bool) -> None:
    reps = sorted({tuple(sorted(k, reverse=True)) for k in itertools.product(range(n_sub), repeat=3)})
    blocks._NEAR_CACHE.clear()
    for c in reps:
        if series and max(c) <= 1:
            blocks.near_block_series(c, h, omega, REF, 10, 4)
        elif series:
            blocks.far_block_series(c, h, omega, REF, 10, 4)
        else:
            blocks.coupling_block(c, h, omega, REF, 10, 4)


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def report(name: str, one_off: float, per_freq: list[float]) -> None:
    mean = float(np.mean(per_freq))
    total = one_off + N_FREQ * mean
    print(
        f"  {name:22s} one-off {one_off:8.2f} s   per frequency {mean:8.3f} s "
        f"(range {min(per_freq):.3f}-{max(per_freq):.3f})   {N_FREQ} frequencies {total / 60:7.2f} min",
        flush=True,
    )


def main(sizes: list[int]) -> int:
    d_list, w_list, asm = hierarchy_lists()
    for n_sub in sizes:
        side = 2.0 * gs.RADIUS / n_sub
        reps = orbits(n_sub)
        print(
            f"n_sub = {n_sub}: {len(reps)} orbits besides the self cell, cell k_S d up to "
            f"{KA_MAX * 2.0 / n_sub:.3f}",
            flush=True,
        )

        # the self table, common to both hierarchy routes, at each frequency
        t_self = [timed(gs.self_array, side, omega_of(j), asm.d_list, asm.w_list)[1] for j in SAMPLES]

        t_gauss = []
        for j in SAMPLES:
            _, t = timed(hierarchy_gauss, reps, side, omega_of(j), d_list, w_list)
            t_gauss.append(t)

        gd.kseries_coefficients.cache_clear()
        _, t_coef = timed(coefficients, reps, d_list, w_list)
        t_kser, errs = [], []
        for j in SAMPLES:
            ks, t = timed(hierarchy_kseries, reps, side, omega_of(j), d_list, w_list)
            t_kser.append(t)
            ga = hierarchy_gauss(reps, side, omega_of(j), d_list, w_list)
            scale = max(np.abs(ga[r]).max() for r in reps)
            errs.append(max(np.abs(ks[r] - ga[r]).max() for r in reps) / scale)
        print(
            "  k-series against Gauss, max entry / max table, at k_S a = "
            + ", ".join(f"{KA_MAX * j / N_FREQ:.3f}: {e:.1e}" for j, e in zip(SAMPLES, errs, strict=True)),
            flush=True,
        )
        report("hierarchy self", 0.0, t_self)
        report("hierarchy Gauss", 0.0, t_gauss)
        report("hierarchy k-series", t_coef, t_kser)

        h = gs.RADIUS / n_sub
        t_quad = [timed(legendre, n_sub, h, omega_of(j), False)[1] for j in SAMPLES]
        blocks.universal_moment.cache_clear()
        blocks.far_series_coefficients.cache_clear()
        # the top frequency of the sweep needs the most terms, so this warm-up computes every moment
        _, t_univ = timed(legendre, n_sub, h, omega_of(N_FREQ), True)
        t_ser = [timed(legendre, n_sub, h, omega_of(j), True)[1] for j in SAMPLES]
        report("Legendre quadrature", 0.0, t_quad)
        report("Legendre series", max(t_univ - float(np.mean(t_ser)), 0.0), t_ser)
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]] or [6, 10]))
