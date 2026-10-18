#!/usr/bin/env python3
"""GATE: the 3-D in-plane sweeps (the thesis's Algorithm 5.2) against the closed-form same-depth sum.

sweep_x_3d (left-right, every same-depth pair with dx != 0) and sweep_y_3d (in-out, dx = 0,
dy != 0) are compared, separately and together, with the explicit O(N^2) pairwise sum of the
closed-form whole-space propagator (pair_propagators.same_depth_table), with a DISTINCT source at
every site of two planes in different media. The error is the transverse quadrature's: it must fall
as the rule is refined, cutoff and spacing together. A partition or convention error would instead
leave an O(1) residual that refinement does not move.

MEASURED (2 x 3 x 4 voxels, pitch 0.25 km, omega = 2 pi 6 Hz (1 + 0.03 i)), relative to the field:

    kr*pitch  n/axis    nodes   left-right   in-out    both    sec
          20     256    51468     2.05e-03  1.86e-03  2.16e-03    2
          20     512   205892     1.03e-04  5.39e-05  1.22e-04    8
          20    1024   823592     1.65e-06  9.58e-07  2.01e-06   30
          30    1536  1853028     1.49e-06  7.94e-07  1.93e-06   71
          20    2048  3294288                         1.13e-06  125
          30    3072  7412064                         8.82e-10  288

BOTH CONSTRAINTS BIND near 1e-6, as stage 2 found for the same kernel. Raising the cutoff from 20
to 30 at a spacing of 0.156/km barely moves the error (the spacing limits it: this lattice has
offsets of up to three pitches, which oscillate faster across the transverse plane than the single
one-pitch offset at which scripts/measure_sweep3d_cost.py measured 3.2e-7); halving the spacing at
kr*pitch = 20 takes it only to 1.1e-6 (the cutoff limits it, as measured there: 1.3e-6). Refined
together, kr*pitch = 30 at a spacing of 0.078/km, the error falls to 8.8e-10: the sweeps converge
to the closed form, and the plateau near 1e-6 is the rule's, not the method's.

LONGER COLUMNS (--lattice 2x3x12: twelve voxels per column, in-out offsets up to eleven pitches,
2.75 km); "in-out/own" is the in-out sweep's error relative to the in-out field itself:

    kr*pitch  n/axis    nodes   left-right   in-out    both    in-out/own    sec
          20     256    51468     3.73e-03  3.00e-03  4.19e-03    3.44e-03      4
          20     512   205892     1.47e-04  1.42e-04  1.59e-04    1.63e-04     17
          20    1024   823592     2.37e-06  1.82e-06  2.60e-06    2.08e-06     72
          30    1536  1853028     2.17e-06  1.71e-06  2.17e-06    1.96e-06    162
          30    3072  7412064     1.01e-09  7.80e-10  1.01e-09    8.95e-10    591

The same convergence as on four-voxel columns, a little slower at the coarse rules (the longer
offsets), and the support count exact; the real-space route takes 0.07 s.

The vacuity control: a uniform source would also be matched by an implementation that summed the
wrong pairs symmetrically, so the gate uses random sources, and a support count checks that a unit
source reaches every other site of its plane, never itself and never another plane.

Timings are reported against the real-space route (same_depth_table plus the pairwise sum): 0.02 s
here. That route remains far the cheaper way to apply this coupling; the sweeps are the thesis's
algorithm, built and gated, not a replacement for it.

Run:  PYTHONPATH=. python -u scripts/gate_in_plane_sweeps_3d.py [--lattice 2x3x12] [--rules 30:3072]
Seismic units (km/s, g/cm3, km), time convention e^{-i omega t}; omega COMPLEX.
"""

import argparse
import time

import numpy as np
from cubic_scattering import directional_sweeps as ds
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.pair_propagators import same_depth_table, separation_index

PITCH = 0.25
OMEGA = 2 * np.pi * 6.0 * (1.0 + 0.03j)
REFS = [ReferenceMedium(5.0, 3.0, 2.5), ReferenceMedium(6.5, 3.6, 2.8)]
CHUNK = 8192
RULES = "20:256,20:512,20:1024,30:1536"


def pairwise(src: np.ndarray, part: str) -> tuple[np.ndarray, float]:
    """The explicit pairwise sum over the same-depth pairs of one sweep ('x', 'y') or both ('xy')."""
    t0 = time.time()
    _, n_x, n_y, _ = src.shape
    out = np.zeros_like(src)
    for z in range(src.shape[0]):
        tab = same_depth_table(n_x, n_y, PITCH, OMEGA, REFS[z])
        for i, j, i2, j2 in np.ndindex(n_x, n_y, n_x, n_y):
            dx, dy = i - i2, j - j2
            if (dx, dy) == (0, 0) or (part == "x" and dx == 0) or (part == "y" and dx != 0):
                continue
            out[z, i, j] += tab[separation_index(dx, n_x), separation_index(dy, n_y)] @ src[z, i2, j2]
    return out, time.time() - t0


def support(shape: tuple[int, int, int]) -> bool:
    """A unit source reaches every other site of its plane, never itself, never another plane."""
    nodes = ds.make_transverse_nodes(20.0 / PITCH, 128)
    ok = True
    for z, i, j in ((0, 1, shape[2] - 1), (1, 0, 0)):
        src = np.zeros((*shape, 9), dtype=complex)
        src[z, i, j, 0] = 1.0
        got = ds.in_plane_sweeps_3d(src, PITCH, OMEGA, REFS, nodes, chunk=CHUNK)
        reached = np.abs(got).max(axis=3) > 0
        want = np.zeros(shape, dtype=bool)
        want[z] = True
        want[z, i, j] = False
        ok &= bool(np.array_equal(reached, want))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lattice", default="2x3x4", help="n_z x n_x x n_y; n_z = 2 (two plane media)")
    parser.add_argument("--rules", default=RULES, help="comma-separated kr*pitch:n_axis pairs")
    args = parser.parse_args()
    shape = tuple(int(v) for v in args.lattice.split("x"))
    if len(shape) != 3 or shape[0] != 2:
        parser.error("--lattice must be 2xNXxNY: the gate has one medium per plane, two planes")
    rules = [(float(a), int(b)) for a, b in (r.split(":") for r in args.rules.split(","))]

    rng = np.random.default_rng(20261018)
    src = rng.standard_normal((*shape, 9)) + 1j * rng.standard_normal((*shape, 9))
    want_x, _ = pairwise(src, "x")
    want_y, _ = pairwise(src, "y")
    want, t_ref = pairwise(src, "xy")
    scale, scale_y = np.abs(want).max(), np.abs(want_y).max()
    n_z, n_x, n_y = shape
    print(
        f"lattice {n_z} x {n_x} x {n_y}, pitch {PITCH} km, two plane media; omega = 2 pi 6 Hz (1 + 0.03 i)"
    )
    print(f"support: unit sources reach exactly the other sites of their plane: {support(shape)}")
    head = f"{'kr*pitch':>8} {'n/axis':>7} {'nodes':>8} {'left-right':>11} {'in-out':>9} {'both':>9}"
    print(f"\n  {head} {'in-out/own':>11} {'sec':>6}", flush=True)
    errors = []
    for mult, n_axis in rules:
        nodes = ds.make_transverse_nodes(mult / PITCH, n_axis)
        t0 = time.time()
        got_x = ds.sweep_x_3d(src, PITCH, OMEGA, REFS, nodes, chunk=CHUNK)
        got_y = ds.sweep_y_3d(src, PITCH, OMEGA, REFS, nodes, chunk=CHUNK)
        dt = time.time() - t0
        ex = np.abs(got_x - want_x).max() / scale
        ey = np.abs(got_y - want_y).max() / scale
        eb = np.abs(got_x + got_y - want).max() / scale
        ey_own = np.abs(got_y - want_y).max() / scale_y
        errors.append(eb)
        print(
            f"  {mult:>8.0f} {n_axis:>7} {nodes.kt.size:>8} {ex:>11.2e} {ey:>9.2e} {eb:>9.2e} "
            f"{ey_own:>11.2e} {dt:>6.1f}",
            flush=True,
        )
    print(f"\nreal-space route (closed-form table and pairwise sum): {t_ref:.2f} s")
    monotone = all(b < a for a, b in zip(errors, errors[1:], strict=False))
    print(f"error falls with every refinement: {monotone}; finest {errors[-1]:.1e}")


if __name__ == "__main__":
    main()
