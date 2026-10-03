#!/usr/bin/env python3
"""Table-building cost: gradient hierarchy (l = 3, linear contrast) against the first-moment Legendre voxel
(p = r = 1), on the graded sphere of Table tab:hiergraded, at k_S a = 0.5.

For each scheme: the wall time to build the whole offset table for an n_sub grid (cold, after a one-off
warm-up that is reported separately), and the number of kernel point evaluations made, counted by
wrapping each scheme's kernel. The Legendre builder evaluates the 9 x 9 point propagator (one call gives
all 81 components); the hierarchy builder evaluates every derivative of g_S and B to sixth order at each
point (compiled), in one call per cell table, one table per orbit of the cube group. Both builders use the
48 signed permutations and the compiled kernels selected by numerics.yml.

Run:  OMP_NUM_THREADS=5 python -u scripts/measure_hierarchy_table_cost.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402
import measure_graded_sphere_gradient_hierarchy_fft as gfft  # noqa: E402
from cubic_scattering.graded_voxel import blocks, fft  # noqa: E402
from cubic_scattering.graded_voxel import derivatives as gd  # noqa: E402

REF, CONTRAST = hier.REF, hier.CONTRAST
OMEGA = 0.5 * REF.beta / gs.RADIUS
COUNT = {"hier_calls": 0, "hier_points": 0, "leg_points": 0}

_fields = gd.scalar_derivative_fields


def counted_fields(X, omega, ref, n_s, n_b):
    COUNT["hier_calls"] += 1
    COUNT["hier_points"] += len(X)
    return _fields(X, omega, ref, n_s, n_b)


gd.scalar_derivative_fields = counted_fields
_kernel = blocks.kernel_9x9


def counted_kernel(X, omega, ref, **kw):
    COUNT["leg_points"] += len(X)
    return _kernel(X, omega, ref, **kw)


blocks.kernel_9x9 = counted_kernel


def hierarchy(n_sub: int) -> float:
    side = 2.0 * gs.RADIUS / n_sub
    asm = gs.Assembler(3, 1, OMEGA, CONTRAST)
    t0 = time.perf_counter()
    gfft.octant_blocks(n_sub, side, OMEGA, asm)
    return time.perf_counter() - t0


def legendre(n_sub: int) -> float:
    h = gs.RADIUS / n_sub
    blocks._NEAR_CACHE.clear()
    t0 = time.perf_counter()
    fft.offset_blocks(n_sub, h, OMEGA, REF, 10, 4)
    return time.perf_counter() - t0


rows = []
for name, fn in (("hierarchy l=3", hierarchy), ("Legendre p=r=1", legendre)):
    t_warm = fn(2)
    print(f"{name}: warm-up (n_sub = 2, includes one-off constants) {t_warm:.1f} s", flush=True)
    for n_sub in (6, 10, 14):
        for k in COUNT:
            COUNT[k] = 0
        t = fn(n_sub)
        pts = COUNT["hier_points"] if name.startswith("hier") else COUNT["leg_points"]
        calls = COUNT["hier_calls"]
        extra = f", {calls} orbit tables" if name.startswith("hier") else ""
        print(f"{name}: n_sub = {n_sub:2d}  {t:8.1f} s   {pts:.3e} kernel points{extra}", flush=True)
        rows.append((name, n_sub, t, pts))
