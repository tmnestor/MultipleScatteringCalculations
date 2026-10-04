#!/usr/bin/env python3
"""Compute once, and store, the unit-cube moments E[m; ds; w] that the hierarchy's self block needs.

The self block (``measure_graded_sphere_gradient_hierarchy.self_array``) is assembled from moments of the
Green's tensor over the cube, each a sum over the series of exp(i k r)/r of the pure numbers
E[m; ds; w] = < d_ds r^m, x_w 1_V > on the unit cube (``measure_cube_gradient_hierarchy.cube_unit``,
evaluated by the ball-and-shell route of ``crosscheck_cube_moments_ball_shell.py``, which agrees with the
closed forms of ``Mathematica/CubeMomentCore.wl`` to 1.1e-12). They depend on neither the cell size nor
the frequency, so they are computed here for every field degree q = 1, 2, 3 and contrast degree
r_c = 1, ..., q, recorded by intercepting the calls, and written to ``scripts/cube_unit_moments.json``,
which ``cube_unit`` then reads. A moment missing from the store is still computed, so a new degree
works, more slowly, until the store is rebuilt.

Run:  python scripts/build_cube_unit_moments.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crosscheck_cube_moments_ball_shell as cm  # noqa: E402
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_cube_gradient_hierarchy as cube  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402


def main() -> int:
    seen: dict[tuple, float] = {}
    original = cm.moment

    def record(m, ds, w):
        key = (int(m), tuple(ds), tuple(w))
        if key not in seen:
            seen[key] = float(original(m, ds, w))
        return seen[key]

    cm.moment = record
    cube.cube_unit.cache_clear()
    cube._stored.cache_clear()
    cube.STORE = Path("/nonexistent")  # compute every value afresh
    t0 = time.perf_counter()
    omega = 0.5 * hier.REF.beta / gs.RADIUS
    for q in (1, 2, 3):
        for r_c in range(1, q + 1):
            asm = gs.Assembler(q, r_c, omega, hier.CONTRAST)
            gs.self_array(1.0, omega, asm.d_list, asm.w_list)
    out = Path(__file__).resolve().parent / "cube_unit_moments.json"
    # the moments with an odd count of some axis vanish by the cube's reflections; cube_unit returns them as
    # zero without the store, so only the others are written
    rows = [
        {"m": k[0], "d": list(k[1]), "w": list(k[2]), "value": v}
        for k, v in sorted(seen.items())
        if not any((k[1].count(ax) + k[2].count(ax)) % 2 for ax in range(3))
    ]
    out.write_text(json.dumps({"cube": "[-1/2, 1/2]^3", "route": "ball and shell", "moments": rows}) + "\n")
    print(f"{len(rows)} moments in {time.perf_counter() - t0:.0f} s, written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
