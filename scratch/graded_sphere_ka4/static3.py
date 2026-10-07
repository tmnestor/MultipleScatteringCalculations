"""The static response to an incident field of a given degree, its stress monopole by self-convergence.

The incident displacement is the plane wave's coefficient of k_S^DEG (POL (i R x.k)^DEG / DEG!), solved with
the STATIC operator only; the measure is the far field's stress term at m = 0, i.e. the stress monopole
sum_c int s_h dC : grad u, projected on the observation directions.
Usage: python static3.py [--deg=3] [--rc=1] [--profile=smoothstep] n1 n2 ...
"""
import inspect
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/MultipleScatteringCalculations")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import measure_graded_sphere_frequency_series as fs  # noqa: E402

gs, gfft, lowf = fs.gs, fs.gfft, fs.lowf


def install(deg: int) -> None:
    src = inspect.getsource(fs.solve_series)
    src = src.replace("for j in range(len(p_idx), J + 1):", f"for j in ([{deg}] if len(p_idx) <= {deg} else []):")
    src = src.replace("rhs[j][tuple(grid[c])]", "rhs[0][tuple(grid[c])]")
    exec(src, fs.__dict__)


if __name__ == "__main__":
    args = sys.argv[1:]
    deg = 3
    for f in [x for x in args if x.startswith("--deg=")]:
        deg = int(f.split("=")[1]); args.remove(f)
    for f in [x for x in args if x.startswith("--rc=")]:
        gfft.R_C = int(f.split("=")[1]); args.remove(f)
    prof = "smoothstep"
    for f in [x for x in args if x.startswith("--profile=")]:
        prof = f.split("=")[1]; args.remove(f)
    born = "--born" in args
    if born:
        args.remove("--born")
    lowf.NEAR = -1
    fs.J = 0
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile(prof)
    install(deg)
    obs = fs.obs_points(gs.R_FAR, gs.THETA)
    prev = []
    for n in [int(v) for v in args]:
        t0 = time.time()
        side, centres, coefs, asm, sols = fs.solve_series(n)
        vox = fs.far_series(side, centres, coefs, asm, sols, obs)
        v = np.concatenate([vox["P"][1].ravel(), vox["S"][1].ravel()])
        out = Path(__file__).parent / "runs" / f"static_d{deg}_rc{gfft.R_C}_{prof}_{n}.npy"
        np.save(out, v)
        msg = f"deg {deg} rc {gfft.R_C} {prof} n={n:2d} |M| {np.abs(v).max():.6e}"
        if prev:
            msg += f"  diff {np.abs(v - prev[-1][1]).max() / np.abs(v).max():.3e}"
        if len(prev) >= 2:
            (n0, v0), (n1, v1) = prev[-2], prev[-1]
            d0, d1 = np.abs(v1 - v0).max(), np.abs(v - v1).max()
            f = lambda r: (n0**-r - n1**-r) / (n1**-r - n**-r) - d0 / d1
            lo, hi = 0.1, 12.0
            if f(lo) * f(hi) < 0:
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
                msg += f"  order {mid:.2f}"
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
        prev.append((n, v))
