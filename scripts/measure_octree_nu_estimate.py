#!/usr/bin/env python3
"""The nonlinear fraction nu of the octree paper, estimated without a solve, against its exact value.

nu = max |u - T1| / max |u| over the nine scattering angles, u the scattered far field and T1 its Born
term. The octree paper's refinement rule needs it before the solve; it was estimated from one solve on
a coarse tree that resolves the medium. Here, on the same coarse tree, from the scheme's Born series
(``octree.born_series_octree``), which costs one product with the assembled matrix per term:

    solve     max |u_coarse - T1| / max |u_coarse|            (the earlier estimate: one dense solve)
    T2        max |T2| / max |T1 + T2|                         (one product)
    T2 / (1 - E)   the same with T2 divided by 1 - E, E the tree's relative projection error: the scheme's
              second-order term is low by E times a factor of order one (section 3.4 of the paper), and E
              is known before the solve; the factor is taken as one
    T2 + T3   max |T2 + T3| / max |T1 + T2 + T3|               (two products)
    (T2 + T3) / (1 - E)   the same, the nonlinear part divided by 1 - E: by the law of section 3.3 the
              whole nonlinear part, not only T2, is low by about E

On the exact solutions T2 alone carries nu to 3-4% and T2 + T3 to 0.2% (the three bodies below).

Bodies (the paper's): the thin-shell sphere (core 0.75 a, k_S a = 0.5) and the halo-and-feature sphere
with the feature between 0.2 a and 0.5 a and between 0.1 a and 0.3 a (k_S a = 1). Coarse tree: the
medium-only rule at tolerance 1e-2 from a uniform 2 x 2 x 2 start, as in the refinement pilot.

Run:  conda run -n seismic python -u scripts/measure_octree_nu_estimate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import measure_graded_voxel_resolution as mres  # noqa: E402
import pilot_octree_two_term_refinement as pil  # noqa: E402
from cubic_scattering.graded_voxel.octree import (  # noqa: E402
    adapt_leaves,
    born_series_octree,
    leaf_energies,
    octree_far_field,
    solve_graded_octree,
    uniform_leaves,
)
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import RADIUS, THETA  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])
EPS = 1e-3


def s5(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def run(name: str, shape: str, core: float, ka: float, profile_r, h_min: float) -> None:
    omega = ka * REF.beta / RADIUS
    rf = 5e8 * RADIUS
    pts = obs_points(rf, THETA)
    dirs = pts / rf
    exact = mres.exact_field(shape, core, omega, 1.0, pts)
    born = (
        mres.exact_field(shape, core, omega, EPS, pts) - mres.exact_field(shape, core, omega, -EPS, pts)
    ) / (2 * EPS)
    nu = float(np.abs(exact - born).max() / np.abs(exact).max())

    def prof_vec(pos: np.ndarray) -> np.ndarray:
        return profile_r(np.linalg.norm(pos, axis=-1))

    def prof(pos: np.ndarray) -> float:
        return float(profile_r(np.linalg.norm(pos)))

    print(f"\n{name}: exact nu = {nu:.4f}", flush=True)
    for p in (0, 1):
        c2, h2 = adapt_leaves(prof_vec, *uniform_leaves(RADIUS, 2), p, 1e-2, h_min)
        defect, _, norm = leaf_energies(prof_vec, c2, h2, p)
        e_proj = float(defect.sum() / norm.sum())
        cache: dict = {}
        series = born_series_octree(
            omega, REF, CONTRAST, c2, h2, prof, K_HAT, K_HAT, "P", 3, p=p, r=p, block_cache=cache
        )
        t1, t2, t3 = (sum(octree_far_field(t, dirs, rf)) for t in series)
        u = sum(
            octree_far_field(
                solve_graded_octree(
                    omega, REF, CONTRAST, c2, h2, prof, K_HAT, K_HAT, "P", p=p, r=p, block_cache=cache
                ),
                dirs,
                rf,
            )
        )
        est = {
            "solve": np.abs(u - t1).max() / np.abs(u).max(),
            "T2": np.abs(t2).max() / np.abs(t1 + t2).max(),
            "T2/(1-E)": np.abs(t2 / (1 - e_proj)).max() / np.abs(t1 + t2 / (1 - e_proj)).max(),
            "T2+T3": np.abs(t2 + t3).max() / np.abs(t1 + t2 + t3).max(),
            "(T2+T3)/(1-E)": np.abs((t2 + t3) / (1 - e_proj)).max()
            / np.abs(t1 + (t2 + t3) / (1 - e_proj)).max(),
        }
        print(
            f"  p = r = {p}: coarse tree {len(h2)} leaves, E = {e_proj:.3e};  "
            + "   ".join(f"{k} {v:.4f} ({v / nu:.3f})" for k, v in est.items()),
            flush=True,
        )


def main() -> int:
    run(
        "thin shell, core 0.75 a, k_S a = 0.5",
        "s5",
        0.75 * RADIUS,
        0.5,
        lambda r: s5((RADIUS - r) / (RADIUS - 0.75 * RADIUS)),
        RADIUS / 16,
    )
    run("halo + feature 0.2 - 0.5 a, k_S a = 1", pil.SHAPE, pil.CORE, 1.0, pil.profile_r, RADIUS / 16)
    pil.CORE, pil.WIDTH = 0.1 * RADIUS, 0.3 * RADIUS
    run("halo + feature 0.1 - 0.3 a, k_S a = 1", pil.SHAPE, pil.CORE, 1.0, pil.profile_r, RADIUS / 32)
    return 0


if __name__ == "__main__":
    sys.exit(main())
