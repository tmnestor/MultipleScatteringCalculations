#!/usr/bin/env python3
"""The pairing question of SphereMieArbiter, settled on a body without a staircase.

The collocation cell closes on its centre value, so the propagator it is paired with should be averaged over
the receiving cell (``cell_averaged_pair``). On the sharp sphere the staircase of the voxelised surface
dominates the error from k a ~ 0.5 and the two propagators cannot be told apart there
(``gate_sphere_cell_average_vs_mie.py``). The smoothly graded sphere has no staircase: its contrast and its
slope vanish at the surface. Here the same cell (the uniform-strain closure exact at every frequency, the
configured lattice single site, contrast at the cell centre) is solved with the point propagator and with
the receiver-averaged one, against the exact solution, with the error of the Error paper's sphere section.

Run:  python -u scripts/measure_pairing_graded_sphere.py 6 8 12 16 24
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, THETA, profile  # noqa: E402

from cubic_scattering.point_cell_self import lattice_single_site  # noqa: E402
from cubic_scattering.sphere_scattering import foldy_lax_far_field, mie_scattered_displacement  # noqa: E402
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft  # noqa: E402

OUT = Path(__file__).parent / "data" / "pairing_graded_sphere.jsonl"
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
R_MULT = 5.0e8


def main() -> int:
    ladder = [int(a) for a in sys.argv[1:]]
    if not ladder:
        print(__doc__)
        return 2
    site = lattice_single_site()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ka_s in (0.5, 1.0):
        omega = ka_s * REF.beta / RADIUS
        r_far = R_MULT * RADIUS
        pts = obs_points(r_far, THETA)
        n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
        exact = mie_scattered_displacement(graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max), pts)
        peak = float(np.max(np.abs(exact)))
        print(f"graded sphere, k_S a = {ka_s}, single site {site}", flush=True)
        for n in ladder:
            errs = {}
            for averaged in (False, True):
                fl = compute_sphere_foldy_lax_fft(
                    omega,
                    RADIUS,
                    REF,
                    CONTRAST,
                    n_sub=n,
                    k_hat=K_HAT,
                    wave_type="P",
                    cell_average=averaged,
                    contrast_profile=profile,
                )
                u_p, u_s = foldy_lax_far_field(fl, pts / r_far, r_far, K_HAT, POL, wave_type="P")
                errs[averaged] = float(np.max(np.abs(u_p + u_s - exact))) / peak
            row = {"ka_s": ka_s, "n": n, "site": site, "point": errs[False], "averaged": errs[True]}
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  n {n:3d}: point {errs[False]:.3e}  averaged {errs[True]:.3e}  "
                f"point / averaged {errs[False] / errs[True]:.2f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
