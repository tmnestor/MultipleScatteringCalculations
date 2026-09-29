#!/usr/bin/env python3
"""Pilot: graded first-moment voxels (T36) on the smoothly graded sphere, against its exact scattering.

Arms, on the same grid (the FFT sphere solver's, cells kept by centre):
  t9  the package's uniform-field collocation voxel, each cell at the profile's centre value;
  g0  Galerkin, p = 0, r = 0: the uniform-field Galerkin voxel (its convention check: it must converge);
  g1  Galerkin, p = 1, r = 1: the graded first-moment voxel.
Error: max |far field - exact| / peak over nine angles at 5e8 radii (see R_MULT).
--weak scales the contrast by 1e-6 (gate G4b: the discretisation alone; the reference is solved at the
same weak contrast, so no Born approximation is involved).
Predicted orders: t9 and g0 -> 2, g1 -> 4.

Run small first:
    conda run -n seismic python -u scripts/pilot_graded_voxel_sphere.py --weak --ka=0.5 --arms=g1 4
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from cubic_scattering import MaterialContrast  # noqa: E402
from cubic_scattering.graded_voxel.farfield import graded_far_field  # noqa: E402
from cubic_scattering.graded_voxel.solver import solve_graded_sphere  # noqa: E402
from cubic_scattering.sphere_scattering import foldy_lax_far_field, mie_scattered_displacement  # noqa: E402
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF, obs_points  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS, THETA, profile  # noqa: E402

K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
#: Observation distance in radii. The arms return the ASYMPTOTIC far field and the reference the exact
#: displacement; their near-field residue is O(1/(kr)). At the gate modules' 5e4 radii it is ~1.4e-4, above
#: the graded voxel's own error (g1 at n = 4: 2.15e-4 there, 7.9e-5 at 5e5, 7.1e-5 at 5e6). At 5e6 it is
#: ~1.4e-6, as large as g1's own error at n = 8 (1.43e-6 at k_S a = 0.5); at 5e8 it is ~1e-8. The phase
#: k r ~ 2.5e8 still carries only ~3e-8 relative error in double precision.
R_MULT = 5.0e8


def main() -> int:
    ka_s, weak, summary, arms = 0.5, False, None, ["t9", "g0", "g1"]
    ladder = []
    for a in sys.argv[1:]:
        if a.startswith("--ka="):
            ka_s = float(a.split("=", 1)[1])
        elif a == "--weak":
            weak = True
        elif a.startswith("--arms="):
            arms = a.split("=", 1)[1].split(",")
        elif a.startswith("--summary="):
            summary = Path(a.split("=", 1)[1])
        else:
            ladder.append(int(a))
    scale = 1e-6 if weak else 1.0
    con = MaterialContrast(CONTRAST.Dlambda * scale, CONTRAST.Dmu * scale, CONTRAST.Drho * scale)
    omega = ka_s * REF.beta / RADIUS
    r_far = R_MULT * RADIUS
    pts = obs_points(r_far, THETA)
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    exact = mie_scattered_displacement(graded_mie_result(omega, RADIUS, CORE, REF, con, n_max), pts)
    peak = float(np.max(np.abs(exact)))
    print(f"graded sphere, k_S a = {ka_s}, weak = {weak}, arms {arms}", flush=True)
    out: dict = {"ka_s": ka_s, "weak": weak, "arms": {}}
    for arm in arms:
        rows = []
        for n in ladder or [4]:
            t0 = time.perf_counter()
            if arm == "t9":
                fl = compute_sphere_foldy_lax_fft(
                    omega, RADIUS, REF, con, n_sub=n, k_hat=K_HAT, wave_type="P", contrast_profile=profile
                )
                u_p, u_s = foldy_lax_far_field(fl, pts / r_far, r_far, K_HAT, POL, wave_type="P")
                unknowns = 9 * fl.n_cells
            else:
                p = r = 0 if arm == "g0" else 1
                res = solve_graded_sphere(omega, RADIUS, REF, con, n, profile, K_HAT, POL, "P", p=p, r=r)
                u_p, u_s = graded_far_field(res, pts / r_far, r_far, K_HAT, POL, "P")
                unknowns = len(res.centres) * 9 * (1 if p == 0 else 4)
            err = float(np.max(np.abs(u_p + u_s - exact))) / peak
            rows.append((n, err, unknowns))
            dt = time.perf_counter() - t0
            print(
                f"  {arm}  n_sub {n:3d}  unknowns {unknowns:6d}  error {err:.4e}  {dt:7.1f} s", flush=True
            )
        for (n1, e1, _), (n2, e2, _) in zip(rows, rows[1:], strict=False):
            print(f"  {arm}  apparent order {n1} -> {n2}: {np.log(e1 / e2) / np.log(n2 / n1):.2f}")
        out["arms"][arm] = {
            "n_sub": [r[0] for r in rows],
            "error": [r[1] for r in rows],
            "unknowns": [r[2] for r in rows],
        }
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(out, indent=2) + "\n")
        print(f"  wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
