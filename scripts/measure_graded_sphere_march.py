#!/usr/bin/env python3
"""The graded sphere by the impedance march (invariant imbedding), on the planes z = -a and z = +a.

The march of ``gate_sphere_vs_impedance_march`` is run on the smoothly graded sphere of the voxel tests
(radius 10 m, homogeneous core r < a/2, contrast falling to zero across a/2 < r < a as the smoothstep),
and scored against the exact partial-wave solution of that sphere. The march returns the reflected
plane-wave amplitudes on the plane z = -a above the sphere and the transmitted ones on z = +a below it.

What is compared. The march needs a laterally periodic grid, so it solves a square ARRAY of spheres. Its
answer is scored against the isolated sphere's exact plane-wave spectrum sampled at the array's orders
(Poisson summation: exact but for scattering between the spheres of the array). The error reported is

    | got - want | / | want |   over the orders of the grid, Nyquist orders excluded,

for the reflected column, and the same for the transmitted column with the direct wave removed from
both, so that both are relative to what the sphere scatters. It is reported on the tangent planes and on
the planes z = -GAP a and +GAP a, GAP = 1.25, where the voxel solution is read
(``measure_graded_sphere_planes.py``): each order is carried across the homogeneous gap by its own
phase exp(i k_z (GAP - 1) a), which is exact, and which damps the evanescent orders.

Three refinement axes, each varied with the others held: the depth step, the lateral pitch, the period.

The march module keeps its problem in module constants; they are set here to the graded sphere, and its
medium and its exact solution are replaced by the graded ones. Nothing else in it is touched.

Run small first:
    python -u scripts/measure_graded_sphere_march.py --ka=0.5 --n=8 --period=5 --steps=8
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_graded_sphere import graded_mie_result  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, REF  # noqa: E402
from pilot_graded_sphere_vs_exact import CORE, RADIUS  # noqa: E402
from scripts import gate_sphere_vs_impedance_march as march  # noqa: E402
from scripts.gate_first_order_lateral_impedance_3d import Slice  # noqa: E402

GAP = 1.25


def smoothstep(r: np.ndarray) -> np.ndarray:
    """The contrast factor: 1 in the core, 10 x^3 - 15 x^4 + 6 x^5 across the shell, 0 outside."""
    x = np.clip((RADIUS - r) / (RADIUS - CORE), 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def graded_slice_at(nx, ny, lx, ly, z_c, radius, *, voxelised=False, amplitude=1.0):
    """The graded sphere's medium on the lateral grid as a function of depth (the march's slice factory)."""
    x = (np.arange(nx) - 0.5 * (nx - 1)) * (lx / nx)
    y = (np.arange(ny) - 0.5 * (ny - 1)) * (ly / ny)
    rho2 = ((x**2)[:, None] + (y**2)[None, :]).reshape(-1)

    def at(z: float) -> Slice:
        s = amplitude * smoothstep(np.sqrt(rho2 + (z - z_c) ** 2))
        rho = REF.rho + s * CONTRAST.Drho
        mu = REF.mu + s * CONTRAST.Dmu
        lam = REF.lam + s * CONTRAST.Dlambda
        return Slice(np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho), rho)

    return at


def configure(ka_s: float) -> None:
    """Point the march module at the graded sphere at this frequency."""
    omega = ka_s * REF.beta / RADIUS
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    mie = graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max)
    march.REF, march.OMEGA, march.RADIUS = REF, omega, RADIUS
    # The march module calls these with its own arguments. Those that describe the SHARP sphere must not
    # reach the graded one: its ``amplitude`` is the sharp sphere's fractional contrast (0.1), and passing
    # it on scales the graded profile to a tenth (an error of 0.89 where the answer is 0.06).
    def slice_factory(nx, ny, lx, ly, z_c, radius, **_sharp_sphere_options):
        return graded_slice_at(nx, ny, lx, ly, z_c, radius, amplitude=1.0)

    march.sphere_slice_at = slice_factory
    march.compute_elastic_mie = lambda *_args, **_kwargs: mie


def errors(n: int, period_radii: float, nstep: int) -> tuple[float, float, float, float, float]:
    """(R error, T error relative to the scattered part, the same two on the planes at GAP, seconds)."""
    lx = period_radii * RADIUS
    t0 = time.perf_counter()
    r_mat, t_mat = march.reflection_transmission(n, n, lx, lx, nstep)
    r_want = march.mie_prediction(n, n, lx, lx)
    t_want = march.mie_transmission(n, n, lx, lx)
    keep = ~np.tile(march.nyquist_orders(n, n), 3)
    direct = np.zeros_like(t_want)
    direct[0] = np.exp(2j * (march.OMEGA / REF.alpha) * RADIUS)
    r_got, t_got = r_mat[:, 0], t_mat[:, 0]
    e_r = float(np.linalg.norm((r_got - r_want)[keep]) / np.linalg.norm(r_want[keep]))
    e_t = float(np.linalg.norm((t_got - t_want)[keep]) / np.linalg.norm((t_want - direct)[keep]))
    kp, ks = march.OMEGA / REF.alpha, march.OMEGA / REF.beta
    q = march.q_grid(n, n, lx, lx)
    kz = np.concatenate([march.kz_of(q, kp), march.kz_of(q, ks), march.kz_of(q, ks)])
    carry = np.exp(1j * kz * (GAP - 1.0) * RADIUS)
    g_r = float(np.linalg.norm((carry * (r_got - r_want))[keep]) / np.linalg.norm((carry * r_want)[keep]))
    g_t = float(
        np.linalg.norm((carry * (t_got - t_want))[keep]) / np.linalg.norm((carry * (t_want - direct))[keep])
    )
    return e_r, e_t, g_r, g_t, time.perf_counter() - t0


def main() -> int:
    ka_s, ns, periods, steps, summary = 0.5, [8], [5.0], [8], None
    for a in sys.argv[1:]:
        key, _, val = a.partition("=")
        if key == "--ka":
            ka_s = float(val)
        elif key == "--n":
            ns = [int(v) for v in val.split(",")]
        elif key == "--period":
            periods = [float(v) for v in val.split(",")]
        elif key == "--steps":
            steps = [int(v) for v in val.split(",")]
        elif key == "--summary":
            summary = Path(val)
        else:
            raise SystemExit(f"unknown argument {a}")
    configure(ka_s)
    print(f"graded sphere by the impedance march, k_S a = {ka_s}", flush=True)
    rows = []
    for period in periods:
        for n in ns:
            for nstep in steps:
                e_r, e_t, g_r, g_t, dt = errors(n, period, nstep)
                rows.append(
                    {
                        "period_radii": period,
                        "n_lateral": n,
                        "pitch_radii": period / n,
                        "nstep": nstep,
                        "unknowns": 3 * n * n,
                        "error_R": e_r,
                        "error_T": e_t,
                        "error_R_at_gap": g_r,
                        "error_T_at_gap": g_t,
                        "seconds": dt,
                    }
                )
                print(
                    f"  period {period:5.1f} a  N {n:3d} (pitch {period / n:.3f} a)  steps {nstep:4d}  "
                    f"unknowns per plane {3 * n * n:6d}   R {e_r:.3e}   T {e_t:.3e}   "
                    f"at {GAP} a: R {g_r:.3e}   T {g_t:.3e}   {dt:7.1f} s",
                    flush=True,
                )
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"ka_s": ka_s, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
