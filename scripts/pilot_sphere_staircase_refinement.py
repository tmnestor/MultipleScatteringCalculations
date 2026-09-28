#!/usr/bin/env python3
"""Pilot: separate a voxelised sphere's SHAPE error from its DISCRETISATION error.

The n0 = 8 staircase sphere is itself a union of cubes, so it has its own exact scattering solution, which
the voxel scheme converges to when each voxel is split into m^3 sub-voxels of the same shape (the
``inside`` mask of ``compute_sphere_foldy_lax_fft``). Hence
  * discretisation error: the change of the far field between refinements m -> 2m (the layer predicts
    second order);
  * shape error: (the converged staircase field) - (exact Mie for the sphere).
Far fields are stored per m in the scratch directory given, so later levels compare against earlier ones.

With --summary=<file>, the per-level errors, the apparent order and the shape error are written as a summary
(the data of the paper's convergence figure; see scripts/plot_convergence_orders.py).

Run:  conda run -n seismic python -u scripts/pilot_sphere_staircase_refinement.py \
        [--ka=0.5] [--summary=f] <outdir> 1 2
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.sphere_scattering import (  # noqa: E402
    compute_elastic_mie,
    foldy_lax_far_field,
    mie_scattered_displacement,
)
from cubic_scattering.sphere_scattering_fft import compute_sphere_foldy_lax_fft  # noqa: E402
from gate_sphere_cell_average_vs_mie import CONTRAST, R_MULT, REF, obs_points  # noqa: E402

KA_S = 0.5
RADIUS = 10.0
OMEGA = KA_S * REF.beta / RADIUS
N0 = 8
K_HAT = np.array([1.0, 0.0, 0.0])
POL = np.array([1.0, 0.0, 0.0])
THETA = np.linspace(0.2, np.pi - 0.2, 9)
R_FAR = R_MULT * RADIUS
PTS = obs_points(R_FAR, THETA)


def staircase(radius: float, n_coarse: int):
    """Cell-centre test for the n_coarse sphere staircase, on any refinement of its grid."""
    dd = 2.0 * radius / n_coarse

    def inside(pos: np.ndarray) -> bool:
        idx = np.clip(np.floor((pos + radius) / dd), 0, n_coarse - 1)
        return bool(np.linalg.norm(-radius + (idx + 0.5) * dd) < radius)

    return inside


def far_field(m: int) -> tuple[np.ndarray, int, float]:
    fl = compute_sphere_foldy_lax_fft(
        OMEGA, RADIUS, REF, CONTRAST, n_sub=N0 * m, k_hat=K_HAT, wave_type="P", inside=staircase(RADIUS, N0)
    )
    u_p, u_s = foldy_lax_far_field(fl, PTS / R_FAR, R_FAR, K_HAT, POL, wave_type="P")
    return u_p + u_s, fl.n_cells, fl.a_sub


def main() -> int:
    global KA_S, OMEGA
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    json_path = None
    for arg in sys.argv[1:]:
        if arg.startswith("--ka="):
            KA_S = float(arg.split("=", 1)[1])
            OMEGA = KA_S * REF.beta / RADIUS
        elif arg.startswith("--summary="):
            json_path = Path(arg.split("=", 1)[1])
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    levels = [int(a) for a in args[1:]] or [1]
    u_mie = mie_scattered_displacement(compute_elastic_mie(OMEGA, RADIUS, REF, CONTRAST), PTS)
    peak = float(np.max(np.abs(u_mie)))
    print(f"staircase n0 = {N0}, sphere k_S a = {KA_S}; errors are max|difference| / Mie peak", flush=True)
    for m in levels:
        f = out / f"staircase_ka{KA_S:g}_n{N0}_m{m}.npy"
        t0 = time.perf_counter()
        if f.exists():
            u = np.load(f)
            note = "(stored)"
        else:
            u, cells, a_sub = far_field(m)
            np.save(f, u)
            note = f"{cells} cells, k_S h = {OMEGA / REF.beta * a_sub:.4f}"
        dt = time.perf_counter() - t0
        err = np.max(np.abs(u - u_mie)) / peak
        print(f"  m = {m}: vs Mie {err:.4e}   {note}   {dt:7.1f} s", flush=True)
    pattern = f"staircase_ka{KA_S:g}_n{N0}_m*.npy"
    stored = sorted(out.glob(pattern), key=lambda p: int(p.stem.split("_m")[1]))
    fields = {int(p.stem.split("_m")[1]): np.load(p) for p in stored}
    ms = sorted(fields)
    for a, b in zip(ms, ms[1:], strict=False):
        d = np.max(np.abs(fields[b] - fields[a])) / peak
        print(f"  change m = {a} -> {b}: {d:.4e}")
    if len(ms) >= 3:
        a, b, c = ms[-3:]
        d1 = np.max(np.abs(fields[b] - fields[a]))
        d2 = np.max(np.abs(fields[c] - fields[b]))
        # an error C m^-p gives changes C (a^-p - b^-p) and C (b^-p - c^-p): solve their ratio for p
        # (bisection; the ratio increases monotonically with p), valid for unequal refinement ratios
        ratio = d1 / d2

        def g(p: float) -> float:
            return (a**-p - b**-p) / (b**-p - c**-p) - ratio

        lo, hi = 0.05, 12.0
        if g(lo) * g(hi) > 0:
            print(f"  apparent order: outside [{lo}, {hi}] (change ratio {ratio:.3f})")
        else:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if g(lo) * g(mid) > 0 else (lo, mid)
            p = 0.5 * (lo + hi)
            limit = fields[c] + (fields[c] - fields[b]) * c**-p / (b**-p - c**-p)
            print(f"  apparent order of the discretisation error (m = {a}, {b}, {c}): {p:.2f}")
            shape_err = np.max(np.abs(limit - u_mie)) / peak
            print(f"  extrapolated staircase field vs Mie (the shape error): {shape_err:.4e}")
            if json_path is not None:
                summary = {
                    "ka_s": KA_S,
                    "n0": N0,
                    "m": ms,
                    "error_vs_mie": [float(np.max(np.abs(fields[k] - u_mie)) / peak) for k in ms],
                    "change_from_previous": [None]
                    + [
                        float(np.max(np.abs(fields[k] - fields[j])) / peak)
                        for j, k in zip(ms, ms[1:], strict=False)
                    ],
                    "error_vs_extrapolated_staircase": [
                        float(np.max(np.abs(fields[k] - limit)) / peak) for k in ms
                    ],
                    "apparent_order": p,
                    "shape_error": float(shape_err),
                }
                json_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"  wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
