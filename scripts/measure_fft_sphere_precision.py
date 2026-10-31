"""Measure: the FFT sphere solver's error from solving in single precision, against its distance from Mie.

The kernel is built in complex128 whatever the solve's dtype (the recursions of the integrals need it): only the
GMRES solve, on the canonical kernel rounded to the device dtype, runs in complex64.  The question is whether that
rounding matters, and the measure of "matters" is the method's own error.  So, for each n_sub, the same sphere is
solved by ``compute_sphere_foldy_lax_fft_gpu``

  * in complex128 at a tight gmres_tol (the reference), and
  * in complex64 at each gmres_tol asked (single precision cannot reach the reference's tolerance),

and two errors are reported, both relative, in the max norm:

  * precision: complex64 against complex128, on the composite 9 x 9 T-matrix and on the far field;
  * discretisation: complex128 against the exact elastic Mie solution, on the far field (raw, and with the voxel
    volume scaled to the sphere's, as scripts/gate_sphere_cell_average_vs_mie.py does: the staircase's dilution).

Single precision is free where the first is far below the second.  The far field is P-incident along z, observed at
9 angles in the z-x plane at R_MULT radii (the distance gate_sphere_cell_average_vs_mie.py established: the
asymptotic and exact fields agree there).

Medium and contrast as tests/test_sphere_scattering_fft_gpu.py; radius 0.5 m; k_S a = --ka.  Each n_sub runs in a
fresh process (its own peak memory, and no allocator cache carried over).  Resumable: rows already in the output file
for the same settings are skipped.  Writes scripts/data/fft_sphere_precision_<host>_<device>.jsonl.

Run:  PYTHONPATH=. python scripts/measure_fft_sphere_precision.py --device cuda --n-sub 32,64,96
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"
RADIUS = 0.5
#: observation distance in sphere radii (gate_sphere_cell_average_vs_mie.py: R_MULT)
R_MULT = 5.0e4
KEYS = ("device", "ka", "tol128", "tols64")


def _rel(a, b) -> float:
    import numpy as np

    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def one(n_sub: int, device_name: str, ka: float, tol128: float, tols64: list[float]) -> dict:
    """Both precisions at one n_sub, and Mie; run in its own process (``--one``)."""
    import numpy as np
    import torch

    from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
    from cubic_scattering.sphere_scattering import (
        compute_elastic_mie,
        foldy_lax_far_field,
        mie_scattered_displacement,
    )
    from cubic_scattering.sphere_scattering_fft_gpu import compute_sphere_foldy_lax_fft_gpu
    from cubic_scattering.torch_gmres import get_device

    device = get_device() if device_name == "auto" else torch.device(device_name)
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
    omega = ka * ref.beta / RADIUS
    k_hat = np.array([1.0, 0.0, 0.0])  # z, axis 0, as the Mie solution's incident wave
    pol = np.array([1.0, 0.0, 0.0])

    theta = np.linspace(0.2, np.pi - 0.2, 9)
    r_far = R_MULT * RADIUS
    pts = np.zeros((len(theta), 3))
    pts[:, 0] = r_far * np.cos(theta)
    pts[:, 1] = r_far * np.sin(theta)

    def solve(dtype, tol):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.perf_counter()
        res = compute_sphere_foldy_lax_fft_gpu(
            omega=omega,
            radius=RADIUS,
            ref=ref,
            contrast=contrast,
            n_sub=n_sub,
            k_hat=k_hat,
            wave_type="P",
            gmres_tol=tol,
            gmres_maxiter=500,
            device=device,
            dtype=dtype,
        )
        seconds = time.perf_counter() - t0
        u_p, u_s = foldy_lax_far_field(res, pts / r_far, r_far, k_hat, pol, wave_type="P")
        return res, u_p + u_s, seconds

    ref128, u128, t128 = solve(torch.complex128, tol128)
    mie = compute_elastic_mie(omega, RADIUS, ref, contrast)
    u_mie = mie_scattered_displacement(mie, pts)
    vol_voxel = ref128.n_cells * (2.0 * ref128.a_sub) ** 3
    vol_true = 4.0 / 3.0 * np.pi * RADIUS**3

    singles = []
    for tol in tols64:
        r64, u64, t64 = solve(torch.complex64, tol)
        singles.append(
            {
                "tol": tol,
                "T_vs_128": _rel(r64.T_comp_9x9, ref128.T_comp_9x9),
                "far_vs_128": _rel(u64, u128),
                "far_vs_mie": _rel(u64, u_mie),
                "seconds": t64,
            }
        )
    return {
        "n_sub": n_sub,
        "device": device.type,
        "ka": ka,
        "tol128": tol128,
        "tols64": tols64,
        "n_cells": ref128.n_cells,
        "seconds128": t128,
        "far128_vs_mie": _rel(u128, u_mie),
        "far128_vs_mie_volume_corrected": _rel(u128 * (vol_true / vol_voxel), u_mie),
        "single": singles,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
    }


def report(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["n_sub"])
    r0 = rows[0]
    print(f"\n{r0['device']}, k_S a = {r0['ka']}; complex128 reference at gmres_tol {r0['tol128']:g}")
    print(
        f"{'n_sub':>6} {'cells':>10} {'128 vs Mie':>11} {'(vol. corr.)':>12} {'64 tol':>7} "
        f"{'T: 64 vs 128':>13} {'far: 64 vs 128':>15} {'64 vs Mie':>10} {'s 128':>7} {'s 64':>7}"
    )
    for r in rows:
        for i, s in enumerate(r["single"]):
            lead = (
                f"{r['n_sub']:>6} {r['n_cells']:>10,} {r['far128_vs_mie']:>11.2e} "
                f"{r['far128_vs_mie_volume_corrected']:>12.2e}"
                if i == 0
                else " " * 42
            )
            t128 = f"{r['seconds128']:>7.1f}" if i == 0 else " " * 7
            print(
                f"{lead} {s['tol']:>7.0e} {s['T_vs_128']:>13.2e} {s['far_vs_128']:>15.2e} {s['far_vs_mie']:>10.2e} "
                f"{t128} {s['seconds']:>7.1f}"
            )
    worst = max((s["far_vs_128"] for r in rows for s in r["single"]), default=0.0)
    disc = min((r["far128_vs_mie_volume_corrected"] for r in rows), default=0.0)
    print(
        f"\nlargest single-precision far-field error {worst:.1e}; smallest discretisation error {disc:.1e} "
        f"(ratio {worst / disc if disc else float('nan'):.1e})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--n-sub", default="32,64,96", help="sizes, comma-separated")
    ap.add_argument("--ka", type=float, default=1.0, help="k_S a")
    ap.add_argument("--tol128", type=float, default=1e-10, help="gmres_tol of the complex128 reference")
    ap.add_argument("--tols64", default="1e-5,1e-6", help="gmres_tol values for complex64, comma-separated")
    ap.add_argument("--out", type=Path, default=None, help="output file (default under scripts/data/)")
    ap.add_argument("--fit-only", action="store_true", help="report the rows in the file; run nothing")
    ap.add_argument("--one", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    tols64 = [float(t) for t in a.tols64.split(",") if t]

    if a.one is not None:
        print(json.dumps(one(a.one, a.device, a.ka, a.tol128, tols64)))
        return

    import torch

    from cubic_scattering.torch_gmres import get_device

    device = get_device() if a.device == "auto" else torch.device(a.device)
    host = os.environ.get("MSC_HOST") or socket.gethostname().split(".")[0]  # on an AWS runner, the instance type
    out = a.out or DATA / f"fft_sphere_precision_{host}_{device.type}.jsonl"
    settings = {"device": device.type, "ka": a.ka, "tol128": a.tol128, "tols64": tols64}
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()] if out.exists() else []
    mine = [r for r in rows if all(r.get(k) == settings[k] for k in KEYS) and "error" not in r]
    done = {r["n_sub"] for r in mine}
    if not a.fit_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        for n in sorted(int(x) for x in a.n_sub.split(",")):
            if n in done:
                continue
            cmd = [sys.executable, __file__, "--one", str(n), "--device", device.type, "--ka", str(a.ka),
                   "--tol128", str(a.tol128), "--tols64", a.tols64]  # fmt: skip
            print(f"n_sub = {n} ...", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                tail = proc.stderr.strip().splitlines()[-40:]
                with out.open("a") as f:
                    f.write(json.dumps({**settings, "n_sub": n, "error": tail}) + "\n")
                print(f"  failed (exit {proc.returncode}); the end of its output:\n    " + "\n    ".join(tail))
                break
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mine.append(row)
            worst = max(s["far_vs_128"] for s in row["single"])
            print(f"  {row['n_cells']:,} cells: complex64 far field within {worst:.1e} of complex128", flush=True)
    if mine:
        report(mine)
        print(f"\nrows in {out}")


if __name__ == "__main__":
    main()
