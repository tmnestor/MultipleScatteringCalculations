#!/usr/bin/env python3
"""A record section of plane-wave scattering by the graded sphere: voxels against the exact solution.

For the thesis's Chapter 7. A P wave travels down (+z, axis 0) onto the graded sphere of
``measure_graded_sphere_planes.py`` (radius a, core a/2, smoothstep shell). Receivers lie on a line in the
plane z = -GAP a above the sphere, at x = 0 ... X_MAX a (y = 0); by the symmetry of the problem about the
axis the section is the same on the other side. At each frequency of the record the scattered displacement
there is computed twice: exactly (``crosscheck_graded_sphere.graded_mie_result``, the per-order T-matrices
of the radial equations, and ``sphere_scattering.mie_scattered_displacement`` at the receivers' finite
distance), and by graded voxels of degree p = r = 1 with n cells across the sphere
(``graded_voxel.fft.solve_graded_sphere_fft``, its kernel in single precision and memory-mapped to --disk,
and ``graded_field`` at the receivers). Each frequency is a row of
scripts/data/graded_sphere_record_n<n>.jsonl, so the run resumes where it stopped.

THE TIME DOMAIN. Time dependence e^{-i omega t}. The record has length T (--T, in a / beta), so the
frequencies are omega_k = k 2 pi / T, k = 1 ... N, up to --fmax times the wavelet's peak. The wavelet is a
Ricker pulse peaking at omega_0 (--k0: k_S a at the peak), W(omega) = (omega / omega_0)^2
exp(-(omega / omega_0)^2), delayed by --delay periods of the peak so that it is causal. Time is counted from
the incident wave's crossing of the receivers' plane, so that it reads as a two-way time: the spectra are
multiplied by exp(-i omega z_r / alpha), z_r = -GAP a. The traces are
    u(x, t) = (1 / pi) Re sum_k W(omega_k) u(x, omega_k) exp(-i omega_k t) d omega,
the exact and the voxel ones alike (``synthesise``).

Run:  python -u scripts/record_graded_sphere.py --n=32 --disk=/tmp [--k0=3] [--fmax=3] [--T=12]
      python -u scripts/record_graded_sphere.py --n=32 --synth    (the traces, from the rows computed)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).parent / "data"
GAP = 1.25  # the receivers' plane, z = -GAP a
X_MAX = 3.0  # the line's extent, in radii


def settings(argv: list[str]) -> dict:
    s = {"n": 32, "disk": None, "k0": 3.0, "fmax": 3.0, "T": 12.0, "nrec": 31, "delay": 1.5, "synth": False}
    for a in argv:
        key, _, val = a.lstrip("-").partition("=")
        if key == "synth":
            s["synth"] = True
        elif key == "disk":
            s["disk"] = val
        elif key in ("n", "nrec"):
            s[key] = int(val)
        elif key in s:
            s[key] = float(val)
        else:
            raise SystemExit(f"record_graded_sphere: unknown option {a!r}")
    return s


def frequencies(s: dict, beta: float, radius: float) -> np.ndarray:
    """omega_k = k 2 pi / T up to fmax omega_0."""
    t_rec = s["T"] * radius / beta
    d_omega = 2 * np.pi / t_rec
    omega0 = s["k0"] * beta / radius
    return d_omega * np.arange(1, int(np.floor(s["fmax"] * omega0 / d_omega)) + 1)


def receivers(s: dict, radius: float) -> np.ndarray:
    return np.array([[-GAP * radius, x * radius, 0.0] for x in np.linspace(0.0, X_MAX, s["nrec"])])


def one_frequency(omega: float, s: dict) -> dict:
    """The exact and the voxel scattered displacement at the receivers, at one frequency."""
    from crosscheck_graded_sphere import graded_mie_result
    from cubic_scattering.graded_voxel.farfield import graded_field
    from cubic_scattering.graded_voxel.fft import solve_graded_sphere_fft
    from cubic_scattering.sphere_scattering import mie_scattered_displacement
    from gate_sphere_cell_average_vs_mie import CONTRAST, REF
    from pilot_graded_sphere_vs_exact import CORE, RADIUS, profile

    rec = receivers(s, RADIUS)
    ka_s = omega * RADIUS / REF.beta
    n_max = max(8, int(np.ceil(ka_s + 4 * ka_s ** (1 / 3) + 6)))
    t0 = time.perf_counter()
    exact = mie_scattered_displacement(graded_mie_result(omega, RADIUS, CORE, REF, CONTRAST, n_max), rec)
    t_exact = time.perf_counter() - t0
    path = None if s["disk"] is None else Path(s["disk"]) / f"graded_sphere_record_kh_{s['n']}.npy"
    t0 = time.perf_counter()
    res = solve_graded_sphere_fft(
        omega,
        RADIUS,
        REF,
        CONTRAST,
        s["n"],
        profile,
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        "P",
        p=1,
        r=1,
        gmres_tol=1e-8,
        kernel_dtype=np.complex64,
        kernel_path=path,
    )
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    voxel, _ = graded_field(res, rec, n_gauss=6, strain=False)  # the record section uses displacement only
    t_read = time.perf_counter() - t0
    if path is not None:
        path.unlink()
    return {
        "omega": omega,
        "ka_s": ka_s,
        "n": s["n"],
        "cells": len(res.centres),
        "exact": [[[z.real, z.imag] for z in row] for row in exact],
        "voxel": [[[z.real, z.imag] for z in row] for row in voxel],
        "error": float(np.linalg.norm(voxel - exact) / np.linalg.norm(exact)),
        "exact_s": t_exact,
        "solve_s": t_solve,
        "readout_s": t_read,
    }


def load(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[round(r["omega"], 9)] = r
    return rows


def synthesise(s: dict, rows: dict, alpha: float, beta: float, radius: float) -> dict:
    """The traces of the exact and the voxel displacement, (n_rec, 3, n_t), and the times (s)."""
    omegas = frequencies(s, beta, radius)
    missing = [w for w in omegas if round(w, 9) not in rows]
    if missing:
        raise SystemExit(
            f"record_graded_sphere: {len(missing)} of {len(omegas)} frequencies not computed yet"
        )
    t_rec = s["T"] * radius / beta
    omega0 = s["k0"] * beta / radius
    d_omega = omegas[0]
    n_t = 512
    t = np.arange(n_t) * t_rec / n_t
    delay = s["delay"] * 2 * np.pi / omega0
    z_r = -GAP * radius
    out = {"t": t}
    for kind in ("exact", "voxel"):
        spec = np.array(
            [
                np.array(rows[round(w, 9)][kind])[..., 0] + 1j * np.array(rows[round(w, 9)][kind])[..., 1]
                for w in omegas
            ]
        )  # (n_omega, n_rec, 3)
        w = (omegas / omega0) ** 2 * np.exp(-((omegas / omega0) ** 2)) * np.exp(1j * omegas * delay)
        w = w * np.exp(-1j * omegas * z_r / alpha)
        phase = np.exp(-1j * np.outer(omegas, t))  # (n_omega, n_t)
        out[kind] = (d_omega / np.pi) * np.real(np.einsum("w,wrc,wt->rct", w, spec, phase))
    return out


#: the figure's traces: every STRIDE-th receiver (0.3a apart), mirrored to x < 0; the difference
#: magnified MAGNIFY times
#: (1e4: the largest difference is 2.5e-5 of the largest trace, so shown at a quarter of it)
STRIDE, MAGNIFY, N_SAMPLES = 3, 1.0e4, 256


def wiggles(s: dict, tr: dict, comp: int) -> str:
    """A pgfplots table of one component's section: columns t (ms) and, per trace, the receiver's offset
    (in radii) plus the exact, the voxel and the magnified difference displacement, one gain for the whole
    section (true relative amplitudes, the largest excursion 0.9 of the traces' spacing); traces separated
    by a row of nan (``unbounded coords=jump``). Mirrored to x < 0: u_z is even in x, u_x odd."""
    x = np.linspace(0.0, X_MAX, s["nrec"])
    keep = np.arange(0, s["nrec"], STRIDE)
    step = max(1, len(tr["t"]) // N_SAMPLES)
    t_ms = tr["t"][::step] * 1e3
    exact, voxel = tr["exact"][:, comp, ::step], tr["voxel"][:, comp, ::step]
    gain = 0.9 * (x[STRIDE] - x[0]) / np.abs(exact[keep]).max()
    lines = ["t xe xv xd"]
    parity = -1.0 if comp == 1 else 1.0
    traces = [(-x[i], i, parity) for i in keep[::-1] if i != 0] + [(x[i], i, 1.0) for i in keep]
    for x0, i, f in traces:
        for j, tj in enumerate(t_ms):
            e, v = f * exact[i, j], f * voxel[i, j]
            lines.append(
                f"{tj:.4f} {x0 + gain * e:.6f} {x0 + gain * v:.6f} {x0 + gain * MAGNIFY * (v - e):.6f}"
            )
        lines.append("nan nan nan nan")
    return "\n".join(lines) + "\n"


def main() -> int:
    s = settings(sys.argv[1:])
    from gate_sphere_cell_average_vs_mie import REF
    from pilot_graded_sphere_vs_exact import RADIUS

    # The rows (one per frequency) are shared by every record length: a frequency's solution does not depend
    # on T, and omega_k = 2 pi k / T puts every frequency of a record of length T among those of 2T. The
    # synthesised traces do depend on T, so for any T but the thesis's 12 a / beta (Chapter 7) they carry it
    # in their names, and the published files are not overwritten.
    path = DATA / f"graded_sphere_record_n{s['n']}.jsonl"
    rows = load(path)
    tag = f"n{s['n']}" if s["T"] == 12.0 else f"n{s['n']}_T{s['T']:g}"
    if s["synth"]:
        tr = synthesise(s, rows, REF.alpha, REF.beta, RADIUS)
        np.savez(DATA / f"graded_sphere_record_{tag}_traces.npz", **tr)
        err = np.linalg.norm(tr["voxel"] - tr["exact"]) / np.linalg.norm(tr["exact"])
        print(f"traces written; voxel against exact over the section: {err:.2e}", flush=True)
        for comp, name in ((0, "z"), (1, "x")):
            out = DATA / f"graded_sphere_record_{tag}_{name}.dat"
            out.write_text(wiggles(s, tr, comp))
            print(f"  {out.name}", flush=True)
        return 0
    omegas = frequencies(s, REF.beta, RADIUS)
    print(
        f"graded sphere record section: n = {s['n']}, {len(omegas)} frequencies, k_S a up to"
        f" {omegas[-1] * RADIUS / REF.beta:.2f}, {s['nrec']} receivers",
        flush=True,
    )
    for omega in omegas[::-1]:  # the highest first: the longest solves, and the size's real test
        if round(omega, 9) in rows:
            continue
        row = one_frequency(float(omega), s)
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  k_S a {row['ka_s']:6.3f}: solve {row['solve_s']:7.1f} s, readout {row['readout_s']:6.1f} s,"
            f" exact {row['exact_s']:5.1f} s; voxel against exact {row['error']:.2e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
