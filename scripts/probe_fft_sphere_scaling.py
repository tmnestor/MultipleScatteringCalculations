"""Measure: memory and time of the GPU FFT sphere solver against n_sub, and extrapolate them, to choose a machine.

``compute_sphere_foldy_lax_fft_gpu`` for a sphere of n_sub cells along the diameter, at each n_sub in a fresh process
so that its peak memory is its own: the kernel's build (``_point_canonical_kernel``), each GMRES solve (iterations,
products, time), the total, the peak host memory (ru_maxrss) and the peak device memory (CUDA: the allocator's peak;
MPS: the driver's allocation, sampled at every product).  Alongside, the two arrays whose size is known in advance:
the Krylov basis ``torch_gmres`` allocates in full, (gmres_maxiter + 1) x 9 n_cells, which is set by gmres_maxiter
and not by the iterations taken, and one padded grid stack of the product, 9 nP^3 with nP = fast_odd_size(2 n_sub - 1).

Then a power law in the number of cells, fitted to the largest three measured sizes, extrapolates the peak memory
and the times to the sizes asked for; the Krylov basis there is exact.  The extrapolation assumes the iteration
count stays as measured: at fixed k_S a the discretisation is refined, not the physics changed, so it should, but a
solve that reached gmres_maxiter without converging makes its times a lower bound (flagged).

Medium and contrast as tests/test_sphere_scattering_fft_gpu.py: (alpha, beta, rho) = (5000, 3000, 2500),
(Dlambda, Dmu, Drho) = (2e9, 1e9, 100); radius 0.5 m; k_S a = --ka (default 1), the same at every n_sub.

The kernel's build reads the k-series coefficient store (numerics.yml: kseries_store), which grows on first use by
the offsets a size needs: a run that grew it is flagged "cold", and its kernel time and memory include that one-time
cost (56 s against 0.6 s at n_sub = 16, measured).  --warm-store grows it first; the fits use the warm rows.

Resumable: the rows already in the output file for the same device, dtype and settings are skipped.  Writes
scripts/data/fft_sphere_scaling_<host>_<device>_<dtype>.jsonl.

Run:  PYTHONPATH=. python scripts/probe_fft_sphere_scaling.py
      PYTHONPATH=. python scripts/probe_fft_sphere_scaling.py --device cuda --dtype complex128 --n-sub 16,24,32,48,64
      PYTHONPATH=. python scripts/probe_fft_sphere_scaling.py --fit-only        (the table from the file, no runs)
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"
RADIUS = 0.5
#: the settings that identify a row, beside n_sub
KEYS = ("device", "dtype", "ka", "maxiter", "tol")


def _peak_host_gb() -> float:
    """Peak resident memory of this process: ru_maxrss is bytes on macOS, kilobytes on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1e3 / 1e9


class _DeviceMemory:
    """The device's peak memory: the CUDA allocator keeps its own; on MPS the driver's figure is sampled."""

    def __init__(self, device) -> None:
        import torch

        self.torch, self.type, self.peak = torch, device.type, 0
        if self.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sync(self) -> None:
        if self.type == "cuda":
            self.torch.cuda.synchronize()
        elif self.type == "mps":
            self.torch.mps.synchronize()

    def sample(self) -> None:
        if self.type == "mps":
            self.peak = max(self.peak, self.torch.mps.driver_allocated_memory())

    def peak_gb(self) -> float | None:
        if self.type == "cuda":
            return self.torch.cuda.max_memory_allocated() / 1e9
        if self.type == "mps":
            return self.peak / 1e9
        return None  # on the CPU the device memory is the host's


def one(n_sub: int, device_name: str, dtype_name: str, ka: float, maxiter: int, tol: float) -> dict:
    """One solve at n_sub, measured; run in its own process (``--one``)."""
    import torch

    from cubic_scattering import sphere_scattering_fft_gpu as G
    from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
    from cubic_scattering.sphere_scattering_fft import _build_grid_index_map, fast_odd_size
    from cubic_scattering.torch_gmres import get_device, select_dtype

    device = get_device() if device_name == "auto" else torch.device(device_name)
    dtype = select_dtype(device) if dtype_name == "auto" else getattr(torch, dtype_name)
    torch.zeros(1, device=device, dtype=dtype)  # the device's context, in the baseline
    baseline = _peak_host_gb()  # the interpreter, the imports and the context: present at any size
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    contrast = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
    omega = ka * ref.beta / RADIUS

    n_cells = len(_build_grid_index_map(RADIUS, n_sub)[1])
    n_p = fast_odd_size(2 * n_sub - 1)
    item = torch.empty(0, dtype=dtype).element_size()

    store = Path.home() / ".cache" / "cubic_scattering"  # numerics.yml: kseries_store
    store_before = sum(f.stat().st_size for f in store.glob("*")) if store.exists() else 0
    mem = _DeviceMemory(device)
    solves: list[dict] = []
    t_kernel = 0.0
    real_kernel, real_gmres = G._point_canonical_kernel, G.torch_gmres

    def timed_kernel(*args, **kwargs):
        nonlocal t_kernel
        mem.sync()
        t0 = time.perf_counter()
        out = real_kernel(*args, **kwargs)
        mem.sync()
        t_kernel += time.perf_counter() - t0
        return out

    def timed_gmres(matvec, b, **kwargs):
        products = 0

        def counted(w):
            nonlocal products
            products += 1
            y = matvec(w)
            mem.sample()
            return y

        mem.sync()
        t0 = time.perf_counter()
        x, n_iter, rel_res = real_gmres(counted, b, **kwargs)
        mem.sync()
        dt = time.perf_counter() - t0
        solves.append({"iterations": n_iter, "products": products, "seconds": dt, "rel_res": float(rel_res)})
        return x, n_iter, rel_res

    # both are looked up as module globals at call time, so patching the module is enough
    G._point_canonical_kernel, G.torch_gmres = timed_kernel, timed_gmres
    t0 = time.perf_counter()
    G.compute_sphere_foldy_lax_fft_gpu(
        omega=omega,
        radius=RADIUS,
        ref=ref,
        contrast=contrast,
        n_sub=n_sub,
        wave_type="P",
        gmres_tol=tol,
        gmres_maxiter=maxiter,
        device=device,
        dtype=dtype,
    )
    mem.sync()
    t_total = time.perf_counter() - t0

    products = sum(s["products"] for s in solves)
    t_solve = sum(s["seconds"] for s in solves)
    return {
        "n_sub": n_sub,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "ka": ka,
        "maxiter": maxiter,
        "tol": tol,
        "n_cells": n_cells,
        "unknowns": 9 * n_cells,
        "n_p": n_p,
        "krylov_gb": (maxiter + 1) * 9 * n_cells * item / 1e9,
        "grid_stack_gb": 9 * n_p**3 * item / 1e9,
        "seconds_total": t_total,
        "seconds_kernel": t_kernel,
        "seconds_solves": t_solve,
        "seconds_per_product": t_solve / products if products else None,
        "solves": solves,
        "max_iterations": max(s["iterations"] for s in solves),
        "converged": all(s["rel_res"] <= tol for s in solves),
        "kseries_store_grew": (sum(f.stat().st_size for f in store.glob("*")) if store.exists() else 0) > store_before,
        "peak_host_gb": _peak_host_gb(),
        "baseline_host_gb": baseline,
        "peak_device_gb": mem.peak_gb(),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
    }


def _fit(rows: list[dict], key, base: float = 0.0) -> tuple[float, float] | None:
    """Least-squares power law y - base = c n_cells^p over the largest three rows; None if fewer than two.

    key is a row's field or a function of the row."""
    get = key if callable(key) else (lambda r: r.get(key))
    pts = sorted((r["n_cells"], get(r) - base) for r in rows if get(r) is not None and get(r) - base > 0)[-3:]
    if len(pts) < 2:
        return None
    xs = [math.log(x) for x, _ in pts]
    ys = [math.log(y) for _, y in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
    return math.exp(my - p * mx), p


def report(rows: list[dict], predict: list[int]) -> None:
    from cubic_scattering.sphere_scattering_fft import _build_grid_index_map, fast_odd_size

    rows = sorted(rows, key=lambda r: r["n_sub"])
    peak = "peak_device_gb" if rows[0]["peak_device_gb"] is not None else "peak_host_gb"
    r0 = rows[0]
    print(
        f"\n{r0['device']} {r0['dtype']}, k_S a = {r0['ka']}, gmres_maxiter = {r0['maxiter']}, gmres_tol = {r0['tol']}"
    )
    print(
        f"{'n_sub':>6} {'cells':>10} {'iters':>6} {'s/product':>10} {'kernel s':>9} {'solves s':>9} {'total s':>9} "
        f"{'Krylov GB':>10} {'host GB':>8} {'device GB':>10}"
    )
    for r in rows:
        dev = f"{r['peak_device_gb']:.2f}" if r["peak_device_gb"] is not None else "-"
        flags = ([] if r["converged"] else ["not converged"]) + (
            ["cold k-series store"] if r.get("kseries_store_grew") else []
        )
        print(
            f"{r['n_sub']:>6} {r['n_cells']:>10,} {r['max_iterations']:>6} {r['seconds_per_product']:>10.3g} "
            f"{r['seconds_kernel']:>9.1f} {r['seconds_solves']:>9.2f} {r['seconds_total']:>9.1f} "
            f"{r['krylov_gb']:>10.2f} {r['peak_host_gb']:>8.2f} {dev:>10}"
            + (f"  ({', '.join(flags)})" if flags else "")
        )
    if not predict or len(rows) < 2:
        return
    warm = [r for r in rows if not r.get("kseries_store_grew")]
    use = warm if len(warm) >= 2 else rows  # a cold store inflates the kernel's time and memory
    if peak == "peak_host_gb":
        base = max(r.get("baseline_host_gb", 0.0) for r in use)
        above = lambda r: r["peak_host_gb"] - r.get("baseline_host_gb", 0.0)
    else:
        base, above = 0.0, (lambda r: r["peak_device_gb"])
    fits = {
        "memory": _fit(use, above),
        "kernel": _fit(use, "seconds_kernel"),
        "product": _fit(use, "seconds_per_product"),
    }
    products = max(sum(s["products"] for s in r["solves"]) for r in use[-3:])
    item = r0["krylov_gb"] / ((r0["maxiter"] + 1) * 9 * r0["n_cells"]) * 1e9
    print(
        f"\nextrapolated from {'warm' if use is warm else 'all'} rows: power laws in cells over the largest 3 "
        f"(memory above the {base:.2f} GB baseline; exponents "
        + ", ".join(f"{k} {f[1]:.2f}" for k, f in fits.items() if f)
        + f"), {products} products a sphere as measured"
    )
    print(f"{'n_sub':>6} {'cells':>10} {'Krylov GB':>10} {'grid GB':>8} {'peak GB':>8} {'kernel s':>9} {'solves s':>9}")
    for n in predict:
        cells = len(_build_grid_index_map(RADIUS, n)[1])
        kry = (r0["maxiter"] + 1) * 9 * cells * item / 1e9
        grid = 9 * fast_odd_size(2 * n - 1) ** 3 * item / 1e9
        fm = fits["memory"]
        mem = base + max(fm[0] * cells ** fm[1] if fm else 0.0, kry)  # the basis alone is a floor
        fk, fp = fits["kernel"], fits["product"]
        tk = f"{fk[0] * cells ** fk[1]:.0f}" if fk else "-"
        ts = f"{products * fp[0] * cells ** fp[1]:.0f}" if fp else "-"
        print(f"{n:>6} {cells:>10,} {kry:>10.1f} {grid:>8.1f} {mem:>8.1f} {tk:>9} {ts:>9}")
    if not all(r["converged"] for r in rows):
        print("some solves reached gmres_maxiter without converging: the solve times are lower bounds")
    if use is not warm:
        print(
            "fewer than two runs had a warm k-series store: the kernel times and memory include its one-time growth "
            "(--warm-store, or run again)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--dtype", default="auto", help="auto, complex64 or complex128")
    ap.add_argument("--n-sub", default="12,16,24,32,40", help="sizes to measure, comma-separated")
    ap.add_argument("--predict", default="64,96,128,160", help="sizes to extrapolate to")
    ap.add_argument("--ka", type=float, default=1.0, help="k_S a, the same at every size")
    ap.add_argument("--maxiter", type=int, default=200, help="gmres_maxiter (sets the Krylov basis' size)")
    ap.add_argument("--tol", type=float, default=1e-8, help="gmres_tol")
    ap.add_argument("--out", type=Path, default=None, help="output file (default under scripts/data/)")
    ap.add_argument("--fit-only", action="store_true", help="report the rows in the file; run nothing")
    ap.add_argument(
        "--warm-store",
        action="store_true",
        help="first grow the k-series store with one unrecorded solve at the largest size, so that every "
        "measured run finds it warm (the store is kept between runs; on AWS too)",
    )
    ap.add_argument("--one", type=int, default=None, help=argparse.SUPPRESS)  # the child's mode
    a = ap.parse_args()

    if a.one is not None:
        print(json.dumps(one(a.one, a.device, a.dtype, a.ka, a.maxiter, a.tol)))
        return

    import torch

    from cubic_scattering.torch_gmres import get_device, select_dtype

    device = get_device() if a.device == "auto" else torch.device(a.device)
    dtype = (select_dtype(device) if a.dtype == "auto" else getattr(torch, a.dtype)).__str__().removeprefix("torch.")
    host = socket.gethostname().split(".")[0]
    out = a.out or DATA / f"fft_sphere_scaling_{host}_{device.type}_{dtype}.jsonl"
    settings = {"device": device.type, "dtype": dtype, "ka": a.ka, "maxiter": a.maxiter, "tol": a.tol}

    rows = []
    if out.exists():
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    mine = [r for r in rows if all(r.get(k) == settings[k] for k in KEYS) and "error" not in r]
    done = {r["n_sub"] for r in mine}
    if not a.fit_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        todo = sorted(int(x) for x in a.n_sub.split(",") if int(x) not in done)
        if a.warm_store and todo:
            print(f"growing the k-series store at n_sub = {todo[-1]} (not recorded) ...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--one",
                    str(todo[-1]),
                    "--device",
                    device.type,
                    "--dtype",
                    dtype,
                    "--ka",
                    str(a.ka),
                    "--maxiter",
                    str(a.maxiter),
                    "--tol",
                    str(a.tol),
                ],
                capture_output=True,
                check=False,
            )
        for n in sorted(int(s) for s in a.n_sub.split(",")):
            if n in done:
                continue
            cmd = [
                sys.executable,
                __file__,
                "--one",
                str(n),
                "--device",
                device.type,
                "--dtype",
                dtype,
                "--ka",
                str(a.ka),
                "--maxiter",
                str(a.maxiter),
                "--tol",
                str(a.tol),
            ]
            print(f"n_sub = {n} ...", flush=True)
            done_proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if done_proc.returncode != 0:
                # most likely out of memory: record it, and stop, since larger sizes need more
                err = {**settings, "n_sub": n, "error": done_proc.stderr.strip().splitlines()[-1:]}
                with out.open("a") as f:
                    f.write(json.dumps(err) + "\n")
                print(f"  failed (exit {done_proc.returncode}): {done_proc.stderr.strip()[-500:]}")
                break
            row = json.loads(done_proc.stdout.strip().splitlines()[-1])
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mine.append(row)
            print(
                f"  {row['n_cells']:,} cells, {row['seconds_total']:.1f} s, peak host {row['peak_host_gb']:.2f} GB",
                flush=True,
            )
    if mine:
        report(mine, [int(s) for s in a.predict.split(",") if s])
        print(f"\nrows in {out}")


if __name__ == "__main__":
    main()
