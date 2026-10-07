"""Self-convergence of variants of the series solve, to find which coupling converges slowly.

variant 'full'    : as the paper
variant 'nodens'  : the density contrast dropped from the OPERATOR (far field and rhs unchanged)
Saves the far-field series (cell model) per grid to runs/<variant>_<rc>_<n>.npz.
Usage: python split_test.py variant [--rc=1] [--J=6] [--spill=DIR] [--workers=4] n1 n2 ...
  --spill=DIR : transformed blocks of each power on disk in DIR (only B_0 in RAM), and the k-series
                coefficients cached in DIR across grids and computed with --workers processes
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/MultipleScatteringCalculations")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import measure_graded_sphere_frequency_series as fs  # noqa: E402

gs, gfft, lowf = fs.gs, fs.gfft, fs.lowf
OUT = Path(__file__).parent / "runs"


class NoDens(gs.Assembler):
    def __init__(self, q, r_c, omega, contrast):
        super().__init__(q, r_c, omega, contrast)
        self.a_rho = 0.0


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args.pop(0)
    for f in [x for x in args if x.startswith("--rc=")]:
        gfft.R_C = int(f.split("=")[1]); args.remove(f)
    for f in [x for x in args if x.startswith("--J=")]:
        fs.J = int(f.split("=")[1]); args.remove(f)
    spill, workers = None, 4
    for f in [x for x in args if x.startswith("--spill=")]:
        spill = Path(f.split("=")[1]); args.remove(f)
    for f in [x for x in args if x.startswith("--workers=")]:
        workers = int(f.split("=")[1]); args.remove(f)
    if spill is not None:
        spill.mkdir(parents=True, exist_ok=True)
        fs.SPILL_DIR = spill
    lowf.NEAR = -1  # Gauss tables: identical results, no cache needed
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile("smoothstep")
    if variant == "rhoonly":
        from cubic_scattering import MaterialContrast
        fs.CONTRAST = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=fs.CONTRAST.Drho)
    if variant.startswith("norhs"):
        # drop the incident wave's coefficient(s) of k_S^j from the right-hand side, e.g. norhs3, norhs23
        import inspect
        drop = [int(c) for c in variant[5:]]
        src = inspect.getsource(fs.solve_series).replace(
            "    psi = []\n", f"    for _j in {drop}:\n        rhs[_j][...] = 0.0\n    psi = []\n", 1)
        exec(src, fs.__dict__)
    if variant == "nodens":
        gs.Assembler = NoDens
    if variant in ("noT2", "noT2self", "noT2pair"):
        # the k_S^2 coefficient of the tables zeroed (strain part at k^2 and, through dens[j-2], nothing else)
        orig_t, orig_s = fs.table_powers, fs.self_powers

        def zero2(tabs):
            tabs = list(tabs)
            tabs[2] = np.zeros_like(tabs[2])
            return tabs

        if variant in ("noT2", "noT2pair"):
            fs.table_powers = lambda *a: zero2(orig_t(*a))
        if variant in ("noT2", "noT2self"):
            fs.self_powers = lambda *a: zero2(orig_s(*a))
    obs = fs.obs_points(gs.R_FAR, gs.THETA)
    ex = fs.exact_series(obs)
    for n in [int(v) for v in args]:
        t0 = time.time()
        if spill is not None:
            asm = gs.Assembler(3, gfft.R_C, 0.0, fs.CONTRAST)
            fs.prefetch_kseries(n, asm.d_list, asm.w_list, workers, spill / f"kseries_rc{gfft.R_C}_J{fs.J}.pkl")
            print(f"  k-series coefficients ready ({len(fs.KSERIES)} orbits, {time.time()-t0:.0f}s)", flush=True)
        side, centres, coefs, asm, sols = fs.solve_series(n)
        vox = fs.far_series(side, centres, coefs, asm, sols, obs)
        np.savez(OUT / (f"{variant}_{gfft.R_C}_{n}" + (f"_J{fs.J}" if fs.J != 6 else "") + ".npz"), P=vox["P"], S=vox["S"], exP=ex["P"], exS=ex["S"])
        print(f"{variant} rc={gfft.R_C} n={n} done {time.time()-t0:.0f}s", flush=True)
