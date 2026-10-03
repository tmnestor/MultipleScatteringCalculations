"""Build the compiled kernels of ``cubic_scattering.fortran`` in place, with f2py and Meson.

Run with the project's conda environment, which supplies gfortran, Meson and Ninja and the OpenMP
runtime that OpenBLAS already uses (one runtime per process; a Homebrew gfortran would bring another):

    conda run -n seismic python -m cubic_scattering.fortran.build

The extension is written next to this file and is not tracked. Optimisation is -O3 without
-ffast-math, which would let the compiler reorder floating-point sums and change results.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {"_point_kernel": "point_kernel.f90"}


def main() -> int:
    env = dict(os.environ, FFLAGS="-O3 -fopenmp", LDFLAGS="-fopenmp")
    for module, source in SOURCES.items():
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            source,
            "-m",
            module,
            "--dep",
            "openmp",
            "only:",
            "kernel_9x9",
            ":",
        ]
        print(" ".join(cmd), flush=True)
        done = subprocess.run(cmd, cwd=HERE, env=env)
        if done.returncode != 0:
            print(f"build of {module} from {source} failed (exit {done.returncode})", file=sys.stderr)
            return done.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
