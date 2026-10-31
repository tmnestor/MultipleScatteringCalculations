"""Import every compiled kernel of cubic_scattering/fortran, optionally after torch, and fail loudly if one does not load.

    python envs/check_kernels_load.py [--after-torch]

The Docker image runs it once the kernels are built.  --after-torch imports torch first, as the GPU solvers and
the scaling probe do: torch's wheels can carry their own copies of the Fortran and OpenMP runtimes (the arm64 one
does, an older libgfortran), and whichever copy loads first is the one every later library gets.
"""

import importlib
import sys

from cubic_scattering.fortran.build import SOURCES


def main() -> int:
    after_torch = "--after-torch" in sys.argv[1:]
    if after_torch:
        importlib.import_module("torch")
    for module in SOURCES:
        try:
            importlib.import_module(f"cubic_scattering.fortran.{module}")
        except ImportError as e:
            print(
                f"cannot load cubic_scattering.fortran.{module}{' after torch' if after_torch else ''}: {e}",
                file=sys.stderr,
            )
            return 1
    print(f"the compiled kernels load{' after torch' if after_torch else ''}: {', '.join(SOURCES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
