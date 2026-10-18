"""Cross-check: the 1996 FFTPROP.F, compiled with gfortran, against its Python port FFTProp.py.

The Fortran is compiled from a COPY in a temporary directory (the original is never modified), with its
commented-out print of PSY restored, in three builds:

  as written       the 1996 source, single precision;
  corrected        the two defects that FFTProp.py/spectral_arrays.py documents and corrects:
                   (1) the loop `Do i=1,Nk` whose body indexes with `ik`, so that the one-layer phase
                       shifts Eavec, Ebvec are set at one stale index only;
                   (2) Eavec, Ebvec used in the free-surface coefficients Rpp, Rsp, Rss before they are set,
                       so that the free-surface reflection is zero and the downsweep adds nothing;
  corrected, -fdefault-real-8   the same in double precision.

PSY (the field at the scatterers), Svec and Rvec (the source and receiver projections) are compared with
the port's, element by element, relative to the largest element of the port's array.

MEASURED (gfortran 15.2):
  as written:          PSY 1.0, Rvec and Svec O(1) -- and compiler-dependent, from the uninitialised phases
  corrected, single:   PSY 3.7e-7, Rvec 2.7e-6, Svec 1.8e-1 (the source projection cancels heavily)
  corrected, double:   PSY 8.3e-16, Rvec 1.2e-15, Svec 1.7e-11

Run:  python scripts/xcheck_thesis_fftprop_fortran.py      (needs gfortran on the PATH)
"""

import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORTRAN = ROOT / "PhD_fortran_code" / "FFTPROP.F"
FLAGS = ["-fdec", "-std=legacy", "-ffixed-line-length-none", "-O0"]


def port_module():
    """FFTProp.py is a package directory whose name ends in .py: load it under an importable name."""
    pkg_dir = ROOT / "FFTProp.py"
    spec = importlib.util.spec_from_file_location(
        "fftprop_port", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    assert spec is not None and spec.loader is not None
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["fftprop_port"] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module("fftprop_port.fftprop_driver")


def source(corrected: bool) -> str:
    lines = FORTRAN.read_text().splitlines(keepends=True)
    text = "".join(lines)
    text = re.sub(
        r"^\*(\s+)type\*,'PSY\(jscat,m,i,iscat\)=',j,PSY\(jscat,m,i,iscat\)",
        r" \1type*,'PSY(jscat,m,i,iscat)=',j,PSY(jscat,m,i,iscat)",
        text,
        flags=re.M,
    )
    if corrected:
        bad = re.compile(r"^(\s*)Do i=1,Nk(\s*\n\s*Eavec\(ik\)=rtEa\(ik\)\*rtEa\(ik\))", re.M)
        text, n = bad.subn(r"\1Do ik=1,Nk\2", text)
        assert n == 1, "defect (1) not found where expected"
        use = "Rpp(ik)=(-R12 + R2)/Rayleigh"
        assert text.count(use) == 1, "defect (2) not found where expected"
        text = text.replace(
            use, "Eavec(ik)=rtEa(ik)*rtEa(ik)\n        Ebvec(ik)=rtEb(ik)*rtEb(ik)\n        " + use
        )
    return text


def run(corrected: bool, double: bool, work: Path) -> str:
    src = work / "fftprop.f"
    src.write_text(source(corrected))
    exe = work / "fftprop"
    extra = ["-fdefault-real-8"] if double else []
    subprocess.run(["gfortran", *FLAGS, *extra, "-o", str(exe), str(src)], check=True, capture_output=True)
    return subprocess.run([str(exe)], check=True, capture_output=True, text=True, cwd=work).stdout


def parse(name: str, text: str, nz: int, nx: int) -> np.ndarray:
    num = r"([-+0-9.Ee]+)"
    pat = re.compile(rf"{name}\(jscat,m,i,iscat\)=\s+(\d+)\s+\(\s*{num}\s*,\s*{num}\s*\)")
    vals = {int(m.group(1)): complex(float(m.group(2)), float(m.group(3))) for m in pat.finditer(text)}
    return np.array([vals[j + 1] for j in range(nz * nx * 2 * 5)]).reshape(nz, nx, 2, 5)


def main() -> None:
    if shutil.which("gfortran") is None:
        sys.exit("gfortran not found on the PATH: install it (e.g. brew install gcc) to run this check.")
    res = port_module().compute_wavefield()
    nx, _, _, nz = res.PSY.shape
    builds = (("as written", False, False), ("corrected", True, False), ("corrected, double", True, True))
    with tempfile.TemporaryDirectory() as tmp:
        for label, corrected, double in builds:
            text = run(corrected, double, Path(tmp))
            cells = []
            for name in ("PSY", "Rvec", "Svec"):
                ref = getattr(res, name).transpose(3, 0, 2, 1)  # (iscat, jscat, i, m), the Fortran order
                err = np.abs(parse(name, text, nz, nx) - ref).max() / np.abs(ref).max()
                cells.append(f"{name} {err:.1e}")
            print(f"{label:<18} " + ", ".join(cells))


if __name__ == "__main__":
    main()
