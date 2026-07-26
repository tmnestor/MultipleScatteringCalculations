"""Check the Fortran/Python lateral propagator P^x (thesis Alg 5.2).

Alg 5.2 replaces an O(N_x^2) pairwise sum by two O(N_x) running sweeps:

    right:  dV_->(chi) = E_x ( Sigma_->(chi-1) + dV_->(chi-1) ),  dV_->(1) = 0
    left:   dV_<-(chi) = E_x ( Sigma_<-(chi+1) + dV_<-(chi+1) ),  dV_<-(Nx) = 0

Unrolling the right sweep gives dV_->(chi) = sum_{chi' < chi} E^(chi-chi') src(chi'),
so the sweep is an exact geometric resummation of a convolution with an
exponential kernel. The knowable-answer test is therefore:

    SWEEP RESULT  ==  DIRECT DOUBLE SUM

evaluated with a DIFFERENT source at every site -- which is precisely the
"disorder-resolved" property being claimed. A uniform-source test would pass
even for an implementation that silently averaged the sites.

From FFTProp.py/propagation.py the accumulation is, per depth layer iscat:

  right: PSY[chi,m] += sum_k 2 i^m PC[k,-m] PU[k]      (PU phased BEFORE readout)
         PU[k]      += sum_m' (-i)^m' PC[k,m'] SY[chi,m']
  left : PSY[chi,m] += sum_k 2 i^m PC[k, m] PD[k]      (PD phased AFTER readout)
         PD[k]      += sum_m' (-i)^m' PC[k,-m'] SY[chi,m']

so the closed forms are

  PSY_right[chi,m] = sum_{chi'<chi}  sum_m' K_R[m,m'](chi-chi') SY[chi',m']
  PSY_left [chi,m] = sum_{chi'>chi}  sum_m' K_L[m,m'](chi'-chi) SY[chi',m']

  K_R[m,m'](d) = 2 i^m (-i)^m' sum_k PC[k,-m] PC[k,m'] E_k^d
  K_L[m,m'](d) = 2 i^m (-i)^m' sum_k PC[k, m] PC[k,-m'] E_k^d

Stubs are used for the spectral arrays: the question is whether the RESUMMATION
is right, not whether the physics inputs are.
"""

# The package directory is literally named "FFTProp.py", so `import FFTProp`
# cannot reach it (and plain pytest collection fails for the same reason --
# it needs --import-mode=importlib). Load it as a package under an alias so
# its internal relative imports still resolve.
import importlib.util  # noqa: E402
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent / "FFTProp.py")
_spec = importlib.util.spec_from_file_location(
    "fftprop", f"{_ROOT}/__init__.py", submodule_search_locations=[_ROOT]
)
if _spec is None or _spec.loader is None:
    msg = f"cannot load the FFTProp package from {_ROOT}"
    raise ImportError(msg)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["fftprop"] = _mod
_spec.loader.exec_module(_mod)
from fftprop.propagation import left_sweep, right_sweep  # noqa: E402

rng = np.random.default_rng(20260726)

NK = 16  # small; sweeps use stride 2 so 8 live wavenumbers
NSCATX = 7
NSCATZ = 1
ISCAT = 0


def make():
    grid = types.SimpleNamespace(Nk=NK, Nscatx=NSCATX, Nscatz=NSCATZ)
    sa = types.SimpleNamespace(
        Eavec=rng.normal(size=NK) + 1j * rng.normal(size=NK),
        Ebvec=rng.normal(size=NK) + 1j * rng.normal(size=NK),
        PC=rng.normal(size=(NK, 5, 2, NSCATZ))
        + 1j * rng.normal(size=(NK, 5, 2, NSCATZ)),
    )
    # damp the phase factors so the geometric series is well conditioned
    sa.Eavec *= 0.4 / np.abs(sa.Eavec)
    sa.Ebvec *= 0.4 / np.abs(sa.Ebvec)
    res = types.SimpleNamespace(
        SY=rng.normal(size=(NSCATX, 5, 2, NSCATZ))
        + 1j * rng.normal(size=(NSCATX, 5, 2, NSCATZ)),
        PSY=np.zeros((NSCATX, 5, 2, NSCATZ), dtype=complex),
    )
    return grid, sa, res


def direct(sa, res, comp, E):
    """O(N^2) reference: explicit double sum over site pairs."""
    out = np.zeros((NSCATX, 5), dtype=complex)
    sl = slice(None, None, 2)
    for chi in range(NSCATX):
        for chip in range(NSCATX):
            if chip == chi:
                continue
            d = abs(chi - chip)
            for mi in range(5):
                m = mi - 2
                acc = 0.0 + 0j
                for mpi in range(5):
                    mp = mpi - 2
                    if chip < chi:  # right sweep contribution
                        k1, k2 = 4 - mi, mpi
                    else:  # left sweep contribution
                        k1, k2 = mi, 4 - mpi
                    kern = np.sum(
                        sa.PC[sl, k1, comp, ISCAT]
                        * sa.PC[sl, k2, comp, ISCAT]
                        * E[sl] ** d
                    )
                    acc += (
                        2.0
                        * (1j**m)
                        * ((-1j) ** mp)
                        * kern
                        * res.SY[chip, mpi, comp, ISCAT]
                    )
                out[chi, mi] += acc
    return out


print("Alg 5.2 lateral sweep: geometric resummation vs direct pairwise sum")
print(f"  N_x = {NSCATX} sites, all sources DISTINCT (disorder-resolved)")
print()

grid, sa, res = make()
right_sweep(sa, grid, ISCAT, res)
left_sweep(sa, grid, ISCAT, res)

for comp, E, name in ((0, sa.Eavec, "P"), (1, sa.Ebvec, "S")):
    ref = direct(sa, res, comp, E)
    got = res.PSY[:, :, comp, ISCAT]
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(
        f"  {name}-channel: ||sweep - direct|| / ||direct|| = {rel:.3e}"
        f"   {'PASS' if rel < 1e-12 else '** FAIL **'}"
    )

print()
print("Control: with a UNIFORM source the test is far weaker -- an")
print("implementation that averaged the sites would still pass it.")
grid2, sa2, res2 = make()
res2.SY[:] = res2.SY[0:1]
right_sweep(sa2, grid2, ISCAT, res2)
left_sweep(sa2, grid2, ISCAT, res2)
spread = np.std(np.abs(res2.PSY[:, :, 0, ISCAT]), axis=0).max()
print(f"  uniform-source PSY still varies across sites by {spread:.3e}")
print("  (non-zero => the operator is genuinely position-dependent, not an average)")
