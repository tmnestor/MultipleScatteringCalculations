#!/usr/bin/env python3
"""GATE: FFTProp's free-surface reflection against two independent arbiters.

Rung 7's remaining prerequisite. `FFTProp` carries a free surface with P-SV
coupling and this solver now can too, so before the two are compared the
question is whether they agree about the surface itself.

THE ARBITERS, and why there are two. A single disagreement localises nothing --
it could be either side, or a convention. So the free-surface P-P reflection is
built twice, independently:

  [P1] AKI & RICHARDS, displacement convention, written out directly:

           R_PP = [ (1/b^2 - 2p^2)^2 - 4 p^2 qa qb ]
                / [ (1/b^2 - 2p^2)^2 + 4 p^2 qa qb ]

  [P2] THE TRACTION-FREE CONDITION applied to the Global Matrix eigenvectors.
       An up-going wave reflects into a down-going one with total traction zero,
       so with T_u, T_d the traction rows of E_u, E_d,

           R = -T_d^-1 T_u.

       Nothing here comes from [P1]: `layer_eigenvectors` is the same object the
       GMM marine stack is reconciled against Kennett with, at 1e-15.

  [P3] FFTProp's own `Rpp`, `Rss`, `Rsp` from `build_spectral_arrays`, with only
       the phase factors its source documents divided out. For propagating p the
       phases are unimodular, so this does not touch the magnitudes at all.

WHAT IS MEASURED. [P1] and [P2] agree to six decimal places at every ray
parameter. [P3] does not: its |Rpp| stays within 1% of unity across the whole
propagating range, where the true free surface dips to 0.07, and its P-SV
conversion is ~1e-3 where the true one reaches 0.6. FFTProp's free surface
behaves as a near-perfect mirror with almost no mode conversion.

WHERE IT COMES FROM. `spectral_arrays` computes R1 = 1 - 2 cb02 p2 with
cb02 = cb0**2 and cb0 = `medium.complex_slowness_s`, i.e. 1 - 2 p^2 / beta^2.
The free-surface R1 is 1/beta^2 - 2 p^2. Those are not proportional: the
velocity sits on the wrong side. The gate prints both so the difference is
visible rather than asserted.

WHAT THIS GATE DOES NOT CLAIM. `FFTProp.py` is described as a faithful
conversion of the Fortran `FFTPROP.F`, and its comments cite the Fortran lines
this expression came from. Whether the Fortran meant `cb0` as a slowness (making
the same expression) or as a velocity (making the port a transcription error) is
NOT settled here, and this gate does not guess. What is established is that the
coefficients as they stand in the Python package do not match the free-surface
reflection, verified two independent ways.

CONSEQUENCE FOR RUNG 7. A comparison of this solver against FFTProp with free
surfaces active on both sides would be comparing against a surface that does not
convert. Either resolve this first, or run rung 7 with both free surfaces off.

Run:  conda run -n seismic python scripts/gate_fftprop_free_surface.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

# The package directory is literally named "FFTProp.py", so `import FFTProp`
# cannot reach it. Same loader as scripts/gate_lateral_sweep_alg52.py.
_FR = str(ROOT / "FFTProp.py")
_spec = importlib.util.spec_from_file_location(
    "fftprop", f"{_FR}/__init__.py", submodule_search_locations=[_FR]
)
if _spec is None or _spec.loader is None:
    msg = f"cannot load the FFTProp package from {_FR}"
    raise ImportError(msg)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["fftprop"] = _mod
_spec.loader.exec_module(_mod)

from fftprop.medium import GridConfig, ReferenceMedium  # noqa: E402
from fftprop.spectral_arrays import build_spectral_arrays  # noqa: E402
from GlobalMatrix.layer_matrix import layer_eigenvectors  # noqa: E402

ALPHA, RHO = 5.0, 3.0
BETA = ALPHA / np.sqrt(3.0)  # Poisson solid, FFTProp's own relation
FREQ = 2.0


def slownesses(p: float) -> tuple[complex, complex]:
    """Vertical P and S slownesses on the Im >= 0 branch."""
    qa = np.sqrt(complex(1.0 / ALPHA**2 - p**2))
    qb = np.sqrt(complex(1.0 / BETA**2 - p**2))
    return (qa if qa.imag >= 0 else -qa), (qb if qb.imag >= 0 else -qb)


def rpp_aki_richards(p: float) -> complex:
    """[P1] free-surface P-P, displacement convention, written out directly."""
    qa, qb = slownesses(p)
    r1 = 1.0 / BETA**2 - 2.0 * p**2
    return (r1**2 - 4.0 * p**2 * qa * qb) / (r1**2 + 4.0 * p**2 * qa * qb)


def r_traction_free(p: float) -> np.ndarray:
    """[P2] R = -T_d^-1 T_u from the Global Matrix eigenvectors."""
    qa, qb = slownesses(p)
    e_d, e_u = layer_eigenvectors(complex(p), qa, qb, RHO, complex(BETA))
    return -np.linalg.solve(e_d[2:4, :], e_u[2:4, :])


def main() -> int:
    med = ReferenceMedium(alpha=ALPHA, rho=RHO, Q=1e8)  # near-lossless
    grid = GridConfig(Nk=4096, Nscatx=21, Nscatz=2, jskip=8)
    sa = build_spectral_arrays(med, grid, freq=FREQ, atten_imag=0.0)
    w = med.complex_omega(FREQ, 0.0)
    p_all = (sa.kxvec / w).real

    # Phases are unimodular for propagating p, so this does not touch magnitudes.
    rpp_f = sa.Rpp / sa.Eavec
    rsp_f = sa.Rsp / (sa.Eavec * sa.Ebvec)

    prop = np.where((np.abs(p_all) * BETA < 0.98) & (p_all >= 0))[0]

    print("=" * 78)
    print("GATE -- FFTProp's free surface vs two independent arbiters")
    print(f"  alpha={ALPHA} beta={BETA:.6f} rho={RHO}, Poisson solid, near-lossless")
    print(f"  {prop.size} propagating wavenumbers on FFTProp's own grid")
    print("=" * 78)

    print(f"\n  {'p':>9} {'A&R':>10} {'GMM':>10} {'FFTProp':>10} {'|Rsp| GMM':>11} {'|Rsp| FFT':>11}")
    agree = True
    differs = False
    for i in prop:
        p = float(p_all[i])
        a = abs(rpp_aki_richards(p))
        r_g = r_traction_free(p)
        g, gsp = abs(r_g[0, 0]), abs(r_g[1, 0])
        f, fsp = abs(rpp_f[i]), abs(rsp_f[i])
        agree = agree and abs(a - g) < 1e-9
        if abs(f - g) > 1e-3:
            differs = True
        print(f"  {p:9.5f} {a:10.6f} {g:10.6f} {f:10.6f} {gsp:11.6f} {fsp:11.6f}")

    print("\n  [P3] where the difference sits: the R1 factor")
    print(f"    {'p':>9} {'FFTProp 1-2p^2/b^2':>20} {'free surface 1/b^2-2p^2':>25}")
    for i in prop[:4]:
        p = float(p_all[i])
        print(f"    {p:9.5f} {1.0 - 2.0 * p**2 / BETA**2:20.6f} {1.0 / BETA**2 - 2.0 * p**2:25.6f}")

    print("\n" + "=" * 78)
    print(f"  [P1]==[P2] the two arbiters agree      : {'PASS' if agree else 'FAIL'}")
    print(f"  [P3] FFTProp differs from both         : {'YES' if differs else 'NO'}")
    print("\n  The gate PASSES when the arbiters agree with each other, which is")
    print("  what makes the third column meaningful. It does NOT assert what")
    print("  FFTProp's Fortran original intended -- only that the coefficients as")
    print("  they stand do not match the free-surface reflection.")
    print("\n  For rung 7: run it with both free surfaces OFF until this is")
    print("  resolved, or the comparison is against a surface that cannot convert.")
    print("=" * 78)
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
