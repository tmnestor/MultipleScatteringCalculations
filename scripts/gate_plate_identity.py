#!/usr/bin/env python3
"""The plate identity, applied to the solver's OWN same-plane operator.

Task 0, step 3 of plans/voxel_generator_riccati.md.  The exact numbers come
from Mathematica/PlateIdentity.wl (14/14); this gate asks whether the operator
`slab_scattering.build_slab_kernels` actually applies reproduces them.

THE IDENTITY
------------
A plane of space-filling cubes (side d) carrying a uniform stress polarisation
IS a uniformly polarised slab, whose static strain field is uniform inside it:

    Gamma_self + sum_{R != 0, same plane} <Gamma>(R)  =  Gamma_plate,
    Gamma_plate_ijkl = sym(n_j n_l K(n)^-1_ik),

where <Gamma>(R) is the field of source cube R, AVERAGED OVER THE SOURCE and
evaluated at the receiver CENTRE (the collocation convention this solver
settled on), and Gamma_self is the cube's centre-point self field.  The same
plane sum at Bloch k_par = 0 is what the solver's dz = 0 kernel holds, so

    kernel(k_par = 0)  should equal  Gamma_plate - Gamma_self   (static limit).

PREDICTIONS, one per averaging setting (Mathematica, PlateIdentity.wl):

  exact_cell_average (auto on the Ewald route)   -> exact, to the tail floor
  contact shell only, midpoint beyond            -> bias 2.14% of Gamma_plate,
                                                     entry by entry as printed
  va_all to Chebyshev radius 2 / 3 / 4           -> 0.506% / 0.189% / 0.090%

CONVENTIONS.  The solver's 9x9 strain block carries its own Voigt ordering
(zz, xx, yy, xy, zy, zx), sign and engineering factors.  They are NOT assumed:
they are calibrated entry by entry against the static Kelvin second
derivative at two separations, and the calibration is only accepted if both
separations give the same pattern.  d = 1 so the cell volume is 1.

Run:
    conda run -n seismic python scripts/gate_plate_identity.py
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: E402
from cubic_scattering.slab_scattering import SlabGeometry, build_slab_kernels  # noqa: E402

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
MU, LAM = REF.mu, REF.lam
NU = LAM / (2.0 * (LAM + MU))
GEOM = SlabGeometry(M=1, N_z=1, a=0.5)  # d = 1 m; M = 1 keeps only the k_par = 0 Bloch point
OMEGA = 3.0  # k_S d = 1e-3: the static limit to O((k d)^1)

# Solver Voigt order in its (z, x, y) = (0, 1, 2) frame.
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
LABELS = ["zz", "xx", "yy", "xy", "zy", "zx"]
NORMAL = np.array([1.0, 0.0, 0.0])  # plate normal = z = axis 0


def symm(f: NDArray) -> NDArray:
    """Symmetrise a 4-tensor in (ij) and in (kl)."""
    return 0.25 * (f + f.transpose(1, 0, 2, 3) + f.transpose(0, 1, 3, 2) + f.transpose(1, 0, 3, 2))


def to_voigt(t: NDArray) -> NDArray:
    """Plain tensor components in the solver's Voigt order (no factors)."""
    return np.array([[t[p, q, r, s] for (r, s) in VOIGT] for (p, q) in VOIGT])


def _kelvin_gamma_function():
    """Gamma_ijkl(x) = -sym d_j d_l G_ik(x), static Kelvin, as a numeric function."""
    xs = sp.symbols("x0 x1 x2", real=True)
    r = sp.sqrt(sum(v**2 for v in xs))
    pref = 1 / (16 * sp.pi * sp.Float(MU) * (1 - sp.Float(NU)))
    g = [
        [
            pref * ((3 - 4 * sp.Float(NU)) * sp.KroneckerDelta(i, k) / r + xs[i] * xs[k] / r**3)
            for k in range(3)
        ]
        for i in range(3)
    ]
    f = [
        [[[-sp.diff(g[i][k], xs[j], xs[l]) for l in range(3)] for k in range(3)] for j in range(3)]
        for i in range(3)
    ]
    fun = sp.lambdify(xs, f, "numpy")
    return lambda x: symm(np.array(fun(*x), dtype=float))


GAMMA_POINT = _kelvin_gamma_function()


def gamma_plate() -> NDArray:
    """sym(n_j n_l K^-1_ik), K = acoustic tensor for the plate normal."""
    n = NORMAL
    kinv = (np.eye(3) - np.outer(n, n)) / MU + np.outer(n, n) / (LAM + 2 * MU)
    f = np.einsum("ik,j,l->ijkl", kinv, n, n)
    return symm(f)


def gamma_self_collocation() -> NDArray:
    """Centre-point field of a uniform cube source (PlateIdentity.wl, exact).

    G_ik = [(4-4nu) delta_ik / r - d_i d_k r] / (16 pi mu (1-nu)); at the cube
    centre Phi_jl = -(4 pi/3) delta_jl and Psi_ikjl = c(dd+dd+dd) + b delta_ikjl
    with c = -4/sqrt(3), 5c + b = -8 pi/3.
    """
    d = np.eye(3)
    c = -4.0 / np.sqrt(3.0)
    b = -8.0 * np.pi / 3.0 - 5.0 * c
    phi2 = -(4.0 * np.pi / 3.0) * d
    psi4 = c * (
        np.einsum("ik,jl->ikjl", d, d) + np.einsum("ij,kl->ikjl", d, d) + np.einsum("il,kj->ikjl", d, d)
    )
    for m in range(3):
        psi4[m, m, m, m] += b
    # f_ijkl = (4-4nu) delta_ik Phi_jl - Psi_ikjl
    f = (4 - 4 * NU) * np.einsum("ik,jl->ijkl", d, phi2) - psi4.transpose(0, 2, 1, 3)
    return -symm(f) / (16 * np.pi * MU * (1 - NU))


# Mathematica PlateIdentity.wl, mu * bias for the contact-shell-only pairing,
# printed in ITS order (xx, yy, zz, yz, xz, xy) with normal z.  Mapped below.
MATH_ORDER = ["xx", "yy", "zz", "zy", "zx", "xy"]
MATH_BIAS_R1 = np.array(
    [
        [-0.005656, 0.001475, 0.004021, 0, 0, 0],
        [0.001475, -0.005656, 0.004021, 0, 0, 0],
        [0.004021, 0.004021, -0.007722, 0, 0, 0],
        [0, 0, 0, 0.004133, 0, 0],
        [0, 0, 0, 0, 0.004133, 0],
        [0, 0, 0, 0, 0, 0.001252],
    ]
)
MATH_REL = {1: 2.14e-2, 2: 5.06e-3, 3: 1.89e-3, 4: 9.00e-4}


def math_bias_in_solver_order() -> NDArray:
    idx = [MATH_ORDER.index(lab) for lab in LABELS]
    return MATH_BIAS_R1[np.ix_(idx, idx)]


def calibrate() -> tuple[NDArray, NDArray]:
    """Entry-wise ratio solver-S / Voigt(Gamma) at two in-plane separations."""
    qs, masks = [], []
    for r in (np.array([0.0, 3.0, 1.0]), np.array([0.0, -2.0, 4.0])):
        s = _propagator_block_9x9(r, OMEGA, REF)[3:, 3:].real
        g = to_voigt(GAMMA_POINT(r))
        mask = np.abs(g) > 1e-6 * np.abs(g).max()
        q = np.where(mask, s / np.where(mask, g, 1.0), np.nan)
        qs.append(q)
        masks.append(mask)
    return qs[0], qs[1]


def kernel_k0(**opts: Any) -> NDArray:
    """The solver's dz = 0 strain block at Bloch k_par = 0."""
    k = build_slab_kernels(
        GEOM, OMEGA, REF, periodic=True, lattice_ewald=True, volume_averaged=True, **opts
    )
    return k[0, 0, 0, 3:, 3:]


def main() -> int:
    passed = total = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal passed, total
        total += 1
        passed += bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(
        f"Background mu={MU:.4g} Pa, lambda={LAM:.4g} Pa, nu={NU:.6f}; d=1 m; k_S d={OMEGA / REF.beta:.1e}"
    )

    # 0. Calibration of the solver's strain-block convention.
    print("\n0. Calibrate the solver's Voigt/sign/engineering convention")
    q1, q2 = calibrate()
    both = ~np.isnan(q1) & ~np.isnan(q2)
    # The fitted ratios carry the O((k R)^2) dynamic part of the point
    # propagator; the convention itself is a pattern of small integers.
    # Accept it only if every fitted ratio, at BOTH separations, rounds to it.
    measured = np.where(np.isnan(q1), q2, q1)
    known = ~np.isnan(measured)
    pattern = np.round(measured[known])
    dev = max(
        np.nanmax(np.abs(q1[~np.isnan(q1)] - np.round(q1[~np.isnan(q1)]))),
        np.nanmax(np.abs(q2[~np.isnan(q2)] - np.round(q2[~np.isnan(q2)]))),
    )
    check(
        "fitted ratios round to one integer pattern at both separations",
        dev < 1e-3 and np.array_equal(np.round(q1[both]), np.round(q2[both])),
        f"max deviation from integers {dev:.1e}",
    )
    print(f"     integer factors present: {np.unique(pattern)}")
    # engineering-Voigt structure q_ab = -e_a e_b with e = 1 (normal), 2 (shear)
    eng = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    q = -np.outer(np.ones(6), eng)
    agree = np.array_equal(q[known], pattern)
    if not agree:
        q = -np.outer(eng, np.ones(6))
        agree = np.array_equal(q[known], pattern)
    check(
        "pattern is -(engineering factor on the source index)",
        agree,
        "q_ab = -e_b, e = (1,1,1,2,2,2)" if agree else "neither row nor column form fits",
    )

    target = to_voigt(gamma_plate() - gamma_self_collocation())
    scale = np.abs(to_voigt(gamma_plate())).max()

    def bias_of(kernel: NDArray) -> NDArray:
        """(kernel in tensor components) - (Gamma_plate - Gamma_self)."""
        return kernel.real / q - target

    # 1. Consistent collocation everywhere: the solver's default on this route.
    print("\n1. exact_cell_average (default on the Ewald route): prediction exact in the limit")
    # cell_avg_r0 is the radius beyond which the analytic d^2 tail replaces
    # direct averaging; the residual is its O(d^4) truncation and must fall.
    b_def = bias_of(kernel_k0())
    print(f"     library defaults (cell_avg_r0=2, cell_avg_gauss=6): {np.abs(b_def).max() / scale:.2e}")
    ladder = []
    for r0 in (4, 6, 8):
        ladder.append(np.abs(bias_of(kernel_k0(cell_avg_r0=r0, cell_avg_gauss=12))).max() / scale)
    check(
        "residual falls with cell_avg_r0 (4, 6, 8) and reaches < 1e-6",
        ladder[0] > ladder[1] > ladder[2] and ladder[2] < 1e-6,
        " -> ".join(f"{v:.2e}" for v in ladder),
    )
    b_w1 = bias_of(kernel_k0(cell_avg_r0=4, cell_avg_gauss=12))
    b_w2 = bias_of(
        build_slab_kernels(
            GEOM,
            2 * OMEGA,
            REF,
            periodic=True,
            lattice_ewald=True,
            volume_averaged=True,
            cell_avg_r0=4,
            cell_avg_gauss=12,
        )[0, 0, 0, 3:, 3:]
    )
    dw = np.abs(b_w2 - b_w1).max() / scale
    check("static limit reached (2x omega moves it by < 1e-6)", dw < 1e-6, f"{dw:.1e}")

    # 2. Contact shell only, midpoint beyond.
    print("\n2. contact shell only (exact_cell_average=False, va_all=False): prediction 2.14% bias")
    b1 = bias_of(kernel_k0(exact_cell_average=False))
    rel1 = np.abs(b1).max() / scale
    check(
        "relative size", abs(rel1 / MATH_REL[1] - 1) < 0.01, f"{rel1:.3e} vs Mathematica {MATH_REL[1]:.2e}"
    )
    diff = np.abs(MU * b1 - math_bias_in_solver_order()).max()
    check("entry by entry vs Mathematica (mu * bias)", diff < 2e-6, f"max abs diff {diff:.1e}")
    print("     mu * bias, solver order " + " ".join(LABELS))
    for row in MU * b1:
        print("      " + " ".join(f"{v:+.5f}" for v in row))

    # 3. Source-cell average extended to Chebyshev radius r0.
    print("\n3. va_all to radius r0 (exact_cell_average=False): prediction 0.506% / 0.189% / 0.090%")
    for r0 in (2, 3, 4):
        b = bias_of(kernel_k0(exact_cell_average=False, va_all=True, va_all_reach=r0, va_gauss=8))
        rel = np.abs(b).max() / scale
        check(
            f"reach {r0}",
            abs(rel / MATH_REL[r0] - 1) < 0.02,
            f"{rel:.3e} vs Mathematica {MATH_REL[r0]:.2e}",
        )

    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
