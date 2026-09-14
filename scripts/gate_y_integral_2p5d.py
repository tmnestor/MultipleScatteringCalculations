#!/usr/bin/env python3
"""GATE: the y-integral of the 3-D closed form IS the 2.5-D propagator.

This is the bridge between the two architectures in this programme, and until
now it was asserted rather than measured.

  * Stage 1 is 2.5-D: heterogeneity in (z, x), invariant along y, with k_y held
    as a parameter on the grid.
  * Stage 2 is 3-D and tabulates the closed-form point propagator
    P(dx, dy, dz) in real space.

The two meet through the y-integral. For a y-INVARIANT model every voxel is a
line of identical voxels along y, so the coupling between two (z, x) columns is
the 3-D propagator summed along that line:

    P_2.5D(dx, dz; k_y) = integral dy  P_3D(dx, y, dz) e^{-i k_y y}

and at k_y = 0 the exponential is 1, so the 2.5-D propagator is literally the
y-integral of the 3-D one. Equivalently: the y-integral is the k_y = 0
component of the y-transform. That is exact, not an approximation -- there is no
2.5-D "closed form" separate from the 3-D one, only a slice of it.

TWO INDEPENDENT ROUTES, which is the evidence standard here:

  [Y1] REAL SPACE. Quadrature of exact_propagator_9x9(dx, y, dz) over y.
  [Y2] SPECTRAL. vertical_kernel_9x9 at k_y = 0 gives P(k_x, k_y = 0; dz);
       inverse-transform over k_x to reach dx.

Neither is derived from the other: [Y1] is the Kupradze closed form, [Y2] is the
k_z-residue construction. Agreement pins the bridge.

Run:  conda run -n seismic python scripts/gate_y_integral_2p5d.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402

REF = ReferenceMedium(5.0, 3.0, 2.5)
# Complex: damping keeps the y-integral absolutely convergent and the branch
# unambiguous. With a real omega the integrand only decays algebraically and
# neither route converges cleanly.
OM = 2 * np.pi * 6.0 * (1 + 0.05j)


def y_integral_real_space(dx: float, dz: float, y_max: float, n_y: int) -> np.ndarray:
    """[Y1] integral dy P_3D(dx, y, dz), by midpoint quadrature."""
    edge = np.linspace(-y_max, y_max, n_y + 1)
    y = 0.5 * (edge[:-1] + edge[1:])
    dy = float(edge[1] - edge[0])
    out = np.zeros((9, 9), dtype=complex)
    for yy in y:
        out += exact_propagator_9x9(dx, float(yy), dz, OM, REF)
    return out * dy


def y_integral_spectral(dx: float, dz: float, kx_max: float, n_kx: int) -> np.ndarray:
    """[Y2] inverse k_x transform of the k_y = 0 spectral kernel."""
    edge = np.linspace(-kx_max, kx_max, n_kx + 1)
    kx = 0.5 * (edge[:-1] + edge[1:])
    dk = float(edge[1] - edge[0])
    p = vertical_kernel_9x9(kx, 0.0, dz, OM, REF)  # (9, 9, n_kx)
    return np.einsum("abk,k->ab", p, np.exp(1j * kx * dx)) * dk / (2 * np.pi)


def main() -> int:
    print("=" * 78)
    print("GATE -- the y-integral of the 3-D closed form is the 2.5-D propagator")
    print(f"  background a={REF.alpha} b={REF.beta} rho={REF.rho}, omega={OM:.3f}")
    print("  [Y1] real-space y-quadrature of the Kupradze closed form")
    print("  [Y2] k_y = 0 spectral kernel, inverse-transformed over k_x")
    print("=" * 78)

    ok = True
    print(f"\n  {'dx':>6} {'dz':>6} {'|Y1|':>12} {'|Y2|':>12} {'rel diff':>11}")
    for dx, dz in ((0.0, 1.0), (0.5, 1.0), (1.0, -1.0), (2.0, 0.5)):
        a = y_integral_real_space(dx, dz, y_max=60.0, n_y=24000)
        b = y_integral_spectral(dx, dz, kx_max=60.0, n_kx=24000)
        rel = float(np.abs(a - b).max() / np.abs(b).max())
        ok = ok and rel < 1e-9
        print(f"  {dx:6.2f} {dz:6.2f} {np.abs(a).max():12.5e} {np.abs(b).max():12.5e} {rel:11.3e}")

    print("\n  Measured 4e-13 to 1.5e-11, so the gate is set at 1e-9. An earlier")
    print("  draft set it at 1e-3 on the assumption that two slowly decaying")
    print("  quadratures could not do better -- which would have admitted a real")
    print("  defect eight orders above the achievable floor. The damping is what")
    print("  makes both integrands converge this well; at real omega they decay")
    print("  only algebraically and neither route is clean.")
    print("\n" + "=" * 78)
    print(f"GATE y-integral / 2.5-D bridge: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
