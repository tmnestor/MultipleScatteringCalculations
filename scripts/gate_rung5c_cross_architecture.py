#!/usr/bin/env python3
"""GATE rung 5c: the 3-D operator against the FFT-convolution architecture.

Deferred from stage 1 for a stated reason -- `slab_scattering` is 3-D on a
finite M x M footprint and cannot represent a y-invariant medium, so it could
not arbitrate a 2.5-D solver. Stage 2 removes that restriction, so the
comparison is finally meaningful. The two architectures agree to 7e-16.

WHY THE COMPARISON IS VALID, checked rather than assumed:

  * `slab_scattering` defaults to `periodic=False`, a ZERO-PADDED LINEAR
    convolution on (2M-1) x (2M-1), not a circular one. Both architectures are
    non-periodic, so this does not measure horizontal periodicity.
  * It defaults to `volume_averaged=False`, the POINT propagator, which is what
    the real-space tables hold. With volume averaging on, the two compute
    genuinely different objects at touching faces -- a converged projection
    difference, not a bug -- and the comparison would be invalid.
  * Both index the lattice (n_z, n_x, n_y) in seismological (z, x, y) order and
    take separations from centres on the same pitch, so no reindexing occurs.

THE CANCELLATION TRAP, which cost a wrong accusation before it was understood.
`_slab_matvec` returns (I - G0 T) psi, so G0 psi is recovered as
psi - matvec(psi) with T = I. That subtraction is CATASTROPHIC whenever
|G0 psi| << |psi|: the intermediate (psi - G0 psi) is stored to a relative
2e-16 of |psi|, so G0 psi comes back with an ABSOLUTE error of eps*|psi|
regardless of how small it is.

In SI units (metres) this propagator is ~1e-13 while psi is ~1, and the
recovered G0 psi is then wrong in its fourth digit -- 4e-4 -- purely from the
cancellation. Run in seismic units (km) the same quantities are ~1e+3 against
~1, there is no cancellation, and the two architectures agree at 7e-16.

This was misdiagnosed once as a defect inside `_slab_matvec`'s FFT convolution.
It is not. The FFT path reproduces a brute-force convolution of its own kernel
to 3e-29 when the comparison is made without the cancellation -- see the
linearity check in [5c-3]. The guard below asserts the magnitudes are
comparable so the trap cannot recur silently.

WHAT IS COMPARED, and why G0 rather than a reflection coefficient: the two
architectures are compared at the OPERATOR level, with a distinct source at
every site. Comparing reflection coefficients would drag in the far-field
projection and the flux normalisation, each a separately gated object with its
own conventions, and a disagreement would not localise.

Run:  conda run -n seismic python scripts/gate_rung5c_cross_architecture.py
Seismic units (km, km/s, g/cm3) -- see the cancellation trap above.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    _slab_matvec,
    build_slab_kernels,
)

# Seismic units throughout, for BOTH sides. See the cancellation trap.
REF = ReferenceMedium(5.0, 3.0, 2.5)
OM = 150.0


def slab_g0(psi: np.ndarray, geom: SlabGeometry, kernel) -> np.ndarray:
    """G0 psi from the FFT-convolution architecture, isolated from the solve."""
    n_z, m = geom.N_z, geom.M
    eye = np.broadcast_to(np.eye(9, dtype=complex), (n_z, m, m, 9, 9)).copy()
    out = _slab_matvec(psi.ravel(), eye, kernel, geom)
    return psi - out.reshape(n_z, m, m, 9)


def main() -> int:
    m, n_z, a = 4, 3, 0.025
    geom = SlabGeometry(M=m, N_z=n_z, a=a)
    pitch = geom.d

    print("=" * 78)
    print("GATE rung 5c -- 3-D real-space operator vs FFT convolution")
    print(f"  lattice N_z={n_z}, M={m}, cube side d={pitch} km, omega={OM} rad/s")
    print(f"  background a={REF.alpha} b={REF.beta} rho={REF.rho} (seismic)")
    print("  slab: periodic=False (linear conv), volume_averaged=False (point)")
    print("=" * 78)

    kernel = build_slab_kernels(geom, OM, REF)
    grid = SweepGrid3D(n_z=n_z, n_x=m, n_y=m, pitch=pitch)
    cache = build_g0_cache_3d(grid, REF, OM)

    rng = np.random.default_rng(20260914)
    shape = (n_z, m, m, 9)
    distinct = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    a_side = slab_g0(distinct, geom, kernel)
    b_side = apply_g0_3d(distinct, cache)
    scale = float(np.abs(b_side).max())

    # [5c-0] THE GUARD. Without it the whole gate is meaningless: a cancelling
    # extraction returns eps*|psi| of noise and no amount of agreement or
    # disagreement downstream means anything.
    ratio = scale / float(np.abs(distinct).max())
    guard_ok = ratio > 1e-3
    print(f"\n  [5c-0] cancellation guard |G0 psi| / |psi| : {ratio:.3e}")
    print("         must exceed 1e-3, else psi - (psi - G0 psi) is noise")
    if not guard_ok:
        print("         FAILED -- the units make G0 negligible beside psi. Nothing")
        print("         below this line can be trusted; fix the units first.")

    rel = float(np.abs(a_side - b_side).max() / scale)
    print(f"\n  [5c-1] G0 psi, distinct sources : rel diff {rel:.3e}")
    print(f"         |G0 psi| (slab)          : {float(np.abs(a_side).max()):.5e}")
    print(f"         |G0 psi| (real space)    : {scale:.5e}")

    # Vacuity control, RELATIVE -- an absolute bound would depend on the unit
    # system rather than on the physics.
    uni = np.ones(shape, dtype=complex)
    avg = np.broadcast_to(distinct.mean(axis=(0, 1, 2)), shape).copy()
    spread = float(np.abs(b_side - apply_g0_3d(avg, cache)).max() / scale)
    blind = float(np.abs(uni - np.broadcast_to(uni.mean(axis=(0, 1, 2)), shape)).max())
    print(f"  [5c-2] vacuity: distinct vs site-averaged : {spread:.3e} (must be LARGE)")
    print(f"         uniform source is its own average  : {blind:.3e} (hence blind)")

    # [5c-3] Localisation, kept even though the gate passes: it is what
    # distinguishes a real defect from a comparison artefact next time.
    print("\n  [5c-3] localisation -- each link against something independent")
    ks = np.fft.ifft2(kernel, axes=(1, 2))
    brute = np.zeros_like(distinct)
    for lz in range(n_z):
        for ix in range(m):
            for iy in range(m):
                for mz in range(n_z):
                    for jx in range(m):
                        for jy in range(m):
                            brute[lz, ix, iy] += (
                                ks[lz - mz + n_z - 1, ix - jx + m - 1, iy - jy + m - 1]
                                @ distinct[mz, jx, jy]
                            )
    k_vs_t = float(np.abs(brute - b_side).max() / np.abs(brute).max())
    fft_vs_brute = float(np.abs(a_side - brute).max() / np.abs(brute).max())
    print(f"         slab kernel, brute-convolved, vs real-space table : {k_vs_t:.3e}")
    print(f"         slab FFT path vs brute force on its OWN kernel    : {fft_vs_brute:.3e}")

    ok = guard_ok and rel < 1e-12 and spread > 1e-6 and blind < 1e-14
    ok = ok and k_vs_t < 1e-12 and fft_vs_brute < 1e-12

    print("\n" + "=" * 78)
    print(f"GATE rung 5c: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("\n  Read [5c-0] FIRST. If the guard failed, the extraction is")
        print("  cancelling and every other number here is noise. Only once it")
        print("  passes do [5c-3]'s two lines mean anything: kernels agreeing with")
        print("  the FFT path disagreeing would indicate _slab_matvec; kernels")
        print("  disagreeing would indicate the propagator, the D4h orbit, the")
        print("  units, or the volume-averaging flag.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
