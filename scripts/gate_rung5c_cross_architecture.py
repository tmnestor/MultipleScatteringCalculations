#!/usr/bin/env python3
"""GATE rung 5c: the 3-D operator against the FFT-convolution architecture.

Deferred from stage 1 for a stated reason -- `slab_scattering` is 3-D on a
finite M x M footprint and cannot represent a y-invariant medium, so it could
not arbitrate a 2.5-D solver. Stage 2 removes that restriction, so the
comparison is finally meaningful.

WHY THE COMPARISON IS VALID, checked rather than assumed:

  * `slab_scattering` defaults to `periodic=False`, which is a ZERO-PADDED
    LINEAR convolution on (2M-1) x (2M-1) -- not a circular one. Both
    architectures are therefore non-periodic and the comparison does not
    measure horizontal periodicity.
  * It defaults to `volume_averaged=False`, the POINT propagator, which is the
    object the real-space tables hold. With volume averaging on, the two
    compute genuinely different objects at touching faces -- a converged
    projection difference, not a bug -- and the comparison would be invalid.
  * Both index the lattice (n_z, n_x, n_y) in seismological (z, x, y) order,
    and both take separations from voxel centres on the same pitch, so no
    reindexing is involved.

WHAT IS COMPARED, and why it is G0 rather than a reflection coefficient. The
two architectures are compared at the OPERATOR level: G0 applied to a random
field with a distinct source at every site. Comparing reflection coefficients
instead would drag in the far-field projection and the flux normalisation, both
of which are separately gated objects with their own convention traps, and a
disagreement would not localise. G0 is the thing the two architectures actually
implement differently -- FFT convolution against a real-space table -- so it is
the thing to compare.

`_slab_matvec` computes (I - G T), so with T = I the operator gives
G psi = psi - matvec(psi). That private function is used deliberately: it is
what isolates G0 from the solve.

UNITS. `slab_scattering` works in SI (metres, m/s, kg/m^3) and so does this
gate, throughout, for both sides. The real-space tables are unit-agnostic
formulas; mixing seismic and SI here would be a silent scale error of the exact
kind this programme has paid for before.

Run:  conda run -n seismic python scripts/gate_rung5c_cross_architecture.py
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

# SI throughout, matching slab_scattering's own convention.
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OM = 150.0


def slab_g0(psi: np.ndarray, geom: SlabGeometry, kernel) -> np.ndarray:
    """G0 psi from the FFT-convolution architecture, isolated from the solve."""
    n_z, m = geom.N_z, geom.M
    eye = np.broadcast_to(np.eye(9, dtype=complex), (n_z, m, m, 9, 9)).copy()
    out = _slab_matvec(psi.ravel(), eye, kernel, geom)
    return psi - out.reshape(n_z, m, m, 9)


def main() -> int:
    m, n_z, a = 4, 3, 25.0
    geom = SlabGeometry(M=m, N_z=n_z, a=a)
    pitch = geom.d

    print("=" * 78)
    print("GATE rung 5c -- 3-D real-space operator vs FFT convolution")
    print(f"  lattice N_z={n_z}, M={m}, cube side d={pitch} m, omega={OM} rad/s")
    print(f"  background a={REF.alpha} b={REF.beta} rho={REF.rho} (SI)")
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

    scale = float(np.abs(a_side).max())
    rel = float(np.abs(a_side - b_side).max() / scale)
    print(f"\n  [5c-1] G0 psi, distinct sources : rel diff {rel:.3e}")
    print(f"         |G0 psi| (slab)          : {scale:.5e}")
    print(f"         |G0 psi| (real space)    : {np.abs(b_side).max():.5e}")

    # Localise any disagreement rather than reporting one number: a self-term
    # convention mismatch shows up on the diagonal alone, a propagator error
    # everywhere.
    worst = np.unravel_index(np.abs(a_side - b_side).argmax(), a_side.shape)
    print(f"         worst entry              : {worst}")

    # Vacuity control. A uniform source would pass for an implementation that
    # silently averaged the sites, so demonstrate that it cannot discriminate.
    # RELATIVE, not absolute: in SI the fields here are ~1e-13, and an absolute
    # threshold would fail a correct implementation on the unit system alone.
    uni = np.ones(shape, dtype=complex)
    avg = np.broadcast_to(distinct.mean(axis=(0, 1, 2)), shape).copy()
    spread = float(np.abs(b_side - apply_g0_3d(avg, cache)).max() / np.abs(b_side).max())
    blind = float(np.abs(uni - np.broadcast_to(uni.mean(axis=(0, 1, 2)), shape)).max())
    print(f"  [5c-2] vacuity: distinct vs site-averaged : {spread:.3e} (must be LARGE)")
    print(f"         uniform source is its own average  : {blind:.3e} (hence blind)")

    # LOCALISATION. A single cross-architecture number says nothing about which
    # side is wrong, so the gate localises before it judges. Each step compares
    # one link in the chain against something already gated independently.
    print("\n  [5c-3] localisation -- which side, and which link")
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

    kernels_agree = k_vs_t < 1e-10
    fft_exact = fft_vs_brute < 1e-10
    ok = rel < 1e-10 and spread > 1e-6 and blind < 1e-14

    print("\n" + "=" * 78)
    print(f"GATE rung 5c: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("\n  DIAGNOSIS from [5c-3]:")
        if kernels_agree and not fft_exact:
            print("    The two architectures build the SAME kernel (agreeing to")
            print("    round-off), and slab_scattering's FFT convolution disagrees")
            print("    with a direct convolution of that very kernel. The defect is")
            print("    therefore inside _slab_matvec, not in the propagator, not in")
            print("    the D4h orbit symmetrisation, and not in the real-space")
            print("    tables -- each of which is separately confirmed here or by")
            print("    gate_g0_3d (pairwise sum, 2.9e-16).")
            print("\n    NOTE ON WHY THIS WAS NOT CAUGHT: slab_scattering is validated")
            print("    against Kennett at ~0.5-1%, and this defect is ~4e-4. It sits")
            print("    two orders BELOW that tolerance and cannot show there.")
        elif not kernels_agree:
            print("    The kernels themselves differ -- check the propagator, the D4h")
            print("    orbit transforms, the units, and the volume-averaging flag")
            print("    before suspecting either convolution.")
        else:
            print("    Both links check out individually; the disagreement is in the")
            print("    comparison itself. Check the self-term convention and the")
            print("    lattice-centre offsets.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
