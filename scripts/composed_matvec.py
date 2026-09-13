"""The composed matvec  (I - [P^z + P^x] dC_eff) psi.

This is the operator of the thesis two-potential formulation (GstratRep
LSstrat): per-voxel contrast screens dC_eff, coupled laterally by P^x and
vertically by the STRATIFIED P^z.

DESIGN.  `slab_scattering._slab_matvec` already has exactly the right shape --

    tau = T psi                (per voxel, real space)
    tau_hat = FFT_xy(tau)
    acc[m] = sum_n K[dz(m,n)] . tau_hat[n]       (9x9 at each kx,ky)
    return psi - IFFT_xy(acc)

-- with the depth kernel indexed by the DIFFERENCE m-n, which is correct only
for a homogeneous background.  A stratified reference is not translation
invariant in z, so the composition replaces that kernel by a PAIR-indexed one:

    K[m, n, kx, ky, 9, 9]

with the diagonal m == n carrying the intra-plane P^x (the laterally invariant
reference makes the existing homogeneous intra-plane kernel correct there) and
the off-diagonal m != n carrying the corrected stratified P^z.  Cost is
unchanged: the depth loop was already O(N_z^2).  The extra work is building
N_z^2 stratified kernels per frequency instead of 2 N_z - 1 homogeneous ones --
which is precisely the "build Q^d once per frequency, sweep cheaply thereafter"
economy the thesis splitting exists for.

NORMALISATION IS NOT ASSUMED.  P^x is a sum over discrete scatterers; P^z is a
continuum Green's function.  Their relative scaling cannot be asserted, so
GATE 1 measures it: with the layered model collapsed to the slab's own uniform
reference, the stratified kernel must reproduce the homogeneous one that
`_build_slab_kernels` already produces.

RESULT: GATE 1 FAILS, AND THE MATVEC IS THEREFORE NOT BUILT HERE.

The ratio of the two kernels is not a constant.  Measured spread (std/median of
|ratio| over all wavenumbers), for inter-plane pairs, as attenuation is
increased to suppress free-surface and seabed reverberation:

    Q       1e5     50      10      3       1
    1->0    9.8     10.0    10.7    11.5    21.5
    2->0    14.1    14.3    15.6    34.9    42.9

It does not shrink -- it grows.  So this is NOT surface leakage (contrast the
translation-invariance test, where damping drove the residual to 1.7e-5); the
two operators are different objects.

WHY.  `_build_slab_kernels` sums the continuum propagator over DISCRETE voxel
positions and then FFTs, so by Poisson summation its spectral kernel is the
continuum kernel ALIASED over reciprocal lattice vectors,
(1/d^2) sum_G Ghat(k+G).  `layered_greens` returns the un-aliased continuum
kernel at a single k.  For a 1/r-type kernel the aliasing sum does not
converge quickly, so the discrepancy varies strongly with k -- exactly what is
measured.

CONSEQUENCE.  P^z must be brought into the DISCRETE representation before it is
added to P^x: build the real-space stratified propagator between voxel centres
(inverse-transform over kx, ky per depth pair), then lattice-sum and FFT it
through the same pipeline `_build_slab_kernels` already uses.  That is the
correct construction and it reuses the existing kernel machinery; it is not a
rescaling.  Adding the spectral P^z to the lattice P^x directly would be wrong
by a k-dependent factor of order ten.

RE-TAKEN 13 September 2026, THROUGH THE CORRECTED WRAPPER
---------------------------------------------------------
The numbers above were measured with the July source convention.  With the
three defects fixed (`resolved_9x9_grid`, second column of GATE 1 below) the
spread roughly HALVES but does NOT close:

    Q            1e5     10      3       1
    1->0  July   9.8    10.7    11.5    21.5
          now    ~8     ~8       8.3    12.2
    2->0  July  14.1    15.6    34.9    42.9
          now    ~6     ~5       5.9     8.7

and it still does not fall with Q.  So the wrapper defects were a real part of
N5 but not the whole of it, and the aliasing diagnosis above stands as the
explanation of what remains.  The CONSEQUENCE is unchanged: lattice-sum P^z
into the discrete representation before adding it to P^x.

An attempt to confirm that by comparing `resolved_9x9_grid` directly against a
freshly written continuum 9x9 was WITHDRAWN, not reported: it showed a uniform
20% offset at every wavenumber (all propagating; not contamination, flat in
frequency), whereas the SAME construction agrees with the validated whole-space
reference in scripts/gate_wrapper_resolution.py to 5e-15.  The fresh reference
was therefore the suspect, not the kernel.  Anyone retrying this should reuse
`gate_wrapper_resolution.whole_space_jump_response` -- promoting it to take the
medium as an argument -- rather than writing a second copy, and should use a
deep, well-separated geometry rather than planes buried 1-3 km under the seabed.

Run: conda run -n seismic python scripts/composed_matvec.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering import ReferenceMedium, SlabGeometry
from cubic_scattering.layered_correction import (
    correct_6x6,
    source_jump_operator,
    strain_from_state,
)
from cubic_scattering.slab_scattering import _build_slab_kernels

from Kennett_Reflectivity.layer_model import LayerModel  # isort: skip
from GlobalMatrix.layered_greens import (  # isort: skip
    _interface_elastic_properties,
    layered_greens_6x6,
    strain_from_displacement_traction,
)

W9 = np.diag(np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float))
J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)


def corrected_9x9_grid(model, w, kx, ky, j, i):
    """Corrected stratified 9x9, vectorised over a (kx, ky) grid.

    Applies both corrections established by
    scripts/gate_9x9_source_convention.py (GATE E, 9.7e-16):
      source normalisation  Q = diag(1,-1,-1, i/w, -i/w, -i/w)
      adjoint operator      B(k) = -J6 A_src(-k)^T W,  SOURCE material.
    """
    shape = kx.shape
    kxf, kyf = kx.ravel(), ky.ravel()

    G = layered_greens_6x6(model, w, kxf, kyf, source_iface=j, receiver_iface=i)
    Q = np.diag(np.array([1, -1, -1, 1j / w, -1j / w, -1j / w], dtype=complex))
    G = G @ Q

    rho_r, al_r, be_r = _interface_elastic_properties(model, i)
    A = strain_from_displacement_traction(kxf, kyf, rho_r, al_r, be_r)

    rho_s, al_s, be_s = _interface_elastic_properties(model, j)
    Am = strain_from_displacement_traction(-kxf, -kyf, rho_s, al_s, be_s)
    B = -np.einsum("ab,xcb,cd->xad", J6, Am, W9)

    out = np.einsum("xab,xbc,xcd->xad", A, G, B)
    return out.reshape(*shape, 9, 9)


def resolved_9x9_grid(model, w, kx, ky, j, i):
    """Stratified 9x9 with the RESOLVED correction (D1, D2, D3).

    Supersedes ``corrected_9x9_grid``, whose Q and B were the July guesses.  See
    cubic_scattering.layered_correction and
    docs/wrapper_problem_state_2026-09-13.md.  K depends on the direction of k,
    so the correction is applied per wavenumber rather than as one matrix.
    """
    shape = kx.shape
    kxf, kyf = kx.ravel(), ky.ravel()
    g = layered_greens_6x6(model, w, kxf, kyf, source_iface=j, receiver_iface=i)

    s_s = model.complex_slowness_s()
    ss_src, ss_rcv = s_s[max(j, 1)], s_s[max(i, 1)]
    rho_r, al_r, be_r = _interface_elastic_properties(model, i)
    rho_s, al_s, be_s = _interface_elastic_properties(model, j)

    out = np.empty(g.shape[:-2] + (9, 9), dtype=complex)
    for t in range(g.shape[0]):
        gc = correct_6x6(g[t], w, ss_src, ss_rcv, kxf[t], kyf[t])
        a = strain_from_state(kxf[t], kyf[t], rho_r, al_r, be_r)
        b = source_jump_operator(kxf[t], kyf[t], rho_s, al_s, be_s)
        out[t] = a @ gc @ b
    return out.reshape(*shape, 9, 9)


def fft_wavenumbers(n, d):
    """FFT-ordered wavenumbers for n samples at spacing d."""
    return 2.0 * np.pi * np.fft.fftfreq(n, d=d)


def gate_homogeneous_limit(M=8, N_z=3, a=0.5, freq=6.0, Q=1e5):
    """GATE 1 -- collapse the layered model to the slab's own uniform medium.

    The stratified kernel must then reproduce the homogeneous inter-plane
    kernel that _build_slab_kernels already produces, up to ONE overall
    constant (the discrete-lattice vs continuum normalisation). Measure that
    constant and check it is the same for every depth pair and wavenumber --
    a single scalar is a normalisation; anything varying is a real mismatch.
    """
    d = 2.0 * a
    ref = ReferenceMedium(alpha=4.0, beta=2.22, rho=2.6)
    omega = 2.0 * np.pi * freq
    geom = SlabGeometry(M=M, N_z=N_z, a=a)

    kern = _build_slab_kernels(geom, omega, ref, periodic=True)  # (2Nz-1, M, M, 9,9)

    model = LayerModel.from_arrays(
        alpha=[1.5, *([ref.alpha] * N_z), ref.alpha],
        beta=[0.0, *([ref.beta] * N_z), ref.beta],
        rho=[1.03, *([ref.rho] * N_z), ref.rho],
        thickness=[3.0, *([d] * N_z), np.inf],
        Q_alpha=[Q, *([Q] * N_z), Q],
        Q_beta=[1e10, *([Q] * N_z), Q],
    )

    kx1 = fft_wavenumbers(M, d)
    KX, KY = np.meshgrid(kx1, kx1, indexing="ij")
    KX = np.where(np.abs(KX) + np.abs(KY) < 1e-12, 1e-6, KX)  # avoid exact k=0

    print("GATE 1 - homogeneous limit: stratified kernel vs the existing")
    print("         homogeneous inter-plane kernel")
    print(f"  M={M}, N_z={N_z}, a={a} km, f={freq} Hz, d={d} km")
    print()
    print(f"  {'pair':>8}   {'JULY median':>13} {'spread':>10}   {'RESOLVED median':>15} {'spread':>10}")

    for m in range(N_z):
        for n in range(N_z):
            if m == n:
                continue
            homo = kern[(m - n) + (N_z - 1)]
            msk = np.abs(homo) > 1e-9 * np.abs(homo).max()
            if msk.sum() < 20:
                continue
            cells = []
            for build in (corrected_9x9_grid, resolved_9x9_grid):
                r = build(model, omega, KX, KY, j=n + 1, i=m + 1)[msk] / homo[msk]
                med = np.median(np.abs(r))
                cells.append((med, np.std(np.abs(r)) / med if med > 0 else np.inf))
            print(
                f"  {f'{n}->{m}':>8}   {cells[0][0]:13.4e} {cells[0][1]:10.3e}   "
                f"{cells[1][0]:15.4e} {cells[1][1]:10.3e}"
            )

    print()
    print("  A single constant ratio would be a pure normalisation. A varying")
    print("  ratio means the two operators are not the same object. If the spread")
    print("  does not fall as Q falls, it is not surface leakage.")


if __name__ == "__main__":
    for _Q in (1e5, 50.0, 10.0, 3.0, 1.0):
        print(f"\n########## Q = {_Q:g} ##########")
        gate_homogeneous_limit(Q=_Q)
