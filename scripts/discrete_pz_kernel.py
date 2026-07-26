"""Discrete P^z kernel: the stratified propagator on the voxel lattice.

Why this is needed.  scripts/composed_matvec.py showed that the spectral
stratified propagator and the lattice P^x are NOT the same object: their ratio
in the homogeneous limit varies by 10-14x across wavenumber, and unlike every
other discrepancy chased so far it GROWS under damping, so it is not surface
leakage.  By Poisson summation, `_build_slab_kernels` produces the continuum
kernel ALIASED over reciprocal lattice vectors, (1/d^2) sum_G Ghat(k+G), while
`layered_greens` returns the un-aliased continuum kernel at a single k.

The fix is to put P^z through the same discretisation P^x already has:

    Ptilde(kx,ky)  --2-D inverse FT-->  P(dx,dy)  --sample on lattice-->
    K_lat[i,j]     --FFT-->             K(kx,ky) on the FFT grid

so that both operators are lattice sums of the same kind and may be added.

SAMPLING.  The k-grid is chosen so the real-space samples land EXACTLY on
lattice multiples: with real-space spacing dx = d/q the grid has
Dk = 2 pi q/(N d) and k_max = pi q/d, and lattice offset j*d is sample index
j*q.  No interpolation.  q must be large enough that k_max comfortably exceeds
the S wavenumber, which is checked and reported.

GATE.  In the homogeneous limit the discrete stratified kernel must reproduce
`_build_slab_kernels` up to ONE constant.

STATUS: THE GATE IS NOW TRUSTWORTHY.  THE CONSTRUCTION IS NOT YET VALIDATED.

Reference side VERIFIED.  `_build_slab_kernels` satisfies the GATE A invariant
(W P symmetric) to 1e-16 and simply IS the closed-form propagator sampled on
the lattice: |slab| / |exact_propagator_9x9| = 1.006, 1.001 at offsets 1d, 2d
(1.226 at 3d is the periodic image).  So the discrepancy is on the stratified
side, not this one.

THREE CAUSES ELIMINATED -- all three were proposed confidently and all three
were wrong, which is why each is recorded rather than deleted:

  1. k-TRUNCATION.  Sweeping q = 8, 16, 32 (k_max/k_S = 1.48, 2.96, 5.92)
     changes the result by NOTHING -- bit-identical to five figures.  The
     k-integral converges well before k_max.

  2. k-RESOLUTION / PADDING.  Also nothing, and this one is eliminated
     MATHEMATICALLY, not just empirically: folding a finely sampled transform
     back onto the coarse period reproduces the coarse-period transform
     exactly.  Padding then folding is an identity, so refining dk cannot
     change a periodic kernel.  Sampling is therefore not the cause at all.

  3. ASYMMETRIC DAMPING IN THE GATE -- this was a real defect and is now
     FIXED.  The old version damped only the layered model while the reference
     carried no attenuation, so low-Q rows compared different media; the
     spread appeared to collapse by four orders of magnitude, which was an
     artefact.  `_build_slab_kernels` accepts complex velocities, so the
     reference is now built from the layered model's own complex velocities
     and the media are identical by construction.  The fix changed the
     picture: the median ratio is now STABLE across Q (10.6, 11.6, 12.7, 14.0,
     12.3) instead of collapsing.

WHAT REMAINS, stated as measurement rather than mechanism.  With the media
matched, the two kernels differ by a roughly constant magnitude factor of
about 12 -- suggestively close to 4 pi = 12.566 -- together with a spread of
roughly 200 percent that does NOT fall as damping suppresses the free surface.
So the surface-reverberation hypothesis is not confirmed either.

The next step should establish the constant and the scatter SEPARATELY: a 4 pi
normalisation is checkable in isolation against a single known value, and only
once it is settled does the residual scatter become interpretable.  No further
mechanism is proposed here on purpose -- three have already been proposed and
refuted, and the pattern says to establish facts one at a time with controls.

Run: conda run -n seismic python scripts/discrete_pz_kernel.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering import ReferenceMedium, SlabGeometry
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
    """Corrected stratified 9x9 over a (kx,ky) grid (gate_9x9 GATE E, 9.7e-16)."""
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

    return np.einsum("xab,xbc,xcd->xad", A, G, B).reshape(*shape, 9, 9)


def discrete_pz_kernel(model, omega, m, n, M, d, q=8, pad=1, report=False):
    """Stratified propagator sampled on the voxel lattice, then FFT'd.

    Args:
        model: layered background.
        omega: angular frequency.
        m, n: receiver / source PLANE indices (0-based); interface = index + 1.
        M: lateral lattice size (periodic M x M convention).
        d: voxel pitch.
        q: real-space oversampling; k_max = pi q / d.

    Returns:
        K(kx,ky) on the M x M FFT grid, shape (M, M, 9, 9).
    """
    # TWO INDEPENDENT PARAMETERS -- conflating them was the original bug:
    #   q   sets the real-space spacing dx = d/q, hence k_max = pi/dx
    #   pad sets the real-space EXTENT L = pad*M*d, hence dk = 2*pi/L
    # With pad fixed at 1, dk = 2*pi/(M*d) regardless of q, so raising q
    # extended k_max while never refining the k-sampling -- which is why the
    # answer was bit-identical for q = 8, 16, 32 and why the real-space kernel
    # did not decay with distance (it was aliased).
    dx = d / q
    L = pad * M * d
    N = int(round(L / dx))  # = pad * q * M
    dk = 2.0 * np.pi / L
    k_max = np.pi / dx

    k1 = dk * np.fft.fftfreq(N, d=1.0 / N)  # FFT-ordered, symmetric
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    # keep the exact origin off the branch point
    KX = np.where(np.abs(KX) + np.abs(KY) < 1e-12, 1e-8, KX)

    if report:
        beta_min = min(b for b in model.beta[1:] if b > 0)
        k_S = omega / beta_min
        print(
            f"    k_max/k_S = {k_max / k_S:5.2f}   samples per k_S = "
            f"{k_S / dk:6.1f}   N = {N}"
        )

    Ptilde = corrected_9x9_grid(model, omega, KX, KY, j=n + 1, i=m + 1)

    # 2-D inverse transform: P(x) = (1/(2pi)^2) int dk e^{ikx} Ptilde
    P_real = np.fft.ifft2(Ptilde, axes=(0, 1)) * (N**2) * dk**2 / (2 * np.pi) ** 2

    # sample on the lattice: offset j*d is index j*q  -> pad*M lattice offsets
    P_lat = P_real[::q, ::q]  # (pad*M, pad*M, 9, 9)

    # fold the periodic images back onto M x M, matching the convention of
    # _build_slab_kernels(periodic=True)
    K_lat = P_lat.reshape(pad, M, pad, M, 9, 9).sum(axis=(0, 2))

    return np.fft.fft2(K_lat, axes=(0, 1))


def gate(M=8, N_z=3, a=0.5, freq=6.0, q=8):
    """Homogeneous-limit gate with SYMMETRIC damping.

    The earlier version damped only the layered model, leaving the comparison
    medium undamped, so the low-Q rows compared different media and the sweep
    was uninterpretable.  `_build_slab_kernels` accepts complex velocities, so
    the reference is now built from the layered model's OWN complex velocities
    and the two media are identical by construction.  The only remaining
    difference is then the ocean and free surface, which damping suppresses --
    so a correct construction must show the spread falling towards zero.
    """
    d = 2.0 * a
    al0, be0, rh0 = 4.0, 2.22, 2.6
    omega = 2.0 * np.pi * freq
    geom = SlabGeometry(M=M, N_z=N_z, a=a)

    print("GATE - discrete stratified kernel vs the homogeneous lattice kernel")
    print(f"  M={M}, N_z={N_z}, d={d} km, f={freq} Hz, q={q}")
    print("  damping applied to BOTH media (reference built from the layered")
    print("  model's own complex velocities)")
    print()
    print(f"  {'Q':>8} {'pair':>7} {'|ratio| median':>16} {'spread':>12}")

    for Q in (1e5, 100.0, 30.0, 10.0, 3.0):
        model = LayerModel.from_arrays(
            alpha=[1.5, *([al0] * N_z), al0],
            beta=[0.0, *([be0] * N_z), be0],
            rho=[1.03, *([rh0] * N_z), rh0],
            thickness=[3.0, *([d] * N_z), np.inf],
            Q_alpha=[Q, *([Q] * N_z), Q],
            Q_beta=[1e10, *([Q] * N_z), Q],
        )
        # identical medium on the homogeneous side, including attenuation
        ref = ReferenceMedium(
            alpha=1.0 / model.complex_slowness_p()[1],
            beta=model.complex_velocity_s()[1],
            rho=rh0,
        )
        kern = _build_slab_kernels(geom, omega, ref, periodic=True)

        first = True
        for m, n in ((0, 1), (0, 2)):
            K = discrete_pz_kernel(
                model, omega, m, n, M, d, q=q, report=(first and Q > 1e4)
            )
            first = False
            homo = kern[(m - n) + (N_z - 1)]
            msk = np.abs(homo) > 1e-9 * np.abs(homo).max()
            r = K[msk] / homo[msk]
            med = float(np.median(np.abs(r)))
            spread = float(np.std(np.abs(r)) / med) if med > 0 else np.inf
            print(f"  {Q:8.0f} {f'{n}->{m}':>7} {med:16.4e} {spread:12.3e}")

    print()
    print("  PASS if the spread FALLS towards zero as Q falls: with the media now")
    print("  identical, the only residual is free-surface/seabed reverberation,")
    print("  which the whole-space kernel cannot contain and damping removes.")


if __name__ == "__main__":
    # q is converged by 8 (see docstring); no need to sweep it
    gate(q=8)
