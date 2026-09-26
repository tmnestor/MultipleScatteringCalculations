#!/usr/bin/env python3
"""MEASUREMENT, not a gate: can the elastic voxel operator be SPARSIFIED as Liu & Ying sparsify Helmholtz?

WHY.  A dense 3-D lateral sweep cannot carry the in-plane Green's tensor compactly
(``measure_sweep_exponential_rank.py``; Engquist & Zhao prove the rank must grow).
Liu & Ying ("Sparsify and sweep", SISC 2018) keep the dense operator for the matvec
and sweep only a PRECONDITIONER, built by first making the system local: for each
3x3x3 neighbourhood mu they find stencils alpha that annihilate the Green's columns
of every voxel OUTSIDE mu,

    alpha = argmin || alpha^H K[mu, mu^c] ||,   ||alpha|| = 1,

so each dense Lippmann-Schwinger row collapses to a 27-point equation.  Their
preconditioner is only as good as that annihilation.

WHAT IS MEASURED.  The annihilation ratio

    r = sigma_(m-th smallest)(K[mu, mu^c]) / sigma_max(K[mu, mu^c]),

with m the number of stencils one voxel needs: m = 1 for scalar Helmholtz (27
unknowns), m = 9 for our 9-component voxel (243 unknowns: displacement + Voigt
strain).  K is the whole-space point propagator on the lattice, the far set
mu^c every lattice point with 1 < |i|_inf <= R.

THE CALIBRATION is Liu & Ying's own case, scalar e^{ikr}/(4 pi r) at the SAME
pitch and wavenumber.  Their method works there, so the question is not whether
r is small but whether the elastic r is comparable to the scalar one at matched
points per wavelength, and whether it holds as R -- the reach the stencil must
cancel -- grows.

Run:  conda run -n seismic python scripts/measure_sparsify_stencil_accuracy.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402

REF = ReferenceMedium(5.0, 3.0, 2.5)
PITCH = 0.25
DAMPING = 0.03  # the solver's omega is complex


def lattice(r: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The 27-point neighbourhood and the far set 1 < |i|_inf <= r.

    Args:
        r: Half-width of the far set, in pitches.

    Returns:
        (neighbourhood offsets, far offsets).
    """
    near = list(itertools.product((-1, 0, 1), repeat=3))
    far = [p for p in itertools.product(range(-r, r + 1), repeat=3) if max(map(abs, p)) > 1]
    return near, far


def block_elastic(near: list, far: list, omega: complex) -> NDArray:
    """K[mu, mu^c] for the 9-component voxel: rows (near point, component).

    Args:
        near: Neighbourhood offsets.
        far: Far offsets.
        omega: Complex angular frequency.

    Returns:
        Shape (9 len(near), 9 len(far)).
    """
    k = np.zeros((9 * len(near), 9 * len(far)), dtype=complex)
    for i, a in enumerate(near):
        for j, b in enumerate(far):
            d = np.subtract(a, b) * PITCH  # receiver minus source, (z, x, y)
            k[9 * i : 9 * i + 9, 9 * j : 9 * j + 9] = exact_propagator_9x9(d[1], d[2], d[0], omega, REF)
    return k


def block_scalar(near: list, far: list, kk: complex) -> NDArray:
    """K[mu, mu^c] for scalar Helmholtz e^{ikr}/(4 pi r).

    Args:
        near: Neighbourhood offsets.
        far: Far offsets.
        kk: Complex wavenumber.

    Returns:
        Shape (len(near), len(far)).
    """
    a = np.array(near, dtype=float)[:, None, :]
    b = np.array(far, dtype=float)[None, :, :]
    rr = np.linalg.norm(a - b, axis=-1) * PITCH
    return np.exp(1j * kk * rr) / (4 * np.pi * rr)


def annihilation(k: NDArray, m: int) -> float:
    """sigma of the m-th smallest over sigma_max: the worst of the m best stencils.

    Args:
        k: The far block, rows the neighbourhood unknowns.
        m: Stencils needed per voxel.

    Returns:
        The ratio.
    """
    s = np.linalg.svd(k, compute_uv=False)
    return float(s[k.shape[0] - m] / s[0])


def diagnostics(freq: float, r: int) -> None:
    """Is the elastic ratio dynamics, or row scaling and strain-displacement redundancy?

    A singular-value ratio is not invariant under row scaling, and the 9-component
    voxel mixes displacement with strain (units 1/km).  And strain is a derivative
    of displacement, so some stencils may be the kinematic relation "strain minus
    a finite difference of displacement" rather than the wave equation.

    Args:
        freq: Frequency, Hz.
        r: Far-set half-width.
    """
    omega = 2 * np.pi * freq * (1.0 + 1j * DAMPING)
    near, far = lattice(r)
    k = block_elastic(near, far, omega)
    rows = np.tile(np.r_[np.ones(3), np.full(6, PITCH)], len(near))  # strain x pitch: displacement units
    s_raw = np.linalg.svd(k, compute_uv=False)
    s_sc = np.linalg.svd(rows[:, None] * k, compute_uv=False)
    disp = k.reshape(len(near), 9, len(far), 9)[:, :3].reshape(3 * len(near), -1)
    s_u = np.linalg.svd(disp, compute_uv=False)
    sc = annihilation(block_scalar(near, far, omega / REF.beta), 1)
    n = k.shape[0]
    print(f"\n  f = {freq} Hz, R = {r}: smallest singular values / largest, smallest first")
    print(f"    9-comp, raw rows       : {' '.join(f'{v:.1e}' for v in s_raw[::-1][:12] / s_raw[0])}")
    print(f"    9-comp, strain x pitch : {' '.join(f'{v:.1e}' for v in s_sc[::-1][:12] / s_sc[0])}")
    print(f"    displacement only (81) : {' '.join(f'{v:.1e}' for v in s_u[::-1][:6] / s_u[0])}")
    print(
        f"    r: raw {s_raw[n - 9] / s_raw[0]:.2e}   strain x pitch {s_sc[n - 9] / s_sc[0]:.2e}   "
        f"displacement-only (3 stencils) {s_u[3 * len(near) - 3] / s_u[0]:.2e}   scalar k_S {sc:.2e}"
    )


def main() -> int:
    """Measure the annihilation ratio, elastic against scalar, over frequency and reach.

    Returns:
        0.
    """
    print("\nDIAGNOSTICS")
    for freq in (1.5, 6.0):
        diagnostics(freq, 4)
    print("=" * 78)
    print("SPARSIFYING STENCILS: ANNIHILATION OF THE FAR FIELD BY A 27-POINT NEIGHBOURHOOD")
    print("  r = sigma_(m-th smallest) / sigma_max of K[mu, mu^c];  scalar m = 1, elastic m = 9")
    print("=" * 78)
    for freq in (1.5, 3.0, 6.0):
        omega = 2 * np.pi * freq * (1.0 + 1j * DAMPING)
        k_s, k_p = omega / REF.beta, omega / REF.alpha
        ppw = 2 * np.pi / (k_s.real * PITCH)
        ppw_p = ppw * REF.alpha / REF.beta
        print(f"\n  f = {freq} Hz: {ppw:.1f} points per S wavelength, {ppw_p:.1f} per P")
        head = ("R", "far pts", "scalar k_S", "scalar k_P", "elastic", "el/sc(k_S)")
        print(f"  {head[0]:>3} {head[1]:>8} " + " ".join(f"{h:>11}" for h in head[2:]))
        for r in (2, 3, 4, 6):
            near, far = lattice(r)
            sc_s = annihilation(block_scalar(near, far, k_s), 1)
            sc_p = annihilation(block_scalar(near, far, k_p), 1)
            el = annihilation(block_elastic(near, far, omega), 9)
            print(f"  {r:3d} {len(far):8d} {sc_s:11.2e} {sc_p:11.2e} {el:11.2e} {el / sc_s:11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
