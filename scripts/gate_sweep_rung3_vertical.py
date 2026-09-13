#!/usr/bin/env python3
"""GATE, rung 3: the Cartesian inter-plane vertical sweep.

  [3]  PHYSICS. The k_y-integrated inter-plane sweep must reproduce the exact
       real-space propagator summed over the same pairs. QUADRATURE-LIMITED, so
       it is reported at two resolutions and passes by FALLING.

  [3b] NON-PERIODICITY. The real Earth is not horizontally periodic, so the
       lateral coupling must not wrap. Reported as the residual between the
       far-edge response and the kernel at the TRUE separation, alongside the
       value a wrapped (circular-convolution) implementation would have given,
       so the two cannot be confused.

  [3c] CONSISTENCY. The k_x-domain accumulation equals an O(N^2) pairwise sum
       over the same kernel -- transform bookkeeping only.

Run serially. Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering.directional_sweeps import (  # noqa: E402
    build_vertical_stack,
    make_sweep_grid,
    sweep_z,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25
N_Z, N_X = 2, 3


def _source() -> np.ndarray:
    s = np.zeros((N_Z, N_X, 9), dtype=complex)
    s[0, 1, 0] = 1.0
    return s


def sweep_over_ky(ky_max: float, n_ky: int, n_kx: int) -> np.ndarray:
    """Integrate the inter-plane sweep over k_y to get the 3-D response."""
    sources = _source()
    ky_nodes = np.linspace(-ky_max, ky_max, n_ky)
    dky = ky_nodes[1] - ky_nodes[0]

    total = np.zeros((N_Z, N_X, 9), dtype=complex)
    for idx, ky in enumerate(ky_nodes):
        grid = make_sweep_grid(N_Z, N_X, PITCH, ky=ky, n_kz=8, n_kx=n_kx)
        vertical = build_vertical_stack(grid, REF, OMEGA)
        weight = 0.5 if idx in (0, n_ky - 1) else 1.0
        total += sweep_z(sources, grid, vertical) * weight * dky / (2 * np.pi)
    return total


def exact_reference() -> np.ndarray:
    """Direct real-space pairwise sum over the inter-plane pairs."""
    sources = _source()
    out = np.zeros_like(sources)
    for lz in range(N_Z):
        for mz in range(N_Z):
            if lz == mz:
                continue
            for i in range(N_X):
                for j in range(N_X):
                    p = exact_propagator_9x9(
                        x=(i - j) * PITCH,
                        y=0.0,
                        z=(lz - mz) * PITCH,
                        omega=OMEGA,
                        ref=REF,
                    )
                    out[lz, i, :] += p @ sources[mz, j, :]
    return out


def check_non_periodicity() -> tuple[float, float]:
    """[3b] far-edge response vs the true separation, and vs the wrapped one."""
    n_x = 8
    grid = make_sweep_grid(2, n_x, PITCH, ky=0.6, n_kz=64, n_kx=512)
    vertical = build_vertical_stack(grid, REF, OMEGA)
    sources = np.zeros((2, n_x, 9), dtype=complex)
    sources[0, 0, 0] = 1.0
    far = sweep_z(sources, grid, vertical)[1, n_x - 1, :]

    def block(d: int) -> np.ndarray:
        ph = np.exp(1j * grid.kx_nodes * d * grid.pitch)
        return np.einsum("k,abk->ab", grid.kx_weights * ph, vertical[1, 0])

    true_sep = block(n_x - 1) @ sources[0, 0, :]
    wrapped = block(-1) @ sources[0, 0, :]
    rel_true = float(np.abs(far - true_sep).max() / np.abs(true_sep).max())
    rel_wrap = float(np.abs(far - wrapped).max() / np.abs(wrapped).max())
    return rel_true, rel_wrap


def check_consistency() -> float:
    """[3c] k_x-domain accumulation vs the O(N^2) pairwise sum."""
    rng = np.random.default_rng(4242)
    n_z, n_x = 3, 6
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=256)
    vertical = build_vertical_stack(grid, REF, OMEGA)
    sources = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    got = sweep_z(sources, grid, vertical)
    want = np.zeros_like(sources)
    for lz in range(n_z):
        for mz in range(n_z):
            if lz == mz:
                continue
            for i in range(n_x):
                for j in range(n_x):
                    ph = np.exp(1j * grid.kx_nodes * (i - j) * grid.pitch)
                    blk = np.einsum("k,abk->ab", grid.kx_weights * ph, vertical[lz, mz])
                    want[lz, i, :] += blk @ sources[mz, j, :]
    return float(np.abs(got - want).max() / np.abs(want).max())


def main() -> int:
    print("=" * 74)
    print("GATE rung 3 -- Cartesian inter-plane vertical sweep")
    print(f"  alpha={REF.alpha} beta={REF.beta} rho={REF.rho} (km/s, g/cm3)")
    print(f"  omega={OMEGA:.4f}  pitch={PITCH} km  n_z={N_Z}  n_x={N_X}")
    print("=" * 74)

    print("\n[3] PHYSICS -- k_y-integrated sweep vs the exact real-space pairwise sum")
    print("      QUADRATURE-LIMITED, not exact: pass = the residual FALLS.")
    want = exact_reference()
    scale = np.abs(want).max()
    print(f"      {'ky_max':>8} {'n_ky':>6} {'n_kx':>6}   {'relative residual':>18}")
    residuals = []
    for ky_max, n_ky, n_kx in ((60.0, 801, 512), (120.0, 2401, 1024)):
        got = sweep_over_ky(ky_max, n_ky, n_kx)
        rel = float(np.abs(got - want).max() / scale)
        residuals.append(rel)
        print(f"      {ky_max:8.1f} {n_ky:6d} {n_kx:6d}   {rel:18.3e}")
    ok3 = residuals[-1] < residuals[0]
    print(f"      residual falls under refinement  ->  {'PASS' if ok3 else 'FAIL'}")

    print("\n[3b] NON-PERIODICITY -- the lateral coupling must not wrap")
    rel_true, rel_wrap = check_non_periodicity()
    print(f"      far edge vs kernel at the TRUE separation (n_x-1 pitches): {rel_true:.3e}")
    print(f"      far edge vs the WRAPPED separation (1 pitch):             {rel_wrap:.3e}")
    okb = rel_true < 1e-12 and rel_wrap > 1e-1
    print("      a circular convolution would have matched the second, not the first")
    print(f"      ->  {'PASS' if okb else 'FAIL'}")

    print("\n[3c] CONSISTENCY -- k_x accumulation vs the O(N^2) pairwise sum")
    relc = check_consistency()
    print(f"      n_z=3  n_x=6  n_kx=256      relative residual = {relc:.3e}")
    okc = relc < 1e-12
    print(f"      target < 1e-12  ->  {'PASS' if okc else 'FAIL'}")

    print("\n" + "=" * 74)
    verdict = ok3 and okb and okc
    print(f"GATE rung 3: {'PASS' if verdict else 'FAIL'}")
    print("=" * 74)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
