"""Gates G1 and G2 of the stratified march: the mode factorisation of Paper 2's coupling.

Plan: ``docs/2026-10-10-stratified-reference-legendre-cells-3d.md``.

G1  The whole-space plane-to-plane spectrum is  sum_m w_m e^{i k_z,m |dz|} d_m d_m^T M  (the source map is
    the transposed receiver map, with the engineering-stress pairing M).  Arbiter:
    ``sweep_kernels.vertical_kernel_9x9`` (k_z residue, gated against the closed-form Kupradze propagator).
G2  With Paper 2's closed-form cell moments on both ends, the mode integral reproduces Paper 2's exact
    Galerkin blocks (``graded_voxel.blocks.coupling_block``) for cells in different, non-touching planes.
    RECORDED, not gated: cells in touching planes, where the mode integral does not converge.

Run:  PYTHONPATH=. python scripts/gate_march_modes.py
Seismic units (km/s, g/cm^3, km), time e^{-i omega t}.
"""

import numpy as np

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel.blocks import coupling_block
from cubic_scattering.stratified_march import (
    SOURCE_PAIRING,
    mode_family,
    mode_integral_block,
)
from cubic_scattering.sweep_kernels import vertical_kernel_9x9

REF = ReferenceMedium(5.0, 3.0, 2.5)


def gate_g1() -> bool:
    """G1: the factorised spectrum against the residue construction."""
    worst = 0.0
    for omega in (10.0, 10.0 * (1 + 0.03j)):
        for kx, ky in [(0.0, 0.0), (0.3, 0.2), (1.5, -0.7), (4.0, 2.5), (9.0, 1.0)]:
            for dz in (0.4, -0.7):
                sign = float(np.sign(dz))
                got = np.zeros((9, 9), dtype=complex)
                for wave in ("P", "S"):
                    for k, d, w in mode_family(
                        np.array([kx]), np.array([ky]), omega, REF, wave, sign
                    ):
                        phase = np.exp(
                            1j * k[0, 0] * dz
                        )  # k[0, 0] = sign k_z, so this is e^{i k_z |dz|}
                        got += w[0] * phase * np.outer(d[0], d[0] * SOURCE_PAIRING)
                want = vertical_kernel_9x9(np.array([kx]), ky, dz, omega, REF)[:, :, 0]
                worst = max(worst, np.abs(got - want).max() / np.abs(want).max())
    ok = worst < 1e-12
    print(
        f"G1  spectrum = W D^T M, 20 cases incl. evanescent and complex omega: {worst:.1e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def gate_g2() -> bool:
    """G2: mode integral with closed-form moments against Paper 2's exact blocks."""
    h, omega = 0.05, 10.0
    ok = True
    for off, ns, nt in [
        ((2, 0, 0), 10, 4),
        ((2, 1, 0), 10, 4),
        ((3, 1, -1), 10, 4),
        ((-2, 1, 0), 20, 10),
        ((2, 1, 1), 35, 10),
    ]:
        want = coupling_block(off, h, omega, REF, n_source=ns, n_test=nt)
        got = mode_integral_block(off, h, omega, REF, n_source=ns, n_test=nt)
        err = np.abs(got - want).max() / np.abs(want).max()
        passed = err < 1e-11
        ok &= passed
        print(
            f"G2  offset {off}, {nt} field x {ns} source functions: {err:.1e}  {'PASS' if passed else 'FAIL'}"
        )
    with np.errstate(all="ignore"):
        for off in [(1, 0, 0), (1, 1, 0)]:
            want = coupling_block(off, h, omega, REF)
            got = mode_integral_block(off, h, omega, REF)
            err = np.abs(got - want).max() / np.abs(want).max()
            print(
                f"G2  RECORD touching planes {off}: {err:.1e}  (does not converge; Paper 2's closed forms carry these)"
            )
    return ok


if __name__ == "__main__":
    results = [gate_g1(), gate_g2()]
    print("ALL PASS" if all(results) else "FAILURES")
