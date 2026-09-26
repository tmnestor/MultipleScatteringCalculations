#!/usr/bin/env python3
"""MEASUREMENT, not a gate: can a 3-D lateral sweep carry the in-plane kernel compactly?

WHY.  ``measure_sweep3d_cost.py`` found that carrying a UNIFORM 2-D transverse
quadrature through a running sweep needs 1.85M nodes per voxel -- 68 GB on a
16^3 lattice -- and the 3-D design moved to real-space tables.  That measured
one quadrature, not the sweep.  A running sweep needs only that the lattice
kernel be a short sum of exponentials in the swept offset: a term z^m is
carried by one amplitude multiplied by z per voxel.

WHAT IS MEASURED.  The exact 9x9 whole-space propagator at equal depth,
A(m, n) = P(m p, n p, 0), on the quadrant m >= 1, 0 <= n < N (the m = 0 axis is
a 1-D sweep of its own).  In two stages, each a multi-channel matrix pencil:

    x:  A(m, n, c) = sum_j  z_j^m  B_j(n, c)       z_j shared by every n and c
    y:  B_j(n, c)  = sum_l  w_l^n  C_jl(c)         w_l shared by every j and c

The x-then-y sweep carries J amplitudes per voxel in its first pass and J L in
its second, against 1.85M for the uniform rule.  THE QUESTION IS WHETHER J AND
L STAY BOUNDED AS N GROWS: if they grow like N, the "sweep" is a dense sum in
disguise and nothing is gained.

RESULT (2026-09-26).  They grow.  At 1e-6, 6 Hz (2 points per S wavelength)
never compresses; 2 Hz needs 15 of 31 x-terms at N = 32 and 21 of 63 at N = 64,
and y then never compresses.  Engquist & Zhao's lower bound on the separability
rank of the Helmholtz Green's function (it must grow with k) says this is
intrinsic.

CONTROLS.  The pencil must recover a synthetic 5-term sequence as exactly 5
terms, and must NOT compress random data (J ~ half the samples).

Run:  conda run -n seismic python scripts/measure_sweep_exponential_rank.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

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
DAMPING = 0.03  # as measure_sweep3d_cost.py: the solver's omega is complex
TARGET = 1e-6
#: Uniform-rule nodes per voxel at 1.3e-6 and 3.2e-7 (measure_sweep3d_cost.py).
UNIFORM_NODES = {1.3e-6: 823_592, 3.2e-7: 1_850_000}


def pencil_fit(s: NDArray, n_terms: int) -> tuple[NDArray, NDArray]:
    """Multi-channel matrix pencil: s[m, c] ~ sum_j coef[j, c] z[j]^m.

    Args:
        s: Samples, shape (M, C).
        n_terms: Number of exponentials J.

    Returns:
        (z of shape (J,), coef of shape (J, C)).
    """
    m_len, n_ch = s.shape
    p = m_len // 2
    rows = m_len - p
    h = np.concatenate([np.stack([s[i : i + p + 1, c] for i in range(rows)]) for c in range(n_ch)])
    _, _, vh = np.linalg.svd(h, full_matrices=False)
    # H = W Z^T with Z[k, j] = z_j^k, so span(Z) is the span of the ROWS of vh
    # (transposed, not conjugated): conjugating returns conj(z) and fails the control.
    v = vh[:n_terms].T  # (p+1, J)
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    vand = z[None, :] ** np.arange(m_len)[:, None]
    coef = np.linalg.lstsq(vand, s, rcond=None)[0]
    return z, coef


def fewest_terms(s: NDArray, scale: float) -> tuple[int, float, NDArray, NDArray]:
    """The smallest J whose pencil fit reproduces s to TARGET of scale.

    Args:
        s: Samples, shape (M, C).
        scale: Error normalisation.

    Returns:
        (J, achieved error, z, coef); J = -1 if no J reaches TARGET.
    """
    best = (-1, np.inf, np.zeros(0), np.zeros((0, s.shape[1])))
    for j in range(1, s.shape[0] // 2 + 1):
        z, coef = pencil_fit(s, j)
        err = float(np.abs(z[None, :] ** np.arange(s.shape[0])[:, None] @ coef - s).max() / scale)
        if err < best[1]:
            best = (j, err, z, coef)
        if err < TARGET:
            return j, err, z, coef
    return -1, best[1], best[2], best[3]


def kernel_quadrant(n: int, omega: complex) -> NDArray:
    """A[m-1, n', c] = P(m p, n' p, 0), m = 1..n-1, n' = 0..n-1, c the 81 entries.

    Args:
        n: Lattice points per axis.
        omega: Complex angular frequency.

    Returns:
        Shape (n-1, n, 81).
    """
    return np.array(
        [
            [exact_propagator_9x9(m * PITCH, k * PITCH, 0.0, omega, REF).ravel() for k in range(n)]
            for m in range(1, n)
        ]
    )


def controls() -> None:
    """The pencil recovers a known sum exactly and does not compress noise."""
    rng = np.random.default_rng(1)
    m = np.arange(40)
    z_true = np.exp(-0.05 * np.arange(1, 6)) * np.exp(1j * np.linspace(0.3, 2.5, 5))
    s = (z_true[None, :] ** m[:, None]) @ rng.standard_normal((5, 3))
    j, err, _, _ = fewest_terms(s, float(np.abs(s).max()))
    print(f"  control, 5-term synthetic:  J = {j}  (error {err:.1e})")
    noise = rng.standard_normal((40, 3)) + 1j * rng.standard_normal((40, 3))
    j, err, _, _ = fewest_terms(noise, float(np.abs(noise).max()))
    print(f"  control, random data:       J = {j}  (best error {err:.1e}; -1 = never reached)")


def measure(freq: float) -> None:
    """Terms needed per axis at one frequency, for growing lattices.

    Args:
        freq: Frequency, Hz.
    """
    omega = 2 * np.pi * freq * (1.0 + 1j * DAMPING)
    ks_p = (2 * np.pi * freq / REF.beta) * PITCH
    print(f"\n  f = {freq} Hz: k_S p = {ks_p:.2f}  ({2 * np.pi / ks_p:.1f} voxels per S wavelength)")
    print(f"  {'N':>4} {'J (x)':>6} {'err':>9} {'L (y)':>6} {'err':>9} {'J*L':>6} {'end-to-end':>11}")
    for n in (16, 32, 64):
        a = kernel_quadrant(n, omega)
        scale = float(np.abs(a).max())
        jx, ex, zx, cx = fewest_terms(a.reshape(n - 1, -1), scale)
        if jx < 0:
            print(f"  {n:4d} {'>' + str((n - 1) // 2):>6} {ex:9.1e}   x-axis does not compress")
            continue
        b = cx.reshape(jx, n, 81).transpose(1, 0, 2).reshape(n, -1)  # (n', j*c)
        ly, ey, wy, cy = fewest_terms(b, scale)
        if ly < 0:
            print(f"  {n:4d} {jx:6d} {ex:9.1e} {'>' + str(n // 2):>6} {ey:9.1e}   y-axis does not compress")
            continue
        rec_b = (wy[None, :] ** np.arange(n)[:, None]) @ cy
        rec = np.einsum("mj,njc->mnc", zx[None, :] ** np.arange(n - 1)[:, None], rec_b.reshape(n, jx, 81))
        e2e = float(np.abs(rec - a).max() / scale)
        print(f"  {n:4d} {jx:6d} {ex:9.1e} {ly:6d} {ey:9.1e} {jx * ly:6d} {e2e:11.1e}")


def main() -> int:
    """Run the controls and the measurement.

    Returns:
        0.
    """
    print("=" * 78)
    print("SUM-OF-EXPONENTIALS RANK OF THE EQUAL-DEPTH LATTICE PROPAGATOR")
    print(f"  target {TARGET:.0e} of max|P|")
    print(f"  uniform rule: {UNIFORM_NODES[1.3e-6]:,} nodes per voxel at 1.3e-6")
    print("=" * 78)
    controls()
    for freq in (6.0, 2.0):
        measure(freq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
