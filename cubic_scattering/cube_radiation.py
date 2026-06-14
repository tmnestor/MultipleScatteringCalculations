"""cube_radiation.py

Validated cube radiation primitives.

These are the reusable, validated building blocks for radiating Cartesian-monomial
fields over the cube ``[-a, a]**3``.  A monomial's far-field (Fourier) integral at
``k_sc = k * r_hat`` separates into a product of three 1D integrals

    I_p(k_j) = int_{-a}^{a} t**p exp(-i k_j t) dt,   p in {0, 1, 2}

with removable ``k -> 0`` limits.  ``radiation_I1d`` is that 1D primitive and
``radiation_monomial`` is the 3D product.

The earlier moment-expansion far-field operator approach (analytic per-mode
equivalent-source radiation and its Gauss-quadrature arbiter) was abandoned — Mie
validation showed it gave no advantage over ``scattered_field.cube_far_field`` — so
only these primitives are retained for potential future multipole work.

Coordinate system (CLAUDE.md): z = axis 0 (down), x = axis 1, y = axis 2.
Voigt order: (e11, e22, e33, 2 e23, 2 e13, 2 e12).
"""

from __future__ import annotations

import numpy as np


def radiation_I1d(p: int, k: float, a: float) -> complex:
    """Return the 1D radiation integral int_{-a}^{a} t**p exp(-i k t) dt.

    Closed form for p in {0, 1, 2} with removable k -> 0 limits.

    Args:
        p: Monomial power (0, 1, or 2).
        k: Wavenumber component along this axis (rad/m); may be zero or negative.
        a: Cube half-width (m); cube extends over [-a, a].

    Returns:
        Complex value of the integral.

    Raises:
        ValueError: If p is not in {0, 1, 2}.
    """
    ka = k * a
    abs_ka = abs(ka)

    if p == 0:
        if abs_ka < 1e-8:
            ka2 = ka * ka
            return complex(2.0 * a * (1.0 - ka2 / 6.0 + ka2 * ka2 / 120.0))
        return complex(2.0 * np.sin(ka) / k)

    if p == 1:
        # int t exp(-ikt) dt = -(2i/k**2)(sin(ka) - ka cos(ka))
        if abs_ka < 1e-6:
            ka2 = ka * ka
            return complex(
                -2j * a**3 * k / 3.0 * (1.0 - ka2 / 10.0 + ka2 * ka2 / 280.0)
            )
        return complex(-2j / k**2 * (np.sin(ka) - ka * np.cos(ka)))

    if p == 2:
        # int t**2 exp(-ikt) dt = (2/k**3)(2 ka cos(ka) + (k**2 a**2 - 2) sin(ka))
        if abs_ka < 1e-5:
            ka2 = ka * ka
            return complex(
                2.0 * a**3 / 3.0 * (1.0 - 3.0 * ka2 / 10.0 + ka2 * ka2 / 56.0)
            )
        return complex(
            2.0 / k**3 * (2.0 * ka * np.cos(ka) + (ka**2 - 2.0) * np.sin(ka))
        )

    msg = f"radiation_I1d: power p={p} not supported (only 0, 1, 2)"
    raise ValueError(msg)


def radiation_monomial(
    exp: tuple[int, int, int], k_sc: np.ndarray, a: float
) -> complex:
    """Radiate a single Cartesian monomial r0**e1 r1**e2 r2**e3 over the cube.

    Computes int_cube r0**e1 r1**e2 r2**e3 exp(-i k_sc . r') d3r' as the product
    of three 1D primitives.  Axis order is (r0, r1, r2) = (z, x, y).

    Args:
        exp: Monomial powers (e1, e2, e3), each in {0, 1, 2}.
        k_sc: Scattered wavevector (3,) = k * r_hat (rad/m).
        a: Cube half-width (m).

    Returns:
        Complex radiation integral.
    """
    e1, e2, e3 = exp
    return (
        radiation_I1d(e1, float(k_sc[0]), a)
        * radiation_I1d(e2, float(k_sc[1]), a)
        * radiation_I1d(e3, float(k_sc[2]), a)
    )
