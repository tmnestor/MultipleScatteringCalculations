"""Special functions with complex arguments, safe against a signed zero.

``scipy.special.spherical_jn`` evaluates j_l(z) for complex z through sqrt(pi / 2z) J_{l+1/2}(z), and
both factors have a branch cut on the negative real axis. j_l is entire, so the cuts cancel
analytically, but numerically a negative real argument whose imaginary part is a NEGATIVE zero,
z = -x - 0j, falls on the other side of both cuts and comes back with the wrong sign:
spherical_jn(0, -0.3 - 0j) = -0.985, where j_0(-0.3) = sin(0.3) / 0.3 = 0.985 (SciPy 1.17.1; +0j and
any nonzero imaginary part are right).

Such arguments arise whenever a real wavenumber held as complex is negated: the source moments'
exp(-i k.x) take -k h, and an up-going mode's vertical wavenumber is sign * k_z. The signed zero then
flipped the sign of the cell moments at about half the quadrature nodes, and the mode factorisation of
Paper 2's blocks (gate G2, ``scripts/gate_march_modes.py``), and with it the matrix-free whole-space
march, was wrong by O(1).

Adding +0j makes the zero positive (IEEE: -0 + 0 = +0) and changes nothing else. Real arguments are
passed through untouched. ``spherical_yn`` has a genuine branch cut on the negative real axis, where the
signed zero is meaningful, and is not wrapped.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import spherical_jn as _spherical_jn

__all__ = ["spherical_jn"]


def spherical_jn(n: ArrayLike, z: ArrayLike, derivative: bool = False) -> NDArray:
    """``scipy.special.spherical_jn``, a complex argument's imaginary zero made positive (see the module).

    Args:
        n: Order(s), non-negative integers.
        z: Argument(s), real or complex.
        derivative: The derivative j_l'(z) instead of j_l(z).

    Returns:
        j_l(z) (or j_l'(z)), as ``scipy.special.spherical_jn``.
    """
    z = np.asarray(z)
    if np.iscomplexobj(z):
        z = z + 0j
    return _spherical_jn(n, z, derivative=derivative)
