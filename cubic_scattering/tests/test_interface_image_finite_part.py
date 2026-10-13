"""Tests for the finite-part potentials in the interface blocks (``interface_image``, order 0 and 1).

Validates:
  1. Each finite-part kernel d^alpha Phi_j is homogeneous up to its log term: f(lam x) = lam^d [f(x) + log(lam) g(x)]
  2. g vanishes for every static (K0-type) family, whose lateral derivatives annihilate the solid harmonic
  3. Corner boxes with a log term: Duffy pyramids (exact rho^e log rho moments) and Euler's identity (with
     -int x^n g) agree, two routes sharing only the kernel
  4. Identical media: the reflected omega^2 block vanishes (slow)
"""

from fractions import Fraction

import numpy as np
import pytest

from cubic_scattering import interface_image as ii
from cubic_scattering.effective_contrasts import ReferenceMedium

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05

#: finite-part families of the omega^2 term (lateral derivatives fewer than j), with and without a log term
FP_FAMILIES = [
    (1, (0, 0, 0)),
    (2, (0, 0, 0)),
    (2, (1, 0, 0)),
    (3, (0, 2, 0)),
    (3, (1, 0, 0)),
    (3, (1, 1, 0)),
    (4, (2, 0, 0)),
    (4, (1, 1, 0)),
]


@pytest.mark.parametrize("fam", FP_FAMILIES)
def test_log_homogeneity(fam):
    j, alpha = fam
    f, g, d = (
        ii.kernel_function(j, alpha),
        ii.kernel_log_coefficient(j, alpha),
        ii.kernel_degree(j, alpha),
    )
    x = (
        np.array([0.3, 1.1, -0.7]),
        np.array([-0.5, 0.2, 0.9]),
        np.array([0.4, 0.05, 1.7]),
    )
    for lam in (0.37, 2.9):
        lhs = f(*(lam * c for c in x))
        rhs = lam**d * (f(*x) + (np.log(lam) * g(*x) if g is not None else 0.0))
        assert np.abs(lhs - rhs).max() < 1e-13 * np.abs(lhs).max()


@pytest.mark.parametrize(
    "fam",
    [(1, (1, 0, 0)), (2, (0, 2, 0)), (2, (1, 1, 0)), (3, (3, 0, 0)), (4, (2, 2, 0))],
)
def test_no_log_term_for_enough_lateral_derivatives(fam):
    assert ii.kernel_log_coefficient(*fam) is None


@pytest.mark.parametrize("fam", FP_FAMILIES)
def test_corner_routes_agree(fam):
    j, alpha = fam
    f, g, d = (
        ii.kernel_function(j, alpha),
        ii.kernel_log_coefficient(j, alpha),
        ii.kernel_degree(j, alpha),
    )
    box = ((0, 2), (-2, 0), (0, 2))
    deg = (2, 2, 2)
    euler = ii._corner_moments(f, d, box, deg, 40, g)
    ws = [
        {m: tuple(Fraction(int(i == m)) for i in range(m + 1)) for m in range(3)}
        for _ in range(3)
    ]
    pairs = [(0, 0, 0), (1, 0, 2), (2, 1, 1), (2, 2, 2)]
    pyr = ii._corner_pyramids(f, d, box, ws, pairs, 40, g)
    for p, v in zip(pairs, pyr, strict=True):
        assert abs(euler[p] - v) < 1e-13 * max(abs(v), np.abs(euler).max())


@pytest.mark.slow
def test_identical_media_reflected_omega2_vanishes():
    k1 = ii.image_block((-1, 0, 0), (-1, 0, 0), H, A, A, order=1)
    scale = np.abs(ii.image_block((-1, 0, 0), (-1, 0, 0), H, A, B, order=1)).max()
    assert np.abs(k1).max() < 1e-14 * scale
