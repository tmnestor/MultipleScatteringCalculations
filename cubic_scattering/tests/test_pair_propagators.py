"""Tests for the real-space pair-propagator tables.

These tables replace the spectral sweep for same-depth coupling. The claim is an
exact identity -- the table entry IS the closed-form propagator at that
separation -- so the target is 1e-14 and a residual above it is a defect, never
a discretisation difference.

The self-term is deliberately poisoned with NaN rather than zeroed. A zero would
let a partition bug contribute silently; NaN makes it fail where it happens.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.horizontal_greens import exact_propagator_9x9
from cubic_scattering.pair_propagators import same_depth_table

# Seismic units: km/s, g/cm3. The damping is not cosmetic -- a real omega puts
# the 1/k branch points on the integration contour for anything spectral, and
# the convention is kept here so the tables compose with those objects.
REF = ReferenceMedium(5.0, 3.0, 2.5)
OM = 2 * np.pi * 6.0 * (1 + 0.03j)
PITCH = 0.25


def test_same_depth_table_matches_the_closed_form() -> None:
    """Every non-self entry equals exact_propagator_9x9 at that separation.

    Note the argument order: exact_propagator_9x9 takes CARTESIAN (x, y, z),
    while the state vector is ordered (z, x, y). Reversing it gives a plausible
    wrong answer rather than an error, so the test spans asymmetric (dx, dy) --
    a symmetric set would pass under the swap.
    """
    n_x, n_y = 4, 5
    tab = same_depth_table(n_x, n_y, PITCH, OM, REF)
    assert tab.shape == (2 * n_x - 1, 2 * n_y - 1, 9, 9)

    checked = 0
    for dx in range(-(n_x - 1), n_x):
        for dy in range(-(n_y - 1), n_y):
            if dx == 0 and dy == 0:
                continue
            got = tab[dx + n_x - 1, dy + n_y - 1]
            want = exact_propagator_9x9(dx * PITCH, dy * PITCH, 0.0, OM, REF)
            err = np.abs(got - want).max() / np.abs(want).max()
            assert err < 1e-14, f"(dx,dy)=({dx},{dy}) off by {err:.3e}"
            checked += 1
    assert checked == (2 * n_x - 1) * (2 * n_y - 1) - 1


def test_same_depth_self_term_is_poisoned_not_zero() -> None:
    """The self entry must be NaN, so a partition bug fails loudly."""
    n_x, n_y = 4, 5
    tab = same_depth_table(n_x, n_y, PITCH, OM, REF)
    self_entry = tab[n_x - 1, n_y - 1]
    assert np.isnan(self_entry).all(), "self-term must be NaN, never zero"
    # And nothing else is NaN: a table that is NaN everywhere would pass the
    # assertion above while being useless.
    others = np.delete(tab.reshape(-1, 9, 9), (n_x - 1) * (2 * n_y - 1) + (n_y - 1), axis=0)
    assert np.isfinite(others).all()


def test_same_depth_table_obeys_x_parity() -> None:
    """P(-dx, dy) = (R9 (x) R9) * P(dx, dy), reflecting x ONLY.

    R9 is the sign vector of each state component under x -> -x, applied as an
    elementwise np.outer -- it is not a matrix, and it is not full parity. Only
    dx is reversed here; dy is held, which is what makes this a test of the
    closed form rather than of a symmetry the table was built to have.

    Each entry is evaluated independently from the closed form, so this checks
    the closed form's consistency across the table, not the table's bookkeeping.
    """
    from cubic_scattering.sweep_kernels import R9

    refl = np.outer(R9, R9)
    n_x, n_y = 3, 3
    tab = same_depth_table(n_x, n_y, PITCH, OM, REF)
    for dx, dy in ((1, 0), (2, 1), (1, -2), (2, 2)):
        fwd = tab[dx + n_x - 1, dy + n_y - 1]
        rev = tab[-dx + n_x - 1, dy + n_y - 1]
        err = np.abs(rev - refl * fwd).max() / np.abs(fwd).max()
        assert err < 1e-13, f"x-parity failed at ({dx},{dy}): {err:.3e}"


@pytest.mark.parametrize(("n_x", "n_y"), [(1, 4), (4, 1), (0, 4)])
def test_same_depth_table_rejects_a_lattice_too_small(n_x: int, n_y: int) -> None:
    """Fail fast, with the four-element diagnostic the project requires."""
    with pytest.raises(ValueError) as exc:
        same_depth_table(n_x, n_y, PITCH, OM, REF)
    msg = str(exc.value)
    assert "Where:" in msg
    assert "Valid:" in msg
    assert "Fix:" in msg
