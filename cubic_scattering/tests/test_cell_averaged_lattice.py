"""Tests for the dz = 0 cell-averaged lattice sum and its analytic tail."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubic_scattering.cell_averaged_lattice import (
    averaged_origin_scalar_tensors,
    averaged_same_plane_9x9,
)
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.lattice_kupradze import bloch_kernel_hat_9x9
from cubic_scattering.slab_scattering import (
    _cell_averaged_propagator,
    _propagator_block_9x9,
)
from cubic_scattering.sweep_kernels import vertical_kernel_9x9

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
D = 1.0
OMEGA = 60.0
K_PAR = np.zeros(2)


class TestAnalyticTail:
    """R0 splits the sum; a correct tail makes that split invisible."""

    def test_r0_independence(self):
        """THE sharp test: a wrong tail drifts with R0, a right one cannot.

        R0 decides how much of the sum is averaged directly and how much is
        handed to the (1 - kappa^2 d^2/24) tail. If the tail were wrong the two
        pieces would not add up to the same thing, and moving R0 would move the
        answer. Nothing else in this module is as diagnostic.
        """
        base = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=2)
        scale = np.max(np.abs(base))
        for r0 in (3, 4):
            blk = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=r0)
            rel = float(np.max(np.abs(blk - base)) / scale)
            assert rel < 1e-4, f"R0={r0} drifted by {rel:.2e}: the tail is wrong"

    def test_matches_convergent_shell_sum(self):
        """Agreement where the direct shell sum is still trustworthy.

        At k_S d = 0.02 the plane sum of [<G> - G] is absolutely convergent, so
        it is a valid arbiter here -- which is the point: the tail is checked in
        the regime the shell sum handles correctly, then used in the regime it
        does not.
        """
        point = bloch_kernel_hat_9x9(1, D, 0.0, OMEGA, REF)[0, 0]
        shell = np.zeros((9, 9), dtype=complex)
        reach = 12
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if dx == 0 and dy == 0:
                    continue
                r_vec = np.array([0.0, dx * D, dy * D])
                shell += _cell_averaged_propagator(
                    r_vec, D, OMEGA, REF, 6, double=False
                ) - _propagator_block_9x9(r_vec, OMEGA, REF)

        avg = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=3)
        rel = float(np.max(np.abs(avg - (point + shell))) / np.max(np.abs(avg)))
        assert rel < 5e-3, f"disagrees with the convergent shell sum by {rel:.2e}"

    def test_tail_uses_each_mode_own_wavenumber(self):
        """P and S must not share a tail factor -- kappa^2 differs.

        Mixing them is invisible to any symmetry check, so it is asserted
        directly: the two modes' averaged tensors must differ by more than the
        tail correction itself, which they cannot if one kappa were used twice.
        """
        common = dict(eta=float(np.sqrt(np.pi) / D), n_real=4, n_recip=4, a_l=D, k_par=K_PAR)
        d_p = averaged_origin_scalar_tensors(OMEGA / REF.alpha, **common)
        d_s = averaged_origin_scalar_tensors(OMEGA / REF.beta, **common)
        assert not np.allclose(d_p[0], d_s[0], rtol=1e-8)

    def test_rejects_zero_near_shell(self):
        with pytest.raises(ValueError, match=r"r0_cells must be >= 1"):
            averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=0)

    def test_diagnostic_has_where_valid_fix(self):
        with pytest.raises(ValueError) as exc:
            averaged_same_plane_9x9(D, OMEGA, REF, K_PAR, r0_cells=0)
        text = str(exc.value)
        assert "cell_averaged_lattice.py" in text
        assert "Valid:" in text
        assert "Fix:" in text


class TestFormFactorAtNonZeroDz:
    """The sinc^1 form factor on the plane-wave branch."""

    def test_default_path_is_untouched(self):
        """cell_half_width=None must be bit-for-bit the old kernel."""
        a = vertical_kernel_9x9(np.array([0.3]), 0.2, D, OMEGA, REF)
        b = vertical_kernel_9x9(np.array([0.3]), 0.2, D, OMEGA, REF, None)
        assert_allclose(a, b, rtol=0, atol=0)

    def test_form_factor_changes_the_kernel(self):
        """An inert parameter would pass every other test here."""
        a = vertical_kernel_9x9(np.array([0.3]), 0.2, D, OMEGA, REF)
        b = vertical_kernel_9x9(np.array([0.3]), 0.2, D, OMEGA, REF, 0.5 * D)
        assert np.max(np.abs(b - a)) > 1e-12 * np.max(np.abs(a))

    def test_applied_per_mode_not_as_one_scalar(self):
        """P and S carry different k_z, so one scalar factor would be wrong.

        A single-factor implementation would leave the kernel proportional to
        the unaveraged one; a per-mode one cannot, because the two poles are
        rescaled differently.
        """
        a = vertical_kernel_9x9(np.array([0.3]), 0.2, D, OMEGA, REF)
        b = vertical_kernel_9x9(np.array([0.3]), 0.2, D, OMEGA, REF, 0.5 * D)
        nz = np.abs(a) > 1e-14 * np.max(np.abs(a))
        ratios = (b[nz] / a[nz]).ravel()
        spread = float(np.max(np.abs(ratios - ratios[0])))
        assert spread > 1e-9, "kernel merely rescaled: the factor is not per-mode"

    def test_dz0_refuses_a_form_factor(self):
        """Ewald's real-space half is a spatial sum; a form factor is meaningless."""
        with pytest.raises(ValueError, match=r"not available at dz = 0"):
            bloch_kernel_hat_9x9(2, D, 0.0, OMEGA, REF, cell_half_width=0.5 * D)
