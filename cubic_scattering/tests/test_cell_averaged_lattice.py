"""Tests for the dz = 0 cell-averaged lattice sum and its analytic tail."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubic_scattering.cell_averaged_lattice import (
    averaged_origin_scalar_tensors,
    averaged_same_plane_9x9,
    cube_self_9x9,
    exact_same_plane_9x9,
    plate_average_9x9,
)
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.lattice_kupradze import bloch_kernel_hat_9x9
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    _cell_averaged_propagator,
    _propagator_block_9x9,
    build_slab_kernels,
)
from cubic_scattering.sweep_kernels import vertical_kernel_9x9

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
D = 1.0
OMEGA = 60.0
K_PAR = np.zeros(2)
# k_par = 0 takes the exact closed form (the tiling identity), so the Ewald + near-shell + tail route
# is exercised at a tiny OBLIQUE Bloch point, which it still serves.
K_TAIL = np.array([1e-6, 0.0])


class TestAnalyticTail:
    """R0 splits the sum; a correct tail makes that split invisible."""

    def test_r0_independence(self):
        """THE sharp test: a wrong tail drifts with R0, a right one cannot.

        R0 decides how much of the sum is averaged directly and how much is
        handed to the (1 - kappa^2 d^2/24) tail. If the tail were wrong the two
        pieces would not add up to the same thing, and moving R0 would move the
        answer. Nothing else in this module is as diagnostic.
        """
        base = averaged_same_plane_9x9(D, OMEGA, REF, K_TAIL, r0_cells=2)
        scale = np.max(np.abs(base))
        for r0 in (3, 4):
            blk = averaged_same_plane_9x9(D, OMEGA, REF, K_TAIL, r0_cells=r0)
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


class TestExactAtNormalIncidence:
    """k_par = 0: the tiling identity, V S(0) = V P0 - self, with no lattice sum."""

    EXACT = Path(__file__).resolve().parents[2] / "Mathematica" / "ContinuumLimit_tiling_exact.json"

    def test_matches_the_analytic_static_value(self):
        """Against Mathematica's exact static sum (analytic face antiderivatives, no quadrature)."""
        ref = json.loads(self.EXACT.read_text())
        med = ref["medium"]
        medium = ReferenceMedium(alpha=med["alpha"], beta=med["beta"], rho=med["rho"])
        mu = medium.rho * medium.beta**2
        exact = np.array(ref["eM_times_mu"]) / mu
        near_static = exact_same_plane_9x9(1.0, 0.6, medium)  # k_S d = 2e-4
        rel = np.max(np.abs(np.real(near_static[3:, 3:]) - exact)) / np.max(np.abs(exact))
        assert rel < 1e-7, f"static strain-from-moment block off the analytic value by {rel:.2e}"

    def test_is_what_the_kernel_returns_at_normal_incidence(self):
        """averaged_same_plane_9x9 dispatches k_par = 0 to the closed form, bit for bit."""
        a = averaged_same_plane_9x9(D, OMEGA, REF, K_PAR)
        b = exact_same_plane_9x9(D, OMEGA, REF)
        assert_allclose(a, b, rtol=0, atol=0)

    def test_is_plate_minus_self(self):
        """The identity itself: all cells (the plate) minus the cube's own."""
        assert_allclose(
            exact_same_plane_9x9(D, OMEGA, REF),
            plate_average_9x9(D, OMEGA, REF) - cube_self_9x9(D, OMEGA, REF) / D**3,
            rtol=0,
            atol=0,
        )

    def test_odd_blocks_vanish(self):
        """u <- M and strain <- f are odd in z and in-plane; the cube and the plate are even."""
        s = exact_same_plane_9x9(D, OMEGA, REF)
        assert np.max(np.abs(s[:3, 3:])) == 0.0
        assert np.max(np.abs(s[3:, :3])) == 0.0

    def test_plate_delta_weights_are_the_static_depolarisation(self):
        """The static plate: e_zz <- M_zz = -1/M_P, 2e_zx <- M_zx = -1/(2 mu), per unit volume."""
        p = plate_average_9x9(D, 1e-3, REF) * D**3
        m_p, mu = REF.rho * REF.alpha**2, REF.rho * REF.beta**2
        assert abs(p[3, 3] + 1.0 / m_p) < 1e-6 / m_p
        assert abs(p[8, 8] + 1.0 / (2.0 * mu)) < 1e-6 / mu

    def test_continuous_with_the_oblique_route(self):
        """At a tiny oblique k_par the tail route must agree, to its own measured accuracy.

        The tail route's static strain-from-moment error at its defaults is 9.4e-4 (the reason for
        this closed form); u <- f it gets to ~1e-6.
        """
        exact = exact_same_plane_9x9(D, OMEGA, REF)
        tail = averaged_same_plane_9x9(D, OMEGA, REF, K_TAIL)
        rel_em = np.max(np.abs(tail[3:, 3:] - exact[3:, 3:])) / np.max(np.abs(exact[3:, 3:]))
        rel_uf = np.max(np.abs(tail[:3, :3] - exact[:3, :3])) / np.max(np.abs(exact[:3, :3]))
        assert 1e-4 < rel_em < 2e-3, f"strain-from-moment {rel_em:.2e}: expected the tail route's ~9e-4"
        assert rel_uf < 1e-5, f"u <- f {rel_uf:.2e}"

    def test_attenuative_medium_tends_to_the_lossless_one(self):
        """Complex velocities are accepted, and Q -> infinity recovers the real medium."""
        q = 1e8
        lossy = ReferenceMedium(
            alpha=REF.alpha * (1 - 0.5j / q), beta=REF.beta * (1 - 0.5j / q), rho=REF.rho
        )
        a = exact_same_plane_9x9(D, OMEGA, lossy)
        b = exact_same_plane_9x9(D, OMEGA, REF)
        assert np.all(np.isfinite(a))
        assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-6

    def test_wired_into_the_slab_kernel(self):
        """build_slab_kernels' same-plane block at the Bloch origin is the closed form."""
        geom = SlabGeometry(M=1, N_z=2, a=D / 2)
        kh = build_slab_kernels(
            geom,
            OMEGA,
            REF,
            periodic=True,
            lattice_ewald=True,
            volume_averaged=True,
            exact_cell_average=True,
        )
        assert_allclose(kh[1, 0, 0], exact_same_plane_9x9(D, OMEGA, REF), rtol=1e-13, atol=0)


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


class TestExactCellAverageWiring:
    """The `exact_cell_average` route through build_slab_kernels."""

    EWALD_KW = {"periodic": True, "lattice_ewald": True, "volume_averaged": True}

    def _geom(self):
        return SlabGeometry(M=2, N_z=2, a=0.5)

    def test_auto_turns_it_on_where_it_exists(self):
        """The DEFAULT is now the exact route on the Bloch path.

        `None` means auto: on wherever the construction is available. It cannot
        be unconditionally True, since it is built at the M^2 Bloch points and
        has no meaning for a truncated real-space kernel or a finite slab.
        """
        g = self._geom()
        auto = build_slab_kernels(g, OMEGA, REF, **self.EWALD_KW)
        on = build_slab_kernels(g, OMEGA, REF, **self.EWALD_KW, exact_cell_average=True)
        assert_allclose(auto, on, rtol=0, atol=0)

    def test_explicit_false_still_gives_the_old_kernel(self):
        """The previous behaviour stays reachable, and genuinely differs."""
        g = self._geom()
        auto = build_slab_kernels(g, OMEGA, REF, **self.EWALD_KW)
        off = build_slab_kernels(g, OMEGA, REF, **self.EWALD_KW, exact_cell_average=False)
        assert np.max(np.abs(auto - off)) > 1e-12 * np.max(np.abs(auto))

    def test_auto_stays_off_without_the_bloch_route(self):
        """Auto must not switch on where the construction does not exist."""
        g = self._geom()
        a = build_slab_kernels(g, OMEGA, REF, periodic=True, volume_averaged=True)
        b = build_slab_kernels(g, OMEGA, REF, periodic=True, volume_averaged=True, exact_cell_average=False)
        assert_allclose(a, b, rtol=0, atol=0)

    def test_auto_defers_to_an_explicit_va_all(self):
        """An explicit request for the old route is honoured, not overridden.

        Auto must not silently replace a choice the caller made, and must not
        raise for it either -- combining them is an error only when BOTH are
        asked for explicitly.
        """
        g = self._geom()
        a = build_slab_kernels(g, OMEGA, REF, **self.EWALD_KW, va_all=True)
        b = build_slab_kernels(g, OMEGA, REF, **self.EWALD_KW, va_all=True, exact_cell_average=False)
        assert_allclose(a, b, rtol=0, atol=0)

    def test_requires_the_ewald_bloch_route(self):
        with pytest.raises(ValueError, match=r"requires lattice_ewald=True"):
            build_slab_kernels(self._geom(), OMEGA, REF, periodic=True, exact_cell_average=True)

    def test_refuses_to_double_count_with_va_all(self):
        """Both routes compute the SAME correction; together it applies twice.

        That failure is silent -- a plausible wrong number, not an error -- so
        it is refused rather than merely documented.
        """
        with pytest.raises(ValueError, match=r"two routes to the SAME"):
            build_slab_kernels(
                self._geom(),
                OMEGA,
                REF,
                **self.EWALD_KW,
                va_all=True,
                exact_cell_average=True,
            )
