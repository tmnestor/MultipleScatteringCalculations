"""Tests for slab voxel-refinement convergence: periodic R_PP and Kennett reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_matrix,
    kennett_reference_rpp,
    slab_reflection_matrix,
    slab_rpp_periodic,
    uniform_slab_material,
)

# ── Shared fixtures ──────────────────────────────────────────────

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
WEAK_CONTRAST = MaterialContrast(
    Dlambda=CONTRAST.Dlambda * 1e-4,
    Dmu=CONTRAST.Dmu * 1e-4,
    Drho=CONTRAST.Drho * 1e-4,
)
A = 1.0
OMEGA = 0.05 * REF.beta / A  # ka_S = 0.05 → ω = 150 rad/s
K_HAT = np.array([1.0, 0.0, 0.0])


# ── 1. test_rpp_periodic_zero_contrast ───────────────────────────


def test_rpp_periodic_zero_contrast():
    """R_PP = 0 for zero-contrast slab."""
    geom = SlabGeometry(M=4, N_z=1, a=A)
    zero = MaterialContrast(0.0, 0.0, 0.0)
    mat = uniform_slab_material(geom, REF, zero)
    result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)
    R_PP = slab_rpp_periodic(result, T_local)
    assert abs(R_PP) < 1e-20


# ── 2. test_rpp_periodic_born_scaling ────────────────────────────


def test_rpp_periodic_born_scaling():
    """Doubling weak contrast approximately doubles |R_PP|."""
    geom = SlabGeometry(M=4, N_z=1, a=A)

    mat1 = uniform_slab_material(geom, REF, WEAK_CONTRAST)
    res1 = compute_slab_scattering(geom, mat1, OMEGA, K_HAT)
    T1 = compute_slab_tmatrices(geom, mat1, OMEGA)
    R1 = slab_rpp_periodic(res1, T1)

    double = MaterialContrast(
        Dlambda=2 * WEAK_CONTRAST.Dlambda,
        Dmu=2 * WEAK_CONTRAST.Dmu,
        Drho=2 * WEAK_CONTRAST.Drho,
    )
    mat2 = uniform_slab_material(geom, REF, double)
    res2 = compute_slab_scattering(geom, mat2, OMEGA, K_HAT)
    T2 = compute_slab_tmatrices(geom, mat2, OMEGA)
    R2 = slab_rpp_periodic(res2, T2)

    if abs(R1) > 1e-30:
        ratio = abs(R2) / abs(R1)
        assert_allclose(ratio, 2.0, rtol=0.05)


# ── 3. test_rpp_periodic_vs_existing ─────────────────────────────


def test_rpp_periodic_weak_matches_kennett():
    """Weak-contrast multi-layer FL R_PP matches Kennett within 5%."""
    geom = SlabGeometry(M=4, N_z=2, a=A)
    mat = uniform_slab_material(geom, REF, WEAK_CONTRAST)
    result = compute_slab_scattering(geom, mat, OMEGA, K_HAT)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)

    R_FL = slab_rpp_periodic(result, T_local)
    R_K = kennett_reference_rpp(REF, WEAK_CONTRAST, H=geom.d * geom.N_z, omega=OMEGA)

    rel_err = abs(R_FL - R_K) / abs(R_K)
    assert rel_err < 0.05, f"Relative error {rel_err:.4f} exceeds 5%"


# ── 4. test_kennett_reference_zero_contrast ──────────────────────


def test_kennett_reference_zero_contrast():
    """Kennett R_PP = 0 for identical layers."""
    zero = MaterialContrast(0.0, 0.0, 0.0)
    R = kennett_reference_rpp(REF, zero, H=2.0, omega=OMEGA)
    assert abs(R) < 1e-12


# ── 5. test_kennett_reference_impedance ──────────────────────────


def test_kennett_reference_impedance():
    """Kennett R_PP scales linearly with weak contrast (Born regime)."""
    R1 = kennett_reference_rpp(REF, WEAK_CONTRAST, H=2.0, omega=OMEGA)
    assert abs(R1) > 0, "R_PP should be nonzero for nonzero contrast"

    double = MaterialContrast(
        Dlambda=2 * WEAK_CONTRAST.Dlambda,
        Dmu=2 * WEAK_CONTRAST.Dmu,
        Drho=2 * WEAK_CONTRAST.Drho,
    )
    R2 = kennett_reference_rpp(REF, double, H=2.0, omega=OMEGA)

    ratio = abs(R2) / abs(R1)
    assert_allclose(ratio, 2.0, rtol=0.02)


# ── 6. test_single_layer_weak_matches_kennett ────────────────────


def test_single_layer_weak_matches_kennett():
    """FL R_PP within 15% of Kennett at ka_S=0.05, M=8."""
    a = 1.0
    geom = SlabGeometry(M=8, N_z=1, a=a)
    mat = uniform_slab_material(geom, REF, CONTRAST)
    result = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)
    R_FL = slab_rpp_periodic(result, T_local)

    H = geom.d  # single layer thickness
    R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)

    rel_err = abs(R_FL - R_K) / abs(R_K)
    assert rel_err < 0.15, f"Relative error {rel_err:.3f} exceeds 15%"


# ── 7. test_convergence_error_decreases ──────────────────────────


def test_convergence_error_decreases():
    """Error monotonically decreases over 3 single-layer refinement levels."""
    a_values = [2.0, 1.0, 0.5]
    M_values = [4, 8, 16]
    errors: list[float] = []

    for a, M in zip(a_values, M_values, strict=True):
        geom = SlabGeometry(M=M, N_z=1, a=a)
        mat = uniform_slab_material(geom, REF, CONTRAST)
        result = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
        T_local = compute_slab_tmatrices(geom, mat, OMEGA)
        R_FL = slab_rpp_periodic(result, T_local)

        H = geom.d
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)

        rel_err = abs(R_FL - R_K) / abs(R_K)
        errors.append(rel_err)

    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], (
            f"Error did not decrease: a={a_values[i]}→{a_values[i + 1]}, "
            f"err={errors[i]:.4e}→{errors[i + 1]:.4e}"
        )


# ── 8. test_periodic_uniform_slab_closer_to_kennett ──────────


def test_periodic_uniform_slab_closer_to_kennett():
    """Periodic mode gives ≤ aperiodic error vs Kennett at M=4."""
    geom = SlabGeometry(M=4, N_z=1, a=A)
    mat = uniform_slab_material(geom, REF, CONTRAST)
    H = geom.d
    R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=OMEGA)

    # Aperiodic (default)
    res_ap = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8)
    T_local = compute_slab_tmatrices(geom, mat, OMEGA)
    R_ap = slab_rpp_periodic(res_ap, T_local)
    err_ap = abs(R_ap - R_K) / abs(R_K)

    # Periodic
    res_p = compute_slab_scattering(geom, mat, OMEGA, K_HAT, gmres_tol=1e-8, periodic=True)
    T_local_p = compute_slab_tmatrices(geom, mat, OMEGA)
    R_p = slab_rpp_periodic(res_p, T_local_p)
    err_p = abs(R_p - R_K) / abs(R_K)

    assert err_p <= err_ap + 1e-10, f"Periodic error {err_p:.4e} should be ≤ aperiodic error {err_ap:.4e}"


# ═══════════════════════════════════════════════════════════════
#  The averaged-propagator slab against the exact layer (Kennett)
# ═══════════════════════════════════════════════════════════════
#
# A uniform slab of N_z = 3 planes of cubes (a = 1 m, so d = 2 m and H = 6 m), periodic, on the exact
# lattice sum, at k_S a = 0.1, 0.2, 0.3 and 0.5 (k_S d = 0.2 to 1.0). The single site is the one
# numerics.yml names (single_site.lattice: exact, the uniform-strain closure, since 31 October 2026).
#
# ONE CUBE TO THE LATERAL PERIOD (M = 1). The kernels are built at the incident wave's lateral
# wavevector, so a uniform slab is the same on any number of cubes across: M = 1 gives M = 4 to 2e-16 in
# a fourteenth of the time (2.6 s against 37.5 s at oblique incidence). That is what lets this class run
# in the gate; while it was marked slow it failed unseen for eleven days.
#
# THE LAW. With the exact closure and the source-cell average of the propagator the scheme is the
# collocation of the continuum equation (Error paper, Eq. 15), so at first order in the contrast its
# reflection is a midpoint rule on the Born integral, whose integrand carries the two-way phase
# exp(2 i k z): a factor 1 / sinc(2 k h). Read with the SYMMETRIC readout (slab_weyl_amplitudes), the
# cell average of the wave, a factor sinc(k h), makes it the paper's Eq. 16,
#
#     R_disc / R = sinc(k h) / sinc(2 k h) = 1 + (k_P d)^2 / 8 + ...  = 1 + 0.045 (k_S d)^2 + ...
#
# for this medium (k = k_P = 0.6 k_S, h = d / 2). The FIRST-ORDER readout, the configured one
# (numerics.yml, slab_readout.exact_site), multiplies the point sum by sinc(2 k h) instead, which
# integrates the first-order term exactly over every cube and leaves the second order in the contrast.
# Measured 11 November 2026 (env seismic; scripts/measure_slab_collocation_law.py), error relative to
# Kennett:
#
#           symmetric readout                                        first-order readout
#   k_S a   law        weak contrast (1e-2)   full contrast          full contrast   package T, full
#   0.1     1.8027e-3  1.8026e-3  (0.9999)    1.7925e-3  (0.994)     1.12e-5         2.72e-4
#   0.2     7.2435e-3  7.2428e-3  (0.9999)    7.1918e-3  (0.993)     6.19e-5         1.12e-3
#   0.3     1.6422e-2  1.6420e-2  (0.9999)    1.6269e-2  (0.991)     1.82e-4         2.61e-3
#   0.5     4.6752e-2  4.6745e-2  (0.9999)    4.6052e-2  (0.985)     7.66e-4         8.11e-3
#
# (in brackets, the error over the law). The symmetric reading's departure from the law is the second
# order in the contrast: 1e-4 at a hundredth of the contrast, 0.6 to 1.5% at the full one, as on the
# paper's layer (0.124 against 1/8, Section 6). The first-order reading keeps only that part: 60 to 160
# times smaller, and 10 to 24 times smaller than the package T-matrix's error (last column:
# single_site.lattice: package, the same kernels, read as points since it carries a form factor of its
# own), which was the more accurate until the law was divided out.
#
# THREE DEFECTS stood between this slab and the law until 11 November 2026; each is pinned below or in
# test_slab_scattering.py and test_cell_averaged_lattice.py, and the figures above are after all three.
#   1. The readout summed point sources: the error was 1 / sinc(k_P d) - 1 = (k_P d)^2 / 6 + ...
#   2. The same-plane kernel at k_par = 0 removed the package T-matrix's self term, whose real part is
#      static, in place of the cube's integral (cell_averaged_lattice.cube_self_9x9). It put the
#      full-contrast error 4.6 to 6.5% below the law where it is now 0.6 to 1.5%.
#   3. Oblique incidence only: the specular average left the incident wave's lateral phase in, and the
#      periodic kernels were built at the lattice's own Bloch points whatever the wave (see the oblique
#      tests).
#
# History, for anyone reading an older number: before 15 September 2026 the class measured the truncated
# lateral sum (errors of 1 to 2%, mostly that artefact); from 15 September the exact sum with the
# single-average contact operator and the package T-matrix (4.2e-4 to 7.8e-3). Neither set is current.
# A test that n_orders = 3 was no worse than n_orders = 2 was removed on 11 November 2026: the exact
# cell-average kernels do not use n_orders, so it compared a computation with itself.

_KA_KENNETT = (0.1, 0.2, 0.3, 0.5)
# A hundredth of CONTRAST: weak enough that the first-order law holds to 1e-4 of the error.
_BORN_CONTRAST = MaterialContrast(
    Dlambda=CONTRAST.Dlambda * 1e-2,
    Dmu=CONTRAST.Dmu * 1e-2,
    Drho=CONTRAST.Drho * 1e-2,
)
# The first-order readout's error at the full contrast, measured (see the table above), and its bound.
_FIRST_ORDER_FULL = {0.1: 1.12e-5, 0.2: 6.19e-5, 0.3: 1.82e-4, 0.5: 7.66e-4}


def _collocation_law(ka: float) -> float:
    """The collocation slab's relative reflection error at first order: sinc(kh) / sinc(2kh) - 1.

    The Error paper's Eq. 16 with k = k_P. ``ka`` is k_S a with a the cube half-width h, so
    k_P d = 2 ka beta / alpha. See the note above.
    """
    k_p_d = 2.0 * ka * REF.beta / REF.alpha
    return float(np.sinc(0.5 * k_p_d / np.pi) / np.sinc(k_p_d / np.pi) - 1.0)


def _slab_kernels(geom, omega, *, volume_averaged, bloch_k=(0.0, 0.0)):
    """The periodic slab kernels on the exact lattice sum.

    ``lattice_ewald=True`` is essential. Without it the lateral sum is truncated to dx, dy in
    [-(M-1), M-1] and wraps, an artefact larger than every error this class measures.
    """
    return build_slab_kernels(
        geom,
        omega,
        REF,
        periodic=True,
        volume_averaged=volume_averaged,
        lattice_ewald=True,
        bloch_k=bloch_k,
    )


def _rpp(geom, mat, omega, *, volume_averaged, kernel_hat=None, readout=None):
    """Periodic normal-incidence specular R_PP on the exact lattice sum (see ``_slab_kernels``).

    The kernels do not depend on the contrast: pass ``kernel_hat`` to solve a second medium on them.
    ``readout`` None is the configured one (first_order).
    """
    if kernel_hat is None:
        kernel_hat = _slab_kernels(geom, omega, volume_averaged=volume_averaged)
    res = compute_slab_scattering(
        geom,
        mat,
        omega,
        K_HAT,
        wave_type="P",
        gmres_tol=1e-11,
        volume_averaged=volume_averaged,
        periodic=True,
        kernel_hat=kernel_hat,
    )
    T_local = compute_slab_tmatrices(geom, mat, omega)
    return slab_rpp_periodic(res, T_local, p=0.0, readout=readout)


class TestSlabAgainstTheExactLayer:
    """The lattice of cubes against the exact layer: the collocation law, and what is left without it."""

    M = 1
    N_z = 3
    A = 1.0  # cube half-width; H = N_z * 2a = 6 m

    def _geom_mat(self, contrast=CONTRAST, M=None, N_z=None):
        geom = SlabGeometry(M=self.M if M is None else M, N_z=self.N_z if N_z is None else N_z, a=self.A)
        mat = uniform_slab_material(geom, REF, contrast)
        return geom, mat

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_symmetric_readout_follows_the_collocation_law(self, ka):
        """Read symmetrically, the R_PP error is sinc(kh) / sinc(2kh) - 1, the Error paper's Eq. 16.

        At a hundredth of the contrast the error is the law to 0.03% of itself (measured: 0.01% below
        it). At the full contrast it is 0.6 to 1.5% below the law, the second order in the contrast; the
        band 0.975 to 0.999 holds that, so a change of the single site, of the kernels or of the readout
        fails here in either direction. (The package T-matrix gives 0.15 to 0.17 of the law, the point
        readout 1.33 times it, and the kernel with the package's self term 0.935 to 0.954.)
        """
        geom, mat = self._geom_mat()
        _, mat_born = self._geom_mat(_BORN_CONTRAST)
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        law = _collocation_law(ka)
        kernel_hat = _slab_kernels(geom, omega, volume_averaged=True)

        R_K = kennett_reference_rpp(REF, _BORN_CONTRAST, H=H, omega=omega)
        R = _rpp(geom, mat_born, omega, volume_averaged=True, kernel_hat=kernel_hat, readout="symmetric")
        err_born = abs(R - R_K) / abs(R_K)
        assert abs(err_born / law - 1.0) < 3e-4, (
            f"ka={ka}: at 1e-2 of the contrast the error {err_born:.4e} is {err_born / law:.4f} of the "
            f"law {law:.4e}; it should be the law to 0.03%"
        )

        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R = _rpp(geom, mat, omega, volume_averaged=True, kernel_hat=kernel_hat, readout="symmetric")
        ratio = R / R_K - 1.0
        assert 0.975 < abs(ratio) / law < 0.999, (
            f"ka={ka}: at the full contrast the error {abs(ratio):.4e} is {abs(ratio) / law:.4f} of the "
            f"law {law:.4e}; measured 0.985 to 0.994"
        )
        # the law's sign: the discrete reflection is too large, in phase with the exact one
        assert ratio.real > 0.0 and abs(ratio.imag) < 0.03 * ratio.real

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_first_order_readout_leaves_the_second_order(self, ka):
        """The configured readout integrates the first-order term exactly: what is left is second order.

        It is the point sum times sinc(2 k h), so its error is the symmetric reading's with the law
        divided out. At a hundredth of the contrast that is under 1.5e-5 at every frequency (the law
        itself is 1.8e-3 to 4.7e-2); at the full contrast it is 1.1e-5 to 7.7e-4, bounded here at 1.5
        times its measured size, 60 to 160 times below the symmetric reading and 10 to 24 times below
        the package T-matrix. The two readings must also differ by exactly their closed-form factor.
        """
        geom, mat = self._geom_mat()
        _, mat_born = self._geom_mat(_BORN_CONTRAST)
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        kernel_hat = _slab_kernels(geom, omega, volume_averaged=True)

        R_K = kennett_reference_rpp(REF, _BORN_CONTRAST, H=H, omega=omega)
        R = _rpp(geom, mat_born, omega, volume_averaged=True, kernel_hat=kernel_hat)
        assert abs(R / R_K - 1.0) < 1.5e-5, f"ka={ka}: weak-contrast error {abs(R / R_K - 1.0):.3e}"

        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R = _rpp(geom, mat, omega, volume_averaged=True, kernel_hat=kernel_hat)
        err = abs(R / R_K - 1.0)
        bound = 1.5 * _FIRST_ORDER_FULL[ka]
        assert err < bound, f"ka={ka}: full-contrast error {err:.3e} exceeds {bound:.3e}"
        assert err < _collocation_law(ka) / 40.0, "the first order should be gone from the error"

        R_sym = _rpp(geom, mat, omega, volume_averaged=True, kernel_hat=kernel_hat, readout="symmetric")
        k_p_h = omega / REF.alpha * self.A
        assert_allclose(R / R_sym, np.sinc(2 * k_p_h / np.pi) / np.sinc(k_p_h / np.pi), rtol=1e-12)

    @pytest.mark.parametrize("ka", _KA_KENNETT)
    def test_averaged_propagator_beats_point_at_every_ka(self, ka):
        """The source-cell average of the propagator is more accurate than the point propagator.

        With the exact closure and the first-order readout (measured 11 November 2026, full contrast):

            ka    averaged    point       ratio
            0.1   1.12e-5     4.02e-3     360
            0.2   6.19e-5     4.08e-3     66
            0.3   1.82e-4     4.20e-3     23
            0.5   7.66e-4     4.82e-3     6.3

        Both are of second order in the contrast; the point propagator's is the larger because its
        coupling between touching cells is a midpoint rule on a singular kernel.
        """
        geom, mat = self._geom_mat()
        omega = ka * REF.beta / self.A
        H = geom.d * geom.N_z
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        err_va = abs(_rpp(geom, mat, omega, volume_averaged=True) / R_K - 1.0)
        err_pt = abs(_rpp(geom, mat, omega, volume_averaged=False) / R_K - 1.0)
        assert err_va < 0.25 * err_pt, (
            f"ka={ka}: averaged-propagator error {err_va:.4e} should be well below the point "
            f"propagator's {err_pt:.4e}"
        )

    def _oblique_errors(self, geom, mat, contrast, omega, p, readout, kernel_hat):
        """R_disc / R - 1 for the four entries of the P-SV matrix (modified convention), after a
        reciprocity check of the matrix read."""
        sm = slab_reflection_matrix(
            geom,
            mat,
            omega,
            p=p,
            volume_averaged=True,
            kernel_hat=kernel_hat,
            gmres_tol=1e-11,
            include_sh=False,
            readout=readout,
        )
        R_mod = sm.to_modified()
        kref = kennett_reference_matrix(REF, contrast, H=geom.d * geom.N_z, omega=omega, p=p)
        exact = np.array([[kref.R_PP, kref.R_SP], [kref.R_PS, kref.R_SS]])
        assert abs(R_mod[0, 1] - R_mod[1, 0]) < 1e-8 * abs(R_mod[0, 1]), "the matrix is not reciprocal"
        return R_mod / exact - 1.0

    def test_oblique_matrix_follows_the_law_and_is_reciprocal(self):
        """Oblique incidence below the P critical slowness: the whole P-SV matrix against Kennett.

        At p = 1e-4 s/m and ka = 0.3 (30 degrees for P), with the exact closure. Read symmetrically, the
        entry (out i, in j) has at first order in the contrast the error

            sqrt(c_i c_j) / sinc((k_z,i + k_z,j) h) - 1,     c_m = sinc(k_z,m h) sinc(omega p h),

        the midpoint rule on the phase exp(i (k_z,i + k_z,j) z) read through the cell (the Error paper's
        Eq. 19). Measured 11 November 2026:

            entry        law        symmetric, weak    symmetric, full       first order, full
            R_PP         1.091e-2   1.091e-2           1.087e-2 + 1.2e-4 i   1.27e-4
            R_SS         4.099e-2   4.099e-2           4.096e-2 + 1.9e-4 i   1.80e-4
            conversions  2.282e-2   2.282e-2           2.259e-2 + 8.7e-5 i   2.41e-4

        The symmetric errors are held to the law within 5e-5 at weak contrast and 5e-4 at the full one;
        the first-order reading's to 5e-5 and 4e-4 outright; and each matrix must be symmetric
        (measured: 1e-10). With six planes of half the size the symmetric full-contrast errors are
        2.70e-3, 9.98e-3 and 5.56e-3: second order, ratios 4.03 to 4.10.

        Before 11 November 2026 none of this held. The specular average left the incident wave's lateral
        phase in (a factor 0.980 on four cubes across), and the kernels were built at the lattice's own
        Bloch points, so the wave was wrapped round the lattice with the wrong phase: the full-contrast
        R_SS error read 3.4e-2 + 9.9e-3 i here, and 25% at p = 1.6e-4, where it is now 3.6%.
        """
        geom, mat = self._geom_mat()
        _, mat_born = self._geom_mat(_BORN_CONTRAST)
        omega = 0.3 * REF.beta / self.A
        h = self.A
        p = 1e-4  # sub-critical: p < 1/alpha = 2e-4
        k_z = omega * np.sqrt(np.array([1.0 / REF.alpha**2, 1.0 / REF.beta**2]) - p**2)
        cell = np.sinc(k_z * h / np.pi) * np.sinc(omega * p * h / np.pi)
        law = np.sqrt(np.outer(cell, cell)) / np.sinc(np.add.outer(k_z, k_z) * h / np.pi) - 1.0
        kernel_hat = _slab_kernels(geom, omega, volume_averaged=True, bloch_k=(omega * p, 0.0))

        born = self._oblique_errors(geom, mat_born, _BORN_CONTRAST, omega, p, "symmetric", kernel_hat)
        assert np.max(np.abs(born - law)) < 5e-5, f"weak contrast: errors {born} against the law {law}"
        full = self._oblique_errors(geom, mat, CONTRAST, omega, p, "symmetric", kernel_hat)
        assert np.max(np.abs(full - law)) < 5e-4, f"full contrast: errors {full} against the law {law}"

        born = self._oblique_errors(geom, mat_born, _BORN_CONTRAST, omega, p, None, kernel_hat)
        assert np.max(np.abs(born)) < 5e-5, f"first-order readout, weak contrast: errors {born}"
        full = self._oblique_errors(geom, mat, CONTRAST, omega, p, None, kernel_hat)
        assert np.max(np.abs(full)) < 4e-4, f"first-order readout, full contrast: errors {full}"

    def test_oblique_answer_does_not_depend_on_the_lattice(self):
        """A uniform slab's oblique reflection is the same on one cube across as on two.

        One plane at the full contrast, omega = 1500 rad/s, p = 1.309e-4 s/m: a lattice of sixteen cubes
        holds one lateral wavelength, so one cube holds a sixteenth and two an eighth. With the kernels
        at the lattice's own Bloch points the R_PP error against Kennett moved between 1.4e-2 and 2.0e-2
        with the number of cubes (M = 8, 16) and its imaginary part between -1.1e-2 and +2e-4
        (M = 4, 16); built at the wave's lateral wavevector the lattices agree to round-off (M = 1
        against M = 4: 2e-16).
        """
        omega = 1500.0
        p = 2.0 * np.pi / (omega * 16 * 2.0 * self.A)
        matrices = []
        # the point propagator's exact lattice sum: the wavevector enters every route alike, and this
        # one is the quickest to build at a general wavevector
        for M in (1, 2):
            geom, mat = self._geom_mat(M=M, N_z=1)
            kernel_hat = _slab_kernels(geom, omega, volume_averaged=False, bloch_k=(omega * p, 0.0))
            sm = slab_reflection_matrix(
                geom, mat, omega, p=p, kernel_hat=kernel_hat, gmres_tol=1e-12, include_sh=False
            )
            matrices.append(sm.to_modified())
        assert_allclose(matrices[0], matrices[1], rtol=1e-9)
