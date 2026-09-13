"""Tests for the directional-sweep kernel algebra.

The claims here are exact identities, not approximations: the same physics
reached through two independent code paths. A residual above the stated target
is a defect, not a discretisation difference, and must never be accommodated by
widening the tolerance.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.horizontal_greens import (
    exact_propagator_9x9,
    post_kx_residue_kernel_9x9_vec,
)
from cubic_scattering.resonance_tmatrix import VOIGT_PAIRS
from cubic_scattering.sweep_kernels import (
    R9,
    LateralSplit,
    lateral_split_9x9,
    vertical_kernel_9x9,
)

# Seismic units: km/s, g/cm3. Slight damping keeps the branch unambiguous.
REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25


def _bundled(ky: float, kz: float, dx: float) -> np.ndarray:
    """The existing bundled kernel at one (ky, kz), as a 9x9."""
    return post_kx_residue_kernel_9x9_vec(
        np.array([ky]),
        kz,
        dx,
        omega=OMEGA,
        rho=REF.rho,
        alpha=REF.alpha,
        beta=REF.beta,
    )[:, :, 0]


@pytest.mark.parametrize("n", [1, 2, 3, 7])
@pytest.mark.parametrize("ky", [0.0, 0.4, 1.7])
def test_split_reproduces_bundled_kernel(ky: float, n: int) -> None:
    """RUNG 1: amp x phase**n == the bundled kernel at separation n*pitch."""
    kz_arr = np.array([0.0, 0.3, 1.1, 2.6, 5.0])
    split = lateral_split_9x9(ky, kz_arr, PITCH, OMEGA, REF, direction="right")

    for m, kz in enumerate(kz_arr):
        got = split.amp_p[:, :, m] * split.phase_p[m] ** n + split.amp_s[:, :, m] * split.phase_s[m] ** n
        want = _bundled(ky, kz, n * PITCH)
        scale = np.abs(want).max()
        # 1e-13, not 1e-15: exp(i kx p)**n and exp(i kx n p) are the same exact
        # quantity evaluated two ways, and differ by a few ULP. The claim that
        # actually matters -- that the amplitude does not accumulate -- is
        # gated by test_residual_does_not_grow_with_separation below, which a
        # loose tolerance here cannot mask.
        assert np.abs(got - want).max() / scale < 1e-13


def test_residual_does_not_grow_with_separation() -> None:
    """RUNG 1, the discriminating half: the residual must be FLAT in separation.

    This is the test that catches the dangerous defect. If any factor that
    should be applied once were instead accumulated along the sweep, the
    residual would grow with n -- linearly for a misplaced polynomial factor,
    exponentially for a misplaced exponential. A reciprocity or symmetry check
    cannot see this, because such checks are homogeneous of degree one.
    """
    ky = 0.4
    kz_arr = np.array([0.0, 0.3, 1.1, 2.6, 5.0])
    split = lateral_split_9x9(ky, kz_arr, PITCH, OMEGA, REF, direction="right")

    residuals = []
    for n in (1, 2, 4, 8, 16, 32):
        worst = 0.0
        for m, kz in enumerate(kz_arr):
            got = (
                split.amp_p[:, :, m] * split.phase_p[m] ** n + split.amp_s[:, :, m] * split.phase_s[m] ** n
            )
            want = _bundled(ky, kz, n * PITCH)
            worst = max(worst, np.abs(got - want).max() / np.abs(want).max())
        residuals.append(worst)

    # Flat, not growing: the residual at 32 pitches is within an order of
    # magnitude of the residual at one. A factor accumulating 32 times over
    # would be nowhere near this.
    assert max(residuals) / min(residuals) < 10.0
    assert max(residuals) < 1e-13


def test_amplitude_is_separation_independent() -> None:
    """The amplitude must NOT depend on separation -- the accumulation trap."""
    ky = 0.4
    kz_arr = np.array([0.0, 1.1, 5.0])
    a = lateral_split_9x9(ky, kz_arr, PITCH, OMEGA, REF, direction="right")
    b = lateral_split_9x9(ky, kz_arr, 4.0 * PITCH, OMEGA, REF, direction="right")
    assert np.abs(a.amp_p - b.amp_p).max() == 0.0
    assert np.abs(a.amp_s - b.amp_s).max() == 0.0
    # ...and the phase must be exactly the pitch-th power relation.
    assert np.abs(b.phase_p - a.phase_p**4).max() < 1e-14


def test_returns_the_declared_type() -> None:
    split = lateral_split_9x9(0.4, np.array([0.0, 1.0]), PITCH, OMEGA, REF)
    assert isinstance(split, LateralSplit)
    assert split.amp_p.shape == (9, 9, 2)
    assert split.phase_s.shape == (2,)
    assert split.direction == "right"


def test_parity_signature_is_derivable_from_the_displacement_parity() -> None:
    """R9 must be the index parity under x -> -x, derived not asserted.

    Rebuilt here from the 3-vector parity and VOIGT_PAIRS, so the test does not
    simply restate the module's hard-coded constant. e_xx carries TWO x indices
    and is therefore EVEN -- the trap is pattern-matching on the letter.
    """
    r3 = np.array([1.0, -1.0, 1.0])  # (z, x, y)
    want = np.concatenate([r3, [r3[p] * r3[q] for (p, q) in VOIGT_PAIRS]])
    np.testing.assert_array_equal(R9, want)
    assert int(np.sum(R9 == -1)) == 3  # u_x, 2e_xy, 2e_zx -- and nothing else
    assert R9[4] == 1.0  # e_xx is EVEN


def test_left_amplitude_is_the_x_reflection_of_the_right() -> None:
    """RUNG 1b: P_left = R9 P_right R9, with R9 the parity under x -> -x.

    Without this, a sign error in the left sweep is invisible: the field stays
    plausible and is simply symmetric in dx where half its components should be
    antisymmetric.
    """
    kz_arr = np.array([0.0, 0.7, 2.2, 6.0])
    ky = 0.9
    right = lateral_split_9x9(ky, kz_arr, PITCH, OMEGA, REF, direction="right")
    left = lateral_split_9x9(ky, kz_arr, PITCH, OMEGA, REF, direction="left")

    # The phase is even in the x-component: same pole, same pitch.
    assert np.abs(left.phase_p - right.phase_p).max() == 0.0
    assert np.abs(left.phase_s - right.phase_s).max() == 0.0

    refl = np.outer(R9, R9)
    for pole_l, pole_r in ((left.amp_p, right.amp_p), (left.amp_s, right.amp_s)):
        for m in range(kz_arr.size):
            want = refl * pole_r[:, :, m]
            scale = np.abs(want).max()
            assert np.abs(pole_l[:, :, m] - want).max() / scale < 1e-15


def test_left_and_right_amplitudes_actually_differ() -> None:
    """Guard against a left sweep that silently equals the right one."""
    kz_arr = np.array([0.0, 0.7, 2.2])
    right = lateral_split_9x9(0.9, kz_arr, PITCH, OMEGA, REF, direction="right")
    left = lateral_split_9x9(0.9, kz_arr, PITCH, OMEGA, REF, direction="left")
    diff = np.abs(left.amp_p - right.amp_p).max() / np.abs(right.amp_p).max()
    assert diff > 1e-2


@pytest.mark.parametrize("dz", [0.25, -0.5])
def test_vertical_kernel_integrates_to_the_exact_propagator(dz: float) -> None:
    """RUNG 3: the k_z-residue kernel, integrated over (kx, ky), is the
    real-space propagator.

    Quadrature-limited, not exact -- the gate script shows it converging. This
    test pins the construction; the gate pins the convergence.
    """
    # The residual is governed by dk alone, not by kmax: (kmax=120, nk=512) and
    # (kmax=240, nk=1024) share dk and give identical errors, and halving dk
    # divides the residual by ~5.4 -- O(dk^2), the trapezoid rate. dk = 0.23 here.
    kmax, nk = 120.0, 1024
    k1d = np.linspace(-kmax, kmax, nk)
    dk = k1d[1] - k1d[0]

    total = np.zeros((9, 9), dtype=complex)
    for ky in k1d:
        kern = vertical_kernel_9x9(k1d, ky, dz, OMEGA, REF)
        total += np.einsum("abk->ab", kern) * dk**2 / (2 * np.pi) ** 2

    want = exact_propagator_9x9(x=0.0, y=0.0, z=dz, omega=OMEGA, ref=REF)
    assert np.abs(total - want).max() / np.abs(want).max() < 1e-2


def test_vertical_kernel_z_sign_flips_the_odd_z_components() -> None:
    """The dz sign must enter the k-vector, not just the exponential.

    Under z -> -z the C and H blocks flip on every odd z index, exactly as the
    lateral kernel does under x -> -x. Same identity, different axis.
    """
    kx = np.array([0.0, 0.8, 2.5])
    ky = 0.4
    up = vertical_kernel_9x9(kx, ky, -0.3, OMEGA, REF)
    down = vertical_kernel_9x9(kx, ky, 0.3, OMEGA, REF)

    # Parity under z -> -z: u_z, e_zy and e_zx are odd; e_zz carries two z
    # indices and is even.
    r3 = np.array([-1.0, 1.0, 1.0])  # (z, x, y)
    rz = np.concatenate([r3, [r3[p] * r3[q] for (p, q) in VOIGT_PAIRS]])
    refl = np.outer(rz, rz)
    for m in range(kx.size):
        want = refl * down[:, :, m]
        assert np.abs(up[:, :, m] - want).max() / np.abs(want).max() < 1e-14


def test_vertical_kernel_rejects_zero_dz() -> None:
    """dz = 0 genuinely diverges -- it must raise, not return a big number."""
    with pytest.raises(ValueError, match="dz"):
        vertical_kernel_9x9(np.array([0.0]), 0.0, 0.0, OMEGA, REF)


def test_rejects_zero_pitch() -> None:
    with pytest.raises(ValueError, match="pitch"):
        lateral_split_9x9(0.4, np.array([0.0]), 0.0, OMEGA, REF)


def test_rejects_zero_frequency() -> None:
    with pytest.raises(ValueError, match="omega"):
        lateral_split_9x9(0.4, np.array([0.0]), PITCH, 0.0, REF)


def test_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        lateral_split_9x9(0.4, np.array([0.0]), PITCH, OMEGA, REF, direction="up")
