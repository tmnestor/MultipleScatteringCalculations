"""The point cell's exact self term (``point_cell_self``): faces against volume quadrature, the sum rule,
the static closed forms of the cube, and the package T-matrix as the closure's static limit."""

import numpy as np
import pytest
import torch

from cubic_scattering.cell_averaged_pair import averaged_pair_block_9x9
from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.kupradze_derivatives import (
    difference_derivative_tensors,
    scalar_derivative_tensors,
)
from cubic_scattering.point_cell_self import (
    contrast_9x9,
    cube_integral_tensors,
    exact_point_tmatrix_9x9,
    offset_block_9x9,
    self_block_9x9,
)
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
MU = REF.rho * REF.beta**2
LAM = REF.rho * REF.alpha**2 - 2.0 * MU
D = 2.0
H = 0.5 * D
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)


def _omega(k_s_h: float) -> float:
    return k_s_h * REF.beta / H


@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 0), (2, 1, 0), (1, 1, 1)])
def test_faces_reproduce_the_volume_quadrature_off_centre(
    offset: tuple[int, int, int],
) -> None:
    s = D * np.array(offset, dtype=float)
    omega = _omega(0.3)
    face = offset_block_9x9(s, omega, REF, D) / D**3
    vol = averaged_pair_block_9x9(s, omega, REF, D, n_gauss=24)
    assert np.max(np.abs(face - vol)) <= 1e-12 * np.max(np.abs(vol))


@pytest.mark.parametrize("k_s_h", [0.01, 0.5, 2.0])
def test_sum_rule_carries_the_delta_at_the_centre(k_s_h: float) -> None:
    # (laplacian + k^2) g = -delta, so the trace of the cube integral of d2 g is -1 - k^2 integral of g;
    # a spherical excision would return -k^2 integral of g alone
    k = _omega(k_s_h) / REF.beta
    t = cube_integral_tensors(np.zeros(3), H, lambda r: scalar_derivative_tensors(r, k, 1), k, 2)
    assert abs(np.trace(t[2]) + 1.0 + k**2 * t[0]) <= 1e-13


def test_the_face_route_is_symmetric_in_every_index() -> None:
    # the first index is peeled onto the faces, so symmetry in it is a check, not a construction
    k_p, k_s = _omega(0.5) / REF.alpha, _omega(0.5) / REF.beta
    t4 = cube_integral_tensors(
        np.zeros(3), H, lambda r: difference_derivative_tensors(r, k_p, k_s, 3), None, 4
    )[4]
    for perm in [(1, 0, 2, 3), (2, 1, 0, 3), (3, 1, 2, 0)]:
        assert np.max(np.abs(t4 - t4.transpose(perm))) <= 1e-13 * np.max(np.abs(t4))


@pytest.mark.parametrize(
    "contrast",
    [
        MaterialContrast(Dlambda=0.0, Dmu=1e9, Drho=0.0),
        MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0),
    ],
)
def test_static_self_term_is_the_cubes_closed_form(contrast: MaterialContrast) -> None:
    # the inverse strain concentrations of the cube on A1g, T2g and Eg (Chiu 1977, Nozaki 2001)
    omega = _omega(1e-4)
    s_self = self_block_9x9(omega, REF, D)
    dm = contrast_9x9(omega, contrast)
    got = np.sort(np.linalg.eigvals(np.eye(6) - s_self[3:, 3:] @ dm[3:, 3:]).real)
    s_off = (np.pi * (LAM + 2 * MU) - np.sqrt(3) * (LAM + MU)) / (3 * np.pi * MU * (LAM + 2 * MU))
    s_diag = (3 * np.sqrt(3) * (LAM + MU) + 2 * np.pi * MU) / (6 * np.pi * MU * (LAM + 2 * MU))
    dil = 1 + (contrast.Dlambda + 2 / 3 * contrast.Dmu) / (LAM + 2 * MU)
    want = np.sort([dil] + [1 + 2 * contrast.Dmu * s_off] * 3 + [1 + 2 * contrast.Dmu * s_diag] * 2)
    assert np.max(np.abs(got - want)) <= 1e-9


def test_the_package_tmatrix_is_the_closure_statically_and_departs_as_kh_squared() -> None:
    def departure(k_s_h: float) -> float:
        omega = _omega(k_s_h)
        exact = exact_point_tmatrix_9x9(omega, D, CONTRAST, self_block_9x9(omega, REF, D))
        package = _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, H, REF, CONTRAST), omega, H)
        return float(np.max(np.abs(exact[3:, 3:] - package[3:, 3:])) / np.max(np.abs(package[3:, 3:])))

    low, mid = departure(0.01), departure(0.1)
    assert low <= 5e-5
    assert 80.0 <= mid / low <= 120.0  # (kh)^2: a hundredfold over a decade


def test_exact_closure_is_block_diagonal_by_parity() -> None:
    omega = _omega(0.5)
    t = exact_point_tmatrix_9x9(omega, D, CONTRAST, self_block_9x9(omega, REF, D))
    assert np.max(np.abs(t[:3, 3:])) + np.max(np.abs(t[3:, :3])) <= 1e-12 * np.max(np.abs(t))


def test_point_solver_refuses_an_unknown_single_site() -> None:
    from cubic_scattering.sphere_scattering_fft_gpu import solve_point_plane_wave

    with pytest.raises(ValueError, match="single_site = 'mie'.*Valid: 'package'.*'exact'.*Fix:"):
        solve_point_plane_wave(
            _omega(0.3),
            10.0,
            REF,
            CONTRAST,
            4,
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "P",
            profile=None,
            single_site="mie",
            gmres_tol=1e-10,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )


def test_static_displacement_block_and_radiation_damping_in_closed_form() -> None:
    # integral of 1/r over the cube is h^2 (12 ln(2 + sqrt 3) - 2 pi); Im g = sin(kr)/(4 pi r) gives the damping
    omega = _omega(1e-4)
    s_uu = self_block_9x9(omega, REF, D)[:3, :3]
    m_p = LAM + 2 * MU
    static = (2 / MU + 1 / m_p) * H**2 * (6 * np.log(2 + np.sqrt(3)) - np.pi) / (6 * np.pi)
    damping = D**3 * omega * (2 / REF.beta**3 + 1 / REF.alpha**3) / (12 * np.pi * REF.rho)
    assert np.max(np.abs(s_uu.real - static * np.eye(3))) <= 1e-8 * static
    assert np.max(np.abs(s_uu.imag - damping * np.eye(3))) <= 1e-7 * damping


@pytest.mark.parametrize("k_s_h", [1e-3, 0.3, 1.0, 2.5, 4.0])
@pytest.mark.parametrize("damping", [0.0, 0.02])
def test_the_series_self_term_is_the_face_rule(k_s_h: float, damping: float) -> None:
    """Within its reach the frequency series of the self term (frequency-free coefficients) is the face rule."""
    from cubic_scattering.point_cell_self import self_block_9x9_series

    omega = _omega(k_s_h) * (1 + 1j * damping)
    series = self_block_9x9_series(omega, REF, D)
    faces = self_block_9x9(omega, REF, D)
    assert np.max(np.abs(series - faces)) <= 5e-13 * np.max(np.abs(faces))
