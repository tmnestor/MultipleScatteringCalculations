"""Stage 2: distant receiver cells by the expansion about the cell centre, vectorised over offsets.

For the offsets the wavenumber series cannot reach (distant cells at high frequency), the average is the
Taylor expansion of the integrand about the receiver cell's centre: the cube's moments (frequency-
independent) times the derivatives of g_S and B at the centre (compiled ladder, one call for every offset).
The blocks must equal a CONVERGED quadrature (24 points per axis) and, offset by offset, the graded-voxel
reference ``moment_table_series``; offsets too close for one expansion, or whose order would pass the cap,
must be left to the quadrature; and the route through ``averaged_pair_blocks_9x9`` must take each offset
from the right method.
"""

import numpy as np
import pytest

from cubic_scattering import cell_average_far as CF
from cubic_scattering import cell_average_kseries as CK
from cubic_scattering import cell_averaged_pair as CAP
from cubic_scattering.cell_average_fortran import averaged_pair_blocks_9x9_fortran, cube_maps
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import derivatives as gd
from cubic_scattering.tests.numerics_yaml import cell_average_yaml

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
#: the series reach used in production (numerics.yml, cell_average.kseries_max_reach)
REACH = 3.0
D = 0.04
FAR = [(3, 2, 1), (6, 4, 2), (12, 7, 3), (24, 24, 24), (-7, 12, 3), (2, -24, 9)]


def _omega(kd: float, eta: float) -> complex:
    return (kd / D) * REF.beta * (1 + 1j * eta) if eta else (kd / D) * REF.beta


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.mark.parametrize("eta", [0.0, 0.05])
@pytest.mark.parametrize("kd", [0.24, 0.4, 1.0])
def test_far_blocks_equal_converged_quadrature(kd: float, eta: float) -> None:
    omega = _omega(kd, eta)
    S = np.array(FAR, dtype=float) * D
    got, covered = CF.averaged_pair_blocks_9x9_far(S, omega, REF, D, tol=1e-15, max_order=40)
    assert covered.all()
    CAP.clear_block_cache()
    want = averaged_pair_blocks_9x9_fortran(S, omega, REF, D, n_gauss=24)
    CAP.clear_block_cache()
    for i in range(len(S)):
        assert _rel(got[i], want[i]) < 1e-12, (FAR[i], _rel(got[i], want[i]))


@pytest.mark.parametrize("kd", [0.24, 1.0])
def test_far_sum_equals_moment_table_series_offset_by_offset(kd: float) -> None:
    omega = _omega(kd, 0.05)
    canonical = np.array([(3, 2, 1), (12, 7, 3), (24, 24, 24)])
    G, Gd, Gdd, ok = CF.canonical_greens_far(canonical, omega, REF, D, tol=1e-15, max_order=40)
    assert ok.all()
    d_list = list(gd.multi_indices(2))
    pos = {a: k for k, a in enumerate(d_list)}
    for j, c in enumerate(canonical):
        T = gd.moment_table_series(c * D, D, omega, REF, d_list, [(0, 0, 0)], 1e-15)[..., 0] / D**3
        assert _rel(G[j], T[..., pos[(0, 0, 0)]]) < 1e-13
        assert _rel(Gd[j, :, :, 2], T[..., pos[(0, 0, 1)]]) < 1e-13
        assert _rel(Gdd[j, :, :, 1, 1], T[..., pos[(0, 2, 0)]]) < 1e-13


def test_offsets_too_close_for_one_expansion_are_left_to_the_quadrature() -> None:
    S = np.array([(1, 0, 0), (2, 1, 0), (3, 2, 1)], dtype=float) * D  # r / d = 1, 2.2, 3.7
    _, covered = CF.averaged_pair_blocks_9x9_far(S, _omega(0.4, 0.0), REF, D, tol=1e-15, max_order=24)
    assert covered.tolist() == [False, False, True]


def test_an_order_beyond_the_cap_is_left_to_the_quadrature() -> None:
    S = np.array([(12, 7, 3)], dtype=float) * D
    _, low = CF.averaged_pair_blocks_9x9_far(S, _omega(0.4, 0.0), REF, D, tol=1e-15, max_order=24)
    _, high = CF.averaged_pair_blocks_9x9_far(S, _omega(4.0, 0.0), REF, D, tol=1e-15, max_order=12)
    assert low.tolist() == [True] and high.tolist() == [False]


def test_off_lattice_separations_are_left_to_the_quadrature() -> None:
    S = (np.array([[12.0, 7.0, 3.0]]) + 0.37) * D
    _, covered = CF.averaged_pair_blocks_9x9_far(S, _omega(0.4, 0.0), REF, D, tol=1e-15, max_order=24)
    assert covered.tolist() == [False]


# ---- the route through averaged_pair_blocks_9x9 and its configuration ------------------------------


def _config(path, far: str, store) -> None:
    path.write_text(cell_average_yaml(kseries_store=store, far_expansion=far))


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "numerics.yml"
    monkeypatch.setattr(CAP, "NUMERICS_YAML", path)
    CAP.cell_average_config.cache_clear()
    yield path
    CAP.cell_average_config.cache_clear()


def test_each_offset_takes_series_then_expansion_then_quadrature(config_file, tmp_path) -> None:
    _config(config_file, "true", tmp_path / "s.npy")
    omega = _omega(1.0, 0.0)  # (1, 0, 0) series; (12, 7, 3) beyond the series, expansion; (2, 1, 0)?
    S = np.array([(1, 0, 0), (12, 7, 3), (2, 0, 0)], dtype=float) * D
    CAP.clear_block_cache()
    got = CAP.averaged_pair_blocks_9x9(S, omega, REF, D)
    series, s_cov = CK.averaged_pair_blocks_9x9_kseries(S, omega, REF, D, tol=1e-15, max_reach=REACH)
    far, f_cov = CF.averaged_pair_blocks_9x9_far(S, omega, REF, D, tol=1e-15, max_order=24)
    CAP.clear_block_cache()
    quadrature = averaged_pair_blocks_9x9_fortran(S, omega, REF, D, n_gauss=24)  # fallback_n_gauss
    CAP.clear_block_cache()
    for i in range(len(S)):
        want = series[i] if s_cov[i] else far[i] if f_cov[i] else quadrature[i]
        np.testing.assert_array_equal(got[i], want)
    assert s_cov[0] and not s_cov[1] and f_cov[1]  # the partition this test is meant to exercise


def test_far_expansion_off_leaves_the_rest_to_the_quadrature(config_file, tmp_path) -> None:
    _config(config_file, "false", tmp_path / "s.npy")
    omega = _omega(1.0, 0.0)
    S = np.array([(12, 7, 3)], dtype=float) * D
    CAP.clear_block_cache()
    got = CAP.averaged_pair_blocks_9x9(S, omega, REF, D)
    CAP.clear_block_cache()
    want = averaged_pair_blocks_9x9_fortran(S, omega, REF, D, n_gauss=24)  # fallback_n_gauss
    CAP.clear_block_cache()
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"far_expansion": None}, "far_expansion"),
        ({"far_expansion": "maybe"}, "far_expansion"),
        ({"far_tol": None}, "far_tol"),
        ({"far_max_order": None}, "far_max_order"),
        ({"far_max_order": 1}, "far_max_order"),
        ({"fallback_n_gauss": None}, "fallback_n_gauss"),
        ({"fallback_n_gauss": 2}, "fallback_n_gauss"),
    ],
)
def test_missing_or_invalid_far_keys_stop_with_a_diagnostic(config_file, change: dict, key: str) -> None:
    config_file.write_text(cell_average_yaml(**change))
    with pytest.raises(ValueError) as err:
        CAP.cell_average_config()
    message = str(err.value)
    assert f"cell_average.{key}" in message  # what
    assert str(config_file) in message  # where
    assert "far_expansion: true" in message and "far_max_order: 24" in message  # a valid example
    assert "Fix:" in message  # how to recover


def test_cube_images_of_far_offsets_share_one_canonical_block() -> None:
    images = np.array([(12, 7, 3), (-7, 12, 3), (3, -12, -7)])
    keys, _, _, _ = cube_maps(images)
    assert keys.tolist() == [[12, 7, 3]]
