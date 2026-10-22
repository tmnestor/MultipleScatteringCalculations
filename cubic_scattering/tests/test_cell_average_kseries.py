"""The frequency-amortised receiver-cell average: universal coefficients, their store, the series blocks.

The coefficients U[t, a](offset) = <d^a r^(t - 1)> over the unit receiver cube are universal constants, so
they are computed once (compiled, ``fortran/kseries_moments.f90``), stored on disk, and serve every
frequency, cell size and medium. The compiled coefficients must equal ``graded_voxel.derivatives.
kseries_coefficients`` (the Python reference) to round-off; the series blocks must equal a CONVERGED
quadrature (24 points per axis, not the automatic orders, which are good to about 1e-8 only: see
``cell_averaged_pair.auto_n_gauss``); offsets the series cannot reach must fall back to the quadrature; an
explicit Gauss order must mean the quadrature; and the store and the configuration must fail fast with a
full diagnostic.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cubic_scattering import cell_average_kseries as CK
from cubic_scattering import cell_averaged_pair as CAP
from cubic_scattering.cell_average_fortran import averaged_pair_blocks_9x9_fortran
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import derivatives as gd
from cubic_scattering.tests.numerics_yaml import cell_average_yaml

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
#: the series reach used in production (numerics.yml, cell_average.kseries_max_reach)
REACH = 3.0
D = 0.25
CANONICAL = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0), (3, 2, 1)]
IMAGES = [(0, -1, 0), (-1, 0, 1), (1, -1, -1), (0, 2, 0), (-1, 0, 2), (1, -3, 2), (-2, -1, 3)]
KD = [1e-3, 0.3, 1.0, 2.0]
ETAS = [0.0, 0.05]


def _omega(kd: float, eta: float) -> complex:
    return (kd / D) * REF.beta * (1 + 1j * eta) if eta else (kd / D) * REF.beta


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh coefficient store in a temporary file, so no test touches the user's cache."""
    path = tmp_path / "receiver_kseries.npy"
    monkeypatch.setattr(CK, "store_path", lambda: path)
    CK.clear_loaded_store()
    yield path
    CK.clear_loaded_store()


# ---- the universal coefficients ------------------------------------------------------------------


@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 1), (2, 1, 0)])
def test_compiled_coefficients_equal_python(offset: tuple[int, int, int]) -> None:
    got = CK.coefficients_fortran(np.array([offset]))[0]  # (T + 1, n_a)
    want, a_list = gd.kseries_coefficients(offset, tuple(gd.multi_indices(2)), ((0, 0, 0),))
    assert a_list == CK.A_LIST
    want = want[..., 0]
    for t in range(want.shape[0]):
        scale = np.max(np.abs(want[t]))
        if scale > 0:
            assert float(np.max(np.abs(got[t] - want[t]))) / scale < 1e-12, t


def test_compiled_coefficients_are_independent_of_thread_count() -> None:
    code = (
        "import numpy as np, sys; sys.path.insert(0, '.');"
        "from cubic_scattering.cell_average_kseries import coefficients_fortran;"
        "sys.stdout.buffer.write(coefficients_fortran(np.array([[1,0,0],[2,1,0],[3,2,1],[4,4,2]])).tobytes())"
    )
    outs = []
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1]


@pytest.mark.parametrize("kd", [1e-3, 1.0, 2.0])
def test_vectorised_sum_equals_moment_table_kseries_offset_by_offset(kd: float) -> None:
    """The batched sum is the graded-voxel one: same terms, same truncation, same refusal."""
    canonical = np.array([(1, 0, 0), (2, 1, 0), (3, 2, 1), (9, 6, 4)])
    U = CK.coefficients_fortran(canonical)
    omega = _omega(kd, 0.05)
    G, Gd, Gdd, ok = CK.canonical_greens(canonical, U, omega, REF, D, tol=1e-15)
    d_list = list(gd.multi_indices(2))
    pos = {a: k for k, a in enumerate(d_list)}
    for j, c in enumerate(canonical):
        key = (int(c[0]), int(c[1]), int(c[2]))
        try:
            T = (
                gd.moment_table_kseries(
                    key,
                    D,
                    omega,
                    REF,
                    d_list,
                    [(0, 0, 0)],
                    1e-15,
                    coefficients=lambda *_, u=U[j]: (u[..., None], CK.A_LIST),
                )[..., 0]
                / D**3
            )
        except ValueError:
            assert not ok[j]
            continue
        assert ok[j]
        assert _rel(G[j], T[..., pos[(0, 0, 0)]]) < 1e-13
        assert _rel(Gd[j, :, :, 1], T[..., pos[(0, 1, 0)]]) < 1e-13
        assert _rel(Gdd[j, :, :, 0, 2], T[..., pos[(1, 0, 1)]]) < 1e-13
    assert ok[:2].all() and (ok[3] == (kd < 1.0))  # near offsets always; the far one only at low k_S d


# ---- the cube's symmetry on finished 9 x 9 blocks ---------------------------------------------------


def test_one_canonical_block_gives_all_48_images() -> None:
    """A signed permutation of the axes is a signed permutation of the block's 9 components.

    Checked against the Python reference computed directly at every image, no symmetry used.
    """
    import itertools

    from cubic_scattering.cell_average_fortran import cube_image_blocks, cube_maps

    omega = _omega(1.0, 0.05)
    canonical = np.array([3.0, 2.0, 1.0]) * D
    images = np.array(
        sorted(
            {
                tuple(sg * (3, 2, 1)[p] for sg, p in zip(signs, perm, strict=True))
                for perm in itertools.permutations(range(3))
                for signs in itertools.product((1, -1), repeat=3)
            }
        ),
        dtype=float,
    )
    assert len(images) == 48
    keys, inverse, pinv, sign = cube_maps(images.astype(int))
    assert keys.tolist() == [[3, 2, 1]]
    CAP.clear_block_cache()
    block_c = CAP.averaged_pair_block_9x9(
        canonical, omega, REF, D, n_gauss=6
    )  # any rule: the cube is symmetric
    got = cube_image_blocks(block_c[None], inverse, pinv, sign)
    for i, s in enumerate(images * D):
        want = CAP.averaged_pair_block_9x9(s, omega, REF, D, n_gauss=6)
        assert _rel(got[i], want) < 1e-12, images[i]
    CAP.clear_block_cache()


# ---- the series blocks ----------------------------------------------------------------------------


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("kd", KD)
def test_series_blocks_equal_converged_quadrature(store, kd: float, eta: float) -> None:
    omega = _omega(kd, eta)
    S = np.array(CANONICAL + IMAGES, dtype=float) * D
    got, covered = CK.averaged_pair_blocks_9x9_kseries(S, omega, REF, D, tol=1e-15, max_reach=REACH)
    CAP.clear_block_cache()
    want = averaged_pair_blocks_9x9_fortran(S, omega, REF, D, n_gauss=24)
    CAP.clear_block_cache()
    # within the configured reach it must cover, beyond it it must decline (the cancellation): exactly
    r_max = (np.linalg.norm(S / D, axis=1) + np.sqrt(3.0) / 2.0) * D
    reach = abs(omega / REF.beta) * r_max
    np.testing.assert_array_equal(covered, reach <= REACH)
    for i in np.flatnonzero(covered):
        assert _rel(got[i], want[i]) < 2e-13, (S[i] / D, _rel(got[i], want[i]))


def test_offsets_beyond_the_series_are_not_covered(store) -> None:
    omega = _omega(1.0, 0.0)  # k_S r_max: 1.9 for the face neighbour, about 26 for the far offset
    S = np.array([[1, 0, 0], [20, 13, 7]], dtype=float) * D
    blocks, covered = CK.averaged_pair_blocks_9x9_kseries(S, omega, REF, D, tol=1e-15, max_reach=REACH)
    assert covered.tolist() == [True, False]
    assert not np.any(blocks[1])


def test_off_lattice_separations_are_not_covered(store) -> None:
    S = (np.array([[2.0, 1.0, 0.0]]) + 0.37) * D
    _, covered = CK.averaged_pair_blocks_9x9_kseries(
        S, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH
    )
    assert covered.tolist() == [False]


# ---- the store ------------------------------------------------------------------------------------


def test_store_is_built_on_first_use_and_reused(store) -> None:
    assert not store.exists()
    S = np.array([[1, 0, 0], [0, 2, -1]], dtype=float) * D
    first, _ = CK.averaged_pair_blocks_9x9_kseries(S, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH)
    assert store.exists()
    assert _stored_offsets(store) == [(1, 0, 0), (2, 1, 0)]  # canonical only
    CK.clear_loaded_store()
    again, _ = CK.averaged_pair_blocks_9x9_kseries(S, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH)
    np.testing.assert_array_equal(again, first)


def _stored_offsets(path) -> list[tuple[int, ...]]:
    rows = np.load(path, mmap_mode="r")
    assert rows.shape[1] == CK.ROW
    return sorted(tuple(int(v) for v in r) for r in np.asarray(rows[:, :3]))


def test_store_grows_with_new_offsets(store) -> None:
    CK.averaged_pair_blocks_9x9_kseries(
        np.array([[1.0, 0, 0]]) * D, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH
    )
    CK.clear_loaded_store()
    CK.averaged_pair_blocks_9x9_kseries(
        np.array([[0, 3.0, 1]]) * D, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH
    )
    assert _stored_offsets(store) == [(1, 0, 0), (3, 1, 0)]


def test_the_store_is_memory_mapped(store) -> None:
    """Sweep workers share one copy through the page cache: the rows must be a map of the file."""
    CK.averaged_pair_blocks_9x9_kseries(
        np.array([[1.0, 0, 0]]) * D, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH
    )
    assert CK._LOADED is not None
    rows, _ = CK._LOADED
    assert isinstance(rows, np.memmap) and not rows.flags.writeable


def test_a_store_from_another_layout_stops_with_a_diagnostic(store) -> None:
    np.save(store, np.zeros((2, 7)))  # rows of the wrong width
    with pytest.raises(ValueError) as err:
        CK.averaged_pair_blocks_9x9_kseries(
            np.array([[1.0, 0, 0]]) * D, _omega(0.3, 0.0), REF, D, tol=1e-15, max_reach=REACH
        )
    text = str(err.value)
    assert "layout" in text and str(store) in text and "Fix:" in text


# ---- the route through averaged_pair_blocks_9x9 and its configuration ------------------------------


def _config(path, route: str, store_file) -> None:
    path.write_text(cell_average_yaml(route=route, kseries_store=store_file, far_expansion="false"))


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "numerics.yml"
    monkeypatch.setattr(CAP, "NUMERICS_YAML", path)
    CAP.cell_average_config.cache_clear()
    CK.clear_loaded_store()
    yield path
    CAP.cell_average_config.cache_clear()
    CK.clear_loaded_store()


def test_kseries_route_uses_the_series_and_falls_back_to_quadrature(config_file, tmp_path) -> None:
    _config(config_file, "kseries", tmp_path / "store.npy")
    omega = _omega(1.0, 0.0)  # the face neighbour within the series reach, the far offset beyond it
    S = np.array([[1, 0, 0], [20, 13, 7]], dtype=float) * D
    CAP.clear_block_cache()
    got = CAP.averaged_pair_blocks_9x9(S, omega, REF, D)
    CAP.clear_block_cache()
    series, _ = CK.averaged_pair_blocks_9x9_kseries(S[:1], omega, REF, D, tol=1e-15, max_reach=REACH)
    quadrature = averaged_pair_blocks_9x9_fortran(S[1:], omega, REF, D, n_gauss=24)  # fallback_n_gauss
    CAP.clear_block_cache()
    np.testing.assert_array_equal(got[0], series[0])
    np.testing.assert_array_equal(got[1], quadrature[0])


def test_explicit_gauss_order_means_quadrature(config_file, tmp_path) -> None:
    _config(config_file, "kseries", tmp_path / "store.npy")
    S = np.array([[1, 0, 0]], dtype=float) * D
    CAP.clear_block_cache()
    got = CAP.averaged_pair_blocks_9x9(S, _omega(0.3, 0.0), REF, D, n_gauss=10)
    CAP.clear_block_cache()
    want = averaged_pair_blocks_9x9_fortran(S, _omega(0.3, 0.0), REF, D, n_gauss=10)
    CAP.clear_block_cache()
    np.testing.assert_array_equal(got, want)


def test_quadrature_route_ignores_the_series(config_file, tmp_path) -> None:
    _config(config_file, "quadrature", tmp_path / "store.npy")
    S = np.array([[1, 0, 0]], dtype=float) * D
    CAP.clear_block_cache()
    got = CAP.averaged_pair_blocks_9x9(S, _omega(0.3, 0.0), REF, D)
    CAP.clear_block_cache()
    want = averaged_pair_blocks_9x9_fortran(S, _omega(0.3, 0.0), REF, D)
    CAP.clear_block_cache()
    np.testing.assert_array_equal(got, want)
    assert not (tmp_path / "store.npy").exists()


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"route": None}, "route"),
        ({"route": "fast"}, "route"),
        ({"kseries_tol": None}, "kseries_tol"),
        ({"kseries_tol": -1}, "kseries_tol"),
        ({"kseries_store": None}, "kseries_store"),
        ({"kseries_max_reach": None}, "kseries_max_reach"),
        ({"kseries_max_reach": 0}, "kseries_max_reach"),
    ],
)
def test_missing_or_invalid_route_keys_stop_with_a_diagnostic(config_file, change: dict, key: str) -> None:
    config_file.write_text(cell_average_yaml(**change))
    with pytest.raises(ValueError) as err:
        CAP.cell_average_config()
    message = str(err.value)
    assert f"cell_average.{key}" in message  # what
    assert str(config_file) in message  # where
    assert "route: kseries" in message and "kseries, quadrature" in message  # a valid example and values
    assert "Fix:" in message  # how to recover


def test_a_float_without_a_decimal_point_is_explained(config_file) -> None:
    """PyYAML (YAML 1.1) reads 1e-15 as text: the diagnostic must say so and show the fix."""
    config_file.write_text(cell_average_yaml(kseries_tol="1e-15"))
    with pytest.raises(ValueError) as err:
        CAP.cell_average_config()
    message = str(err.value)
    assert "cell_average.kseries_tol" in message and "decimal point" in message and "1.0e-15" in message


def test_the_shipped_configuration_is_valid() -> None:
    CAP.cell_average_config.cache_clear()
    cfg = CAP.cell_average_config()
    assert cfg.route in ("kseries", "quadrature") and cfg.kseries_tol > 0
