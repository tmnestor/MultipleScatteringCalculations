"""Derivatives of the Green's tensor to sixth order, and the hierarchy's moment tables by cube symmetry.

The references are independent of the module: a 40-digit sympy evaluation of the derivatives of
e^{ikr}/r (in which the P - S difference is harmless), the package's 9 x 9 point kernel for orders 0-2,
and, for the symmetry, tables integrated directly at every image offset.
"""

import itertools

import mpmath
import numpy as np
import pytest
import sympy as sp

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import derivatives as gd
from cubic_scattering.graded_voxel.kernel import greens_tensors

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
DIRECTION = np.array([0.48, -0.36, 0.8])  # unit vector: no component vanishes, so no term is trivially 0


@pytest.fixture(scope="module")
def symbolic():
    """d^a of e^{ikr}/r as lambdas evaluated at 40 digits, for a spread of multi-indices to order 6."""
    x, y, z, k = sp.symbols("x y z k")
    g = sp.exp(sp.I * k * sp.sqrt(x**2 + y**2 + z**2)) / sp.sqrt(x**2 + y**2 + z**2)
    picks = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 1), (2, 2, 0)]
    picks += [(1, 1, 1), (3, 1, 1), (2, 2, 2), (4, 0, 2)]
    out = {}
    for a in picks:
        expr = sp.diff(g, x, a[0], y, a[1], z, a[2])
        out[a] = sp.lambdify((x, y, z, k), expr, "mpmath")
    return out


def exact(fn, X, k) -> complex:
    with mpmath.workdps(40):
        return complex(fn(*(mpmath.mpf(float(v)) for v in X), mpmath.mpf(float(k))))


@pytest.mark.parametrize("ks_r", [1e-3, 0.05, 0.3, 0.49, 0.51, 2.0, 12.0])
def test_scalar_derivatives_against_40_digits(symbolic, ks_r: float) -> None:
    """Both g_S and B = (g_S - g_P)/k_S^2, on either side of the series switch, to 1e-12 relative."""
    omega = 300.0
    ks, kp = omega / REF.beta, omega / REF.alpha
    X = DIRECTION * (ks_r / ks)
    idx = gd.multi_indices(6)
    s, b = gd.scalar_derivative_fields_python(X[None, :], omega, REF, len(idx), len(idx))
    for a, fn in symbolic.items():
        n = idx.index(a)
        es = exact(fn, X, ks)
        with mpmath.workdps(40):
            eb = complex(
                (
                    fn(*(mpmath.mpf(float(v)) for v in X), mpmath.mpf(ks))
                    - fn(*(mpmath.mpf(float(v)) for v in X), mpmath.mpf(kp))
                )
                / mpmath.mpf(ks) ** 2
            )
        assert abs(s[n, 0] - es) <= 1e-12 * abs(es), (a, s[n, 0], es)
        assert abs(b[n, 0] - eb) <= 1e-11 * abs(eb), (a, b[n, 0], eb)


@pytest.mark.parametrize("ks_r", [1e-2, 0.4, 0.6, 5.0])
def test_orders_0_to_2_equal_the_point_kernel(ks_r: float) -> None:
    omega = 300.0
    X = np.stack([DIRECTION, -DIRECTION[[2, 0, 1]], [0.0, 0.6, -0.8]]) * (ks_r * REF.beta / omega)
    G, Gd, Gdd = greens_tensors(X, omega, REF)
    full = gd.green_derivative_fields(X, omega, REF, 2)
    idx = gd.multi_indices(2)
    scale = np.abs(G).max(axis=(1, 2))
    for p in range(len(X)):
        for i, n in itertools.product(range(3), repeat=2):
            assert abs(full[i, n, idx.index((0, 0, 0)), p] - G[p, i, n]) <= 1e-12 * scale[p]
            for k in range(3):
                e = [0, 0, 0]
                e[k] += 1
                got = full[i, n, idx.index(tuple(e)), p]
                assert abs(got - Gd[p, i, n, k]) <= 1e-12 * np.abs(Gd[p]).max()
                for m in range(3):
                    e2 = list(e)
                    e2[m] += 1
                    got = full[i, n, idx.index(tuple(e2)), p]
                    assert abs(got - Gdd[p, i, n, k, m]) <= 1e-12 * np.abs(Gdd[p]).max()


def test_the_origin_is_refused() -> None:
    with pytest.raises(ValueError, match="r = 0"):
        gd.scalar_derivative_fields(np.zeros((1, 3)), 300.0, REF, 4, 4)


def test_compiled_equals_reference() -> None:
    from cubic_scattering.graded_voxel.derivatives_fortran import scalar_derivative_fields_fortran

    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 3)) * np.geomspace(1e-3, 30.0, 500)[:, None]
    for omega in (300.0, 300.0 + 6.0j):
        n_s, n_b = len(gd.multi_indices(4)), len(gd.multi_indices(6))
        ps, pb = gd.scalar_derivative_fields_python(X, omega, REF, n_s, n_b)
        fs, fb = scalar_derivative_fields_fortran(X, omega, REF, n_s, n_b)
        assert np.all(np.abs(fs - ps) <= 1e-13 * np.abs(ps).max(axis=0, keepdims=True))
        assert np.all(np.abs(fb - pb) <= 1e-13 * np.abs(pb).max(axis=0, keepdims=True))


def test_tables_by_symmetry_equal_tables_integrated_directly() -> None:
    """Every offset of two orbits, from the representative alone, against direct integration."""
    side, omega, n_gauss = 2.0, 300.0, 6
    d_list = list(gd.multi_indices(3))
    w_list = list(gd.multi_indices(2))
    for rep in ((2, 1, 0), (3, 2, 1)):
        table = gd.moment_table(side * np.array(rep, float), side, omega, REF, d_list, w_list, n_gauss)
        canonical = {rep: table}
        images = {gd.apply(pi, sigma, rep) for pi, sigma in gd.signed_permutations()}
        assert len(images) == 48 if len(set(rep)) == 3 and 0 not in rep else len(images) == 24
        for off in images:
            derived = gd.table_for_offset(off, canonical, d_list, w_list)
            direct = gd.moment_table(side * np.array(off, float), side, omega, REF, d_list, w_list, n_gauss)
            assert np.abs(derived - direct).max() <= 1e-13 * np.abs(direct).max(), off


def test_canonical_offset_and_mapping() -> None:
    for off in itertools.product(range(-3, 4), repeat=3):
        pi, sigma = gd.mapping_to(off)
        assert gd.apply(pi, sigma, gd.canonical_offset(off)) == off


def test_blocks_by_symmetry_equal_blocks_from_direct_tables() -> None:
    """A covariant block B[V, (i, P), (j, W)] = T[i, j, P, W + V] transforms as the tables do."""
    side, omega, n_gauss = 2.0, 300.0, 6
    u_list = list(gd.multi_indices(2))
    v_list = list(gd.multi_indices(1))
    w_list = list(gd.multi_indices(3))
    w_pos = {w: n for n, w in enumerate(w_list)}
    nu = len(u_list)

    def block(table):
        out = np.zeros((len(v_list), 3 * nu, 3 * nu), dtype=complex)
        for vi, v in enumerate(v_list):
            for wi, w in enumerate(u_list):
                col = w_pos[(w[0] + v[0], w[1] + v[1], w[2] + v[2])]
                for i, j in itertools.product(range(3), repeat=2):
                    out[vi, i * nu : (i + 1) * nu, j * nu + wi] = table[i, j, :, col]
        return out

    rep = (3, 1, 0)
    base = block(gd.moment_table(side * np.array(rep, float), side, omega, REF, u_list, w_list, n_gauss))
    for pi, sigma in gd.signed_permutations():
        off = gd.apply(pi, sigma, rep)
        table = gd.moment_table(side * np.array(off, float), side, omega, REF, u_list, w_list, n_gauss)
        direct = block(table)
        derived = gd.transform_block(base, pi, sigma, v_list, u_list)
        assert np.abs(derived - direct).max() <= 1e-13 * np.abs(direct).max(), off


@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_dispatch_follows_the_configuration(tmp_path, monkeypatch, backend: str) -> None:
    from cubic_scattering.graded_voxel import kernel as K
    from cubic_scattering.graded_voxel.derivatives_fortran import scalar_derivative_fields_fortran

    path = tmp_path / "numerics.yml"
    path.write_text(f"point_kernel:\n  backend: {backend}\n")
    monkeypatch.setattr(K, "NUMERICS_YAML", path)
    K.point_kernel_backend.cache_clear()
    try:
        X = np.array([[1.0, 2.0, 3.0], [30.0, -20.0, 10.0]])
        chosen = (
            gd.scalar_derivative_fields_python if backend == "python" else scalar_derivative_fields_fortran
        )
        dispatched = gd.scalar_derivative_fields(X, 60.0, REF, 20, 35)
        for got, want in zip(dispatched, chosen(X, 60.0, REF, 20, 35), strict=True):
            np.testing.assert_array_equal(got, want)
    finally:
        K.point_kernel_backend.cache_clear()
