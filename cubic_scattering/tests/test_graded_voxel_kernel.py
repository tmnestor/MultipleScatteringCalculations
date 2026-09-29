"""The graded voxel's point propagator: equal to the package's, accurate as k r -> 0."""

import numpy as np
import pytest

from cubic_scattering import ReferenceMedium
from cubic_scattering.graded_voxel import kernel as kn
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 150.0  # k_S = 0.05 / m
W9 = np.diag([1.0, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5])
PI9 = np.diag([-1.0, -1, -1, 1, 1, 1, 1, 1, 1])


def _points(kr_lo, kr_hi, n=30, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * (rng.uniform(kr_lo, kr_hi, n) / (OMEGA / REF.beta))[:, None]


def test_equals_package_propagator_away_from_the_origin():
    X = _points(0.6, 5.0)
    got = kn.kernel_9x9(X, OMEGA, REF)
    for x, g in zip(X, got, strict=True):
        want = _propagator_block_9x9(x, OMEGA, REF)
        assert np.linalg.norm(g - want) / np.linalg.norm(want) < 1e-11


def test_series_and_closed_form_agree_across_the_switch(monkeypatch):
    X = _points(0.3, 0.9, seed=1)
    monkeypatch.setattr(kn, "SERIES_LIMIT", 10.0)
    series = kn.kernel_9x9(X, OMEGA, REF)
    monkeypatch.setattr(kn, "SERIES_LIMIT", 0.0)
    closed = kn.kernel_9x9(X, OMEGA, REF)
    rel = np.linalg.norm(series - closed, axis=(1, 2)) / np.linalg.norm(closed, axis=(1, 2))
    assert rel.max() < 1e-12


def test_static_part_is_kelvin():
    X = _points(1e-4, 1e-2, seed=2)
    lam, mu = REF.lam, REF.mu
    got = kn.kernel_9x9(X, OMEGA, REF, dynamic=False)[:, :3, :3]
    r = np.linalg.norm(X, axis=1)
    xh = X / r[:, None]
    want = ((lam + 3 * mu) * np.eye(3)[None] + (lam + mu) * np.einsum("ni,nj->nij", xh, xh)) / (
        8 * np.pi * mu * (lam + 2 * mu) * r[:, None, None]
    )
    np.testing.assert_allclose(got, want, rtol=1e-13)


def test_static_plus_dynamic_is_total():
    X = _points(0.05, 3.0, seed=3)
    tot = kn.kernel_9x9(X, OMEGA, REF)
    parts = kn.kernel_9x9(X, OMEGA, REF, dynamic=False) + kn.kernel_9x9(X, OMEGA, REF, static=False)
    np.testing.assert_allclose(parts, tot, rtol=1e-12, atol=1e-12 * np.abs(tot).max())


def test_reciprocity_and_parity():
    X = _points(0.01, 4.0, seed=4)
    P = kn.kernel_9x9(X, OMEGA, REF)
    Pm = kn.kernel_9x9(-X, OMEGA, REF)
    for p, pm in zip(P, Pm, strict=True):
        wp = W9 @ p
        assert np.linalg.norm(wp - wp.T) / np.linalg.norm(wp) < 1e-12
        assert np.linalg.norm(pm - PI9 @ p @ PI9) / np.linalg.norm(p) < 1e-12


def test_dynamic_remainder_is_weakly_singular():
    # r * |P_dyn| stays bounded as r -> 0 (at most 1/r)
    d = np.array([[0.3, 0.5, 0.81]]) / np.linalg.norm([0.3, 0.5, 0.81])
    vals = [r * np.abs(kn.kernel_9x9(d * r, OMEGA, REF, static=False)).max() for r in (1e-2, 1e-4, 1e-6)]
    assert vals[2] < 2 * vals[0]


def test_origin_is_refused():
    with pytest.raises(ValueError, match="r = 0"):
        kn.kernel_9x9(np.zeros((1, 3)), OMEGA, REF)
