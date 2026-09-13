"""Tests for the stratified-wrapper correction.

Pins the three defects found in the 9x9 wrapper work (see
``docs/wrapper_problem_state_2026-09-13.md``) and the placement restriction that
makes the correction well defined.
"""

import numpy as np
import pytest

from cubic_scattering.layered_correction import (
    J6,
    W9,
    assert_interface_continuous,
    correct_6x6,
    k_operator,
    source_jump_operator,
    strain_from_state,
)

RHO, ALPHA, BETA = 2.6, 4.0, 2.22
S_S = 1.0 / BETA


class _StubModel:
    """Minimal duck-typed stand-in for LayerModel."""

    def __init__(self, alpha, beta, rho):
        self.alpha = list(alpha)
        self.beta = list(beta)
        self.rho = list(rho)
        self.n_layers = len(self.alpha)


# ---------------------------------------------------------------- A operator


def test_strain_operator_matches_hooke():
    """A maps (u, T) -> (u, eps) exactly, checked against the definitions."""
    rng = np.random.default_rng(11)
    mu = RHO * BETA**2
    lam = RHO * ALPHA**2 - 2 * mu
    for _ in range(6):
        kx, ky, kz = rng.normal(size=3) * 3.0
        amp = rng.normal(size=3) + 1j * rng.normal(size=3)
        k = np.array([kz, kx, ky])  # (z, x, y)

        eps = 0.5j * (np.outer(k, amp) + np.outer(amp, k))
        sig = lam * np.trace(eps) * np.eye(3) + 2 * mu * eps
        state = np.array([amp[0], amp[1], amp[2], sig[0, 0], sig[0, 1], sig[0, 2]])
        want = np.array(
            [
                amp[0],
                amp[1],
                amp[2],
                eps[0, 0],
                eps[1, 1],
                eps[2, 2],
                2 * eps[1, 2],
                2 * eps[0, 2],
                2 * eps[0, 1],
            ]
        )
        got = strain_from_state(kx, ky, RHO, ALPHA, BETA) @ state
        assert np.allclose(got, want, atol=1e-12)


# ---------------------------------------------------------------- K operator


def test_k_operator_is_projector_form():
    """K = 1 (+) (-P_par + P_perp) = 1 (+) (I2 - 2 khat khat^T).

    Symmetric, unit on the zz slot, and a Householder reflection in plane.
    """
    w, p, c, s = 2 * np.pi * 48.0, 0.12, 0.6, 0.8
    kx, ky = w * p * c, w * p * s
    k = k_operator(S_S, kx, ky, w)
    assert abs(k[0, 0] - 1.0) < 1e-14
    assert np.allclose(k[0, 1:], 0.0) and np.allclose(k[1:, 0], 0.0)
    assert abs(k[1, 2] - k[2, 1]) < 1e-14

    khat = np.array([kx, ky]) / np.hypot(kx, ky)
    assert np.allclose(k[1:, 1:], np.eye(2) - 2.0 * np.outer(khat, khat))


def test_k_operator_is_involutive():
    """A reflection applied twice is the identity. The eta_S form was not."""
    w, p = 2 * np.pi * 48.0, 0.12
    k = k_operator(S_S, w * p * 0.6, w * p * 0.8, w)
    assert np.allclose(k @ k, np.eye(3), atol=1e-13)


def test_k_operator_does_not_depend_on_the_medium_or_frequency():
    """REGRESSION GUARD for the SH impedance fix (2026-09-14).

    K's in-plane block is purely geometric. It previously carried a factor
    eta_S = sqrt(s_s^2 - p^2) on the perpendicular component, which is the SH
    direction -- and that factor was not physics: it compensated a missing eta
    in the sibling repository's SH layer eigenvector. The two errors cancelled
    in uniform media, so every whole-space gate passed at 1e-15 while the
    layered SH reflection was angle-INDEPENDENT and wrong.

    If K ever depends on s_s or omega again, that compensation has been
    reintroduced. See scripts/gate_sh_impedance.py, which holds the uniform
    limit and the interface reflection at the same time.
    """
    kx, ky = 2 * np.pi * 48.0 * 0.12 * 0.6, 2 * np.pi * 48.0 * 0.12 * 0.8
    base = k_operator(S_S, kx, ky, 2 * np.pi * 48.0)
    for s_s in (S_S, 2.0 * S_S, 0.3 * S_S, complex(0.5, -0.2)):
        for w in (2 * np.pi * 6.0, 2 * np.pi * 48.0, 2 * np.pi * 200.0):
            assert np.allclose(k_operator(s_s, kx, ky, w), base, atol=1e-14), (
                f"K varied with the medium/frequency (s_s={s_s}, w={w})"
            )


def test_k_operator_is_not_diagonal_off_axis():
    """The reason no diagonal source correction could ever have existed."""
    w, p = 2 * np.pi * 48.0, 0.12
    k = k_operator(S_S, w * p * 0.6, w * p * 0.8, w)
    off = np.linalg.norm(k - np.diag(np.diag(k))) / np.linalg.norm(k)
    assert off > 0.6, f"K should be strongly off-diagonal, got {off}"


def test_k_operator_is_diagonal_on_axis():
    """On k_y = 0 or k_x = 0 it does happen to be diagonal."""
    w, p = 2 * np.pi * 48.0, 0.12
    for kx, ky in ((w * p, 0.0), (0.0, w * p)):
        k = k_operator(S_S, kx, ky, w)
        off = np.linalg.norm(k - np.diag(np.diag(k))) / np.linalg.norm(k)
        assert off < 1e-12


def test_k_operator_rejects_the_origin():
    """kx = ky = 0 has no propagation direction, so the projectors are undefined."""
    with pytest.raises(ValueError, match="undefined at kx = ky = 0"):
        k_operator(S_S, 0.0, 0.0, 2 * np.pi * 48.0)


def test_k_operator_is_even_in_k():
    """P_par is invariant under khat -> -khat, so K(-k) = K(+k)."""
    w, p = 2 * np.pi * 48.0, 0.12
    kx, ky = w * p * 0.6, w * p * 0.8
    assert np.allclose(k_operator(S_S, kx, ky, w), k_operator(S_S, -kx, -ky, w))


# ------------------------------------------------------------ source operator


def test_source_operator_matches_thesis_box_53():
    """The derived jump vector reproduces the thesis's worked point explosion.

    Box 5.3 gives the effective delta coefficient F_1 + A_S F_2 for a point
    explosion.  This repository's 9-vector source carries the stress glut as
    ``Dsigma* = -m`` relative to the thesis's ``m`` (that relative sign is
    defect D3, arbitrated by GATE F), so an isotropic thesis moment m = 1
    enters here as Dsigma* = -1.
    """
    kx, ky = 0.9, 0.7
    mu = RHO * BETA**2
    lam = RHO * ALPHA**2 - 2 * mu
    gam = lam / (lam + 2 * mu)

    src = np.zeros(9)
    src[3] = src[4] = src[5] = -1.0  # Dsigma* = -m for an isotropic m = 1
    got = source_jump_operator(kx, ky, RHO, ALPHA, BETA) @ src

    want = np.array(
        [
            1.0 / (RHO * ALPHA**2),
            0.0,
            0.0,
            0.0,
            2j * kx * BETA**2 / ALPHA**2,
            2j * ky * BETA**2 / ALPHA**2,
        ]
    )
    assert np.allclose(got, want, atol=1e-14)
    # and the closed form the thesis value equals: ik(1 - gamma) = 2ik beta^2/alpha^2
    assert abs((1 - gam) - 2 * BETA**2 / ALPHA**2) < 1e-14


def test_source_operator_is_the_symplectic_adjoint():
    """B = J6 A_src(-k)^T W, with the glut columns carrying the D3 sign."""
    kx, ky = 0.9, 0.7
    a_minus = strain_from_state(-kx, -ky, RHO, ALPHA, BETA)
    base = J6 @ a_minus.T @ W9
    got = source_jump_operator(kx, ky, RHO, ALPHA, BETA)
    assert np.allclose(got[:, :3], base[:, :3], atol=1e-14)
    assert np.allclose(got[:, 3:], -base[:, 3:], atol=1e-14)


# ------------------------------------------------------------ the 6x6 fix (D1, D2)


def test_correct_6x6_applies_d1_to_the_stress_block_only():
    """D1 multiplies only the stress-row x stress-column block by (-i w)."""
    w = 2 * np.pi * 48.0
    raw = np.arange(36, dtype=complex).reshape(6, 6) + 1.0
    out = correct_6x6(raw, w, S_S, S_S, kx=5.0, ky=7.0)
    # undo D2 to isolate D1: D2 is a similarity on the traction half only
    assert out.shape == (6, 6)
    # the displacement-row x displacement-column block sees only the -1
    assert np.allclose(out[:3, :3], -raw[:3, :3], atol=1e-10)


# ------------------------------------------------------------ placement guard


def test_accepts_an_interface_inside_a_uniform_zone():
    """Interfaces between identical layers are fine."""
    mod = _StubModel([1.5, 4.0, 4.0, 4.0], [0.0, 2.22, 2.22, 2.22], [1.03, 2.6, 2.6, 2.6])
    assert_interface_continuous(mod, 1, "source")  # layers 1|2, identical


def test_rejects_a_plane_on_a_material_discontinuity():
    """eta_S is two-valued there, so K is not defined."""
    mod = _StubModel([1.5, 4.0, 6.5, 6.5], [0.0, 2.22, 3.7, 3.7], [1.03, 2.6, 3.3, 3.3])
    with pytest.raises(ValueError, match="material discontinuity"):
        assert_interface_continuous(mod, 1, "source")  # layers 1|2 differ


def test_discontinuity_error_is_diagnostic():
    """The message must name what, where, why and how to recover."""
    mod = _StubModel([1.5, 4.0, 6.5, 6.5], [0.0, 2.22, 3.7, 3.7], [1.03, 2.6, 3.3, 3.3])
    with pytest.raises(ValueError) as exc:
        assert_interface_continuous(mod, 1, "receiver")
    text = str(exc.value)
    assert "receiver interface 1" in text
    assert "eta_S" in text
    assert "Fix:" in text
    assert "subdivid" in text.lower()


def test_half_space_interface_is_accepted():
    """Nothing lies below the deepest interface to disagree with."""
    mod = _StubModel([1.5, 4.0, 6.5], [0.0, 2.22, 3.7], [1.03, 2.6, 3.3])
    assert_interface_continuous(mod, 2, "source")


# ------------------------------------------------------------ integration


def test_gate_f_passes_with_the_module():
    """W M symmetric to machine precision -- the point of the whole exercise."""
    import sys

    # the stratified solver lives in a sibling repository
    sibling = "/Users/tod/Desktop/SeismicInversion"
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    pytest.importorskip("GlobalMatrix.layered_greens")
    from cubic_scattering.layered_correction import corrected_layered_6x6

    lm = pytest.importorskip("Kennett_Reflectivity.layer_model")
    n = 6
    model = lm.LayerModel.from_arrays(
        alpha=[1.5, *([ALPHA] * n), ALPHA],
        beta=[0.0, *([BETA] * n), BETA],
        rho=[1.03, *([RHO] * n), RHO],
        thickness=[3.0, *([1.0] * n), np.inf],
        Q_alpha=[2.0] * (n + 2),
        Q_beta=[1e10, *([2.0] * n), 2.0],
    )
    w = 2 * np.pi * 48.0
    p, c, s = 0.12, 0.6, 0.8
    kx, ky = w * p * c, w * p * s

    g = corrected_layered_6x6(model, w, kx, ky, source_iface=4, receiver_iface=3)
    a = strain_from_state(kx, ky, RHO, ALPHA, BETA)
    b = source_jump_operator(kx, ky, RHO, ALPHA, BETA)
    m = a @ g @ b
    wm = W9 @ m
    resid = np.linalg.norm(wm - wm.T) / np.linalg.norm(wm)
    assert resid < 1e-10, f"GATE F residual {resid:.3e}"
