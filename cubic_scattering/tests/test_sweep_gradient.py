"""Tests for the adjoint-state misfit gradient on the directional-sweep solver.

The gradient is note (IV), AdjointStateGradients: a forward Foldy-Lax solve, one
adjoint solve (the forward solve at -k_y conjugated by W_pm = I_pm W, gate A1), the
residual carried back by the dense transpose of the receiver map, and a local
9x9 contraction through dT0/dm (Richardson differences, gate A2).

The arbiter throughout is the misfit itself, differenced by brute force: every
check below would fail if the adjoint shortcut, the weight W_pm, the receiver
transpose, or the T0 derivative were wrong.
"""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import LayeredBackground, apply_g0, build_g0_cache, make_sweep_grid
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_gradient import (
    adjoint_solve,
    frechet_adjoint_product,
    frechet_vector_product,
    layered_receiver_map,
    misfit_and_gradient,
    rayleigh_t0_and_derivative,
    whole_space_receiver_map,
)

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25
KY = 0.6
N_Z, N_X = 2, 3
SOLVE = {"tol": 1e-12, "max_iter": 500}
STEP = np.array([1e-2, 1e-2, 1e-3])  # Richardson base steps for (Dlambda, Dmu, Drho)


def _caches():
    """Whole-space sweep caches at +k_y and -k_y."""
    out = []
    for ky in (KY, -KY):
        grid = make_sweep_grid(N_Z, N_X, PITCH, ky=ky, n_kz=128, n_kx=128)
        out.append(build_g0_cache(grid, REF, OMEGA))
    return out


def _receiver(grid):
    """Three displacement receivers on a line 1 km above plane 0."""
    return whole_space_receiver_map(
        grid, REF, OMEGA, receiver_x=np.array([0.1, 0.35, 0.6]), receiver_dz=-1.0, rows=(0, 1, 2)
    )


def _contrasts(seed: int, scale: float) -> np.ndarray:
    """Per-voxel (Dlambda, Dmu, Drho), shape (N_Z, N_X, 3)."""
    rng = np.random.default_rng(seed)
    base = np.array([2.0, 1.0, 0.1])
    return scale * base * (1.0 + 0.5 * rng.standard_normal((N_Z, N_X, 3)))


def _incident(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_Z, N_X, 9)) + 1j * rng.standard_normal((N_Z, N_X, 9))


def _misfit_only(cache_p, cache_m, contrasts, psi_inc, receiver, d_obs) -> float:
    t0, dt0 = rayleigh_t0_and_derivative(contrasts, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    return misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, d_obs, **SOLVE).misfit


def test_adjoint_solve_is_the_dense_transpose() -> None:
    """The -k_y shortcut solves (I - G0^T T0^T) psi_adj = b_tilde. Fails if W_pm or the flip is wrong."""
    cache_p, cache_m = _caches()
    t0, _ = rayleigh_t0_and_derivative(_contrasts(1, 1.0), [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    size = N_Z * N_X * 9
    g = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        g[:, c] = apply_g0(e.reshape(N_Z, N_X, 9), cache_p).ravel()
    t_full = np.zeros((size, size), dtype=complex)
    for k, blk in enumerate(t0.reshape(-1, 9, 9)):
        t_full[9 * k : 9 * k + 9, 9 * k : 9 * k + 9] = blk
    b = _incident(11)
    want = np.linalg.solve(np.eye(size) - g.T @ t_full.T, b.ravel()).reshape(b.shape)

    got = adjoint_solve(cache_m, t0, b, **SOLVE)

    assert np.abs(got - want).max() / np.abs(want).max() < 1e-9


def test_t0_derivative_matches_a_plain_central_difference() -> None:
    """Richardson dT0/dm against an independent small-step difference. Fails on a wrong step or index."""
    contrasts = _contrasts(2, 1.0)
    t0, dt0 = rayleigh_t0_and_derivative(contrasts, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    assert t0.shape == (N_Z, N_X, 9, 9)
    assert dt0.shape == (N_Z, N_X, 3, 9, 9)
    iz, ix, a = 1, 2, 2
    h = 1e-6
    up, dn = contrasts.copy(), contrasts.copy()
    up[iz, ix, a] += h
    dn[iz, ix, a] -= h
    t_up, _ = rayleigh_t0_and_derivative(up, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    t_dn, _ = rayleigh_t0_and_derivative(dn, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    fd = (t_up[iz, ix] - t_dn[iz, ix]) / (2 * h)
    assert np.abs(dt0[iz, ix, a] - fd).max() / np.abs(fd).max() < 1e-6


@pytest.mark.parametrize("scale", [0.1, 1.0])
def test_gradient_matches_finite_difference_of_the_misfit(scale: float) -> None:
    """The whole chain against brute force, weak and moderate contrast.

    Fails if any link is wrong: the adjoint operator, W_pm, the receiver transpose,
    the conjugate on the residual, the Re, or dT0/dm.
    """
    cache_p, cache_m = _caches()
    receiver = _receiver(cache_p.grid)
    psi_inc = _incident()
    true_m = _contrasts(3, scale)
    t0_true, dt0_true = rayleigh_t0_and_derivative(true_m, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    d_obs = misfit_and_gradient(
        cache_p, cache_m, t0_true, dt0_true, psi_inc, receiver, np.zeros(receiver.n_data, complex), **SOLVE
    ).data
    model = _contrasts(4, scale)
    t0, dt0 = rayleigh_t0_and_derivative(model, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    res = misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, d_obs, **SOLVE)
    assert res.grad.shape == (N_Z, N_X, 3)

    for iz, ix, a in ((0, 0, 0), (1, 1, 1), (0, 2, 2), (1, 0, 0)):
        h = 1e-4 * STEP[a] / 1e-2
        up, dn = model.copy(), model.copy()
        up[iz, ix, a] += h
        dn[iz, ix, a] -= h
        fd = (
            _misfit_only(cache_p, cache_m, up, psi_inc, receiver, d_obs)
            - _misfit_only(cache_p, cache_m, dn, psi_inc, receiver, d_obs)
        ) / (2 * h)
        err = abs(res.grad[iz, ix, a] - fd) / abs(fd)
        assert err < 1e-6, f"voxel ({iz},{ix}) param {a}: adjoint {res.grad[iz, ix, a]:.6e} fd {fd:.6e}"


def _forward(cache_p, cache_m, contrasts, psi_inc, receiver):
    """Forward field and predicted data at a model, through the production path."""
    t0, dt0 = rayleigh_t0_and_derivative(contrasts, [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    res = misfit_and_gradient(
        cache_p, cache_m, t0, dt0, psi_inc, receiver, np.zeros(receiver.n_data, complex), **SOLVE
    )
    return t0, dt0, res


def test_jacobian_vector_product_matches_finite_difference_of_the_data() -> None:
    """F dm against central differences of the predicted data along dm.

    Fails if the tangent-linear model drops the multiple-scattering term
    T0 (I - G0 T0)^-1 G0 dT0 psi, or uses the wrong operator order.
    """
    cache_p, cache_m = _caches()
    receiver = _receiver(cache_p.grid)
    psi_inc = _incident()
    model = _contrasts(4, 1.0)
    t0, dt0, res = _forward(cache_p, cache_m, model, psi_inc, receiver)
    rng = np.random.default_rng(21)
    dm = rng.standard_normal(model.shape) * np.array([1.0, 1.0, 0.1])

    jv = frechet_vector_product(cache_p, t0, dt0, res.psi, receiver, dm, **SOLVE)

    h = 1e-5
    d_up = _forward(cache_p, cache_m, model + h * dm, psi_inc, receiver)[2].data
    d_dn = _forward(cache_p, cache_m, model - h * dm, psi_inc, receiver)[2].data
    fd = (d_up - d_dn) / (2 * h)
    assert np.abs(jv - fd).max() / np.abs(fd).max() < 1e-7


def test_dot_product_identity_holds() -> None:
    """Re(q^H F dm) = dm . Re(F^H q) for random dm and q, to solver tolerance.

    Fails if the adjoint is not the transpose of the tangent-linear model --
    independently of any finite-difference step.
    """
    cache_p, cache_m = _caches()
    receiver = _receiver(cache_p.grid)
    t0, dt0, res = _forward(cache_p, cache_m, _contrasts(4, 1.0), _incident(), receiver)
    rng = np.random.default_rng(22)
    dm = rng.standard_normal((N_Z, N_X, 3))
    q = rng.standard_normal(receiver.n_data) + 1j * rng.standard_normal(receiver.n_data)

    lhs = float(np.vdot(q, frechet_vector_product(cache_p, t0, dt0, res.psi, receiver, dm, **SOLVE)).real)
    rhs = float(np.sum(dm * frechet_adjoint_product(cache_m, t0, dt0, res.psi, receiver, q, **SOLVE)))
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


def test_misfit_gradient_is_the_adjoint_product_of_the_residual() -> None:
    """grad misfit = Re(F^H r): misfit_and_gradient must route through the same adjoint product."""
    cache_p, cache_m = _caches()
    receiver = _receiver(cache_p.grid)
    psi_inc = _incident()
    d_obs = _forward(cache_p, cache_m, _contrasts(3, 1.0), psi_inc, receiver)[2].data
    t0, dt0 = rayleigh_t0_and_derivative(_contrasts(4, 1.0), [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    res = misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, d_obs, **SOLVE)
    via = frechet_adjoint_product(cache_m, t0, dt0, res.psi, receiver, res.data - d_obs, **SOLVE)
    assert np.abs(res.grad - via).max() <= 1e-12 * np.abs(via).max()


def test_zero_residual_gives_zero_gradient() -> None:
    """At the true model the misfit and its gradient vanish."""
    cache_p, cache_m = _caches()
    receiver = _receiver(cache_p.grid)
    psi_inc = _incident()
    t0, dt0 = rayleigh_t0_and_derivative(_contrasts(3, 1.0), [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    zero = np.zeros(receiver.n_data, complex)
    d = misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, zero, **SOLVE).data
    res = misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, d, **SOLVE)
    assert res.misfit < 1e-24
    assert np.abs(res.grad).max() < 1e-12


def test_receiver_map_agrees_with_the_sweep_operator() -> None:
    """R must be the same coupling the forward operator uses between planes.

    A receiver on voxel column j of a THIRD plane sees the sources of planes 0
    and 1 exactly as that plane's voxel does through apply_g0. Offsets are
    asymmetric, so a flipped lateral phase or depth sign cannot pass.
    """
    grid3 = make_sweep_grid(3, N_X, PITCH, ky=KY, n_kz=128, n_kx=128)
    cache3 = build_g0_cache(grid3, REF, OMEGA)
    grid2 = make_sweep_grid(2, N_X, PITCH, ky=KY, n_kz=128, n_kx=128)
    cols = np.array([2, 0])
    receiver = whole_space_receiver_map(
        grid2, REF, OMEGA, receiver_x=cols * PITCH, receiver_dz=2 * PITCH, rows=tuple(range(9))
    )
    got = receiver.matrix.reshape(cols.size, 9, 2, N_X, 9)
    for iz in range(2):
        for jx in range(N_X):
            for b in range(9):
                src = np.zeros((3, N_X, 9), complex)
                src[iz, jx, b] = 1.0
                field = apply_g0(src, cache3)
                for r, col in enumerate(cols):
                    want = field[2, col]
                    assert np.abs(got[r, :, iz, jx, b] - want).max() <= 1e-12 * np.abs(want).max() + 1e-300


def test_receiver_on_a_voxel_plane_is_rejected() -> None:
    """dz = 0 is the kernel's divergent case. Fail fast, with the diagnostic fields."""
    cache_p, _ = _caches()
    with pytest.raises(ValueError) as exc:
        whole_space_receiver_map(
            cache_p.grid, REF, OMEGA, receiver_x=np.array([0.1]), receiver_dz=PITCH, rows=(0, 1, 2)
        )
    msg = str(exc.value)
    for field in ("Where:", "Valid:", "Fix:"):
        assert field in msg


def test_observed_data_of_the_wrong_length_is_rejected() -> None:
    """A mismatched d_obs is a caller error, not something to broadcast."""
    cache_p, cache_m = _caches()
    receiver = _receiver(cache_p.grid)
    t0, dt0 = rayleigh_t0_and_derivative(_contrasts(3, 1.0), [REF] * N_Z, OMEGA, 0.5 * PITCH, step=STEP)
    too_long = np.zeros(receiver.n_data + 1, complex)
    with pytest.raises(ValueError) as exc:
        misfit_and_gradient(cache_p, cache_m, t0, dt0, _incident(), receiver, too_long, **SOLVE)
    msg = str(exc.value)
    for field in ("Where:", "Valid:", "Fix:"):
        assert field in msg


# ── Stratified background ──────────────────────────────────────────────────


def _layer_module():
    """The sibling repo's layer model, skipped rather than failed if absent."""
    import sys

    sibling = "/Users/tod/Desktop/SeismicInversion"
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    pytest.importorskip("GlobalMatrix.layered_greens")
    return pytest.importorskip("Kennett_Reflectivity.layer_model")


def _stratified_model(slab: bool):
    """16 one-km layers under 3 km of water, Q = 2; optionally the fast slab in layers 9-10."""
    lm = _layer_module()
    n_lay, q = 24, 2.0
    al, be, rh = 4.0, 2.22, 2.6
    a = [1.5, *([al] * n_lay), al]
    b = [0.0, *([be] * n_lay), be]
    r = [1.03, *([rh] * n_lay), rh]
    if slab:
        for lay in (9, 10):
            a[lay], b[lay], r[lay] = 6.5, 3.7, 3.3
    return lm.LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([1.0] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def _plane_ref(model, iface: int) -> ReferenceMedium:
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    j = max(iface, 1)
    return ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], model.rho[j])


def test_layered_receiver_map_reduces_to_whole_space_on_a_uniform_model() -> None:
    """On a uniform model, deep enough that the seabed return is damped, layered R = whole-space R.

    Fails if the layered kernel's orientation, source/receiver order or plane
    map is wrong -- the stratified path must reproduce an independent
    construction in the one limit where the answer is known.
    """
    # 3 Hz, not a low frequency: on a uniform model the layered and whole-space
    # maps differ by the seabed return, which Q = 2 damps per WAVELENGTH. Measured
    # residual, flat in the k_x rule (96 vs 192 nodes) and falling with frequency:
    # 9.7e-4 at 0.3 Hz, 1.1e-4 at 0.6, 3.0e-7 at 1.2, 1.7e-15 at 3 Hz. The first
    # version of this test ran at 0.3 Hz and failed on that physical return.
    model = _stratified_model(slab=False)
    planes, rcv_iface, om = (18, 19), 15, 2 * np.pi * 3.0
    grid = make_sweep_grid(2, 3, 1.0, ky=0.3, n_kz=64, n_kx=96, kx_max=6.0)
    ref = _plane_ref(model, planes[0])
    x_r = np.array([0.2, 1.3])

    layered = layered_receiver_map(
        grid, model, om, receiver_x=x_r, receiver_iface=rcv_iface, plane_ifaces=planes, rows=(0, 1, 2)
    )
    whole = whole_space_receiver_map(
        grid, ref, om, receiver_x=x_r, receiver_dz=float(rcv_iface - planes[0]), rows=(0, 1, 2)
    )

    err = np.abs(layered.matrix - whole.matrix).max() / np.abs(whole.matrix).max()
    assert err < 1e-9, f"layered receiver map does not reduce: {err:.3e}"


def test_gradient_matches_finite_difference_in_a_stratified_background() -> None:
    """End to end in the fast-slab crust, receiver INSIDE the slab (cross-material).

    Stratified forward and adjoint caches, a layered receiver map, and per-plane
    media for T0. The misfit is differenced by brute force, so the check holds
    for the discrete model whatever the quadrature accuracy.
    """
    model = _stratified_model(slab=True)
    planes, rcv_iface, om = (7, 11), 9, 2 * np.pi * 0.3
    refs = [_plane_ref(model, p) for p in planes]
    background = LayeredBackground(model=model, plane_ifaces=planes)
    caches = []
    for ky in (0.3, -0.3):
        grid = make_sweep_grid(2, 3, 1.0, ky=ky, n_kz=64, n_kx=64, kx_max=3.0)
        caches.append(build_g0_cache(grid, refs[0], om, background=background))
    cache_p, cache_m = caches
    receiver = layered_receiver_map(
        cache_p.grid,
        model,
        om,
        receiver_x=np.array([0.4, 1.6]),
        receiver_iface=rcv_iface,
        plane_ifaces=planes,
        rows=(0, 1, 2),
    )
    rng = np.random.default_rng(9)
    psi_inc = rng.standard_normal((2, 3, 9)) + 1j * rng.standard_normal((2, 3, 9))
    a = 0.5
    step = np.array([1e-2, 1e-2, 1e-3])

    def chi_grad(m):
        t0, dt0 = rayleigh_t0_and_derivative(m, refs, om, a, step=step)
        return t0, dt0

    true_m = np.array([2.0, 1.0, 0.1]) * (1.0 + 0.4 * rng.standard_normal((2, 3, 3)))
    t0, dt0 = chi_grad(true_m)
    zero = np.zeros(receiver.n_data, complex)
    d_obs = misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, zero, **SOLVE).data

    model_m = np.array([2.0, 1.0, 0.1]) * (1.0 + 0.4 * rng.standard_normal((2, 3, 3)))
    t0, dt0 = chi_grad(model_m)
    res = misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, d_obs, **SOLVE)

    for iz, ix, p in ((0, 1, 0), (1, 2, 1), (1, 0, 2)):
        h = 1e-4 * step[p] / 1e-2
        chis = []
        for sgn in (1, -1):
            m = model_m.copy()
            m[iz, ix, p] += sgn * h
            t0_s, dt0_s = chi_grad(m)
            res_s = misfit_and_gradient(cache_p, cache_m, t0_s, dt0_s, psi_inc, receiver, d_obs, **SOLVE)
            chis.append(res_s.misfit)
        fd = (chis[0] - chis[1]) / (2 * h)
        err = abs(res.grad[iz, ix, p] - fd) / abs(fd)
        assert err < 1e-6, f"voxel ({iz},{ix}) param {p}: adjoint {res.grad[iz, ix, p]:.6e} fd {fd:.6e}"


def test_receiver_on_a_voxel_plane_interface_is_rejected() -> None:
    """The layered map refuses a receiver line on a voxel plane, with the diagnostic fields."""
    model = _stratified_model(slab=False)
    grid = make_sweep_grid(2, 3, 1.0, ky=0.3, n_kz=32, n_kx=32, kx_max=3.0)
    with pytest.raises(ValueError) as exc:
        layered_receiver_map(
            grid, model, 1.0, receiver_x=np.array([0.1]), receiver_iface=18, plane_ifaces=(18, 19), rows=[0]
        )
    msg = str(exc.value)
    for field in ("Where:", "Valid:", "Fix:"):
        assert field in msg
