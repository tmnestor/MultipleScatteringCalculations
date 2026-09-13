"""Tests for the Cartesian directional sweeps."""

import numpy as np
import pytest

from cubic_scattering.directional_sweeps import (
    LayeredBackground,
    apply_g0,
    build_g0_cache,
    build_vertical_stack,
    build_vertical_stack_layered,
    make_sweep_grid,
    sweep_x,
    sweep_y,
    sweep_z,
)
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_kernels import lateral_split_9x9, same_depth_kernel_9x9

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)
PITCH = 0.25


def _splits(grid):
    """Both lateral splits for a grid."""
    right = lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, OMEGA, REF, direction="right")
    left = lateral_split_9x9(grid.ky, grid.kz_nodes, grid.pitch, OMEGA, REF, direction="left")
    return right, left


def _pairwise_lateral(sources, grid, split_right, split_left):
    """O(N^2) reference: sum the split kernel over every ordered pair.

    Deliberately a different algorithm from the sweep -- a double loop over
    pairs, not a running accumulation -- so agreement is evidence.
    """
    n_z, n_x, _ = sources.shape
    out = np.zeros_like(sources)
    w = grid.kz_weights
    for i in range(n_x):
        for j in range(n_x):
            if i == j:
                continue
            n = abs(i - j)
            split = split_right if j < i else split_left
            kern = np.einsum("abk,k->abk", split.amp_p, split.phase_p**n) + np.einsum(
                "abk,k->abk", split.amp_s, split.phase_s**n
            )
            block = np.einsum("k,abk->ab", w, kern)
            out[:, i, :] += sources[:, j, :] @ block.T
    return out


def test_sweep_x_equals_pairwise_sum_distinct_sources() -> None:
    """RUNG 2: the running sweep resums the pairwise double sum exactly.

    DISTINCT source at every site -- that is the disorder-resolved property
    being claimed. See test_uniform_source_control_would_be_vacuous.
    """
    rng = np.random.default_rng(20260913)
    n_z, n_x = 2, 9
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=512)
    right, left = _splits(grid)

    sources = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    got = sweep_x(sources, grid, right, left)
    want = _pairwise_lateral(sources, grid, right, left)
    # 1e-12, not 1e-14: both paths sum n_kz quadrature terms in different orders,
    # and the residual grows as sqrt(n_kz) (measured: 4e-14 at n_kz=64 rising to
    # 2e-13 at n_kz=2048) while staying FLAT in n_x. That is round-off in the
    # quadrature, not error in the resummation -- which is what
    # test_residual_is_flat_in_lattice_size pins down.
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_residual_is_flat_in_lattice_size() -> None:
    """RUNG 2, the discriminating half: the residual must not grow with n_x.

    The sweep replaces an O(n_x^2) pairwise sum by two O(n_x) recursions. If the
    recursion were wrong -- a factor accumulated that should be applied once, a
    term added before being read instead of after -- the discrepancy would grow
    with the number of sites swept. Holding n_kz fixed isolates that from the
    quadrature round-off.
    """
    rng = np.random.default_rng(99)
    residuals = []
    for n_x in (4, 8, 16, 32):
        grid = make_sweep_grid(1, n_x, PITCH, ky=0.6, n_kz=128)
        right, left = _splits(grid)
        sources = rng.standard_normal((1, n_x, 9)) + 1j * rng.standard_normal((1, n_x, 9))
        got = sweep_x(sources, grid, right, left)
        want = _pairwise_lateral(sources, grid, right, left)
        residuals.append(np.abs(got - want).max() / np.abs(want).max())

    # An eight-fold increase in lattice size must not move the residual by an
    # order of magnitude.
    assert max(residuals) / min(residuals) < 10.0
    assert max(residuals) < 1e-12


def test_uniform_source_control_would_be_vacuous() -> None:
    """MANDATORY CONTROL: show the uniform-source version cannot discriminate.

    An implementation that averaged the sites before sweeping still matches the
    pairwise sum when every source is identical. This test asserts that the weak
    version of rung 2 passes for a deliberately WRONG implementation, so a
    future edit cannot quietly downgrade the real test to the weak one.
    """
    n_z, n_x = 1, 6
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    uniform = np.ones((n_z, n_x, 9), dtype=complex)
    averaged = np.broadcast_to(uniform.mean(axis=1, keepdims=True), uniform.shape).copy()

    a = sweep_x(uniform, grid, right, left)
    b = sweep_x(averaged, grid, right, left)
    assert np.abs(a - b).max() == 0.0  # indistinguishable -- hence vacuous

    rng = np.random.default_rng(7)
    varied = rng.standard_normal((n_z, n_x, 9)) + 0j
    v_avg = np.broadcast_to(varied.mean(axis=1, keepdims=True), varied.shape).copy()
    c = sweep_x(varied, grid, right, left)
    d = sweep_x(v_avg, grid, right, left)
    assert np.abs(c - d).max() / np.abs(c).max() > 1e-2  # the real test discriminates


def test_self_term_is_never_formed() -> None:
    """A source at one site alone must produce no field AT that site."""
    n_z, n_x = 1, 5
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    sources = np.zeros((n_z, n_x, 9), dtype=complex)
    sources[0, 2, :] = 1.0
    out = sweep_x(sources, grid, right, left)
    assert np.abs(out[0, 2, :]).max() == 0.0
    assert np.abs(out[0, 1, :]).max() > 0.0  # neighbours DO see it
    assert np.abs(out[0, 3, :]).max() > 0.0


def test_lateral_coupling_decays_with_separation() -> None:
    """Sanity: a single source is felt less further away, in both directions."""
    n_z, n_x = 1, 7
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=256)
    right, left = _splits(grid)

    sources = np.zeros((n_z, n_x, 9), dtype=complex)
    sources[0, 3, :] = 1.0
    out = np.abs(sweep_x(sources, grid, right, left)[0]).max(axis=1)
    assert out[4] > out[5] > out[6]
    assert out[2] > out[1] > out[0]


def _vertical_block(grid, vertical, lz, mz, d_index):
    """K_{lz,mz}(d_index * pitch) as a 9x9, by explicit k_x quadrature."""
    ph = np.exp(1j * grid.kx_nodes * d_index * grid.pitch)
    return np.einsum("k,abk->ab", grid.kx_weights * ph, vertical[lz, mz])


def _pairwise_vertical_same_kernel(sources, grid, vertical):
    """O(N^2) reference: a double loop over inter-plane pairs.

    Uses the SAME kernel as sweep_z but a different algorithm -- pairwise rather
    than an accumulation in the k_x domain -- so agreement tests the transform
    bookkeeping, not the physics. The physics is rung 3, in the gate script.
    """
    n_z, n_x, _ = sources.shape
    out = np.zeros_like(sources)
    for lz in range(n_z):
        for mz in range(n_z):
            if lz == mz:
                continue
            for i in range(n_x):
                for j in range(n_x):
                    block = _vertical_block(grid, vertical, lz, mz, i - j)
                    out[lz, i, :] += block @ sources[mz, j, :]
    return out


def test_sweep_z_equals_the_pairwise_inter_plane_sum() -> None:
    """The k_x-domain accumulation resums the pairwise inter-plane sum."""
    rng = np.random.default_rng(4242)
    n_z, n_x = 3, 6
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=256)
    vertical = build_vertical_stack(grid, REF, OMEGA)

    sources = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))
    got = sweep_z(sources, grid, vertical)
    want = _pairwise_vertical_same_kernel(sources, grid, vertical)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-12


def test_sweep_z_does_not_wrap_around_the_lattice() -> None:
    """RUNG 3b: the lateral coupling must NOT be periodic.

    The real Earth is not horizontally periodic. A circular convolution -- which
    is what an unpadded FFT along x would give -- would couple site 0 to site
    n_x-1 as though they were ONE pitch apart rather than n_x-1 pitches. This
    test puts a lone source at site 0 and checks the field at the far edge
    against the kernel at the true separation, and separately confirms it is
    nowhere near the wrapped value.
    """
    n_z, n_x = 2, 8
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=512)
    vertical = build_vertical_stack(grid, REF, OMEGA)

    sources = np.zeros((n_z, n_x, 9), dtype=complex)
    sources[0, 0, 0] = 1.0
    out = sweep_z(sources, grid, vertical)

    far = out[1, n_x - 1, :]
    true_sep = _vertical_block(grid, vertical, 1, 0, n_x - 1) @ sources[0, 0, :]
    wrapped = _vertical_block(grid, vertical, 1, 0, -1) @ sources[0, 0, :]

    assert np.abs(far - true_sep).max() / np.abs(true_sep).max() < 1e-12
    # And the two are genuinely different, so the test above is not vacuous.
    assert np.abs(true_sep - wrapped).max() / np.abs(wrapped).max() > 1e-1


def test_sweep_z_excludes_the_same_plane() -> None:
    """Same-depth coupling belongs to sweep_x; sweep_z must not double-count it."""
    n_z, n_x = 2, 4
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=256)
    vertical = build_vertical_stack(grid, REF, OMEGA)
    sources = np.zeros((n_z, n_x, 9), dtype=complex)
    sources[0, :, :] = 1.0
    out = sweep_z(sources, grid, vertical)
    assert np.abs(out[0]).max() == 0.0
    assert np.abs(out[1]).max() > 0.0


def test_sweep_z_rejects_a_stack_from_another_grid() -> None:
    grid = make_sweep_grid(2, 4, PITCH, ky=0.6, n_kz=64, n_kx=256)
    other = make_sweep_grid(2, 4, PITCH, ky=0.6, n_kz=64, n_kx=128)
    vertical = build_vertical_stack(other, REF, OMEGA)
    with pytest.raises(ValueError, match="vertical"):
        sweep_z(np.zeros((2, 4, 9), dtype=complex), grid, vertical)


def test_g0_covers_every_off_diagonal_pair_exactly_once() -> None:
    """RUNG 4: the partition.

    Asserted on the SUPPORT of the operator -- which pairs light up -- not on
    magnitudes. It is the one rung in the ladder that an overall scale error
    cannot pass, because it does not look at values at all.
    """
    n_z, n_x = 3, 5
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=128)
    cache = build_g0_cache(grid, REF, OMEGA)

    for mz in range(n_z):
        for j in range(n_x):
            src = np.zeros((n_z, n_x, 9), dtype=complex)
            src[mz, j, 0] = 1.0
            out = apply_g0(src, cache)
            lit = np.abs(out).max(axis=2) > 0.0
            assert not lit[mz, j], f"self-term leaked at ({mz}, {j})"
            expected = np.ones((n_z, n_x), dtype=bool)
            expected[mz, j] = False
            np.testing.assert_array_equal(lit, expected)


def test_layered_g0_lights_the_self_site_but_whole_space_does_not() -> None:
    """The self-term is background-dependent, and that is physics, not a leak.

    In a whole space, T0 already closes the self-interaction, so G0 must not
    touch the source site -- that is the partition gate above. In a LAYERED
    background a wave can leave a voxel, reflect off a layer boundary and return
    to that same voxel; T0 is the whole-space T-matrix and does not contain that
    path, so G0 must supply it. A solver that zeroed the diagonal here would
    silently drop every layer-return-to-self.
    """
    grid = make_sweep_grid(1, 3, 1.0, ky=0.3, n_kz=32, n_kx=64, kx_max=24.0)
    omega = 2 * np.pi * 6.0
    lay = _uniform_layer_model(contrast_layers=(10, 11))
    s_p, s_s = lay.complex_slowness_p(), lay.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], lay.rho[1])

    src = np.zeros((1, 3, 9), dtype=complex)
    src[0, 1, 0] = 1.0

    cache_w = build_g0_cache(grid, ref, omega)
    cache_l = build_g0_cache(grid, ref, omega, background=LayeredBackground(model=lay, plane_ifaces=(8,)))
    assert np.abs(apply_g0(src, cache_w)[0, 1, :]).max() == 0.0
    assert np.abs(apply_g0(src, cache_l)[0, 1, :]).max() > 1e-9


def test_self_energy_is_equivalent_to_a_dressed_tmatrix() -> None:
    """The diagonal behaves as a genuine self-energy, checked by moving it.

    With G0 = G_off + D and D block-diagonal, the sources b must be identical
    whether the self-energy sits in the propagator or is absorbed into T:

        A:  b = T0 psi,   (I - (G_off + D) T0) psi = psi_inc
        B:  b = Td psi',  (I - G_off Td) psi' = psi_inc,  Td = T0 (I - D T0)^-1

    Compare b, NOT psi: the two fields differ by exactly the self-return. This
    is an exact identity, so it catches a self-energy applied on the wrong side,
    double-counted, or sign-flipped -- though it cannot confirm D's value.
    """
    grid = make_sweep_grid(1, 3, 1.0, ky=0.3, n_kz=32, n_kx=64, kx_max=24.0)
    omega = 2 * np.pi * 6.0
    lay = _uniform_layer_model(contrast_layers=(10, 11))
    s_p, s_s = lay.complex_slowness_p(), lay.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], lay.rho[1])
    cache = build_g0_cache(grid, ref, omega, background=LayeredBackground(model=lay, plane_ifaces=(8,)))

    size = 1 * 3 * 9
    g0 = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        g0[:, c] = apply_g0(e.reshape(1, 3, 9), cache).ravel()

    d_self = np.zeros_like(g0)
    for s in range(3):
        sl = slice(9 * s, 9 * s + 9)
        d_self[sl, sl] = g0[sl, sl]
    g_off = g0 - d_self

    rng = np.random.default_rng(20260913)
    t0 = np.zeros((size, size), dtype=complex)
    for s in range(3):
        sl = slice(9 * s, 9 * s + 9)
        t0[sl, sl] = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    psi_inc = rng.standard_normal(size) + 0j
    eye = np.eye(size)

    b_direct = t0 @ np.linalg.solve(eye - g0 @ t0, psi_inc)
    t_d = t0 @ np.linalg.inv(eye - d_self @ t0)
    b_dressed = t_d @ np.linalg.solve(eye - g_off @ t_d, psi_inc)
    rel_eq = np.abs(b_direct - b_dressed).max() / np.abs(b_direct).max()
    assert rel_eq < 1e-12

    # ...and the control: the identity must beat the term it moves, by a lot.
    b_dropped = t0 @ np.linalg.solve(eye - g_off @ t0, psi_inc)
    rel_drop = np.abs(b_direct - b_dropped).max() / np.abs(b_direct).max()
    assert rel_drop / max(rel_eq, 1e-300) > 1e6


def test_self_energy_vanishes_as_the_reflector_recedes() -> None:
    """A magnitude statement at solve level, not just on the kernel."""
    grid = make_sweep_grid(1, 3, 1.0, ky=0.3, n_kz=32, n_kx=64, kx_max=24.0)
    omega = 2 * np.pi * 6.0
    mags = []
    for layers in ((10, 11), (12, 13), (14, 15)):
        lay = _uniform_layer_model(contrast_layers=layers)
        stack = build_vertical_stack_layered(grid, LayeredBackground(model=lay, plane_ifaces=(8,)), omega)
        mags.append(float(np.abs(stack[0, 0]).max()))
    assert mags == sorted(mags, reverse=True), f"not receding: {mags}"
    assert mags[-1] < mags[0] * 1e-6


def test_g0_is_exactly_the_sum_of_its_two_sweeps() -> None:
    """No double counting: sweep_x is same-plane only, sweep_z different-plane."""
    rng = np.random.default_rng(11)
    n_z, n_x = 2, 6
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=128)
    cache = build_g0_cache(grid, REF, OMEGA)
    src = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    want = sweep_x(src, grid, cache.split_right, cache.split_left) + sweep_z(src, grid, cache.vertical)
    assert np.abs(apply_g0(src, cache) - want).max() == 0.0


def test_g0_is_linear() -> None:
    """G0 is a pure forward summation: no inversion, no state carried between calls."""
    rng = np.random.default_rng(13)
    n_z, n_x = 2, 5
    grid = make_sweep_grid(n_z, n_x, PITCH, ky=0.6, n_kz=64, n_kx=128)
    cache = build_g0_cache(grid, REF, OMEGA)
    a = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))
    b = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    lhs = apply_g0(3.0 * a - 2.0j * b, cache)
    rhs = 3.0 * apply_g0(a, cache) - 2.0j * apply_g0(b, cache)
    # Algebraically exact; numerically ~250 ULP, because both quadratures sum
    # summands much larger than their total (the 1/kx pole factors), so the sum
    # is ill-conditioned even though the result is not. Same 1e-12 floor as the
    # other sweep identities.
    assert np.abs(lhs - rhs).max() / np.abs(lhs).max() < 1e-12


def test_sweep_y_is_an_explicit_stage_two_refusal() -> None:
    """Omitting the in-out coupling silently would be far worse than failing."""
    grid = make_sweep_grid(2, 4, PITCH, ky=0.6, n_kz=64, n_kx=128)
    cache = build_g0_cache(grid, REF, OMEGA)
    with pytest.raises(NotImplementedError, match="stage 2"):
        sweep_y(np.zeros((2, 4, 9), dtype=complex), grid, cache)


def _uniform_layer_model(
    n_lay: int = 16,
    pitch: float = 1.0,
    q: float = 2.0,
    contrast_layers: tuple[int, ...] = (),
):
    """Ocean over n_lay identical crust layers; interface k at the bottom of layer k.

    Matches the model the wrapper resolution was validated on. n_lay must be
    large enough that the plane pair is not the half-space boundary -- there
    layered_greens_6x6 returns zeros, which reads as a 1.000 residual rather
    than as an error.
    """
    import sys

    sibling = "/Users/tod/Desktop/SeismicInversion"
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    pytest.importorskip("GlobalMatrix.layered_greens")
    lm = pytest.importorskip("Kennett_Reflectivity.layer_model")
    al, be, rh = 4.0, 2.22, 2.6
    alpha = [1.5, *([al] * n_lay), al]
    beta = [0.0, *([be] * n_lay), be]
    rho = [1.03, *([rh] * n_lay), rh]
    # A fast slab, placed so it is NOT adjacent to the plane used in the tests
    # (interface 8) -- a material jump on the plane itself makes the correction
    # operator K two-valued and is rightly refused.
    for lay in contrast_layers:
        alpha[lay], beta[lay], rho[lay] = 6.5, 3.7, 3.3
    return lm.LayerModel.from_arrays(
        alpha=alpha,
        beta=beta,
        rho=rho,
        thickness=[3.0, *([pitch] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def test_layered_vertical_stack_reduces_to_the_whole_space_stack() -> None:
    """RUNG 3L: the production vertical operator reduces to the arbiter.

    build_vertical_stack_layered is the thesis Ch.5 stratified propagator;
    build_vertical_stack is an INDEPENDENT construction (k_z residue of the
    whole-space Green's tensor, gated against the closed-form Kupradze
    propagator). On a uniform model with the planes deep enough that the free
    surface is attenuated, they must agree.
    """
    model = _uniform_layer_model()
    pitch = 1.0
    grid = make_sweep_grid(2, 3, pitch, ky=0.3, n_kz=32, n_kx=64, kx_max=3.0)
    omega = 2 * np.pi * 6.0

    background = LayeredBackground(model=model, plane_ifaces=(8, 9))
    got = build_vertical_stack_layered(grid, background, omega)

    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], model.rho[1])
    want = build_vertical_stack(grid, ref, omega)

    # Off-diagonal blocks reduce to the whole-space stack.
    scale = np.abs(want).max()
    for lz in range(2):
        for mz in range(2):
            if lz != mz:
                assert np.abs(got[lz, mz] - want[lz, mz]).max() / scale < 1e-13
    # The DIAGONAL now carries the same-plane layer reverberation. With a
    # uniform background there is nothing to reverberate off, so it must vanish.
    # Normalise against the SAME-DEPTH kernel, not the off-diagonal block: the
    # claim is that two O(1) quantities cancel, and measuring that against an
    # unrelated 0.011 scale would overstate the residual by two orders.
    s_p2, s_s2 = model.complex_slowness_p(), model.complex_slowness_s()
    ref2 = ReferenceMedium(1.0 / s_p2[1], 1.0 / s_s2[1], model.rho[1])
    direct = same_depth_kernel_9x9(grid.kx_nodes, grid.ky, omega, ref2)
    cancelled = np.abs(direct).max()
    assert cancelled > 0.1, "the cancelled term should be O(1), else this is vacuous"
    assert np.abs(got[0, 0]).max() / cancelled < 1e-14
    assert np.abs(got[1, 1]).max() / cancelled < 1e-14


def test_same_plane_reverberation_appears_only_with_layering() -> None:
    """The intra-plane gap: same-depth coupling must see the background.

    The lateral sweep supplies the direct whole-space term. What it cannot
    supply is the wave that leaves a voxel, reflects off a layer boundary and
    returns to the same plane. That is the diagonal of the vertical stack, and
    it must be zero without layering and non-zero with it.
    """
    pitch = 1.0
    grid = make_sweep_grid(1, 3, pitch, ky=0.3, n_kz=32, n_kx=64, kx_max=24.0)
    omega = 2 * np.pi * 6.0

    uni = _uniform_layer_model()
    flat = build_vertical_stack_layered(grid, LayeredBackground(model=uni, plane_ifaces=(8,)), omega)

    lay = _uniform_layer_model(contrast_layers=(10, 11))
    bumpy = build_vertical_stack_layered(grid, LayeredBackground(model=lay, plane_ifaces=(8,)), omega)

    assert np.abs(flat[0, 0]).max() < 1e-12
    assert np.abs(bumpy[0, 0]).max() > 1e-6


def test_same_plane_reverberation_decays_in_kx() -> None:
    """It must decay, or the k_x quadrature would not converge.

    Every reverberation path travels at least twice the distance to the nearest
    interface, so it carries e^{-kappa 2H}. Neither the layered kernel nor the
    whole-space one is integrable on its own -- both grow like |k_x| -- so this
    decay is what makes the split usable rather than merely tidy.
    """
    grid = make_sweep_grid(1, 3, 1.0, ky=0.3, n_kz=32, n_kx=256, kx_max=40.0)
    lay = _uniform_layer_model(contrast_layers=(10, 11))
    stack = build_vertical_stack_layered(
        grid, LayeredBackground(model=lay, plane_ifaces=(8,)), 2 * np.pi * 6.0
    )
    mag = np.abs(stack[0, 0]).max(axis=(0, 1))
    low = mag[np.abs(grid.kx_nodes) < 2.0].max()
    high = mag[np.abs(grid.kx_nodes) > 25.0].max()
    assert low > 1e-6
    assert high < low * 1e-6, f"not decaying: low={low:.3e} high={high:.3e}"


def test_layered_stack_rejects_a_wrong_length_plane_map() -> None:
    model = _uniform_layer_model()
    grid = make_sweep_grid(3, 3, 1.0, ky=0.3, n_kz=32, n_kx=32, kx_max=3.0)
    background = LayeredBackground(model=model, plane_ifaces=(8, 9))
    with pytest.raises(ValueError, match="plane_ifaces"):
        build_vertical_stack_layered(grid, background, 2 * np.pi * 6.0)


def test_grid_rejects_single_site_row() -> None:
    with pytest.raises(ValueError, match="n_x"):
        make_sweep_grid(2, 1, PITCH, ky=0.6)


def test_sweep_rejects_mismatched_pitch() -> None:
    grid = make_sweep_grid(1, 4, PITCH, ky=0.6, n_kz=64)
    wrong_r = lateral_split_9x9(grid.ky, grid.kz_nodes, 2 * PITCH, OMEGA, REF, direction="right")
    wrong_l = lateral_split_9x9(grid.ky, grid.kz_nodes, 2 * PITCH, OMEGA, REF, direction="left")
    with pytest.raises(ValueError, match="pitch"):
        sweep_x(np.zeros((1, 4, 9), dtype=complex), grid, wrong_r, wrong_l)


def test_sweep_rejects_two_splits_of_the_same_direction() -> None:
    """Passing the same split twice silently symmetrises the field."""
    grid = make_sweep_grid(1, 4, PITCH, ky=0.6, n_kz=64)
    right, _ = _splits(grid)
    with pytest.raises(ValueError, match="direction"):
        sweep_x(np.zeros((1, 4, 9), dtype=complex), grid, right, right)


def test_sweep_rejects_wrong_source_shape() -> None:
    grid = make_sweep_grid(1, 4, PITCH, ky=0.6, n_kz=64)
    right, left = _splits(grid)
    with pytest.raises(ValueError, match="sources"):
        sweep_x(np.zeros((1, 4, 6), dtype=complex), grid, right, left)
