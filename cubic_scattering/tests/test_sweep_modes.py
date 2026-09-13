"""Tests for the (u, eps) <-> P/SV/SH mode bridge.

Representation conversion is where this project's defects have actually lived:
all three defects resolved in the 9x9 wrapper work were conversion-convention
errors, and one survived months because a symmetry gate passed it. Hence this
module is isolated and gated hard.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_kernels import vertical_kernel_9x9
from cubic_scattering.sweep_modes import (
    ModeBasis,
    mode_basis,
    modes_to_state,
    state_to_modes,
    vertical_factorisation,
)

REF = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
OMEGA = 2 * np.pi * (1.0 + 0.03j)

# Oblique, normal-incidence, and a strongly evanescent case.
CASES = [(0.8, 0.4), (0.0, 0.0), (3.0, 1.5), (0.0, 0.9)]


@pytest.mark.parametrize(("kx", "ky"), CASES)
def test_mode_basis_is_orthonormal_and_on_shell(kx: float, ky: float) -> None:
    """Polarisations: P along k, SV and SH transverse; each mode on its own shell."""
    mb = mode_basis(kx, ky, OMEGA, REF)
    assert isinstance(mb, ModeBasis)

    kp2 = (OMEGA / REF.alpha) ** 2
    ks2 = (OMEGA / REF.beta) ** 2
    for m in range(6):
        k = mb.k_vectors[m]
        pol = mb.polarisations[m]
        assert abs(pol @ pol - 1.0) < 1e-12  # unit (complex-bilinear, not conjugate)
        shell = kp2 if m % 3 == 0 else ks2
        assert abs(k @ k - shell) / abs(shell) < 1e-12
        if m % 3 == 0:
            # P: polarisation parallel to k
            assert abs(abs(pol @ k) ** 2 - abs(k @ k)) / abs(k @ k) < 1e-10
        else:
            # SV, SH: transverse
            assert abs(pol @ k) / abs(np.sqrt(k @ k)) < 1e-10


@pytest.mark.parametrize(("kx", "ky"), CASES)
def test_sv_and_sh_are_mutually_orthogonal(kx: float, ky: float) -> None:
    mb = mode_basis(kx, ky, OMEGA, REF)
    for base in (0, 3):
        sv, sh = mb.polarisations[base + 1], mb.polarisations[base + 2]
        assert abs(sv @ sh) < 1e-12


@pytest.mark.parametrize(("kx", "ky"), CASES)
def test_mode_round_trip_is_a_rank_six_projector(kx: float, ky: float) -> None:
    """RUNG 6a: exact on modes; a PROJECTOR, not the identity, on the 9-state.

    The nine-component state has more components than the six modes -- three
    combinations are fixed by the equations of motion -- so the round trip
    projects. Asserting identity on R^9 would be wrong and would send an
    implementer hunting a defect that is not there.
    """
    to_s = modes_to_state(kx, ky, OMEGA, REF)
    to_m = state_to_modes(kx, ky, OMEGA, REF)
    assert to_s.shape == (9, 6)
    assert to_m.shape == (6, 9)

    assert np.abs(to_m @ to_s - np.eye(6)).max() < 1e-10
    proj = to_s @ to_m
    assert np.abs(proj @ proj - proj).max() < 1e-10
    assert np.linalg.matrix_rank(proj, tol=1e-8) == 6


@pytest.mark.parametrize(("kx", "ky"), CASES)
def test_up_and_down_modes_differ(kx: float, ky: float) -> None:
    """Guard against a basis that silently uses one direction for both halves."""
    mb = mode_basis(kx, ky, OMEGA, REF)
    for m in range(3):
        assert abs(mb.k_vectors[m][0] + mb.k_vectors[m + 3][0]) < 1e-12
        assert abs(mb.k_vectors[m][0]) > 1e-9


@pytest.mark.parametrize(("kx", "ky"), CASES)
@pytest.mark.parametrize("dz", [0.25, -0.5])
def test_factorisation_fitted_at_one_dz_predicts_every_other(kx: float, ky: float, dz: float) -> None:
    """RUNG 6b: the whole-space kernel is D diag(phase) S, exactly.

    The source side is fitted at ONE separation and then used to predict the
    kernel at four others. A wrong mode embedding cannot survive this: it would
    reproduce the fitted separation by construction and fail the rest.
    """
    fac = vertical_factorisation(kx, ky, np.sign(dz) * 0.25, OMEGA, REF)

    for scale in (1.0, 2.0, 3.0, 7.0):
        sep = np.sign(dz) * 0.25 * scale
        want = vertical_kernel_9x9(np.array([kx]), ky, sep, OMEGA, REF)[:, :, 0]
        got = fac.evaluate(sep)
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-11


@pytest.mark.parametrize(("kx", "ky"), CASES)
def test_factorisation_uses_only_the_propagating_half(kx: float, ky: float) -> None:
    """For dz > 0 only the down-going modes carry amplitude, and vice versa."""
    down = vertical_factorisation(kx, ky, 0.25, OMEGA, REF)
    up = vertical_factorisation(kx, ky, -0.25, OMEGA, REF)
    assert np.abs(down.source[3:]).max() == 0.0
    assert np.abs(up.source[:3]).max() == 0.0
    assert np.abs(down.source[:3]).max() > 0.0
    assert np.abs(up.source[3:]).max() > 0.0


@pytest.mark.parametrize(("kx", "ky"), CASES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_both_halves_decay_with_separation(kx: float, ky: float, sign: float) -> None:
    """REGRESSION: the up-going half must decay too.

    The propagation wavenumber is on the Im >= 0 branch for ALL six modes, so
    e^{+i kz |dz|} decays in both directions. Storing the SIGNED z-component
    here instead makes the up-going half GROW with separation -- which leaves
    every dz > 0 case exact and silently breaks every dz < 0 one.
    """
    fac = vertical_factorisation(kx, ky, sign * 0.25, OMEGA, REF)
    assert np.all(np.imag(fac.kz) >= -1e-15)

    mags = [np.abs(fac.evaluate(sign * d)).max() for d in (0.25, 0.5, 1.0, 2.0)]
    assert mags == sorted(mags, reverse=True), f"not decaying: {mags}"


def test_evaluate_rejects_the_wrong_direction() -> None:
    """A factorisation is per-direction; the two halves radiate differently."""
    fac = vertical_factorisation(0.8, 0.4, 0.25, OMEGA, REF)
    with pytest.raises(ValueError, match="does not match"):
        fac.evaluate(-0.25)
    with pytest.raises(ValueError, match="does not match"):
        fac.evaluate(0.0)


def test_conditioning_is_sane_in_seismic_units() -> None:
    """Seismic units, not SI.

    In metres and pascals the mode matrix looks ill-conditioned at ~1e10, which
    is a units artefact scaling as rho*omega*v, not a defect -- and the wrong
    response to it is regularisation. In km/s and g/cm3 it is unremarkable.
    """
    for kx, ky in CASES:
        cond = np.linalg.cond(modes_to_state(kx, ky, OMEGA, REF))
        assert cond < 1e4, f"cond={cond:.3e} at (kx, ky)=({kx}, {ky})"
