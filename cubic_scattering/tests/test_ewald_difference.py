"""The Ewald lattice sums of D = g_S - g_P, formed without subtracting the per-mode sums.

The per-mode sums are each accurate to round-off, but D is smaller than either by about (kappa a)^2, so
their difference loses digits as eps / (kappa a)^2; in the strain-strain block the measured loss is
2e-12 at kappa_S a = 0.03 and 2e-9 at 1e-3. At k_par = 0 the per-mode sums are worse still, by the
(eta / kappa)^(n-1) cancellation of the q = 0 order (7e-5 in the strain block at kappa_S a = 1e-3).
The difference form evaluates each Ewald term's mode difference as one quantity, and takes the q = 0
order's plane-wave pole out analytically (theory: docs/ewald-mode-difference.md).

Measured floor of the difference form: an eta spread of 4e-12 at kappa_S a = 1e-3 and 3e-13 at 0.03,
set by the accuracy of the Faddeeva function rather than by any subtraction; the thresholds below sit
above that floor and far below the per-mode figures.

The checks share nothing with the construction:
- the Ewald split is exact for every eta, so the sum must not depend on eta. The plain subtraction does,
  at the round-off level it amplifies, which ``test_plain_subtraction_is_eta_dependent`` confirms, so
  the eta check discriminates;
- where the subtraction is benign (kappa a = 0.5) the two must agree.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.lattice_kupradze import (
    bloch_block_ewald_9x9,
    lattice_difference_tensors,
    lattice_scalar_tensors,
    origin_difference_tensors,
    origin_scalar_tensors,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
A_L = 10.0
ETA0 = float(np.sqrt(np.pi) / A_L)
ETAS = [0.75 * ETA0, ETA0, 1.3 * ETA0]
CUT = 6
KPARS = [np.array([0.0, 0.0]), np.array([0.2 * np.pi / A_L, 0.07 * np.pi / A_L])]
R_OFF = np.array([0.0, 0.31 * A_L, -0.22 * A_L])  # in plane, off-site


def _k(ka: float) -> tuple[float, float]:
    ks = ka / A_L
    return ks * REF.beta / REF.alpha, ks


def _spread(results: list[list[np.ndarray]]) -> float:
    """Largest relative spread across eta, over every derivative order that is not identically zero."""
    # Orders carry different powers of length, so compare them as dimensionless (times a^n). Orders that
    # vanish by symmetry (the odd ones at the origin) hold round-off only and are skipped.
    scales = [max(float(np.max(np.abs(r[n]))) for r in results) * A_L**n for n in range(len(results[0]))]
    worst = 0.0
    for n, scale in enumerate(scales):
        if scale <= 1e-12 * max(scales):
            continue
        for r in results[1:]:
            worst = max(worst, float(np.max(np.abs(r[n] - results[0][n]))) * A_L**n / scale)
    return worst


def _difference_cases():
    return [
        ("origin", lambda kp, ks, eta, kpar: origin_difference_tensors(kp, ks, eta, CUT, CUT, A_L, kpar)),
        (
            "lattice",
            lambda kp, ks, eta, kpar: lattice_difference_tensors(R_OFF, kp, ks, eta, CUT, CUT, A_L, kpar),
        ),
    ]


def _plain_cases():
    return [
        (
            "origin",
            lambda kp, ks, eta, kpar: [
                s - p
                for s, p in zip(
                    origin_scalar_tensors(ks, eta, CUT, CUT, A_L, kpar),
                    origin_scalar_tensors(kp, eta, CUT, CUT, A_L, kpar),
                    strict=True,
                )
            ],
        ),
    ]


@pytest.mark.parametrize("kpar_index", [0, 1])
@pytest.mark.parametrize("ka", [1e-3, 1e-2, 0.03])
@pytest.mark.parametrize("case", range(2))
def test_difference_sum_is_eta_independent(case: int, ka: float, kpar_index: int) -> None:
    name, fn = _difference_cases()[case]
    kp, ks = _k(ka)
    results = [fn(kp, ks, eta, KPARS[kpar_index]) for eta in ETAS]
    assert _spread(results) < 1e-11, f"{name}, kappa_S a = {ka}: eta spread {_spread(results):.2e}"


def test_plain_subtraction_is_eta_dependent() -> None:
    """The discrimination check: at kappa_S a = 1e-3 the subtraction's noise moves with eta."""
    _, fn = _plain_cases()[0]
    kp, ks = _k(1e-3)
    results = [fn(kp, ks, eta, KPARS[1]) for eta in ETAS]
    assert _spread(results) > 1e-11


@pytest.mark.parametrize("kpar_index", [0, 1])
def test_difference_matches_subtraction_where_benign(kpar_index: int) -> None:
    kp, ks = _k(0.5)
    kpar = KPARS[kpar_index]
    pairs = [
        (
            origin_difference_tensors(kp, ks, ETA0, CUT, CUT, A_L, kpar),
            [
                s - p
                for s, p in zip(
                    origin_scalar_tensors(ks, ETA0, CUT, CUT, A_L, kpar),
                    origin_scalar_tensors(kp, ETA0, CUT, CUT, A_L, kpar),
                    strict=True,
                )
            ],
        ),
        (
            lattice_difference_tensors(R_OFF, kp, ks, ETA0, CUT, CUT, A_L, kpar),
            [
                s - p
                for s, p in zip(
                    lattice_scalar_tensors(R_OFF, ks, ETA0, CUT, CUT, A_L, kpar),
                    lattice_scalar_tensors(R_OFF, kp, ETA0, CUT, CUT, A_L, kpar),
                    strict=True,
                )
            ],
        ),
    ]
    # A wrong formula would differ at order one (or at O(kappa^2)); the plain subtraction itself loses
    # up to 1e-12 in the fourth order here, so that is the bar.
    for got, plain in pairs:
        assert _spread([plain, got]) < 2e-12


@pytest.mark.parametrize("kpar_index", [0, 1])
@pytest.mark.parametrize("ka", [1e-3, 0.03])
def test_ewald_block_is_eta_independent(ka: float, kpar_index: int) -> None:
    """The assembled 9x9 at dz = 0, the strain block included; k_par = 0 is in every Bloch grid."""
    omega = ka * REF.beta / A_L
    kpar = KPARS[kpar_index]
    blocks = [bloch_block_ewald_9x9(kpar, 0.0, A_L, omega, REF, eta=eta, cutoff=CUT) for eta in ETAS]
    # Per sub-block, made dimensionless (G, C and S carry 1, 1/a and 1/a^2): the strain block is the one
    # that suffered, and it is far smaller than G. A sub-block that vanishes by symmetry (C at k_par = 0)
    # holds round-off only and is skipped.
    parts = [
        ((slice(0, 3), slice(0, 3)), 1.0),
        ((slice(0, 3), slice(3, 9)), A_L),
        ((slice(3, 9), slice(3, 9)), A_L**2),
    ]
    scales = [float(np.max(np.abs(blocks[1][rc]))) * w for rc, w in parts]
    worst = 0.0
    for (rc, w), scale in zip(parts, scales, strict=True):
        if scale <= 1e-12 * max(scales):
            continue
        worst = max(worst, max(float(np.max(np.abs(b[rc] - blocks[1][rc]))) for b in blocks) * w / scale)
    assert worst < 1e-11, f"kappa_S a = {ka}: block eta spread {worst:.2e}"
