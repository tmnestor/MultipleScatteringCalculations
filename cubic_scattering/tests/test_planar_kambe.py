"""Planar lattice structure constants D[q, s] to high order, for the layer-KKR solve.

WHY A NEW ROUTE.  The Phase-3b chain (``Mathematica/IntraPlaneKambe*.wl``,
``tests/test_intraplane_kambe.py``) projects the Ewald lattice field onto
spherical harmonics on a SMALL sphere, rho0 = a_L/4, with 16 x 32 quadrature,
and checks to 1e-4.  That is enough at N_max = 2 but not at q ~ 20: dividing by
j_q(kappa rho0) there loses everything, and the 1e-5 energy floor of that chain
is this quadrature.  The field of the R != 0 images is regular for r < a_L, so
the projection sphere can be taken at rho0 = 0.7 a_L, where D_q j_q(kappa rho0)
behaves like (rho0/a_L)^q and nothing is lost.

THE CHECKS, each against something the construction cannot fake:
  * the Phase-3b Mathematica dump, to its own (quadrature-limited) accuracy;
  * at DAMPED kappa, the projection of the plain direct lattice sum, which uses
    no Ewald split at all;
  * independence of the projection radius, to q = 20 -- the check that fails
    first if digits are being lost;
  * independence of the Ewald splitting parameter.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_planar_kambe.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.planar_ewald import direct_sum
from cubic_scattering.planar_kambe import lattice_field, structure_constants

REF = Path(__file__).resolve().parents[2] / "Mathematica" / "IntraPlaneKambe_reference.json"

#: The target: a 600 m square lattice at the march gate's frequency, normal incidence.
L_TARGET = 600.0
KP_TARGET, KS_TARGET = 60.0 / 5000.0, 60.0 / 3000.0
QMAX = 20


def _shell_relative(got: dict, want: dict) -> float:
    """Worst |got - want| relative to the largest |want| in the neighbouring q-shells.

    Per-entry relative error is the wrong measure, because symmetry makes
    whole families of entries vanish: on a square lattice at normal incidence
    D[q, s] is zero unless s is a multiple of 4, and at k_par = 0 the lattice
    is inversion-symmetric, so every ODD q-shell vanishes entirely.  Those are
    round-off with no scale of their own.  The largest entry among shells
    q-1, q, q+1 is the magnitude any use of shell q sees.

    Args:
        got: {(q, s): value}.
        want: {(q, s): value}.

    Returns:
        The worst shell-relative difference.
    """
    shell_max = {q: max(abs(v) for (qq, _), v in want.items() if qq == q) for q in {k[0] for k in want}}
    worst = 0.0
    for q in sorted(shell_max):
        scale = max(shell_max.get(qq, 0.0) for qq in (q - 1, q, q + 1))
        keys = [k for k in want if k[0] == q]
        worst = max(worst, max(abs(got[k] - want[k]) for k in keys) / scale)
    return worst


def test_lattice_field_matches_the_scalar_ewald_it_vectorises() -> None:
    """The vectorised field is the existing validated scalar routine, point by point."""
    from cubic_scattering.planar_ewald import ewald_total

    rng = np.random.default_rng(3)
    pts = rng.normal(size=(12, 3)) * 0.3 * L_TARGET
    kpar = np.array([0.0, 0.0])
    eta = 2.0 / L_TARGET
    got = lattice_field(KS_TARGET, pts, L_TARGET, kpar, eta)
    want = np.array([ewald_total(KS_TARGET, p, eta, 6, 6, L_TARGET, kpar) for p in pts])
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-12


def test_matches_the_phase3b_mathematica_dump() -> None:
    """To the dump's own accuracy, which its small projection sphere limits to ~1e-5."""
    par = json.loads(REF.read_text())
    p = par["params"]
    d = structure_constants(p["kappa"], 3, p["aL"], np.array([p["kx"], p["ky"]]))
    worst = max(abs(d[e["q"], e["s"]] - complex(*e["val"])) for e in par["Dstruct"] if e["q"] <= 3)
    assert worst < 1e-4, f"worst |D - Mathematica| = {worst:.2e}"


def test_damped_matches_the_direct_lattice_sum_without_ewald() -> None:
    """At Im(kappa) > 0 the plain sum converges; project it and compare, to q = 12."""
    a_l, kpar = 2.0, np.array([0.2, 0.1])
    kappa = 1.5 + 0.35j
    d_ewald = structure_constants(kappa, 12, a_l, kpar)
    d_direct = structure_constants(
        kappa, 12, a_l, kpar, field=lambda pts: np.array([direct_sum(kappa, p, 60, a_l, kpar) for p in pts])
    )
    worst = _shell_relative(d_ewald, d_direct)
    assert worst < 1e-8, f"worst |Ewald - direct|, relative to the q-shell = {worst:.2e}"


@pytest.mark.parametrize("kappa", [KP_TARGET, KS_TARGET], ids=["kP", "kS"])
def test_independent_of_the_projection_radius_to_q20(kappa: float) -> None:
    kpar = np.array([0.0, 0.0])
    d1 = structure_constants(kappa, QMAX, L_TARGET, kpar, rho_frac=0.6)
    d2 = structure_constants(kappa, QMAX, L_TARGET, kpar, rho_frac=0.75)
    worst = _shell_relative(d1, d2)
    assert worst < 1e-8, f"worst radius dependence, relative to the q-shell = {worst:.2e}"


@pytest.mark.parametrize("kappa", [KP_TARGET, KS_TARGET], ids=["kP", "kS"])
def test_independent_of_the_ewald_parameter(kappa: float) -> None:
    kpar = np.array([0.0, 0.0])
    # Both choices must be PRECISE for the comparison to test the split: the
    # halves carry e^{kappa^2/(4 eta^2)} and cancel, so at eta a_L = 1.5 and
    # kappa a_L = 12 that is e^16 of round-off amplification (measured: 6e-8),
    # which tests the arithmetic, not the construction.  At 3 and 4.5 it is e^4
    # and e^1.8.
    d1 = structure_constants(kappa, QMAX, L_TARGET, kpar, eta=3.0 / L_TARGET)
    d2 = structure_constants(kappa, QMAX, L_TARGET, kpar, eta=4.5 / L_TARGET)
    worst = _shell_relative(d1, d2)
    assert worst < 1e-8, f"worst eta dependence, relative to the q-shell = {worst:.2e}"
