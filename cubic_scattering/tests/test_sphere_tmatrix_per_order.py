"""The per-order Mie T-matrix, checked order by order at every order a lattice needs.

WHY PER ORDER.  The isolated sphere's far field is dominated by the low orders,
so it cannot vouch for the high ones; a lattice (layer-KKR) solve multiplies the
T-matrix by structure constants that grow like h_q, and would amplify any error
there.  ``mie_tmatrix_psv`` and ``mie_tmatrix_sh`` expose the T-matrix per order
so that it can be checked directly.

THE DIRECT 4x4 IS ACCURATE DESPITE ITS CONDITION NUMBER.  It solves continuity
of (u_r, u_theta, sigma_rr, sigma_rtheta) with the scattered and interior
amplitudes as unknowns; its condition number is 4.5e9 at n = 1 and 2.1e33 at
n = 12 at k_S a = 2.4.  That is ill-SCALING -- j_n inside against h_n outside,
displacement rows against traction rows -- and partial pivoting is insensitive
to it.  An interior-impedance (2x2) elimination was built as the alternative and
measured: it was no more accurate at any order up to 80 or at k_S a up to 10.

THE ARBITERS:
  * ``Mathematica/MieTmatrixReference.wl`` -- the 4x4 at fifty digits, with
    the stresses derived symbolically from the potentials rather than taken
    from the Python's hand-simplified radial formulas;
  * flux unitarity of the P-SV S-matrix S = W (I + 2T) W^-1,
    W = diag(sqrt(alpha), sqrt(beta n(n+1))), per order, for a lossless sphere;
  * reciprocity, S_PS = S_SP in that normalisation (read off the arbiter, where
    it holds exactly);
  * zero contrast gives zero scattering.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_sphere_tmatrix_per_order.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.sphere_scattering import compute_elastic_mie, mie_tmatrix_psv, mie_tmatrix_sh

REF_JSON = Path(__file__).resolve().parents[2] / "Mathematica" / "MieTmatrixReference.json"


def _scaled_sphere(
    alpha: float, beta: float, rho: float, s: float
) -> tuple[ReferenceMedium, MaterialContrast]:
    """Background and contrast for alpha, beta, rho all scaled by s inside.

    Args:
        alpha: Background P speed.
        beta: Background S speed.
        rho: Background density.
        s: Scale factor inside the sphere.

    Returns:
        (reference medium, contrast).
    """
    ref = ReferenceMedium(alpha, beta, rho)
    return ref, MaterialContrast(
        Dlambda=(s**3 - 1.0) * ref.lam, Dmu=(s**3 - 1.0) * ref.mu, Drho=(s - 1.0) * ref.rho
    )


GATE = dict(omega=60.0, radius=120.0, alpha=5000.0, beta=3000.0, rho=2500.0, s=1.1)
SOFT = dict(omega=60.0, radius=120.0, alpha=5000.0, beta=3000.0, rho=2500.0, s=0.7)


def _flux_s(tm: np.ndarray, n: int, ref: ReferenceMedium) -> np.ndarray:
    """The flux-normalised P-SV S-matrix of one order.

    Args:
        tm: 2x2 T-matrix, potential coefficients (P, S).
        n: Order.
        ref: Background medium.

    Returns:
        2x2 S-matrix.
    """
    w = np.diag([np.sqrt(ref.alpha), np.sqrt(ref.beta * n * (n + 1))])
    return w @ (np.eye(2) + 2.0 * tm) @ np.linalg.inv(w)


@pytest.fixture(scope="module")
def arbiter() -> dict:
    assert REF_JSON.exists(), f"missing {REF_JSON}: run Mathematica/MieTmatrixReference.wl"
    return json.loads(REF_JSON.read_text())


def test_matches_the_fifty_digit_arbiter_at_every_order(arbiter: dict) -> None:
    """Every entry of every order, n = 0..25, for three spheres.

    ⚠ RELATIVE PER ENTRY IS THE WRONG MEASURE FOR AN ENTRY NEAR ZERO.  The soft
    sphere's SH n = 1 element is 4.5e-5 -- that partial wave barely scatters --
    and it carries an absolute error of 6.8e-15, round-off on the scale of its
    S-matrix element 1 + 2T, which unitarity pins at modulus one.  Relative to
    itself that is 1.5e-10, and no formulation removes it: the inputs are right
    to 1e-15 and the cancellation is in the physics.  So an entry passes if it
    is right RELATIVELY (which the tiny high orders need) or right on the
    S-matrix's own O(1) scale -- in the flux normalisation, where both
    polarisations share that scale.
    """
    rtol, atol = 1e-10, 1e-13
    worst = 0.0
    for key, par in arbiter["sets"].items():
        ref, con = _scaled_sphere(par["alpha"], par["beta"], par["rho"], par["scale"])
        for row in arbiter["tmatrices"][key]:
            n = row["n"]
            want = np.array([[complex(*e) for e in r] for r in row["Tpsv"]])
            got = mie_tmatrix_psv(n, par["omega"], par["radius"], ref, con)
            nn1 = max(n * (n + 1), 1)
            w = np.array([np.sqrt(ref.alpha), np.sqrt(ref.beta * nn1)])
            for i in range(2):
                for j in range(2):
                    if abs(want[i, j]) > 0.0:
                        err = abs(got[i, j] - want[i, j])
                        flux_err = 2.0 * err * w[i] / w[j]
                        worst = max(worst, min(err / abs(want[i, j]) / rtol, flux_err / atol))
            if n >= 1:
                want_sh = complex(*row["Tsh"])
                err = abs(mie_tmatrix_sh(n, par["omega"], par["radius"], ref, con) - want_sh)
                worst = max(worst, min(err / abs(want_sh) / rtol, 2.0 * err / atol))
    assert worst < 1.0, f"worst entry exceeds both tolerances by a factor {worst:.3e}"


@pytest.mark.parametrize("par", [GATE, SOFT], ids=["gate", "soft"])
def test_psv_is_unitary_in_flux_normalisation_at_every_order(par: dict) -> None:
    ref, con = _scaled_sphere(par["alpha"], par["beta"], par["rho"], par["s"])
    worst = 0.0
    for n in range(1, 31):
        s = _flux_s(mie_tmatrix_psv(n, par["omega"], par["radius"], ref, con), n, ref)
        worst = max(worst, float(np.linalg.norm(s.conj().T @ s - np.eye(2))))
    assert worst < 1e-12, f"worst unitarity defect over n = 1..30: {worst:.3e}"


@pytest.mark.parametrize("par", [GATE, SOFT], ids=["gate", "soft"])
def test_psv_is_reciprocal_at_every_order(par: dict) -> None:
    ref, con = _scaled_sphere(par["alpha"], par["beta"], par["rho"], par["s"])
    worst = 0.0
    for n in range(1, 31):
        s = _flux_s(mie_tmatrix_psv(n, par["omega"], par["radius"], ref, con), n, ref)
        worst = max(worst, abs(s[0, 1] - s[1, 0]) / max(abs(s[0, 1]), 1e-300))
    assert worst < 1e-10, f"worst relative asymmetry S_PS vs S_SP: {worst:.3e}"


@pytest.mark.parametrize("par", [GATE, SOFT], ids=["gate", "soft"])
def test_sh_is_unimodular_at_every_order(par: dict) -> None:
    ref, con = _scaled_sphere(par["alpha"], par["beta"], par["rho"], par["s"])
    worst = max(
        abs(abs(1.0 + 2.0 * mie_tmatrix_sh(n, par["omega"], par["radius"], ref, con)) - 1.0)
        for n in range(1, 31)
    )
    assert worst < 1e-13, f"worst SH defect: {worst:.3e}"


def test_zero_contrast_scatters_nothing() -> None:
    ref, con = _scaled_sphere(5000.0, 3000.0, 2500.0, 1.0)
    worst = max(
        max(
            float(np.max(np.abs(mie_tmatrix_psv(n, 60.0, 120.0, ref, con)))),
            abs(mie_tmatrix_sh(n, 60.0, 120.0, ref, con)) if n >= 1 else 0.0,
        )
        for n in range(0, 26)
    )
    assert worst < 1e-14, f"zero contrast scattered {worst:.3e}"


def test_compute_elastic_mie_is_the_impedance_t_matrix_times_the_incident_coefficient() -> None:
    """The public coefficients are T times the plane wave's own expansion coefficient."""
    par = GATE
    ref, con = _scaled_sphere(par["alpha"], par["beta"], par["rho"], par["s"])
    mie = compute_elastic_mie(par["omega"], par["radius"], ref, con)
    kp, ks = par["omega"] / ref.alpha, par["omega"] / ref.beta
    for n in range(0, mie.n_max + 1):
        tm = mie_tmatrix_psv(n, par["omega"], par["radius"], ref, con)
        cp = (2 * n + 1) * 1j**n / (1j * kp)
        cs = (2 * n + 1) * 1j**n / (1j * ks)
        assert mie.a_n[n] == pytest.approx(tm[0, 0] * cp, rel=1e-14, abs=1e-300)
        if n >= 1:
            assert mie.b_n[n] == pytest.approx(tm[1, 0] * cp, rel=1e-14, abs=1e-300)
            assert mie.a_n_sv[n] == pytest.approx(tm[0, 1] * cs, rel=1e-14, abs=1e-300)
            assert mie.b_n_sv[n] == pytest.approx(tm[1, 1] * cs, rel=1e-14, abs=1e-300)
            t_sh = mie_tmatrix_sh(n, par["omega"], par["radius"], ref, con)
            assert mie.c_n[n] == pytest.approx(t_sh * cs, rel=1e-14, abs=1e-300)
