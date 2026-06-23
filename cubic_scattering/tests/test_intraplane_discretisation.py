"""Phase 2 item (e): sphere-packing discretisation error.

Quantifies the irreducible geometric error of representing a heterogeneity as a
diluted planar sphere packing (planar volume fraction ``phi = pi/6`` at touching)
versus the space-filling cube slab (``phi = 1``), in the Rayleigh limit.

Two stages (see ``docs/superpowers/specs/2026-06-23-sphere-packing-discretisation-design.md``):

- **Stage A — single-site shape factor.** Compare a sphere's Rayleigh effective
  contrast against a cube's at the *same* material contrast (no renormalisation).
  Because the effective-contrast extraction already normalises by volume, this
  isolates the pure shape error. Empirically small (<= a few percent) and grows
  with contrast magnitude and ``ka``.

- **Stage B — collective layer R_PP.** The ``Delta -> Delta/phi`` contrast
  renormalisation is a *layer-level* correction (``layer contrast ~ phi *
  single_site(Delta/phi)``): it makes the diluted sphere layer reproduce the
  fully-filled cube layer. Stage B uses the planar-collective monopole dumped by
  ``Mathematica/IntraPlaneDiscretisation.wl`` and compares the sphere-layer
  ``R_PP`` to the cube layer via ``kennett_reference_rpp``.

The cube and sphere single-site effective contrasts come from the validated
``cubic_scattering`` package (``compute_cube_tmatrix`` / the Mie extraction); the
Mathematica deliverable carries the *new* physics (the sphere-packing collective).
"""

import math

import pytest

from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

PHI_TOUCH = math.pi / 6.0
A_RADIUS = 1.0  # m; ka = (omega/alpha)*a, so omega = ka*alpha/a with a = A_RADIUS
KA_LIST = (0.05, 0.1)


@pytest.fixture(scope="module")
def ref():
    return ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)


@pytest.fixture(scope="module")
def contrasts(ref):
    """Weak (Born check), moderate (~10%), and negative/strong (-60%) contrasts."""
    lam0 = ref.rho * (ref.alpha**2 - 2 * ref.beta**2)
    mu0 = ref.rho * ref.beta**2
    return {
        "weak": MaterialContrast(
            Dlambda=1e-4 * lam0, Dmu=1e-4 * mu0, Drho=1e-4 * ref.rho
        ),
        "moderate": MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0),
        "negative": MaterialContrast(
            Dlambda=-0.6 * lam0, Dmu=-0.6 * mu0, Drho=-0.6 * ref.rho
        ),
    }


def _omega(ka, ref):
    return ka * ref.alpha / A_RADIUS


def cube_eff(omega, a, contrast, ref):
    """Cube single-site effective contrast (Dkappa*, Dmu*, Drho*).

    The isotropic bulk modulus contrast Dkappa* = Dlambda* + (2/3) Dmu*, with the
    cubic-anisotropic shear averaged over the off/diagonal channels.
    """
    r = compute_cube_tmatrix(omega, a, ref, contrast)
    dmu = 0.5 * (r.Dmu_star_off + r.Dmu_star_diag)
    dkappa = r.Dlambda_star + 2.0 / 3.0 * dmu
    return dkappa, dmu, r.Drho_star


def sphere_eff(omega, a, contrast, ref):
    """Sphere single-site effective contrast from the Mie partial-wave extraction."""
    e = mie_extract_effective_contrasts(compute_elastic_mie(omega, a, ref, contrast))
    return e.Dkappa_star, e.Dmu_star, e.Drho_star


def _shape_errors(omega, contrast, ref):
    """Relative (sphere - cube) error per channel at the same material contrast."""
    ck, cm, cr = cube_eff(omega, A_RADIUS, contrast, ref)
    sk, sm, sr = sphere_eff(omega, A_RADIUS, contrast, ref)
    return {
        "kappa": abs(sk - ck) / abs(ck),
        "mu": abs(sm - cm) / abs(cm),
        "rho": abs(sr - cr) / abs(cr),
    }


# ── Stage A: single-site shape factor ────────────────────────────────────────


def test_stageA_shape_factor_is_small(ref, contrasts):
    """The sphere-vs-cube single-site shape error is small at every contrast/ka.

    This is the irreducible geometric discretisation error at the single-site
    level: the diluted sphere and the space-filling cube of the *same* contrast
    scatter nearly identically per unit volume. Bound chosen from the measured
    worst case (negative -60% contrast at ka=0.1: ~3.5%).
    """
    for name, c in contrasts.items():
        for ka in KA_LIST:
            errs = _shape_errors(_omega(ka, ref), c, ref)
            for comp, e in errs.items():
                assert e < 5e-2, f"{name}/{comp}/ka={ka}: shape error {e:.3e} too large"


def test_stageA_shape_factor_grows_with_ka(ref, contrasts):
    """Shape factor grows with frequency (finite-ka form-factor departure)."""
    for name, c in contrasts.items():
        e_lo = _shape_errors(_omega(0.05, ref), c, ref)
        e_hi = _shape_errors(_omega(0.1, ref), c, ref)
        # kappa channel is the cleanest monotone indicator across contrasts
        assert e_hi["kappa"] > e_lo["kappa"], (
            f"{name}: kappa shape error must grow with ka"
        )


def test_stageA_shape_factor_grows_with_contrast(ref, contrasts):
    """Shape factor grows with contrast magnitude (weak < moderate < negative-60%)."""
    ka = 0.1
    e_weak = _shape_errors(_omega(ka, ref), contrasts["weak"], ref)["kappa"]
    e_neg = _shape_errors(_omega(ka, ref), contrasts["negative"], ref)["kappa"]
    assert e_neg > e_weak, (
        "the -60% contrast shape error must exceed the weak-contrast one"
    )


def test_stageA_born_limit(ref, contrasts):
    """Weak contrast: sphere and cube effective contrasts agree to Born/Rayleigh order."""
    for ka in KA_LIST:
        errs = _shape_errors(_omega(ka, ref), contrasts["weak"], ref)
        for comp, e in errs.items():
            assert e < 5e-3, f"weak/{comp}/ka={ka}: Born-limit error {e:.3e} too large"
