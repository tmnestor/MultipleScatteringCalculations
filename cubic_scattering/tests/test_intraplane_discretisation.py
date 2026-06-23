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

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.slab_scattering import kennett_reference_rpp
from cubic_scattering.sphere_scattering import (
    compute_elastic_mie,
    mie_extract_effective_contrasts,
)

PHI_TOUCH = math.pi / 6.0
A_RADIUS = 1.0  # m; ka = (omega/alpha)*a, so omega = ka*alpha/a with a = A_RADIUS
KA_LIST = (0.05, 0.1)
REF_JSON = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneDiscretisation_reference.json"
)


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


# ── Stage B: collective layer R_PP ───────────────────────────────────────────
#
# The Mathematica deliverable ``IntraPlaneDiscretisation.wl`` dumps, per
# (contrast, ka, aL), the sphere-packing collective monopole renormalisation
# r_ms = mono_coll / mono_single = tcoll[1,1] / T0[0] of the Delta/phi-renormalised
# sphere packing, plus the spectral radius / conditioning of (I - G0 T0).  Python
# maps r_ms onto the layer effective contrast phi * sphere_eff(Delta/phi) and forms
# the normal-incidence specular R_PP via ``kennett_reference_rpp``; the cube ground
# truth is the fully-filled (phi=1) layer of the raw contrast.


@pytest.fixture(scope="module")
def dump():
    assert REF_JSON.exists(), (
        f"missing {REF_JSON} (run IntraPlaneDiscretisation.wl first)"
    )
    d = json.loads(REF_JSON.read_text())
    d["_contrasts"] = {c["name"]: c for c in d["contrasts"]}
    return d


def _layer_H():
    return 2.0 * A_RADIUS  # one plane of cubes/spheres of half-width a -> thickness 2a


def _sphere_eff_mc(om, mc, ref):
    e = mie_extract_effective_contrasts(compute_elastic_mie(om, A_RADIUS, ref, mc))
    return e.Dkappa_star, e.Dmu_star, e.Drho_star


def _cube_layer_rpp(dump, name, ka, ref):
    """Space-filling cube slab: exact R_PP of a uniform layer of the raw contrast."""
    c = dump["_contrasts"][name]
    mc = MaterialContrast(Dlambda=c["Dlambda"], Dmu=c["Dmu"], Drho=c["Drho"])
    return kennett_reference_rpp(ref, mc, _layer_H(), ka * ref.alpha / A_RADIUS)


def _sphere_layer_rpp(dump, rec, ref, *, renorm):
    """Sphere-packing layer R_PP = kennett(phi * sphere_eff(Delta or Delta/phi) * r_ms)."""
    c = dump["_contrasts"][rec["name"]]
    phi, ka = rec["phi"], rec["ka"]
    om = ka * ref.alpha / A_RADIUS
    scale = (1.0 / phi) if renorm else 1.0
    mc = MaterialContrast(
        Dlambda=c["Dlambda"] * scale, Dmu=c["Dmu"] * scale, Drho=c["Drho"] * scale
    )
    dk, dm, dr = _sphere_eff_mc(om, mc, ref)
    rms = complex(*rec["r_ms"]) if renorm else 1.0
    dk_layer, dm_layer, dr_layer = phi * dk * rms, phi * dm, phi * dr
    layer = MaterialContrast(
        Dlambda=(dk_layer - 2.0 / 3.0 * dm_layer).real,
        Dmu=dm_layer.real,
        Drho=dr_layer.real,
    )
    return kennett_reference_rpp(ref, layer, _layer_H(), om)


def _phys(dump, name):
    return [r for r in dump["stageB"] if r["name"] == name and r["physical"] == 1]


def test_stageB_dump_self_consistent(dump):
    """r_ms equals mono_coll / mono_single (sanity check on the Mathematica dump)."""
    for r in dump["stageB"]:
        mc = complex(*r["mono_coll"])
        ms = complex(*r["mono_single"])
        rms = complex(*r["r_ms"])
        assert abs(rms - mc / ms) < 1e-9, f"{r['name']}/aL={r['aL']}: r_ms inconsistent"


def test_stageB_renorm_recovers_cube_layer(dump, ref):
    """Delta->Delta/phi collapses the ~48% dilution error to the irreducible residual.

    The raw (un-renormalised) sphere packing is diluted by phi=pi/6, so its layer
    R_PP is ~(1-phi)~48% below the cube.  The renormalisation recovers the cube
    layer to within the single-site shape factor plus the nonlinear-mixing residual
    (~0.4% weak, ~4% moderate).
    """
    for name in ("weak", "moderate"):
        for rec in _phys(dump, name):
            rc = _cube_layer_rpp(dump, name, rec["ka"], ref)
            e_ren = abs(_sphere_layer_rpp(dump, rec, ref, renorm=True) - rc) / abs(rc)
            e_raw = abs(_sphere_layer_rpp(dump, rec, ref, renorm=False) - rc) / abs(rc)
            assert e_raw > 0.4, (
                f"{name}/aL={rec['aL']}: raw dilution error {e_raw:.3e} too small"
            )
            assert e_ren < 6e-2, (
                f"{name}/aL={rec['aL']}: renorm error {e_ren:.3e} too large"
            )
            assert e_ren < 0.2 * e_raw, (
                f"{name}/aL={rec['aL']}: renorm did not recover the cube"
            )


def test_stageB_collective_negligible_at_rayleigh(dump, ref):
    """The multiple-scattering correction barely moves the layer R_PP at Rayleigh.

    r_ms - 1 <= a few 1e-4 for the physical contrasts, so the discretisation error
    is the single-site shape factor + dilution, NOT inter-sphere multiple scattering:
    the renormalised layer error is essentially aL-independent.
    """
    for name in ("weak", "moderate"):
        for rec in _phys(dump, name):
            assert abs(complex(*rec["r_ms"]) - 1.0) < 5e-4
        errs = [
            abs(
                _sphere_layer_rpp(dump, rec, ref, renorm=True)
                - _cube_layer_rpp(dump, name, rec["ka"], ref)
            )
            / abs(_cube_layer_rpp(dump, name, rec["ka"], ref))
            for rec in _phys(dump, name)
            if rec["ka"] == 0.1
        ]
        assert max(errs) - min(errs) < 1e-3, (
            f"{name}: collective drives the error (not aL-flat)"
        )


def test_stageB_collective_grows_toward_touching(dump):
    """Spectral radius and |r_ms-1| increase as aL -> touching (matches item (c))."""
    for name in ("moderate", "negative"):
        recs = sorted(
            [r for r in dump["stageB"] if r["name"] == name and r["ka"] == 0.1],
            key=lambda r: -r["aL"],  # aL descending: dilute -> touching
        )
        specrad = [r["specrad"] for r in recs]
        rmsdev = [abs(complex(*r["r_ms"]) - 1.0) for r in recs]
        assert np.all(np.diff(specrad) > 0), (
            f"{name}: specrad must grow toward touching"
        )
        assert np.all(np.diff(rmsdev) > 0), (
            f"{name}: |r_ms-1| must grow toward touching"
        )


def test_stageB_conditioning_boundary(dump):
    """Conditioning of (I - G0 T0) grows toward touching; -60% degrades far worse."""
    for name in ("moderate", "negative"):
        recs = sorted(
            [r for r in dump["stageB"] if r["name"] == name and r["ka"] == 0.1],
            key=lambda r: -r["aL"],
        )
        cond = [r["cond"] for r in recs]
        assert np.all(np.diff(cond) > 0), f"{name}: cond must grow toward touching"
    # the negative -60% case is far more ill-conditioned than the physical moderate one
    cond_mod = max(r["cond"] for r in dump["stageB"] if r["name"] == "moderate")
    cond_neg = max(r["cond"] for r in dump["stageB"] if r["name"] == "negative")
    assert cond_neg > 10 * cond_mod


def test_stageB_negative_beyond_renorm_floor(dump):
    """The -60% contrast exceeds the renorm validity floor |Delta| < phi*background.

    Delta/phi pushes the inner moduli/density negative (phi=pi/6 => Delta/phi ~ 1.9*Delta,
    so -60% -> -115%), so every negative record is flagged unphysical in the dump.
    """
    neg = [r for r in dump["stageB"] if r["name"] == "negative"]
    assert neg, "expected negative-contrast records in the dump"
    assert all(r["physical"] == 0 for r in neg), (
        "negative -60% must be flagged unphysical"
    )
    # weak/moderate stay physical under the renormalisation
    assert all(
        r["physical"] == 1 for r in dump["stageB"] if r["name"] in ("weak", "moderate")
    )
