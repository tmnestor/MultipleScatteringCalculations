"""Tests for the layered incident field.

The field at every voxel due to a source in the stratified background. Built as
a row of the layered propagator rather than as new physics: it is
`corrected_layered_9x9` from one source plane to each scattering plane,
transformed to real space by the same subtract-transform-add route
`layered_stack_table` uses, so the whole-space part stays exact and only the
layer reverberation is ever integrated.

WHY IT MATTERS that this is layered rather than a homogeneous plane wave. The
3-D solver carries the stratified background inside G0. Driving it with an
incident field computed in a HOMOGENEOUS medium would half-dress the problem --
the propagator would know about the layering and the illumination would not --
and any comparison against a dress-after architecture would then measure that
inconsistency rather than the two orderings.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.horizontal_greens import exact_propagator_9x9
from cubic_scattering.pair_propagators import TransverseRule, layered_incident_field

OM = 2 * np.pi * 6.0
PITCH = 1.0
SRC = np.array([1.0, 0.2, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)


def _layer_module():
    """The sibling repo's layer model, skipped rather than failed if absent."""
    import sys

    sibling = "/Users/tod/Desktop/SeismicInversion"
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    pytest.importorskip("GlobalMatrix.layered_greens")
    return pytest.importorskip("Kennett_Reflectivity.layer_model")


def _uniform_model(n_lay: int = 24, q: float = 2.0):
    lm = _layer_module()
    a, b, r = 5.0, 3.0, 2.5
    return lm.LayerModel.from_arrays(
        alpha=[1.5, *([a] * n_lay), a],
        beta=[0.0, *([b] * n_lay), b],
        rho=[1.03, *([r] * n_lay), r],
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


def _model_reference(model, iface: int) -> ReferenceMedium:
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    j = max(iface, 1)
    return ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], model.rho[j])


# Deep enough that the seabed reverberation is damped out of the reduction test.
# On a uniform model the layered and whole-space fields differ by exactly that
# reverberation, so the residual is NOT zero in principle -- it is the surface
# path, suppressed by q = 2 over its extra distance. At source 5 / planes 8-9
# the surface path is only ~10 km longer than the direct one and the residual
# sits at 3.5e-9, flat in the quadrature rule. Deeper, the extra path is ~28 km
# and the reduction is clean. The residual's dependence on that geometry is the
# evidence it is the reverberation rather than an error.
PLANES = (18, 19)
SOURCE_IFACE = 14


def test_whole_space_case_matches_the_closed_form() -> None:
    """STRUCTURE and ORIENTATION: with no model it is the closed form applied to SRC.

    Asymmetric offsets on purpose -- exact_propagator_9x9 takes Cartesian
    (x, y, z) while the state is ordered (z, x, y), so a symmetric set would
    pass under the swap.
    """
    ref = ReferenceMedium(5.0, 3.0, 2.5)
    n_z, n_x, n_y = 2, 3, 2
    src_xy = (1, 0)
    dz_planes = (3.0, 4.0)  # depth of each plane below the source

    got = layered_incident_field(
        n_z, n_x, n_y, PITCH, OM, ref, source_xy=src_xy, source_vec=SRC, dz_planes=dz_planes
    )
    assert got.shape == (n_z, n_x, n_y, 9)

    for lz in range(n_z):
        for ix in range(n_x):
            for iy in range(n_y):
                want = (
                    exact_propagator_9x9(
                        (ix - src_xy[0]) * PITCH, (iy - src_xy[1]) * PITCH, dz_planes[lz], OM, ref
                    )
                    @ SRC
                )
                err = np.abs(got[lz, ix, iy] - want).max() / np.abs(want).max()
                assert err < 1e-14, f"({lz},{ix},{iy}) off by {err:.3e}"


def test_uniform_background_reduces_to_the_whole_space_field() -> None:
    """PHYSICS: the layered path on a uniform model must give whole space back.

    Runs the stratified propagator and the transverse transform, then demands
    the closed form. FLAT in the quadrature rule, which is the discriminating
    property: on a uniform model the reverberation integrand vanishes at every
    node, so refining cannot move the answer.
    """
    model = _uniform_model()
    ref = _model_reference(model, PLANES[0])
    n_z, n_x, n_y = 2, 2, 2
    src_xy = (0, 0)
    dz_planes = tuple(float(PLANES[i] - SOURCE_IFACE) * PITCH for i in range(n_z))

    want = layered_incident_field(
        n_z, n_x, n_y, PITCH, OM, ref, source_xy=src_xy, source_vec=SRC, dz_planes=dz_planes
    )

    errs = []
    for n_axis in (24, 48):
        rule = TransverseRule(kr_max=10.0 / PITCH, n_axis=n_axis)
        got = layered_incident_field(
            n_z,
            n_x,
            n_y,
            PITCH,
            OM,
            ref,
            source_xy=src_xy,
            source_vec=SRC,
            dz_planes=dz_planes,
            model=model,
            plane_ifaces=PLANES,
            source_iface=SOURCE_IFACE,
            transverse=rule,
        )
        errs.append(float(np.abs(got - want).max() / np.abs(want).max()))

    assert max(errs) < 1e-12, f"did not reduce: {errs}"
    assert max(errs) / max(min(errs), 1e-300) < 10.0, f"not flat in the rule: {errs}"


def test_rejects_a_model_without_its_companions() -> None:
    """Fail fast, with the four-element diagnostic the project requires."""
    ref = ReferenceMedium(5.0, 3.0, 2.5)
    with pytest.raises(ValueError) as exc:
        layered_incident_field(
            2,
            2,
            2,
            PITCH,
            OM,
            ref,
            source_xy=(0, 0),
            source_vec=SRC,
            dz_planes=(3.0, 4.0),
            model=_uniform_model(),
            plane_ifaces=None,
            source_iface=None,
            transverse=None,
        )
    msg = str(exc.value)
    assert "Where:" in msg
    assert "Valid:" in msg
    assert "Fix:" in msg
