"""Tests for the real-space layered plane-to-plane table.

This is the 3-D counterpart of `directional_sweeps.build_vertical_stack_layered`,
and it deliberately mirrors that function's SEMANTICS rather than inventing new
ones:

  * off-diagonal ``[lz, mz]``, lz != mz -- the FULL layered propagator from
    plane mz to plane lz;
  * diagonal ``[lz, lz]`` -- the layer REVERBERATION only, i.e. the full layered
    kernel minus the same-depth whole-space term, because the whole-space part
    of same-depth coupling is carried by the closed-form same-depth table.

The diagonal is not an afterthought. Stage 1 established, and
`scripts/gate_dg0_absolute_magnitude.py` validated absolutely, that same-depth
coupling on a layered background needs the reverberation added -- INCLUDING the
path that leaves a voxel, reflects off a layer boundary and returns to that same
voxel. That self-return is not inside T0, which is the whole-space single-site
T-matrix. So the diagonal here carries Dx = Dy = 0 as well, where the
whole-space same-depth table poisons it.

Test strategy, stated because one of these is weak on its own:

  * the background=None case is a STRUCTURE and ORIENTATION test. The
    implementation evaluates the closed form directly there, so comparing it
    against the closed form is near-tautological; what it actually pins is the
    shape, the zero diagonal, and the lz/mz orientation.
  * the uniform-background case is the PHYSICS test. It runs the full layered
    path -- the stratified propagator plus the separable transverse transform --
    and demands the whole-space answer back. A transposed lz/mz, a sign error in
    Dz, or a botched transform all fail here.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.horizontal_greens import exact_propagator_9x9
from cubic_scattering.pair_propagators import (
    TransverseRule,
    layered_stack_table,
    separation_index,
)

OM = 2 * np.pi * 6.0
PITCH = 1.0


def _layer_module():
    """The sibling repo's layer model, skipped rather than failed if absent.

    Mirrors the idiom in test_directional_sweeps: n_lay must be large enough
    that the plane pair is not the half-space boundary, where layered_greens
    returns zeros and a reduction test reads 1.000 rather than erroring.
    """
    import sys

    sibling = "/Users/tod/Desktop/SeismicInversion"
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    pytest.importorskip("GlobalMatrix.layered_greens")
    return pytest.importorskip("Kennett_Reflectivity.layer_model")


def _uniform_model(n_lay: int = 16, q: float = 2.0):
    """A uniform stack with one interface per pitch; planes sit in interiors."""
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


def _layered_model(n_lay: int = 16, q: float = 2.0):
    """The same stack with a genuine contrast below, so the diagonal is alive."""
    lm = _layer_module()

    a, b, r = 5.0, 3.0, 2.5
    al = [1.5, *([a] * n_lay), a]
    be = [0.0, *([b] * n_lay), b]
    rh = [1.03, *([r] * n_lay), r]
    for j in range(11, n_lay + 2):
        al[j], be[j], rh[j] = 6.8, 3.9, 3.2
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )


REF = ReferenceMedium(5.0, 3.0, 2.5)
PLANES = (8, 9)


def _model_reference(model) -> ReferenceMedium:
    """The model's own COMPLEX medium at the first plane.

    Attenuation is not optional here. The stack has a free surface above it, and
    the layered propagator carries surface reverberations that the whole-space
    kernel does not; only damping (q = 2, as stage 1's reduction test uses)
    attenuates them enough for the two to agree. With q large the reduction
    fails at ~0.5 for a correct implementation -- a test artefact that reads
    exactly like a physics defect.

    The reference must therefore be built from the model's COMPLEX slownesses,
    not from nominal real velocities, or the subtraction leaves the attenuation
    behind.
    """
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    j = max(PLANES[0], 1)
    return ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], model.rho[j])


def test_whole_space_case_has_the_right_shape_orientation_and_zero_diagonal() -> None:
    """STRUCTURE: shape, zero diagonal, and the lz/mz orientation.

    Near-tautological against the closed form by construction -- see the module
    docstring. Its real content is that [lz, mz] means "field at lz from a
    source at mz", i.e. Dz = (lz - mz) * pitch, checked with ASYMMETRIC indices
    so a transpose would fail.
    """
    n_z, n_x, n_y = 3, 2, 2
    tab = layered_stack_table(n_z, n_x, n_y, PITCH, OM, REF)
    assert tab.shape == (n_z, n_z, 2 * n_x - 1, 2 * n_y - 1, 9, 9)

    for lz in range(n_z):
        assert np.abs(tab[lz, lz]).max() == 0.0, "whole-space diagonal must be zero"

    for lz, mz, dx, dy in ((0, 2, 1, 0), (2, 0, 1, 0), (0, 1, -1, 1), (1, 0, 0, -1)):
        got = tab[lz, mz, separation_index(dx, n_x), separation_index(dy, n_y)]
        want = exact_propagator_9x9(dx * PITCH, dy * PITCH, (lz - mz) * PITCH, OM, REF)
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err < 1e-14, f"[{lz},{mz}] d=({dx},{dy}) off by {err:.3e}"


def test_uniform_background_reduces_to_the_whole_space_table() -> None:
    """PHYSICS: the layered path on a uniform model must give whole space back.

    Runs the stratified propagator and the separable transverse transform, then
    demands the closed form. This is the check that catches the reverberation
    being added with the wrong sign or scale -- the 3-D analogue of the
    reduction stage 1 pinned at 1.1e-15.

    The uniform diagonal must additionally vanish: with no layering there is no
    reverberation, so the integrand is zero at every node and the quadrature
    rule cannot matter.
    """
    n_z, n_x, n_y = 2, 2, 2
    model = _uniform_model()
    ref = _model_reference(model)
    want = layered_stack_table(n_z, n_x, n_y, PITCH, OM, ref)
    scale = np.abs(want[0, 1]).max()

    offs, diags = [], []
    for n_axis in (24, 48):
        rule = TransverseRule(kr_max=10.0 / PITCH, n_axis=n_axis)
        got = layered_stack_table(
            n_z, n_x, n_y, PITCH, OM, ref, model=model, plane_ifaces=PLANES, transverse=rule
        )
        offs.append(float(np.abs(got[0, 1] - want[0, 1]).max() / scale))
        diags.append(float(np.abs(got[0, 0]).max() / scale))

    assert max(offs) < 1e-12, f"off-diagonal did not reduce: {offs}"
    assert max(diags) < 1e-12, f"uniform diagonal should vanish: {diags}"
    # FLAT in the rule, which is the discriminating property. On a uniform model
    # the reverberation integrand is zero at every node, so refining the
    # quadrature cannot change the answer -- a residual that DID move with the
    # rule would mean something other than the reverberation is being
    # transformed, and would pass a magnitude-only assertion.
    assert max(offs) / max(min(offs), 1e-300) < 10.0, f"not flat in the rule: {offs}"


def test_layered_diagonal_is_alive() -> None:
    """DISCRIMINATOR: with a real contrast the diagonal must NOT be zero.

    A diagonal that stays zero for a layered model means the same-plane
    reverberation was dropped -- which would leave the solver silently missing
    every path that reflects off a layer and returns, including the self-return.
    The uniform test above cannot see that, since there the diagonal is zero
    either way.
    """
    n_z, n_x, n_y = 2, 2, 2
    rule = TransverseRule(kr_max=10.0 / PITCH, n_axis=64)
    model = _layered_model()
    ref = _model_reference(model)
    ws = layered_stack_table(n_z, n_x, n_y, PITCH, OM, ref)
    got = layered_stack_table(
        n_z, n_x, n_y, PITCH, OM, ref, model=model, plane_ifaces=PLANES, transverse=rule
    )
    rel = np.abs(got[0, 0]).max() / np.abs(ws[0, 1]).max()
    assert rel > 1e-6, f"layered diagonal is dead ({rel:.3e}) -- reverberation dropped"


def test_rejects_a_plane_map_of_the_wrong_length() -> None:
    """Fail fast, with the four-element diagnostic the project requires."""
    with pytest.raises(ValueError) as exc:
        layered_stack_table(
            3, 2, 2, PITCH, OM, REF, model=_uniform_model(), plane_ifaces=(8, 9), transverse=None
        )
    msg = str(exc.value)
    assert "Where:" in msg
    assert "Valid:" in msg
    assert "Fix:" in msg
