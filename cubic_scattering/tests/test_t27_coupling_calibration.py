"""Step-2 calibration test for the T27 inter-voxel coupling study.

Only the cheap calibration is asserted here (the decision table lives in
scripts/t27_coupling_study.py — it is a study, not a regression).

Calibration facts asserted (measured in the study):
  1. The vectorised Green's tensor matches elastodynamic_greens.
  2. Reciprocity: P(-R) = P(R)^T.
  3. Normalisation: M9^-1 P[0:9,0:9] W^-1 with W = diag(V,V,V, -V a^2/3 x6)
     converges to exact_propagator_9x9 in the point limit a/R -> 0
     (machinery + sign/factor conventions are exact in the limit).
  4. At face separation R = (2a, 0, 0) (a/R = 0.5) the deviation from the
     point propagator is a genuine O((a/R)^2) volume-averaging effect with
     measured envelopes: G ~ 12%, C/H ~ 34%, S ~ 60%.  A normalisation bug
     (sign or factor-2) would blow well past these envelopes.
  5. Step 2b outcome vs the existing analytic volume-averaged propagator
     (inter_voxel_propagator_9x9, unit pitch): its G block matches the
     quadrature truth at face separation to ~1%, while its S block does NOT
     (measured ~0.5 of block scale: S[0,0] ~3x low, shear-shear sign flip),
     and H = C^T drops the engineering 2x on shear rows.
"""

import importlib.util
from pathlib import Path

import numpy as np

from cubic_scattering.resonance_tmatrix import elastodynamic_greens

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "t27_coupling_study.py"
_spec = importlib.util.spec_from_file_location("t27_coupling_study", _SCRIPT)
assert _spec is not None and _spec.loader is not None
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

REF = study.REF
KA = 0.1


def test_vectorised_greens_matches_kupradze():
    """greens_tensor reproduces elastodynamic_greens at scattered points."""
    omega = KA * REF.beta / 1.0
    rng = np.random.default_rng(3)
    for r_vec in rng.uniform(1.5, 5.0, size=(4, 3)):
        G_ref = elastodynamic_greens(r_vec, omega, REF)
        G_vec = study.greens_tensor(r_vec[None, :], omega, REF)[0]
        np.testing.assert_allclose(G_vec, G_ref, rtol=1e-12)


def test_propagator_reciprocity():
    """P(-R) = P(R)^T (G is even and symmetric)."""
    omega = KA * REF.beta / 1.0
    R = np.array([2.0, 1.0, 0.5])
    Pp = study.quad_propagator_27(R, omega, REF, 1.0, n=5)
    Pm = study.quad_propagator_27(-R, omega, REF, 1.0, n=5)
    err = np.max(np.abs(Pm - Pp.T)) / np.max(np.abs(Pp))
    assert err < 1e-12, f"Reciprocity violated: {err:.2e}"


def test_point_limit_normalisation():
    """M9^-1 P W^-1 matches exact_propagator_9x9 to <1% at a/R = 0.025.

    Measured deviation at a = 0.05, R = 2 (n=6, as run here): max 2.6e-3
    (S block scale; n=8 gives the same value to two digits).
    Tolerance 1e-2 gives ~4x margin; any sign or factor error in W fails.
    """
    omega = KA * REF.beta / 1.0  # fixed omega -> fixed point propagator
    a = 0.05
    R = np.array([2.0, 0.0, 0.0])
    dev = study.nine_block_deviation(R, omega, a, n=6)
    for block, value in dev.items():
        assert value < 1e-2, f"Block {block} deviation {value:.2e} exceeds 1e-2"


def test_face_separation_volume_averaging_envelope():
    """Face-separation deviations stay within the measured physical envelopes.

    Measured at a = 1, R = (2a,0,0), ka = 0.1, n = 8:
      G = 0.119, C = H = 0.343, S = 0.596.
    Envelopes at ~1.3x; a broken normalisation gives O(1)-O(2) deviations.
    """
    a = 1.0
    omega = KA * REF.beta / a
    R = np.array([2.0 * a, 0.0, 0.0])
    dev = study.nine_block_deviation(R, omega, a, n=8)
    envelopes = {"G": 0.16, "C": 0.45, "H": 0.45, "S": 0.75}
    for block, env in envelopes.items():
        assert dev[block] < env, (
            f"Block {block} deviation {dev[block]:.3f} exceeds envelope {env} — "
            "either the normalisation broke or the volume-averaging physics changed"
        )
    # And the deviation is genuinely nonzero (volume averaging is real):
    assert dev["S"] > 0.3, "S-block deviation unexpectedly small — check setup"


def test_analytic_volume_averaged_propagator_face_outcome():
    """Step-2b measured outcome vs inter_voxel_propagator_9x9 at face contact.

    At unit pitch (a = 0.5, R = (1,0,0)), quasi-static: the analytic module's
    G block IS the volume-averaged Green's tensor (matches the quadrature
    truth to ~1%), while its S block does not match the volume-averaged
    object (measured deviation ~0.48 of block scale at n = 6: S[0,0] ~3x
    low and shear-shear entries sign-flipped).  If the S block is ever
    fixed, the second assertion should fail — update it then.
    """
    from cubic_scattering.inter_voxel_propagator import inter_voxel_propagator_9x9

    a = 0.5
    omega = 1e-3 * REF.beta / a  # quasi-static (analytic evaluated at omega=0)
    R = np.array([1.0, 0.0, 0.0])
    D = study.galerkin_9_block(R, omega, a, n=6)
    P9 = inter_voxel_propagator_9x9((1, 0, 0), REF.alpha, REF.beta, REF.rho, 0.0, 0)
    dev = study.block_devs(np.real(P9), np.real(D))
    assert dev["G"] < 0.02, f"Analytic G no longer matches quadrature: {dev['G']:.3e}"
    assert dev["S"] > 0.3, (
        f"Analytic S unexpectedly matches now ({dev['S']:.3e}) — the face S "
        "mismatch documented in the study may have been fixed; update this test"
    )
