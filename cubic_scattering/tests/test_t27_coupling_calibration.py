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
     quadrature truth at face separation to ~1%.  The study originally
     reported its S block as defective (S[0,0] "3x low", shear-shear
     "sign flip" vs the FD-avg arbiter) — that report was OVERTURNED by
     the 2026-06-13 rederivation (scripts/face_s_rederivation.py): the
     module's face S is exact to 13+ digits against three bias-free
     routes; the FD/direct arbiters are invalid at face contact (tensor-
     product double-cube Gauss diagonal bias on the 1/w^3 kernel; see
     TestFaceSBlockArbiter in tests/test_inter_voxel_propagator.py).
     The remaining measured S deviation vs THIS study's Galerkin object
     (~0.5-0.7 of block scale, converged to ~1e-3 in n) is a GENUINE
     projection difference between the linear-basis Galerkin propagator
     and the point-derivative propagator at touching faces — analogous
     to (but larger than) the ~8% measured at corner.
     The H bug the study measured (H = C^T dropped the engineering 2x on
     shear rows, exactly 0.5x) has since been FIXED in
     inter_voxel_propagator_9x9 (H = W C^T with W = diag(1,1,1,2,2,2));
     the fix is regression-tested by TestHEngineeringConvention in
     tests/test_inter_voxel_propagator.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

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


# Bias-free dyadic-shell reference for the face S block at unit pitch
# (study medium), Gdd index order d2<G>_ij/dR_k dR_l.  Computed 2026-06-13
# by scripts/face_s_rederivation.py dyadic_contact_table(ng=12); agrees
# with the subdivision fixed point to ~1.5e-8 and with the delta-collapse
# defining integrals to ~1e-16.  The Voigt S entries below follow from
# these via the C4v relations (see TestFaceSBlockArbiter).
_FACE_S_TRUTH = {
    (0, 0): +3.0089307689e-12,  # -(A11 - eta B1111)/mu
    (1, 1): -1.2292204359e-12,  # -(A22 - eta B2222)/mu
    (0, 1): -4.2432795288e-13,  # eta B1122/mu
    (1, 2): +5.7341094441e-13,  # eta B2233/mu
    (3, 3): -1.8536137666e-12,  # -(A22 - 2 eta B2233)/mu
    (4, 4): +6.5153185437e-13,  # -(A11 + A22 - 4 eta B1122)/(2 mu)
}


def test_analytic_volume_averaged_propagator_face_outcome():
    """Step-2b face-contact outcome — RESOLVED 2026-06-13.

    History: the study originally measured the module's face S block as
    deviating ~0.5 of block scale from both its arbiters (S[0,0] "3x
    low", shear-shear "sign flip") and this test pinned that defect.
    The rederivation (scripts/face_s_rederivation.py) overturned it:

    - The module's face S IS the volume-averaged point-propagator object,
      to 13+ digits against three bias-free routes (delta-collapse
      defining integrals 1e-16, subdivision fixed point 1e-13,
      dyadic-shell 3D quadrature 1e-8).  Asserted here against the
      pinned dyadic truth values (first assertion block).
    - The FD-avg/direct arbiters are INVALID at face contact for the S
      kernel (1/w^3): tensor-product double-cube Gauss samples the
      singular ray w_perp = 0 with O(1) spurious weight (S00 drifts
      8.45e-12 -> 1.02e-11 over n = 4..16; FD2 at h = 0.005 drifts
      9.2e-12 -> 11.1e-12 as n grows, h <~ 1/n^2 regime).
    - The deviation vs THIS study's Galerkin object is a GENUINE,
      converged projection difference (Galerkin S00 = +2.87e-12,
      S44 = -7.2e-13 vs point-object +3.01e-12, +6.52e-13; n = 8 -> 10
      drift <= 1.3%): the linear-basis Galerkin propagator is a
      different functional at touching faces.  Pinned as an envelope
      (second assertion block) so a normalisation regression (which
      would blow past it) is still caught.
    """
    from cubic_scattering.inter_voxel_propagator import inter_voxel_propagator_9x9

    a = 0.5
    omega = 1e-3 * REF.beta / a  # quasi-static (analytic evaluated at omega=0)
    R = np.array([1.0, 0.0, 0.0])
    P9 = inter_voxel_propagator_9x9(
        (1, 0, 0), REF.alpha, REF.beta, REF.rho, 0.0, 0, d=1.0
    )

    # 1. The module's S block matches the bias-free volume-averaged truth.
    S_mod = np.real(P9[3:, 3:])
    for (i, j), ref in _FACE_S_TRUTH.items():
        assert S_mod[i, j] == pytest.approx(ref, rel=1e-6), (
            f"face S[{i},{j}] = {S_mod[i, j]:.6e} no longer matches the "
            f"bias-free volume-averaged truth {ref:.6e}"
        )

    # 2. G matches the Galerkin quadrature and the S projection difference
    #    stays within its measured envelope.
    D = study.galerkin_9_block(R, omega, a, n=6)
    dev = study.block_devs(np.real(P9), np.real(D))
    assert dev["G"] < 0.02, f"Analytic G no longer matches quadrature: {dev['G']:.3e}"
    assert 0.2 < dev["S"] < 0.9, (
        f"Galerkin-vs-point S projection difference {dev['S']:.3e} left its "
        "measured envelope [0.2, 0.9] (measured ~0.5 at n = 6; a sign or "
        "factor regression in either object would blow past 0.9)"
    )
