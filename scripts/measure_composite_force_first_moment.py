"""MEASURE: does the resonance composite need the first moment of its sub-cell forces?

THE QUESTION
------------
``compute_resonance_tmatrix`` builds the cube's 9x9 composite ``T_comp`` on phase-free Taylor patterns
about the cube centre (uniform displacement; uniform strain with its linear displacement) and collects
its output as the plain sum of the sub-cell sources: force F = sum_n F_n, stress sigma = sum_n sigma_n.

Seen from the centre, a sub-cell force F_n at offset dx_n radiates as

    F_n exp(-i k r.dx_n)  ~  F_n  -  i k (r.dx_n) F_n,

and the far-field formula carries a stress sigma as + i k (sigma r).  So the force's first moment enters
exactly like a stress dipole, sigma_eff = - sum_n F_n (x) dx_n.  The plain sum omits it.  Its symmetric
part fits the 9-component (Voigt) output; its antisymmetric part (a torque) does not.

The reciprocity gate cannot see the omission (the symmetric term keeps T_comp W^-1 symmetric, and its
coupling to a uniform displacement vanishes by inversion symmetry), so it is measured here instead.

THE MEASUREMENT
---------------
Reference: the per-cell far field (``resonance_far_field``: every sub-cell radiates from its own centre,
driven by the plane wave itself, ``psi_pw``).  Against it, the composite radiated as ONE point at the cube
centre, driven by the plane wave's Taylor input [pol, eps]:

    plain      T_comp as built
    +sym       T_comp + the symmetric first moment (what a Voigt output can hold)
    +full      T_comp + the full first moment, torque included (not representable in 9 components)

Same sub-cells and same n_sub in every arm, so the only difference is the point representation.
Controls, fixed before running:
  * a harness self-check: summing this script's point far field over the sub-cells reproduces
    resonance_far_field to rounding;
  * Drho = 0: no sub-cell force, so the three arms must coincide;
  * the error of every arm must fall as ka -> 0; a first-moment term with the wrong SIGN would make
    +full worse than plain at small ka.

Run:  conda run -n seismic python scripts/measure_composite_force_first_moment.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.resonance_tmatrix import compute_resonance_tmatrix  # noqa: E402
from cubic_scattering.scattered_field import (  # noqa: E402
    _incident_voigt_strain,
    _voigt_to_tensor,
    resonance_far_field,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
A = 10.0
N_SUB = 4
THETA = np.linspace(0.0, np.pi, 25)


def point_far_field(forces, sigmas, centres, omega, k_hat):
    """Far field of point sources (force F_n, stress TENSOR sigma_n, any symmetry) at centres.

    The same expressions as resonance_far_field, with the same scattering-plane basis, but taking a full
    3x3 stress so the antisymmetric first moment can be radiated too.
    """
    kP, kS = omega / REF.alpha, omega / REF.beta
    ref_vec = np.array([1.0, 0.0, 0.0]) if abs(k_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 = ref_vec - np.dot(ref_vec, k_hat) * k_hat
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(k_hat, perp1)
    out = np.zeros((3, len(THETA)), dtype=complex)
    for i, th in enumerate(THETA):
        r = np.sin(th) * perp1 + np.cos(th) * k_hat
        sv = np.cos(th) * perp1 - np.sin(th) * k_hat
        ph_P = np.exp(-1j * kP * centres @ r)
        ph_S = np.exp(-1j * kS * centres @ r)
        rF = forces @ r
        rSr = np.einsum("i,nij,j->n", r, sigmas, r)
        out[0, i] = np.sum((rF + 1j * kP * rSr) * ph_P) / (4 * np.pi * REF.rho * REF.alpha**2)
        Sr = np.einsum("nij,j->ni", sigmas, r)
        Q = (forces - np.outer(rF, r)) + 1j * kS * (Sr - np.outer(rSr, r))
        u_S = np.sum(Q * ph_S[:, None], axis=0) / (4 * np.pi * REF.rho * REF.beta**2)
        out[1, i] = sv @ u_S
        out[2, i] = perp2 @ u_S
    return out


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def run(ka, wave, con):
    omega = ka * REF.beta / A
    k_hat = np.array([0.3, 0.5, 0.81])
    k_hat /= np.linalg.norm(k_hat)
    k = omega / (REF.alpha if wave == "P" else REF.beta)
    k_vec = k * k_hat
    s_pol = np.cross(k_hat, [1.0, 0.0, 0.0])
    pol = k_hat if wave == "P" else s_pol / np.linalg.norm(s_pol)
    res = compute_resonance_tmatrix(omega, A, REF, con, n_sub=N_SUB, k_hat=k_hat, wave_type=wave)
    ref_ff = np.array(resonance_far_field(res, THETA, REF, con, omega, A, k_vec, pol))

    # harness self-check: the per-cell sum through this script's evaluator
    x_in = np.concatenate([pol, _incident_voigt_strain(k_vec, pol)])
    N = len(res.centres)
    src = np.array([res.T_loc_9x9 @ (res.psi_pw[9 * n : 9 * n + 9] @ x_in) for n in range(N)])
    sig = np.array([_voigt_to_tensor(s[3:]) for s in src])
    self_check = rel(point_far_field(src[:, :3], sig, res.centres, omega, k_hat), ref_ff)

    # the composite's output for the Taylor input, and the first moment of its sub-cell forces
    x_c = res.centres.mean(axis=0)
    dx = res.centres - x_c
    src_t = np.array([res.T_loc_9x9 @ (res.psi_exc[9 * n : 9 * n + 9] @ x_in) for n in range(N)])
    F = src_t[:, :3].sum(axis=0)
    sigma = _voigt_to_tensor(res.T_comp_9x9[3:] @ x_in)
    moment = -np.einsum("ni,nj->ij", src_t[:, :3], dx)
    sym = 0.5 * (moment + moment.T)
    tol = 1e-12 * (np.linalg.norm(F) + 1e-300)
    np.testing.assert_allclose(F, res.T_comp_9x9[:3] @ x_in, rtol=0, atol=tol)

    one = x_c[None, :]
    arms = {
        "plain": point_far_field(F[None], sigma[None], one, omega, k_hat),
        "+sym": point_far_field(F[None], (sigma + sym)[None], one, omega, k_hat),
        "+full": point_far_field(F[None], (sigma + moment)[None], one, omega, k_hat),
    }
    size = np.linalg.norm(sym) / np.linalg.norm(sigma)
    return self_check, size, {k_: rel(v, ref_ff) for k_, v in arms.items()}


def main() -> int:
    base = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
    cases = [
        ("gate contrast", base),
        ("density only", MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=100.0)),
        ("Drho = 0 (control)", MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=0.0)),
    ]
    print(f"cube half-width {A} m, n_sub = {N_SUB}, oblique incidence; errors vs the per-cell far field")
    worst_self = 0.0
    for label, con in cases:
        print(f"\n== {label}")
        head = f"{'wave':>4} {'k_S a':>6} {'|sym M|/|sigma|':>16} {'plain':>10} {'+sym':>10} {'+full':>10}"
        print(f"   {head}")
        for wave in ("P", "S"):
            for ka in (0.1, 0.2, 0.5, 1.0):
                sc, size, e = run(ka, wave, con)
                worst_self = max(worst_self, sc)
                print(
                    f"   {wave:>4} {ka:>6} {size:>16.3e} "
                    f"{e['plain']:>10.3e} {e['+sym']:>10.3e} {e['+full']:>10.3e}"
                )
    print(f"\nharness self-check (per-cell sum vs resonance_far_field): worst {worst_self:.1e}")
    return 0 if worst_self < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
