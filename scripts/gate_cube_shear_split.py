#!/usr/bin/env python3
"""Gate: the cube's two shear channels split about the sphere's, with the sphere's isotropic average.

Python half of a two-implementation check (the symbolic half is ``Mathematica/CubeShearSplit.wl``).

The static strain self-term of a cube has three O_h channels. From the moment integrals
(``Mathematica/CubeA22Block.wl``) they are

    A1g:  1 + (3 dlam + 2 dmu) / (3 (lam + 2 mu))
    T2g:  1 + 2 dmu S_shear,   S_shear = (pi (lam + 2 mu) - sqrt3 (lam + mu)) / (3 pi mu (lam + 2 mu))
    Eg:   1 + 2 dmu S_diag,    S_diag  = (3 sqrt3 (lam + mu) + 2 pi mu) / (6 pi mu (lam + 2 mu))

Claims checked here, through the PRODUCTION route (``effective_contrasts._static_eshelby_ABC``: surface
constants j1, j2, k1 and an explicit delta term -- sharing nothing with the moment engine but the
definition of the object):

  [1] the production T2c, T3c equal -dmu S_shear and 2 dmu (S_shear - S_diag);
  [2] the isotropic average (3 S_shear + 2 S_diag)/5 equals the SPHERE's deviatoric depolarisation,
      2 mu . avg = 2 (4 - 5 nu) / (15 (1 - nu))  -- the sqrt3 terms cancel;
  [3] the split S_diag - S_shear = (lam + mu)(5 sqrt3 - 2 pi) / (6 pi mu (lam + 2 mu)) > 0;
  [4] the bulk channel equals the sphere's, 1 + dK / (lam + 2 mu), dK = dlam + 2 dmu / 3;
  [C] control: T2g ALONE is not the sphere's (the gate can fail).

Run:  conda run -n seismic python scripts/gate_cube_shear_split.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cubic_scattering.effective_contrasts import _compute_T123, _static_eshelby_ABC  # noqa: E402

SQ3 = np.sqrt(3.0)
TOL = 1e-13
# (alpha, beta, rho): the validated background, then Poisson ratios from ~0 to ~0.45
BACKGROUNDS = [(5.0, 3.0, 2.5), (2.0, 1.4, 1.0), (6.0, 2.0, 3.3), (4.0, 1.2, 2.0), (3.0, 1.5, 1.7)]


def closed_forms(lam: float, mu: float) -> tuple[float, float]:
    s_shear = (np.pi * (lam + 2 * mu) - SQ3 * (lam + mu)) / (3 * np.pi * mu * (lam + 2 * mu))
    s_diag = (3 * SQ3 * (lam + mu) + 2 * np.pi * mu) / (6 * np.pi * mu * (lam + 2 * mu))
    return s_shear, s_diag


def main() -> int:
    worst = {k: 0.0 for k in ("1", "2", "3", "4")}
    control = np.inf
    for alpha, beta, rho in BACKGROUNDS:
        mu, lam = rho * beta**2, rho * (alpha**2 - 2 * beta**2)
        nu = lam / (2 * (lam + mu))
        s_shear, s_diag = closed_forms(lam, mu)
        ac, bc, cc = _static_eshelby_ABC(alpha, beta, rho)

        dmu = 0.01 * mu
        _, t2c, t3c = _compute_T123(ac, bc, cc, 0.0, dmu)
        e1 = max(abs(t2c + dmu * s_shear), abs(t3c - 2 * dmu * (s_shear - s_diag))) / (dmu * s_shear)
        worst["1"] = max(worst["1"], e1)

        # production S's, not the closed forms, carry the remaining claims
        p_shear = -t2c.real / dmu
        p_diag = p_shear - t3c.real / (2 * dmu)
        sphere = 2 * (4 - 5 * nu) / (15 * (1 - nu))
        worst["2"] = max(worst["2"], abs(2 * mu * (3 * p_shear + 2 * p_diag) / 5 - sphere) / sphere)
        split = (lam + mu) * (5 * SQ3 - 2 * np.pi) / (6 * np.pi * mu * (lam + 2 * mu))
        worst["3"] = max(worst["3"], abs((p_diag - p_shear) - split) / split)
        control = min(control, abs(2 * mu * p_shear - sphere) / sphere)

        dlam = 0.02 * lam
        t1c, t2c, t3c = _compute_T123(ac, bc, cc, dlam, dmu)
        amp_theta = 1.0 / (1.0 - 3 * t1c - 2 * t2c - t3c)
        bulk_sphere = 1.0 / (1.0 + (dlam + 2 * dmu / 3) / (lam + 2 * mu))
        worst["4"] = max(worst["4"], abs(amp_theta - bulk_sphere) / abs(bulk_sphere))

        avg = 2 * mu * (3 * p_shear + 2 * p_diag) / 5
        print(
            f"  nu = {nu:.4f}:  2 mu S_shear = {2 * mu * p_shear:.10f}"
            f"   2 mu S_diag = {2 * mu * p_diag:.10f}"
            f"   average = {avg:.10f}   sphere = {sphere:.10f}"
        )

    oks = []
    labels = {
        "1": "production T2c, T3c == the moment closed forms",
        "2": "isotropic average of the cube's shear channels == sphere",
        "3": "split S_diag - S_shear == (lam+mu)(5 sqrt3 - 2 pi)/(6 pi mu (lam+2mu))",
        "4": "bulk channel == sphere's",
    }
    for key, label in labels.items():
        ok = worst[key] < TOL
        oks.append(ok)
        print(f"  [{key}] {'PASS' if ok else 'FAIL'}  {label}: worst rel. error {worst[key]:.2e}")
    ok = control > 1e-2
    oks.append(ok)
    print(
        f"  [C] {'PASS' if ok else 'FAIL'}  control: T2g alone differs from the sphere by >= {control:.2e}"
    )
    verdict = f"ALL {len(oks)} CHECKS PASS" if all(oks) else "CHECKS FAILED"
    print(f"==== gate_cube_shear_split: {verdict}")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
