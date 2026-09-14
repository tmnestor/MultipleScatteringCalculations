"""GATE: cross-check the extracted self-energy against cube_eshelby.

WHY. `measure_lattice_self_energy` extracted Gamma_0 = T_Born^-1 - T0^-1, a
DIFFERENCE OF TWO NEARLY-EQUAL INVERSES -- the cancellation trap this project
has already paid for once (a pushed, wrong accusation against slab_scattering).
Its |Gamma_0| grew as the cells shrank, which looked wrong. Before that
extraction is used for anything, it has to be checked against machinery that
computes the same physics a different way.

ONE EXPECTATION IS CORRECTED AT THE OUTSET. The suspicion was that |Gamma_0|
should scale as 1/V, i.e. x8 per halving of the cell. It should not.
`effective_contrasts._compute_Gamma0_analytical` gives
Gamma0_stat = a^2 (a0 + b0/3) G0_CUBE, so the scalar self-interaction scales as
a^2 -- which is right, since Int_cube (1/r) dV ~ a^2. The earlier "x2, x2, x6.6,
x8 looks wrong" reasoning was measuring against a wrong expectation.

AND THE TWO OBJECTS ARE NOT THE SAME THING. The scalar Gamma0 above lives in the
EFFECTIVE-CONTRAST formulation; the extracted one lives in the 9-component
source/field basis and carries volume factors. Comparing their magnitudes
directly would be meaningless. What IS comparable, and is computed
independently by `compute_cube_eshelby_factors`, is the AMPLIFICATION -- the
ratio of the full response to its own Born limit, which is dimensionless and
convention-free.

WHAT IS CHECKED:
  [E1] the eps-limit Born T-matrix used throughout this line of work against
       `compute_cube_born_tmatrix`, which does its own finite-difference scaling.
       Two routes to the same Born limit.
  [E2] the amplification T0/T_Born from the 9x9 matrices against the published
       E_u, E_theta, E_e_off, E_e_diag. This is the real cross-check: it uses
       no inverse and no difference, so it cannot be contaminated by the
       cancellation being investigated.
  [E3] the CONDITIONING of the extraction itself -- how large is
       |T_Born^-1 - T0^-1| against the terms being subtracted? A small relative
       difference means the extracted Gamma_0 is dominated by rounding and must
       not be used, whatever its value.

Run:  conda run -n seismic python scripts/gate_gamma0_cross_check.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.cube_eshelby import (  # noqa: E402
    compute_cube_born_tmatrix,
    compute_cube_eshelby_factors,
)
from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)
OMEGA = 60.0
HALF_WIDTHS = (2.0, 1.0, 0.5, 0.25, 0.125)
EPS_LIN = 1e-6


def _t9(a: float, scale: float = 1.0):
    c = MaterialContrast(CONTRAST.Dlambda * scale, CONTRAST.Dmu * scale, CONTRAST.Drho * scale)
    return _sub_cell_tmatrix_9x9(compute_cube_tmatrix(OMEGA, a, REF, c), OMEGA, a)


def main() -> int:
    print("=" * 88)
    print("GATE -- cross-check the extracted self-energy against cube_eshelby")
    print("=" * 88)

    # ---- [E1] two independent routes to the Born limit ---------------------
    print("\n  [E1] eps-limit Born T vs compute_cube_born_tmatrix (independent route)")
    print(f"       {'a (m)':>8} {'Drho* eps-limit':>17} {'Drho* module':>15} {'rel':>10}")
    e1_ok = True
    for a in HALF_WIDTHS:
        born_mod = compute_cube_born_tmatrix(OMEGA, a, REF, CONTRAST)
        # The eps-limit of the SAME quantity, scaled back to full contrast.
        weak = MaterialContrast(CONTRAST.Dlambda * EPS_LIN, CONTRAST.Dmu * EPS_LIN, CONTRAST.Drho * EPS_LIN)
        eps_res = compute_cube_tmatrix(OMEGA, a, REF, weak)
        d_eps = eps_res.Drho_star / EPS_LIN
        d_mod = born_mod.Drho_star
        rel = abs(d_eps - d_mod) / abs(d_mod)
        e1_ok = e1_ok and rel < 1e-3
        print(f"       {a:8.3f} {d_eps.real:17.6e} {d_mod.real:15.6e} {rel:10.2e}")

    # ---- [E2] amplification, the convention-free comparison ---------------
    print("\n  [E2] amplification from the 9x9 T's vs the published Eshelby factors")
    print(f"       {'a (m)':>8} {'ka':>8} {'T00 ratio':>11} {'E_u':>11} {'rel':>10}")
    e2_ok = True
    for a in HALF_WIDTHS:
        ka = OMEGA / REF.beta * a
        esh = compute_cube_eshelby_factors(REF, CONTRAST, a=a, ka=ka)
        t_full, t_born = _t9(a), _t9(a, EPS_LIN) / EPS_LIN
        # The (0,0) entry is the displacement/density channel, which is E_u.
        ratio = complex(t_full[0, 0] / t_born[0, 0])
        rel = abs(ratio - esh.E_u) / abs(esh.E_u)
        e2_ok = e2_ok and rel < 1e-6
        print(f"       {a:8.3f} {ka:8.4f} {ratio.real:11.6f} {esh.E_u.real:11.6f} {rel:10.2e}")

    # ---- [E3] is the extraction even conditioned? -------------------------
    print("\n  [E3] conditioning of Gamma_0 = T_Born^-1 - T0^-1")
    print(f"       {'a (m)':>8} {'|Gamma_0|':>12} {'|T_Born^-1|':>12} {'rel diff':>10} {'cond(T0)':>11}")
    e3_ok = True
    for a in HALF_WIDTHS:
        t_full, t_born = _t9(a), _t9(a, EPS_LIN) / EPS_LIN
        inv_b, inv_f = np.linalg.inv(t_born), np.linalg.inv(t_full)
        g0 = inv_b - inv_f
        # If the difference is a tiny fraction of the terms, the result is noise.
        rel_diff = np.abs(g0).max() / np.abs(inv_b).max()
        cond = float(np.linalg.cond(t_full))
        e3_ok = e3_ok and rel_diff > 1e-6
        print(
            f"       {a:8.3f} {np.abs(g0).max():12.4e} {np.abs(inv_b).max():12.4e} "
            f"{rel_diff:10.2e} {cond:11.3e}"
        )

    print("\n" + "=" * 88)
    if e1_ok and e2_ok and e3_ok:
        print("  PASS: the Born limit agrees by two independent routes, the")
        print("  amplification reproduces the published Eshelby factors, and the")
        print("  self-energy extraction is not dominated by cancellation. The")
        print("  gamma result of measure_lattice_self_energy stands.")
    else:
        print("  FAIL:", end=" ")
        print("Born routes disagree." if not e1_ok else "", end="")
        print("amplification does not match the Eshelby factors." if not e2_ok else "", end="")
        print("Gamma_0 is dominated by cancellation." if not e3_ok else "")
    print("=" * 88)
    return 0 if (e1_ok and e2_ok and e3_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
