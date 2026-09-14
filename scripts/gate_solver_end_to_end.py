#!/usr/bin/env python3
"""GATE: the 3-D real-space solver on a PHYSICAL problem, end to end.

Everything gated before this drove the 3-D machinery with synthetic inputs --
random T-matrices, random sources. This is the first run on a physical model:
real material contrasts, a real cube T-matrix per voxel, a plane wave in, and a
reflection coefficient out.

NOTHING NEW IS BUILT FOR IT, and that is the point of the survey that preceded
it. The lattice `SweepGrid3D(n_z, n_x, n_y, pitch)` is the same lattice as
`SlabGeometry(M, N_z, a)` with pitch = 2a, so the existing builders drop
straight in:

  * `compute_slab_tmatrices`      -> (N_z, M, M, 9, 9), what solve_foldy_lax_3d wants
  * `_build_slab_incident_field`  -> (N_z, M, M, 9),    likewise
  * `slab_reflected_field`        -> (R_PP, R_PS, R_SP) from a solved field

The observable is reused rather than rewritten: the solved field is wrapped in a
`SlabResult` and handed to the validated far-field summation. Writing a second
far-field projection would have been the easiest way to introduce a defect that
no existing gate could see.

WHAT IS ACTUALLY COMPARED. Both architectures are given the SAME T-matrices and
the SAME incident field, so the only difference between them is how G0 is
applied -- real-space tables against FFT convolution. `gate_rung5c` already
showed those agree at 5e-16 on the operator; this shows the agreement survives
a full Krylov solve and a far-field projection, which it need not have: a solver
can amplify a small operator difference, and the far-field sum weights the
voxels unequally.

  [E1] the exciting fields agree
  [E2] the reflection amplitudes agree. TWO independent channels, not three:
       `slab_reflected_field` sets R_SP = R_PP by construction, so on a
       P-incident run the third return is a copy.
  [E3] the scattering is NOT trivial, so [E1] and [E2] have something to say. A
       model that barely scatters would let both architectures agree on
       approximately the incident field and prove nothing.

BOTH SIDES AT THE SAME GMRES TOLERANCE, and tight. The reference defaults to
1e-6; running it there against a 1e-10 solve compares two solves converged to
different residuals, and the disagreement is then the looser tolerance. Measured
at 6e-7 before this was matched, which is exactly that scale.

Run:  conda run -n seismic python scripts/gate_solver_end_to_end.py
SI units (m, m/s, kg/m3, Pa) -- the validated parameters for this machinery.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    SlabResult,
    _build_slab_incident_field,
    compute_slab_scattering,
    compute_slab_tmatrices,
    slab_reflected_field,
)
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402

# Validated test parameters for this machinery.
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OM = 150.0
A_HALF = 1.0  # cube half-width, m; ka = 0.03, Rayleigh
M, N_Z = 6, 3
K_HAT = np.array([1.0, 0.0, 0.0])  # straight down, in (z, x, y)
GMRES_TOL = 1e-12  # both sides, see the note at the reference solve


def build_model() -> tuple:
    """Geometry and a heterogeneous material -- a DISTINCT contrast per voxel."""
    geom = SlabGeometry(M=M, N_z=N_Z, a=A_HALF)
    rng = np.random.default_rng(20260914)
    shape = (N_Z, M, M)
    # Moderate contrast, modulated per voxel so the medium is genuinely
    # heterogeneous: a uniform slab would let a site-averaging implementation
    # pass, the same vacuity the lateral gates guard against.
    scale = 0.5 + rng.random(shape)
    material = SlabMaterial(
        Dlambda=2.0e9 * scale,
        Dmu=1.0e9 * scale,
        Drho=100.0 * scale,
        ref=REF,
    )
    return geom, material


def main() -> int:
    geom, material = build_model()
    pitch = geom.d

    print("=" * 78)
    print("GATE -- the 3-D real-space solver on a physical problem, end to end")
    print(f"  lattice {N_Z} x {M} x {M}, cube side {pitch} m, ka = {OM / REF.alpha * A_HALF:.3f}")
    print(f"  background a={REF.alpha} b={REF.beta} rho={REF.rho}; per-voxel contrast")
    print("=" * 78)

    t_local = compute_slab_tmatrices(geom, material, OM)
    psi_inc = _build_slab_incident_field(geom, OM, REF, K_HAT, "P")
    print(f"\n  T-matrices {t_local.shape}, incident field {psi_inc.shape}")

    # --- the FFT-convolution architecture, as the reference -----------------
    # THE SAME GMRES TOLERANCE ON BOTH SIDES, and tight. The reference defaults
    # to 1e-6; leaving it there and asking the other side for 1e-10 compares two
    # solves converged to different residuals, and the disagreement is then the
    # looser tolerance rather than anything about the operators. Measured: 6e-7
    # at the default, 1e-6 being exactly that scale.
    ref_res = compute_slab_scattering(geom, material, OM, K_HAT, "P", psi0=psi_inc, gmres_tol=GMRES_TOL)
    r_ref = slab_reflected_field(ref_res, t_local)

    # --- the 3-D real-space solver ------------------------------------------
    grid = SweepGrid3D(n_z=N_Z, n_x=M, n_y=M, pitch=pitch)
    cache = build_g0_cache_3d(grid, REF, OM)
    got = solve_foldy_lax_3d(cache, t_local, psi_inc, tol=GMRES_TOL)

    # Reuse the validated far-field summation rather than writing a second one.
    new_res = SlabResult(
        psi=got.psi,
        psi0=psi_inc,
        geometry=geom,
        material=material,
        omega=OM,
        k_hat=K_HAT,
        wave_type="P",
        n_gmres_iter=got.n_matvec,
        gmres_residual=got.residual,
    )
    r_new = slab_reflected_field(new_res, t_local)

    # [E1] the exciting fields
    e1 = float(np.abs(got.psi - ref_res.psi).max() / np.abs(ref_res.psi).max())
    print(f"\n  [E1] exciting field, real-space vs FFT : {e1:.3e}")
    print(f"       matvecs: real-space {got.n_matvec}, FFT {ref_res.n_gmres_iter}")

    # [E2] the observable.
    #
    # Only the first two channels are independent here. slab_reflected_field
    # sets R_SP = R_PP by construction -- "same P-projection, meaningful for
    # S-wave incidence" -- so on a P-incident run it is a copy, not a third
    # measurement. Reporting it as one would overstate the evidence by half.
    print("\n  [E2] reflection amplitudes (P incident; R_SP is a copy of R_PP)")
    print(f"    {'':>6} {'real-space':>26} {'FFT convolution':>26} {'rel diff':>11}")
    e2 = 0.0
    for name, a, b in zip(("R_PP", "R_PS"), r_new[:2], r_ref[:2], strict=True):
        d = abs(a - b) / max(abs(b), 1e-300)
        e2 = max(e2, d)
        print(f"    {name:>6} {a.real:+.6e}{a.imag:+.6e}j {b.real:+.6e}{b.imag:+.6e}j {d:11.3e}")

    # [E3] is there anything to agree about?
    scat = float(np.abs(got.psi - psi_inc).max() / np.abs(psi_inc).max())
    print("\n  [E3] multiple scattering is non-trivial")
    print(f"       |psi - psi_inc| / |psi_inc| : {scat:.3e}  (must be well above 0)")
    print(f"       |R_PP|                      : {abs(r_new[0]):.6e}")

    ok = e1 < 1e-9 and e2 < 1e-9 and scat > 1e-3
    print("\n" + "=" * 78)
    print(f"GATE solver end to end: {'PASS' if ok else 'FAIL'}")
    print("  Both architectures were given the same T-matrices and the same")
    print("  incident field, so this isolates the G0 application through a full")
    print("  Krylov solve and a far-field projection -- neither of which is")
    print("  guaranteed to preserve an operator-level agreement.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
