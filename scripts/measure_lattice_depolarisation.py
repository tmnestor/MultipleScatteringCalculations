"""MEASUREMENT: is the residual the ISOLATED-cube depolarisation, applied in a
SPACE-FILLING lattice where it does not belong?

═══ ANSWERED: NO, AND EMPHATICALLY ════════════════════════════════════════════
Removing the depolarisation makes the answer 24.6x WORSE -- 8.51e-4 becomes
2.09e-2, at every rung of the refinement ladder and at every contrast. The
isolated-cube amplification is not a correction the lattice has already made; it
is most of the physics, and it stays.

So the hypothesis below is REFUTED, and the refutation is the useful part: the
residual is not in the cube's self-response. The partition is right -- Gamma_0 is
the voxel's own self-field and the lattice carries everything else through G0 --
and the remaining ~8.6e-4 is about what the voxel is responding TO, not about how
it responds. The script is kept because the measurement is the evidence, and
because a future reading of "quadratic in contrast" will suggest this same
mechanism again.
═══════════════════════════════════════════════════════════════════════════════

THE CHAIN OF EVIDENCE THAT LEADS HERE. With the lateral sum exact and the
Galerkin contact correction applied, the refinement ladder is flat and a residual
of ~8.6e-4 survives. `measure_residual_origin` then established three things
about it:

  * it is NOT the averaged propagator's dynamic truncation. n_orders 0, 1, 2 --
    static, +w^2, +w^4 -- agree to 6e-3 relative, so even the STATIC table gives
    the same answer;
  * it is FREQUENCY-INDEPENDENT. The log-log slope in omega is -0.02, where a
    dynamic truncation would give +2;
  * it is QUADRATIC IN CONTRAST, c^1.98 over the range where the measurement is
    above the pipeline's numerical floor. A quadrature error is linear -- the
    truncation artifact fixed earlier measured c^1.00 exactly.

Static, frequency-independent, quadratic in contrast, scale-invariant. That is
the signature of a LOCAL-FIELD effect, and this script tests the specific
mechanism.

THE MECHANISM. `compute_cube_tmatrix` returns EFFECTIVE contrasts

    Drho* = Drho . A_u,   Dmu*  = Dmu . A_e,   Dlambda* = (...) . A_theta,

where the amplification factors A are built from Gamma_0, the self-interaction
of an isolated cube in an INFINITE HOST. A is the cube's depolarisation: the
statement that the field inside an inclusion differs from the field that would
be there without it, because the inclusion polarises and acts back on itself.
It is correct, and it is exactly right for one cube in a host.

BUT THESE CUBES TILE SPACE. A cube's neighbours are not host material -- they are
identical cubes with identical contrast. The cavity the depolarisation assumes is
not there. So T0 applies a correction the lattice has already cancelled, and it
does so once per cell, at every refinement, which is precisely the scale-
invariant per-cell error being chased. A enters as 1 + O(Gamma_0 Delta), so the
error it leaves is O(Delta^2) -- the measured quadratic.

THE TEST. Run the identical problem with the BARE contrasts, A = 1, and nothing
else changed. The test is decisive in both directions:

  * if the residual collapses, the depolarisation is the residual, and the 9x9
    T-matrix formulation needs a lattice-aware amplification rather than the
    isolated-cube one;
  * if it does not move, or gets worse, the local-field story is wrong and the
    quadratic scaling has another source.

WHAT MAKES THIS A FAIR TEST AND NOT A FITTED ONE: nothing is tuned. A = 1 is not
a fitted parameter, it is the OTHER canonical choice, and the medium here is a
UNIFORM slab -- every cube identical -- where the exact answer is a homogeneous
layer and Kennett gives it in closed form. There is no freedom anywhere.

Run:  conda run -n seismic python scripts/measure_lattice_depolarisation.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
)
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
D_LAM, D_MU, D_RHO = 2.0e9, 1.0e9, 100.0
OMEGA = 60.0
H_PHYS = 4.0
M = 4


def _bare_tmatrices(geom: SlabGeometry, c: float, omega: float) -> np.ndarray:
    """T with A = 1: the bare contrast, no isolated-cube depolarisation.

    Same construction as `_sub_cell_tmatrix_9x9`, with the STARRED contrasts
    replaced by the bare ones and nothing else touched -- same volume factor,
    same Voigt assembly, same block structure.
    """
    v_cell = (2.0 * geom.a) ** 3
    t = np.zeros((9, 9), dtype=complex)
    t[:3, :3] = omega**2 * (c * D_RHO) * v_cell * np.eye(3)
    t[3:, 3:] = v_cell * effective_stiffness_voigt(c * D_LAM, c * D_MU, c * D_MU)
    return np.broadcast_to(t, (geom.N_z, M, M, 9, 9)).copy()


def _run(c: float, n_z: int, omega: float, *, bare: bool) -> float:
    a_half = H_PHYS / (2.0 * n_z)
    geom = SlabGeometry(M=M, N_z=n_z, a=a_half)
    ones = np.ones((n_z, M, M))
    mat = SlabMaterial(Dlambda=c * D_LAM * ones, Dmu=c * D_MU * ones, Drho=c * D_RHO * ones, ref=REF)
    t0 = _bare_tmatrices(geom, c, omega) if bare else compute_slab_tmatrices(geom, mat, omega)
    kh = build_slab_kernels(
        geom, omega, REF, periodic=True, lattice_ewald=True, volume_averaged=True, n_orders=2
    )
    res = compute_slab_scattering(
        geom,
        mat,
        omega,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-13,
        kernel_hat=kh,
        T_local=t0,
    )
    r_lat = slab_rpp_periodic(res, t0)
    r_ken = kennett_reference_rpp(REF, MaterialContrast(c * D_LAM, c * D_MU, c * D_RHO), H_PHYS, omega)
    return float(abs(r_lat - r_ken) / abs(r_ken))


def main() -> int:
    print("=" * 88)
    print("MEASUREMENT -- is the residual the isolated-cube depolarisation?")
    print(f"  uniform slab H = {H_PHYS} m, M = {M}, exact lateral sum + Galerkin contact")
    print("  the medium is UNIFORM, so the exact answer is a homogeneous layer")
    print("=" * 88)

    # How large is the depolarisation at this contrast? Report it, so the
    # comparison below is read against the size of the thing being removed.
    res_t = compute_cube_tmatrix(OMEGA, H_PHYS / 8.0, REF, MaterialContrast(D_LAM, D_MU, D_RHO))
    print("\n  the amplification factors being switched off (A = 1):")
    print(f"       A_u     = {res_t.amp_u:.6f}")
    print(f"       A_theta = {res_t.amp_theta:.6f}")
    print(f"       A_e_off = {res_t.amp_e_off:.6f}   A_e_diag = {res_t.amp_e_diag:.6f}")

    print("\n  [D1] refinement ladder, starred (isolated-cube A) vs bare (A = 1)")
    print(f"       {'n_z':>4} {'d (m)':>8} {'STARRED':>13} {'BARE':>13} {'ratio':>9}")
    for n_z in (1, 2, 4, 8):
        star = _run(1.0, n_z, OMEGA, bare=False)
        bare = _run(1.0, n_z, OMEGA, bare=True)
        print(
            f"       {n_z:4d} {H_PHYS / n_z:8.4f} {star:13.5e} {bare:13.5e} {star / max(bare, 1e-300):9.2f}"
        )

    print("\n  [D2] contrast scaling of each, at n_z = 4")
    print(f"       {'c':>8} {'STARRED':>13} {'BARE':>13} {'ratio':>9}")
    cs = [0.03, 0.1, 0.3, 1.0]
    stars, bares = [], []
    for c in cs:
        s = _run(c, 4, OMEGA, bare=False)
        b = _run(c, 4, OMEGA, bare=True)
        stars.append(s)
        bares.append(b)
        print(f"       {c:8.3g} {s:13.5e} {b:13.5e} {s / max(b, 1e-300):9.2f}")
    sl_s = float(np.polyfit(np.log(cs), np.log(stars), 1)[0])
    sl_b = float(np.polyfit(np.log(cs), np.log(bares), 1)[0])
    print(f"       relative-error slope in c:  starred {sl_s:.3f}   bare {sl_b:.3f}")

    print("\n" + "=" * 88)
    improved = stars[-1] / max(bares[-1], 1e-300)
    if improved > 3.0:
        print("  THE DEPOLARISATION IS THE RESIDUAL. Removing it improves the answer")
        print(f"  by {improved:.1f}x at full contrast. The isolated-cube amplification is")
        print("  the wrong object for a SPACE-FILLING lattice: it corrects for a host")
        print("  cavity that the neighbouring cubes fill. The 9x9 T-matrix formulation")
        print("  is right for an isolated cube and needs a LATTICE-AWARE amplification")
        print("  here -- the same class of correction as the sphere-packing")
        print("  Delta -> Delta/phi already established in this project.")
    elif improved < 0.33:
        print(f"  THE OPPOSITE: removing the depolarisation makes it {1 / improved:.1f}x WORSE,")
        print("  so the isolated-cube amplification is closer to right than A = 1 and")
        print("  the residual is something else. The local-field story is refuted in")
        print("  the direction that matters -- A = 1 is not the answer either.")
    else:
        print(f"  INCONCLUSIVE: the two differ by only {improved:.2f}x. The depolarisation")
        print("  is not the dominant term in the residual, whatever else it is.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
