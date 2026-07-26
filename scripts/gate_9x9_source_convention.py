"""Gate: may GlobalMatrix's stratified P^z be composed with this repo's P^x?

The composition (I - [P^z + P^x] dC_eff)^-1 is only meaningful if both
propagators speak the same 9-component language.  Both claim a state
``(u_z,u_x,u_y, e_zz,e_xx,e_yy, 2e_xy, 2e_zy, 2e_zx)``, but they are built
from opposite ends:

  * this repo   ``_propagator_block_9x9`` / ``exact_propagator_9x9``
                maps a source ``(F_i, dsigma*_Va)`` to ``(u_i, e_Va)``
  * GlobalMatrix ``layered_greens_9x9 = A @ G6 @ B`` wraps a 6x6
                displacement-traction Green's function

Reciprocity is the right discriminator: it holds in ANY elastic medium, so no
uniform-limit or k<->r transform is needed, and it is sensitive to exactly the
source-side weighting that differs.

GATE A (calibration)  establishes the invariant on the VALIDATED closed-form
    propagator.  Measured, not assumed: the elementwise ratio P[a,b]/P[b,a] is
    an exact outer product w_a/w_b with w = (1,1,1,1,1,1,2,2,2) -- the Voigt
    engineering factor, strain rows carrying 2e on the shears while stress
    columns carry plain sigma.  Hence

        W P  is symmetric,   W = diag(1,1,1, 1,1,1, 1/2,1/2,1/2)

    A gate whose calibration leg is not itself checked is worthless: the first
    version of this script asserted plain symmetry and failed at 10-16%.

GATE B (the gate)  requires the same weighted relation of the layered
    propagator, in the (kx,ky) domain of a laterally invariant medium:

        G9(i<-j)(+k)  ==  W^-1 [G9(j<-i)(-k)]^T W

GATE C (localisation)  if B fails, drop to the 6x6 core to decide whether the
    defect is in the A/B wrapper or upstream of it.

Run:  conda run -n seismic python scripts/gate_9x9_source_convention.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering import ReferenceMedium
from cubic_scattering.horizontal_greens import exact_propagator_9x9

from Kennett_Reflectivity.layer_model import LayerModel  # isort: skip
from GlobalMatrix.layered_greens import (  # isort: skip
    _interface_elastic_properties,
    layered_greens_6x6,
    layered_greens_9x9,
    strain_from_displacement_traction,
)

W = np.diag(np.array([1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5], dtype=float))
WINV = np.linalg.inv(W)

BLOCKS = {
    "G (force->disp,    3x3)": (slice(0, 3), slice(0, 3)),
    "C (stress->disp,   3x6)": (slice(0, 3), slice(3, 9)),
    "H (force->strain,  6x3)": (slice(3, 9), slice(0, 3)),
    "S (stress->strain, 6x6)": (slice(3, 9), slice(3, 9)),
}

TOL_A = 1e-10
TOL_B = 1e-6


def marine_model() -> LayerModel:
    """Ocean + 4 crust layers + stiffer half-space, seismic units."""
    return LayerModel.from_arrays(
        alpha=[1.5, 3.2, 3.8, 4.4, 5.0, 6.5],
        beta=[0.0, 1.8, 2.1, 2.45, 2.8, 3.7],
        rho=[1.03, 2.3, 2.5, 2.65, 2.8, 3.3],
        thickness=[2.0, 0.5, 0.5, 0.5, 0.5, np.inf],
        Q_alpha=[20000, 600, 600, 600, 600, 600],
        Q_beta=[1e10, 300, 300, 300, 300, 300],
    )


def gate_a() -> float:
    """Calibrate the invariant on the validated closed-form propagator."""
    print("=" * 74)
    print("GATE A - calibrate the invariant on the closed-form 9x9 propagator")
    print("=" * 74)
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    omega = 150.0
    worst = 0.0
    for z, x, y in (
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (1.5, -2.5, 3.5),
        (-4.0, 1.0, 2.0),
    ):
        P = exact_propagator_9x9(x, y, z, omega, ref)
        A = W @ P
        rel = np.linalg.norm(A - A.T) / np.linalg.norm(A)
        worst = max(float(worst), float(rel))
        print(
            f"  (z,x,y)=({z:5.1f},{x:5.1f},{y:5.1f})  ||WP-(WP)^T||/||WP|| = {rel:.3e}"
        )
    print(f"\n  GATE A: {'PASS' if worst < TOL_A else 'FAIL'}  (worst {worst:.3e})")
    print("  invariant:  P = W^-1 P^T W,  W = diag(1,1,1,1,1,1,.5,.5,.5)")
    return worst


def gate_b() -> float:
    """Require the same weighted reciprocity of the layered 9x9."""
    print()
    print("=" * 74)
    print("GATE B - layered_greens_9x9 must obey  G9(i<-j)(+k) = W^-1 G9(j<-i)(-k)^T W")
    print("=" * 74)
    model = marine_model()
    omega = 2.0 * np.pi * 10.0
    worst = 0.0
    for j, i in ((1, 3), (2, 4)):
        for p in (0.08, 0.15):
            kx, ky = np.array([omega * p]), np.array([0.0])
            M1 = layered_greens_9x9(
                model, omega, kx, ky, source_iface=j, receiver_iface=i
            )[0]
            M2 = layered_greens_9x9(
                model, omega, -kx, -ky, source_iface=i, receiver_iface=j
            )[0]
            target = WINV @ M2.T @ W
            rel = np.linalg.norm(M1 - target) / np.linalg.norm(target)
            worst = max(float(worst), float(rel))
            print(f"\n  {j}<->{i}, p={p}:  overall rel = {rel:.3e}")
            for name, (rs, cs) in BLOCKS.items():
                a, b = M1[rs, cs], target[rs, cs]
                den = max(np.linalg.norm(a), np.linalg.norm(b))
                print(f"    {name}: {np.linalg.norm(a - b) / den if den else 0.0:.3e}")
    print(f"\n  GATE B: {'PASS' if worst < TOL_B else 'FAIL'}  (worst {worst:.3e})")
    return worst


def gate_d() -> float:
    """The SOLVED reciprocity law of the 6x6, in closed form.

    layered_greens_6x6 is correct, but in a MIXED basis: the traction half is
    carried as sigma/(-i w). Solving for the weight (jointly over interface
    pairs, slownesses and frequencies) gives, exactly,

        M(i<-j)(+k) = (i/w) . SD . J6 [M(j<-i)(-k)]^T J6 . SD

        SD = diag(1, -1, -1, -i w, +i w, +i w)

    where diag(1,1,1,-iw,-iw,-iw) is the sigma/(-i w) scaling (uniform across
    P-SV and SH) and diag(1,-1,-1,1,-1,-1) is the parity of the x,y components
    under k -> -k. Verified to ~1e-15.

    NOTE: recovering the SH entries REQUIRES ky != 0. At ky = 0 the SH channel
    decouples, components 2 and 5 vanish identically, and a fit will silently
    return an unconstrained (wrong) value for them.
    """
    print()
    print("=" * 74)
    print("GATE D - the solved closed-form reciprocity law of the 6x6")
    print("=" * 74)
    model = marine_model()
    J6 = np.zeros((6, 6))
    J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)

    worst = 0.0
    for f in (5.0, 10.0, 25.0):
        w = 2.0 * np.pi * f
        SD = np.diag(np.array([1, -1, -1, -1j * w, 1j * w, 1j * w], dtype=complex))
        for j, i, p in ((1, 3, 0.08), (2, 4, 0.15), (1, 4, 0.12)):
            kx = np.array([w * p * 0.6])
            ky = np.array([w * p * 0.8])  # ky != 0: keep SH alive
            M1 = layered_greens_6x6(model, w, kx, ky, source_iface=j, receiver_iface=i)[
                0
            ]
            M2 = layered_greens_6x6(
                model, w, -kx, -ky, source_iface=i, receiver_iface=j
            )[0]
            pred = (1j / w) * (SD @ (J6 @ M2.T @ J6) @ SD)
            rel = np.linalg.norm(M1 - pred) / np.linalg.norm(M1)
            worst = np.inf if not np.isfinite(rel) else max(float(worst), float(rel))
            print(f"  f={f:5.1f} Hz  {j}->{i}  p={p:.2f}:  rel = {rel:.3e}")
    print(f"\n  GATE D: {'PASS' if worst < 1e-9 else 'FAIL'}  (worst {worst:.3e})")
    return worst


def corrected_9x9(model, w, kx, ky, j, i):
    """layered_greens_9x9 with BOTH corrections applied.

    (1) SOURCE-SIDE NORMALISATION of the 6x6.  riccati_greens injects a unit
        state JUMP in a mixed basis; multiplying on the right by

            Q = diag(1, -1, -1, i/w, -i/w, -i/w)

        converts it to a self-adjoint (force, stress-glut) source, after which
        the 6x6 obeys CLEAN symplectic reciprocity  N1 = J6 N2^T J6.
        Q is not guessed: it is forced by the GATE D law (see solve_correction).

    (2) ADJOINT SOURCE OPERATOR.  `traction_from_strain` is plain Hooke and
        wavenumber-independent, but the operator conjugate to A must carry the
        same k-dependence. Requiring A1 N1 B1 = W^-1 (A2 N2 B2)^T W given
        N1 = J6 N2^T J6 forces

            B(k) = -J6 . A_source(-k)^T . W

        built with the SOURCE interface's material (using the receiver's
        leaves an 83% residual).
    """
    J6 = np.zeros((6, 6))
    J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)
    Q = np.diag(np.array([1, -1, -1, 1j / w, -1j / w, -1j / w], dtype=complex))

    G = (
        layered_greens_6x6(
            model, w, np.array([kx]), np.array([ky]), source_iface=j, receiver_iface=i
        )[0]
        @ Q
    )

    rho_r, al_r, be_r = _interface_elastic_properties(model, i)
    A = strain_from_displacement_traction(
        np.array([kx]), np.array([ky]), rho_r, al_r, be_r
    )[0]

    rho_s, al_s, be_s = _interface_elastic_properties(model, j)
    A_src_minus = strain_from_displacement_traction(
        np.array([-kx]), np.array([-ky]), rho_s, al_s, be_s
    )[0]
    B = -J6 @ A_src_minus.T @ W

    return A @ G @ B


def gate_e() -> float:
    """The corrected construction must satisfy the GATE B invariant."""
    print()
    print("=" * 74)
    print("GATE E - corrected 9x9 (source normalisation + adjoint B) vs GATE B")
    print("=" * 74)
    model = marine_model()
    worst = 0.0
    for f in (5.0, 10.0, 25.0):
        w = 2.0 * np.pi * f
        for j, i, p in ((1, 3, 0.08), (2, 4, 0.15), (1, 4, 0.12)):
            kx, ky = w * p * 0.6, w * p * 0.8  # ky != 0 keeps SH alive
            M1 = corrected_9x9(model, w, kx, ky, j, i)
            M2 = corrected_9x9(model, w, -kx, -ky, i, j)
            target = WINV @ M2.T @ W
            rel = np.linalg.norm(M1 - target) / np.linalg.norm(target)
            worst = np.inf if not np.isfinite(rel) else max(float(worst), float(rel))
            print(f"  f={f:5.1f} Hz  {j}->{i}  p={p:.2f}:  rel = {rel:.3e}")
    print(f"\n  GATE E: {'PASS' if worst < TOL_B else 'FAIL'}  (worst {worst:.3e})")
    return worst


def gate_f() -> float:
    """GATE F - WITHIN-MATRIX block structure, in the homogeneous limit.

    GATE E compares M(i<-j)(+k) against M(j<-i)(-k) -- two DIFFERENT matrices.
    It therefore constrains neither the overall scale (the invariant is
    homogeneous of degree one) nor the relation between the G, C, H, S blocks
    INSIDE one matrix.  Both blind spots are real: an absolute comparison
    against the closed form showed ratios varying 3.9 -> 16.2 with offset and
    differing per block (G 15.3, C 4.5, H 2.6, S 3.9).

    The missing invariant is GATE A's own, applied to the layered propagator:

        W . M(i<-j)(kx,ky)   must be SYMMETRIC

    Justification.  For the closed form, W P(r) is symmetric at EVERY
    separation r individually (GATE A, four separations including non-zero dz,
    residuals 0 to 4e-18).  The 2-D transverse transform acts only on (x,y)
    while W acts on components, so the property carries into the (kx,ky)
    domain unchanged.

    RESTRICTION: this is licensed only in the HOMOGENEOUS limit, where the
    source<->receiver swap is equivalent to the identity on the medium.  For a
    genuinely stratified reference M(i<-j) and M(j<-i) see different media and
    no within-matrix symmetry should be expected.  The gate therefore builds a
    uniform crust and damps it, and is a statement about the A/B wrapper only.
    """
    print()
    print("=" * 74)
    print("GATE F - within-matrix block structure (homogeneous limit)")
    print("=" * 74)

    al, be, rh, Q = 4.0, 2.22, 2.6, 20.0
    n_lay, dz = 16, 1.0
    model = LayerModel.from_arrays(
        alpha=[1.5, *([al] * n_lay), al],
        beta=[0.0, *([be] * n_lay), be],
        rho=[1.03, *([rh] * n_lay), rh],
        thickness=[3.0, *([dz] * n_lay), np.inf],
        Q_alpha=[Q, *([Q] * n_lay), Q],
        Q_beta=[1e10, *([Q] * n_lay), Q],
    )

    worst = 0.0
    for f in (6.0, 12.0):
        w = 2.0 * np.pi * f
        for p in (0.05, 0.12):
            kx = np.array([w * p * 0.6])
            ky = np.array([w * p * 0.8])
            # deep pair: ocean and free surface ~8 km away and attenuated
            M9 = corrected_9x9_grid(model, w, kx, ky, j=9, i=8)[0]
            A = W @ M9
            rel = np.linalg.norm(A - A.T) / np.linalg.norm(A)
            worst = np.inf if not np.isfinite(rel) else max(float(worst), float(rel))
            print(f"  f={f:5.1f} Hz  p={p:.2f}:  ||WM-(WM)^T||/||WM|| = {rel:.3e}")

    print(f"\n  GATE F: {'PASS' if worst < 1e-6 else 'FAIL'}  (worst {worst:.3e})")
    return worst


def corrected_9x9_grid(model, w, kx, ky, j, i):
    """Corrected stratified 9x9 over a (kx,ky) grid (same as GATE E's build)."""
    shape = kx.shape
    kxf, kyf = kx.ravel(), ky.ravel()

    J6 = np.zeros((6, 6))
    J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)

    G = layered_greens_6x6(model, w, kxf, kyf, source_iface=j, receiver_iface=i)
    Q = np.diag(np.array([1, -1, -1, 1j / w, -1j / w, -1j / w], dtype=complex))
    G = G @ Q

    rho_r, al_r, be_r = _interface_elastic_properties(model, i)
    A = strain_from_displacement_traction(kxf, kyf, rho_r, al_r, be_r)
    rho_s, al_s, be_s = _interface_elastic_properties(model, j)
    Am = strain_from_displacement_traction(-kxf, -kyf, rho_s, al_s, be_s)
    B = -np.einsum("ab,xcb,cd->xad", J6, Am, W)

    return np.einsum("xab,xbc,xcd->xad", A, G, B).reshape(*shape, 9, 9)


def gate_c() -> None:
    """Localise: is the defect in the A/B wrapper, or already in the 6x6?"""
    print()
    print("=" * 74)
    print("GATE C - localise to the 6x6 core")
    print("=" * 74)
    model = marine_model()
    omega = 2.0 * np.pi * 10.0
    kx, ky = np.array([omega * 0.10]), np.array([0.0])
    M1 = layered_greens_6x6(model, omega, kx, ky, source_iface=1, receiver_iface=3)[0]
    M2 = layered_greens_6x6(model, omega, -kx, -ky, source_iface=3, receiver_iface=1)[0]

    print(
        f"  plain      ||M1-M2^T||/||M2^T||     = "
        f"{np.linalg.norm(M1 - M2.T) / np.linalg.norm(M2.T):.3e}"
    )

    best = min(
        (
            (
                np.linalg.norm(M1 - np.diag(s) @ M2.T @ np.diag(s))
                / np.linalg.norm(M2.T),
                s,
            )
            for s in (
                np.array(t, dtype=float)
                for t in itertools.product([1.0, -1.0], repeat=6)
            )
        ),
        key=lambda t: t[0],
    )
    print(
        f"  best +-1 diagonal signature {tuple(int(v) for v in best[1])}: {best[0]:.3e}"
    )

    J6 = np.zeros((6, 6))
    J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)
    print(
        f"  symplectic J6                        = "
        f"{np.linalg.norm(M1 - J6 @ M2.T @ J6) / np.linalg.norm(M2.T):.3e}"
    )

    print()
    print("  If the 6x6 also fails, the defect is NOT the A/B wrapper: it is")
    print("  upstream, in the 6x6 source convention. Note riccati_greens_psv")
    print("  documents its basis as [u_x, u_z, sigma_zz/(-iw), sigma_xz/(-iw)] --")
    print("  a MIXED convention whose correct reciprocity weight is not yet derived.")


if __name__ == "__main__":
    a = gate_a()
    b = gate_b()
    if b >= TOL_B:
        gate_c()
    d = gate_d()
    e = gate_e()
    fblk = gate_f()

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(
        f"  A  closed-form 9x9 invariant W P symmetric : {'PASS' if a < TOL_A else 'FAIL'}"
    )
    print(
        f"  B  layered 9x9 obeys that invariant       : {'PASS' if b < TOL_B else 'FAIL'}"
    )
    print(
        f"  D  solved 6x6 reciprocity law             : {'PASS' if d < 1e-9 else 'FAIL'}"
    )
    print(
        f"  E  CORRECTED 9x9 obeys the invariant      : {'PASS' if e < TOL_B else 'FAIL'}"
    )
    print(
        f"  F  within-matrix block structure          : {'PASS' if fblk < TOL_B else 'FAIL'}"
    )
    print()
    if a < TOL_A and e < TOL_B and fblk < TOL_B:
        print("  Conventions RECONCILED. Use corrected_9x9() and compose.")
    elif e < TOL_B and fblk >= TOL_B:
        print("  NOT SAFE TO COMPOSE.  E passes but F fails: the corrections fix")
        print("  the behaviour under source<->receiver SWAP while leaving the")
        print("  WITHIN-MATRIX block structure wrong.  E cannot see this -- it")
        print("  compares two different matrices -- which is why an earlier run")
        print("  of this script wrongly reported the conventions reconciled.")
        print("  Symptom: against the closed form the ratio varies 3.9 -> 16.2")
        print("  with offset and differs per block (G 15.3, C 4.5, H 2.6, S 3.9).")
        print("  Fix the A/B wrapper until F passes; only then compose.")
    else:
        print("  NOT SAFE to compose. Resolve the source convention first.")
    print("=" * 74)
    sys.exit(0 if (a < TOL_A and e < TOL_B and fblk < TOL_B) else 1)
