#!/usr/bin/env python3
"""GATE: does carrying the stratification in the REFERENCE make the multiple-scattering sum converge faster?

THE CLAIM (thesis, Introduction): "we have concentrated on the two-way Riccati factorization approach,
where the reference Green's tensor is the exact solution for a plane stratified halfspace, and none of
the scattered field has been incorporated within it. The burden of convergence of the multiple
scattering series is then borne by the method used to sum the multiple scattering partial sums."
``gate_variational_summation`` confirms the summation method (Alg 6.2) on a representative operator but
says itself that pairing it with the stratified reference "is a separate build".  This is that build,
against a CORRECT competitor.

ONE MEDIUM, TWO CORRECT FORMULATIONS.  A uniform background (below a matched fluid layer 0), a weak
slab (the gate's contrast) of thickness D, and a strong but physical reflector LAYER (the gate's jump,
scaled by c) of thickness D, a gap D below the slab.  Normal incidence, so the specular chain is exact.
  [A] thesis:      reference = the LAYERED background (reflector layer included); scatterers = the slab
                   only; the reverberation added per plane pair
                   (``measure_thesis_refinement.run_pairwise``).
  [B] homogeneous: reference = the whole space; scatterers = the slab AND the reflector layer, voxelised
                   at the same pitch.
Both are discretisations of the same exact Lippmann-Schwinger equation, so both must reproduce the
exact layered answer (checked, [1]); only then is their CONVERGENCE compared.

MEASURED per arm: the spectral radius of K T (the Born series converges iff < 1); unrestarted GMRES
iterations to a relative residual of 1e-10; and BCGVAR (Alg 6.2, the thesis's own summation) iterations
to 1e-6 on the transition <b~, Omega b>, with b the incident field and b~ the field of a unit force at
the observation plane.

PRE-REGISTERED, before any run.  CONFIRMED if, at every reflector strength c in {0.25, 0.5, 1} and
refinement n in {1, 2, 4}: both arms reproduce the exact scattered field to < 10% (so the comparison is
between correct formulations), and [A] needs FEWER GMRES iterations than [B].  REFUTED if both arms are
correct and [A] is not faster somewhere.  The BCGVAR counts and spectral radii are reported, not gated.

Run:  conda run -n seismic python scripts/gate_summation_stratified_reference.py
SI units.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.layered_correction as LC  # noqa: E402
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import plane_reference_medium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_kernels import same_depth_kernel_9x9, vertical_kernel_9x9  # noqa: E402
from scripts import gate_thesis_formulation_periodic as gate  # noqa: E402

A0, B0, R0 = gate.A0, gate.B0, gate.R0
OM, EPS = gate.OM, gate.EPS
D = 2.0  # slab thickness = reflector-layer thickness = the gap between them
Z_SRC = -12.0
M = 2  # only the specular block acts
KERNEL_KW = gate.KERNEL_KW
MJ = np.diag([1.0, 1, 1, -1, -1, -1, -0.5, -0.5, -0.5])  # the 9-component pairing, gated elsewhere
GMRES_TOL, BCG_TOL = 1e-10, 1e-6
STRENGTHS = (0.25, 0.5, 1.0)
LADDER = (1, 2, 4)


def lame(al: float, be: float, rh: float) -> tuple[float, float]:
    """(lambda, mu)."""
    return rh * (al**2 - 2 * be**2), rh * be**2


class Setup:
    """Depths, models and plane sets for one reflector strength c and refinement n (z down, slab top 0)."""

    def __init__(self, c: float, n: int) -> None:
        self.c, self.n = c, n
        self.d = D / n
        self.h = self.d / 2
        self.z_refl = 2 * D  # reflector layer [2D, 3D]
        self.n_fine = int(round((3 * D + self.d - Z_SRC) / self.h))
        d_al, d_be, d_rh = gate.REFL_JUMP
        self.refl = (A0 + c * d_al, B0 + c * d_be, R0 + c * d_rh)
        lam_s, mu_s = lame(A0, B0, R0)
        self.slab_contrast = (gate.D_LAM, gate.D_MU, gate.D_RHO)
        self.slab = (
            float(np.sqrt((lam_s + gate.D_LAM + 2 * (mu_s + gate.D_MU)) / (R0 + gate.D_RHO))),
            float(np.sqrt((mu_s + gate.D_MU) / (R0 + gate.D_RHO))),
            R0 + gate.D_RHO,
        )
        lr, mr = lame(*self.refl)
        self.refl_contrast = (lr - lam_s, mr - mu_s, self.refl[2] - R0)

    def iface(self, z: float) -> int:
        return 1 + int(round((z - Z_SRC) / self.h))

    def z_iface(self, j: int) -> float:
        return Z_SRC + (j - 1) * self.h

    def centres(self, top: float, count: int) -> list[float]:
        return [top + (k + 0.5) * self.d for k in range(count)]

    def model(self, *, slab: bool, refl: bool):
        """Matched fluid layer 0, background to the source, fine layers, background half-space."""
        import Kennett_Reflectivity.layer_model as lm

        n_l = 2 + self.n_fine
        al = [A0] * (n_l + 1)
        be = [0.0, *([B0] * n_l)]
        rh = [R0] * (n_l + 1)
        th = [3000.0, 300.0, *([self.h] * self.n_fine), np.inf]
        for lay in range(2, n_l):
            top, bot = self.z_iface(lay - 1), self.z_iface(lay)
            if slab and top >= -1e-9 and bot <= D + 1e-9:
                al[lay], be[lay], rh[lay] = self.slab
            if refl and top >= self.z_refl - 1e-9 and bot <= self.z_refl + D + 1e-9:
                al[lay], be[lay], rh[lay] = self.refl
        return lm.LayerModel.from_arrays(
            alpha=al,
            beta=be,
            rho=rh,
            thickness=th,
            Q_alpha=[1e4] * (n_l + 1),
            Q_beta=[1e10, *([1e4] * n_l)],
        )


def p_tilde(model, src: int, rcv: int) -> np.ndarray:
    return LC.corrected_layered_9x9(model, OM, np.array([EPS]), np.array([0.0]), src, rcv)[0]


def ws(dz: int, d: float, ref_c: ReferenceMedium) -> np.ndarray:
    e = np.array([EPS])
    if dz == 0:
        return same_depth_kernel_9x9(e, 0.0, OM, ref_c)[:, :, 0]
    return vertical_kernel_9x9(e, 0.0, dz * d, OM, ref_c)[:, :, 0]


def build_arm(s: Setup, arm: str) -> dict:
    """The chain H = I - K T, its weight W, the incident and dual fields, and the exact obs field."""
    ref = ReferenceMedium(A0, B0, R0)
    obs = [-0.5 * s.d]
    slab = s.centres(0.0, s.n)
    if arm == "A":
        zs = obs + slab
        m_ref = s.model(slab=False, refl=True)
    else:  # empty planes through the gap keep the plane spacing uniform, as the Toeplitz kernel needs
        zs = obs + slab + s.centres(D, s.n) + s.centres(s.z_refl, s.n)
        m_ref = s.model(slab=False, refl=False)
    m_full = s.model(slab=True, refl=True)
    planes = [s.iface(z) for z in zs]
    n_z = len(planes)
    geom = SlabGeometry(M=M, N_z=n_z, a=s.d / 2)
    con = np.zeros((3, n_z, M, M))
    for lz, z in enumerate(zs):
        if 0.0 < z < D:
            con[:, lz] = np.array(s.slab_contrast)[:, None, None]
        elif s.z_refl < z < s.z_refl + D:
            con[:, lz] = np.array(s.refl_contrast)[:, None, None]
    mat = SlabMaterial(Dlambda=con[0], Dmu=con[1], Drho=con[2], ref=ref)
    t0 = compute_slab_tmatrices(geom, mat, OM)[:, 0, 0]
    kh = build_slab_kernels(geom, OM, ref, **KERNEL_KW)[:, 0, 0]
    ref_c = plane_reference_medium(m_ref, planes[1])
    big_k = np.zeros((9 * n_z, 9 * n_z), dtype=complex)
    for i in range(n_z):
        for j in range(n_z):
            k = kh[i - j + n_z - 1].copy()
            if arm == "A":
                k += (p_tilde(m_ref, planes[j], planes[i]) - ws(i - j, s.d, ref_c)) / s.d**2
            big_k[9 * i : 9 * i + 9, 9 * j : 9 * j + 9] = k
    t_blk = np.zeros_like(big_k)
    w = np.zeros_like(big_k)
    for j in range(n_z):
        t_blk[9 * j : 9 * j + 9, 9 * j : 9 * j + 9] = t0[j]
        w[9 * j : 9 * j + 9, 9 * j : 9 * j + 9] = MJ @ t0[j]
    ez = np.zeros(9, dtype=complex)
    ez[0] = 1.0
    si = s.iface(Z_SRC)
    b = np.concatenate([p_tilde(m_ref, si, j) @ ez for j in planes])
    b_dual = np.concatenate([p_tilde(m_ref, planes[0], j) @ ez for j in planes])
    exact = p_tilde(m_full, si, planes[0]) @ ez
    return {
        "kt": big_k @ t_blk,
        "w": w,
        "b": b,
        "b_dual": b_dual,
        "exact": exact,
        "incident": b[:9],
        "n_z": n_z,
    }


def gmres_iterations(h: np.ndarray, b: np.ndarray) -> int:
    """Unrestarted GMRES inner iterations to GMRES_TOL (restart = n, one cycle)."""
    count = [0]

    def cb(_: object) -> None:
        count[0] += 1

    n = h.shape[0]
    op = LinearOperator((n, n), matvec=lambda x: h @ x, dtype=complex)
    _, info = gmres(
        op, b, rtol=GMRES_TOL, atol=0.0, restart=n, maxiter=1, callback=cb, callback_type="pr_norm"
    )
    if info != 0:
        raise RuntimeError(f"GMRES did not reach {GMRES_TOL} in {n} iterations (info={info})")
    return count[0]


def bcgvar_iterations(
    h: np.ndarray, w: np.ndarray, b: np.ndarray, b_dual: np.ndarray
) -> tuple[int | None, float]:
    """Alg 6.2 (as corrected in ``gate_variational_summation``): iterations to BCG_TOL on <b~, Omega b>.

    Returns:
        (iterations or None, the W H asymmetry that licenses H_dag = H).
    """
    wh = w @ h
    asym = float(np.abs(wh - wh.T).max() / np.abs(wh).max())

    def bil(u: np.ndarray, v: np.ndarray) -> complex:
        return complex(u @ (w @ v))

    exact = bil(b_dual, np.linalg.solve(h, b))
    r, rd = b.copy(), b_dual.copy()
    p = pd = np.zeros_like(b)
    ds, rho_prev = 0.0 + 0.0j, 0.0 + 0.0j
    for it in range(h.shape[0]):
        rho_n = bil(rd, r)
        if it == 0:
            p, pd = r.copy(), rd.copy()
        else:
            beta = rho_n / rho_prev
            p, pd = r + beta * p, rd + beta * pd
        q, qd = h @ p, h @ pd
        alpha = rho_n / bil(pd, q)
        ds += alpha * rho_n
        r, rd = r - alpha * q, rd - alpha * qd
        rho_prev = rho_n
        if abs(ds - exact) / abs(exact) < BCG_TOL:
            return it + 1, asym
    return None, asym


def main() -> int:
    """Both arms at every strength and refinement; the gate on GMRES iterations.

    Returns:
        0 if CONFIRMED, else 1.
    """
    print("=" * 100)
    print("GATE -- does the stratified reference make the multiple-scattering sum converge faster?")
    print(
        f"  slab {gate.D_LAM:.0e}/{gate.D_MU:.0e} Pa, {gate.D_RHO} kg/m3;"
        f"  reflector layer jump c x {gate.REFL_JUMP}"
    )
    print(
        "  [A] thesis: layered reference, slab scatters"
        "   [B] whole-space reference, slab + reflector scatter"
    )
    print("=" * 100)
    head = (
        f"  {'c':>5} {'n':>2} {'arm':>3} {'unknowns':>8} {'rho(KT)':>8}"
        f" {'GMRES':>6} {'BCGVAR':>7} {'WH asym':>8} {'err/scat':>9}"
    )
    print(head)
    confirmed, refuted = True, False
    for c in STRENGTHS:
        for n in LADDER:
            s = Setup(c, n)
            row = {}
            for arm in ("A", "B"):
                a = build_arm(s, arm)
                n_u = a["kt"].shape[0]
                h = np.eye(n_u) - a["kt"]
                psi = np.linalg.solve(h, a["b"])
                err = float(np.abs(psi[:9] - a["exact"]).max() / np.abs(a["exact"] - a["incident"]).max())
                rho = float(np.abs(np.linalg.eigvals(a["kt"])).max())
                it_g = gmres_iterations(h, a["b"])
                it_b, asym = bcgvar_iterations(h, a["w"], a["b"], a["b_dual"])
                row[arm] = (it_g, err)
                print(
                    f"  {c:5.2f} {n:2d} {arm:>3} {n_u:8d} {rho:8.4f}"
                    f" {it_g:6d} {str(it_b):>7} {asym:8.1e} {err:9.2e}",
                    flush=True,
                )
            valid = row["A"][1] < 0.1 and row["B"][1] < 0.1
            faster = row["A"][0] < row["B"][0]
            confirmed = confirmed and valid and faster
            refuted = refuted or (valid and not faster)
    print("=" * 100)
    if confirmed:
        print("  CONFIRMED: both formulations reproduce the medium, and the stratified reference converges")
        print("  in fewer iterations at every strength and refinement.")
    elif refuted:
        print(
            "  REFUTED: both formulations are correct, and the stratified reference"
            " is NOT faster somewhere."
        )
    else:
        print(
            "  INCONCLUSIVE: an arm failed to reproduce the medium;"
            " the iteration counts are not comparable."
        )
    print("=" * 100)
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
