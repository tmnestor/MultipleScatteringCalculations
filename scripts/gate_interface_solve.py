"""Gate: the Galerkin scattering solve near a welded interface (step 2).

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (step 2).
Solver: cubic_scattering/interface_solver.py.

THE SCATTERER.  Eight cells straddling the interface (iz = -1, 0; ix, iy = 0, 1), each with a contrast of 10% of
its own side's lambda, mu and rho; h = 0.05 km, media A (5, 3, 2.5) over B (6.5, 3.6, 2.8), omega h / beta_A = 1/6
(Q0 h = 10).  Receiver cells (no contrast) above at iz = -4 (ix, iy = 0, 0 and 3, 1) and below at iz = 3
(0, 0 and 1, 3): planes clear of the scatterer's, so a plane-wave route reaches them.  A P wave incident from A
with lateral slowness (0.3, 0.1) / alpha_A.

CHECKS.
1. IDENTICAL MEDIA.  A = B: the solve equals the homogeneous one assembled from Paper 2's blocks alone, with the
   incident plane wave as right-hand side (``graded_voxel.solver.plane_wave_moments``): the interface blocks,
   the reference field and the per-side bookkeeping all reduce exactly.
2. BORN.  The scattered field at the receivers, s = psi - M^-1 <Q, u_ref>, from solves at contrast eps and 2 eps
   (Richardson: (4 s(eps) - s(2 eps)) / 2, the O(eps^2) term removed), against the Born field
   M^-1 sum_n K_rn E_n M^-1 <Q, u_ref>_n with every coupling K_rn by an independent route: the plane-wave
   (mode) integral of each block, direct, reflected or transmitted, from ``interface_rt`` alone (no closed
   forms, no split), on the radial rule with all four branch points as panel ends.  The receivers' planes do
   not touch the scatterer's, so the integrals converge exponentially.
3. CONVERGENCE IN p.  The scattered field at the receivers for p = 0, 1, 2 (r = 0).
4. COST.  The set-up (frequency-independent parts, once) against one frequency (blocks and solve).

Run:  PYTHONPATH=. python scripts/gate_interface_solve.py [checks...]   (default: all; e.g. "identical born")
"""

import sys
import time

import numpy as np
from numpy.typing import NDArray

from cubic_scattering import interface_solver as iso
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.graded_voxel.basis import gram_test, source_expansion
from cubic_scattering.graded_voxel.blocks import coupling_block
from cubic_scattering.graded_voxel.solver import field_sizes, plane_wave_moments
from cubic_scattering.interface_spectral import radial_rule
from cubic_scattering.sphere_scattering import _plane_wave_strain_voigt
from cubic_scattering.stratified_march import (
    SOURCE_PAIRING,
    interface_rt,
    mode_family,
    receiver_moments,
    source_moments_local,
)

A = ReferenceMedium(5.0, 3.0, 2.5)
B = ReferenceMedium(6.5, 3.6, 2.8)
H = 0.05
OMEGA = A.beta / (6 * H)
SCATTERER = [(iz, ix, iy) for iz in (-1, 0) for ix in (0, 1) for iy in (0, 1)]
RECEIVERS = [(-4, 0, 0), (-4, 3, 1), (3, 0, 0), (3, 1, 3)]
KX, KY = 0.3 * OMEGA / A.alpha, 0.1 * OMEGA / A.alpha


def contrast(medium: ReferenceMedium, eps: float) -> MaterialContrast:
    return MaterialContrast(eps * medium.lam, eps * medium.mu, eps * medium.rho)


def contrasts(grid: iso.InterfaceGrid, eps: float) -> list:
    return [
        contrast(grid.medium(m), eps) if m < len(SCATTERER) else None
        for m in range(len(grid.idx))
    ]


# ---------------------------------------------------------------------------
# The independent coupling: the plane-wave (mode) integral of a block
# ---------------------------------------------------------------------------


def _modes(kx, ky, ref, sign):
    out = []
    for w in ("P", "S"):
        out += mode_family(kx, ky, OMEGA, ref, w, sign)
    return out  # (k, d, w) for P, SV, SH


def mode_block(grid, r, s, n_source, n_test, n_phi=192):
    """K_rs by the plane-wave integral: sum over the modes leaving the source cell towards the receiver's side
    and reaching it directly (same side) and through the interface (reflected or transmitted)."""
    zr, zs = grid.centres[r][0], grid.centres[s][0]
    r_in_a, s_in_a = zr < 0, zs < 0
    src_med = A if s_in_a else B
    rec_med = A if r_in_a else B
    gap = min(abs(zr - zs) if r_in_a == s_in_a else np.inf, abs(zr) + abs(zs)) - 2 * H
    qh_max = 45.0 * H / gap + 1.0
    branch = [OMEGA * H / c for c in (A.alpha, A.beta, B.alpha, B.beta)]
    qh, wq = radial_rule(qh_max, branch, 1.0, order=24)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    qq, pp = np.meshgrid(qh / H, phi, indexing="ij")
    weight = (np.repeat((wq / H * qh / H)[:, None], n_phi, 1) / (2 * np.pi * n_phi)).ravel()
    kx, ky = (qq * np.cos(pp)).ravel(), (qq * np.sin(pp)).ravel()
    cr, cs = grid.centres[r], grid.centres[s]
    rt = interface_rt(kx, ky, OMEGA, A, B)

    def side(k, d, centre, n, receiver):
        ph = np.exp((1j if receiver else -1j) * (k @ centre))
        mom = receiver_moments(k, H, n) if receiver else source_moments_local(k, H, n)
        return ph[:, None] * mom, d

    out = np.zeros((n_test, n_source, 9, 9), dtype=complex)
    # the modes the source radiates: towards the interface (the reflected or transmitted path), and directly
    s_sign_to_interface = 1.0 if s_in_a else -1.0
    src_to_if = _modes(kx, ky, src_med, s_sign_to_interface)
    paths = []  # (receiver modes, coupling (N, 3, 3) or None for direct, source modes)
    if r_in_a == s_in_a:
        rec_from_if = _modes(kx, ky, rec_med, -s_sign_to_interface)  # back away from the interface
        coupling = rt["R_d"] if s_in_a else rt["R_u"]
        paths.append((rec_from_if, coupling, src_to_if))
        direct = _modes(kx, ky, src_med, float(np.sign(zr - zs)))
        paths.append((direct, None, direct))
    else:
        rec_from_if = _modes(kx, ky, rec_med, s_sign_to_interface)  # onward through the interface
        coupling = rt["T_d"] if s_in_a else rt["T_u"]
        paths.append((rec_from_if, coupling, src_to_if))
    for rec_modes, coupling, src_modes in paths:
        for n, (ks, ds, ws) in enumerate(src_modes):
            sc, dsv = side(ks, ds, cs, n_source, False)
            for m, (kr, dr, _) in enumerate(rec_modes):
                if coupling is None and m != n:
                    continue
                c = 1.0 if coupling is None else coupling[:, m, n]
                ra, drv = side(kr, dr, cr, n_test, True)
                coef = weight * ws * c
                out += np.einsum("n,na,nc,ni,nj->acij", coef, ra, sc, drv, dsv * SOURCE_PAIRING)
    return out


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def scattered(res: iso.InterfaceResult, na: int) -> NDArray:
    m = np.diag(gram_test(H, res.reference.shape[1]))[:na]
    return res.psi[:, :na] - res.reference[:, :na] / m[None, :, None]


def check_identical() -> bool:
    grid = iso.InterfaceGrid(np.array(SCATTERER), H, A, A)
    c = iso.InterfaceCouplings(grid)
    res = iso.solve_interface(c, OMEGA, contrasts(grid, 0.1), KX, KY, "P", p=1)
    # the homogeneous solve: Paper 2's blocks, the incident plane wave
    blocks = {
        (m, n): coupling_block(tuple(int(v) for v in grid.idx[m] - grid.idx[n]), H, OMEGA, A, 10, 4)
        for m in range(len(SCATTERER))
        for n in range(len(SCATTERER))
    }
    a = iso.assemble(grid, blocks, res.delta, 1, 0)
    k_vec = np.array([np.sqrt((OMEGA / A.alpha) ** 2 - KX**2 - KY**2), KX, KY])
    k_hat = k_vec / np.linalg.norm(k_vec)
    amp = np.concatenate([k_hat + 0j, _plane_wave_strain_voigt(k_hat, k_hat, OMEGA / A.alpha)])
    rhs = plane_wave_moments(grid.centres, H, k_vec, amp, 4)
    psi = np.linalg.solve(a, rhs.ravel()).reshape(res.psi.shape)
    ref_err = np.abs(res.reference - rhs).max() / np.abs(rhs).max()
    err = np.abs(res.psi - psi).max() / np.abs(psi).max()
    ok = err < 1e-8 and ref_err < 1e-14
    print(f"1. identical media: reference field {ref_err:.1e}, solution vs homogeneous solve {err:.1e}"
          f"  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def check_born() -> bool:
    grid = iso.InterfaceGrid(np.array(SCATTERER + RECEIVERS), H, A, B)
    t0 = time.time()
    c = iso.InterfaceCouplings(grid)
    print(f"   set-up {time.time() - t0:.0f} s, {len(c.classes)} classes", flush=True)
    blocks = c.blocks(OMEGA)
    eps = 1e-6
    s1 = scattered(iso.solve_interface(c, OMEGA, contrasts(grid, eps), KX, KY, "P", p=1, blocks=blocks), 4)
    s2 = scattered(iso.solve_interface(c, OMEGA, contrasts(grid, 2 * eps), KX, KY, "P", p=1, blocks=blocks), 4)
    solve = (4 * s1 - s2) / (2 * eps)
    # Born with the mode-route couplings
    unit = iso.cell_contrasts(grid, OMEGA, contrasts(grid, 1e-3), 0) / 1e-3  # linear in the contrast
    ref = iso.reference_moments(grid, OMEGA, KX, KY, "P", 4)
    m = np.diag(gram_test(H, 4))
    psi0 = ref / m[None, :, None]
    ok = True
    worst = 0.0
    for i, r in enumerate(range(len(SCATTERER), len(grid.idx))):
        born = np.zeros((4, 9), dtype=complex)
        for n in range(len(SCATTERER)):
            k = mode_block(grid, r, n, 10, 4)
            e = source_expansion(unit[n], 4)  # (n_source, 4, 9, 9)
            born += np.einsum("acij,cbjk,bk->ai", k, e, psi0[n])
            if i == 0 and n == 0:
                kb = blocks[r, n]
                print(f"   block ({RECEIVERS[0]} <- {SCATTERER[0]}): mode route vs solver's "
                      f"{np.abs(k - kb).max() / np.abs(kb).max():.1e}", flush=True)
        born /= m[:, None]
        err = np.abs(solve[r] - born).max() / np.abs(born).max()
        worst = max(worst, err)
        print(f"   receiver {RECEIVERS[i]}: solve (eps -> 0) vs Born by plane waves {err:.1e}", flush=True)
    ok = worst < 1e-8
    print(f"2. Born: worst {worst:.1e}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def check_p() -> bool:
    grid = iso.InterfaceGrid(np.array(SCATTERER + RECEIVERS), H, A, B)
    _, n_field, _, n_source = field_sizes(2, 0)
    t0 = time.time()
    c = iso.InterfaceCouplings(grid, n_source=n_source, n_test=n_field)
    t1 = time.time()
    blocks = c.blocks(OMEGA)
    t2 = time.time()
    out = {}
    for p in (0, 1, 2):
        ts = time.time()
        res = iso.solve_interface(c, OMEGA, contrasts(grid, 0.1), KX, KY, "P", p=p, blocks=blocks)
        out[p] = scattered(res, 1)[len(SCATTERER):, 0]  # the receivers' cell means
        print(f"   p = {p}: solve {time.time() - ts:.2f} s", flush=True)
    d01 = np.abs(out[0] - out[2]).max() / np.abs(out[2]).max()
    d12 = np.abs(out[1] - out[2]).max() / np.abs(out[2]).max()
    print(f"3. convergence in p: receivers' mean scattered field, |p0 - p2| {d01:.1e}, |p1 - p2| {d12:.1e}"
          f"  {'PASS' if d12 < d01 else 'FAIL'}", flush=True)
    print(f"4. cost: set-up {t1 - t0:.0f} s ({len(c.classes)} classes, n_source {n_source}, n_test {n_field});"
          f" one frequency: blocks {t2 - t1:.1f} s + solve (above)", flush=True)
    return d12 < d01


def main() -> None:
    which = sys.argv[1:] or ["identical", "born", "p"]
    checks = {"identical": check_identical, "born": check_born, "p": check_p}
    ok = all([checks[w]() for w in which])
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
