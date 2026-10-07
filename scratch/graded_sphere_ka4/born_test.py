"""Born-limit test: is the slow (ka)^4 convergence present at first order in the contrast?

Cell model: psi_j = rhs_j (incident Taylor coefficients, no solve), far field by fs.far_series.
Exact Born: the same far-field formula with the exact profile and the exact plane wave, by a
spherical quadrature that integrates the polynomial-times-profile integrands exactly.
Usage: python born_test.py [--rc=1] [--profile=smoothstep] n1 n2 ...
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/MultipleScatteringCalculations")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import measure_graded_sphere_frequency_series as fs  # noqa: E402

gs, gfft, hier = fs.gs, fs.gfft, fs.hier
J, RATIO, REF, CONTRAST = fs.J, fs.RATIO, fs.REF, fs.CONTRAST


def cell_rhs(n_sub):
    side, centres, coefs, grid = gs.build_cells(n_sub, gfft.R_C)
    asm = gs.Assembler(3, gfft.R_C, 0.0, CONTRAST)
    sols = [np.zeros((len(centres), 3 * asm.nu), dtype=complex) for _ in range(J + 1)]
    for c, xc in enumerate(centres):
        kx = float(gs.K_HAT @ xc)
        for pi, p_idx in enumerate(asm.u_list):
            kp_pow = np.prod([gs.K_HAT[ax] for ax in p_idx]) if p_idx else 1.0
            for j in range(len(p_idx), J + 1):
                m = j - len(p_idx)
                val = (1j * RATIO) ** j * kp_pow * kx**m / math.factorial(m)
                for i in fs.AXES:
                    sols[j][c, i * asm.nu + pi] += gs.POL[i] * val
    return fs.far_series(side, centres, coefs, asm, sols, OBS)


def exact_born():
    a, b = gs.RADIUS, gs.CORE
    xr = []
    for lo, hi in ((0.0, b), (b, a)):
        x, w = np.polynomial.legendre.leggauss(80)
        xr.append((0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w))
    r = np.concatenate([p[0] for p in xr]); wr = np.concatenate([p[1] for p in xr])
    ct, wt = np.polynomial.legendre.leggauss(30)
    nph = 60
    ph = 2 * np.pi * np.arange(nph) / nph
    R, CT, PH = np.meshgrid(r, ct, ph, indexing="ij")
    W = (wr[:, None, None] * r[:, None, None] ** 2) * wt[None, :, None] * (2 * np.pi / nph)
    st = np.sqrt(1 - CT**2)
    pts = np.stack([R * CT, R * st * np.cos(PH), R * st * np.sin(PH)], -1).reshape(-1, 3)
    W = np.broadcast_to(W, R.shape).ravel()
    s = gs.smoothstep(np.linalg.norm(pts, axis=1))
    dc = hier.stiffness(CONTRAST)
    dirs = OBS / np.linalg.norm(OBS, axis=1)[:, None]
    n_pow = J + 3
    out = {m: np.zeros((n_pow, len(OBS), 3), dtype=complex) for m in "PS"}
    x0 = pts @ gs.K_HAT
    for j in range(J + 1):
        # incident coefficient of k_S^j
        u = (1j * RATIO * x0) ** j / math.factorial(j)
        u = u[:, None] * gs.POL[None, :]
        g = np.zeros((len(pts), 3, 3), dtype=complex)  # grad[g, r, j]
        if j >= 1:
            gj = 1j * RATIO * (1j * RATIO * x0) ** (j - 1) / math.factorial(j - 1)
            g = gj[:, None, None] * gs.K_HAT[None, :, None] * gs.POL[None, None, :]
        force = REF.beta**2 * CONTRAST.Drho * s[:, None] * u
        tau = s[:, None, None] * np.einsum("nkrj,grj->gnk", dc, g)
        for o, rh in enumerate(dirs):
            proj = pts @ rh
            for mode, kr, speed in (("P", RATIO, REF.alpha), ("S", 1.0, REF.beta)):
                pref = 1.0 / (4.0 * math.pi * REF.rho * speed**2)
                for m in range(n_pow - j):
                    phw = (-1j * kr * proj) ** m / math.factorial(m) * W
                    terms = []
                    if j + m + 2 < n_pow:
                        terms.append((j + m + 2, phw @ force))
                    if j + m + 1 < n_pow:
                        terms.append((j + m + 1, 1j * kr * np.einsum("g,gnk,k->n", phw, tau, rh)))
                    for p, amp in terms:
                        amp = rh * (rh @ amp) if mode == "P" else amp - rh * (rh @ amp)
                        out[mode][p, o] += pref * amp
    return out


def report(tag, vox, ex):
    a = gs.RADIUS
    lead = max(np.abs(ex[m][2]).max() for m in "PS") * a**-2
    row = []
    for p in range(2, J + 2):
        diff = max(np.abs(vox[m][p] - ex[m][p]).max() for m in "PS") * a**-p
        size = max(np.abs(ex[m][p]).max() for m in "PS") * a**-p
        rel = diff / size if size > 1e-12 * lead else diff / lead
        row.append(f"{p}:{rel:.2e}")
    print(tag, " ".join(row), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    for f in [x for x in args if x.startswith("--rc=")]:
        gfft.R_C = int(f.split("=")[1]); args.remove(f)
    prof = "smoothstep"
    for f in [x for x in args if x.startswith("--profile=")]:
        prof = f.split("=")[1]; args.remove(f)
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile(prof)
    OBS = fs.obs_points(gs.R_FAR, gs.THETA)
    ex = exact_born()
    for n in [int(v) for v in args]:
        report(f"born n={n:2d} rc={gfft.R_C} {prof}", cell_rhs(n), ex)
