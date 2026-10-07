"""The density-density second-order term, against its exact value.

Density-only contrast. At k_S^2 the solution's second-order part is psi_2 - rhs_2 = M(B_2 * psi_0) with
psi_0 = POL (A_0 = I, B_1 = 0 without a stiffness contrast). Its force moment
    F2 = beta^2 Drho sum_c int_c s_h(x) u2(x) dx
is compared with the exact
    F2 = beta^4 Drho^2 g E POL,  g = [1 - (1 - R^2)/3] / (4 pi mu),  E = int int s(x) s(y) / |x - y|,
(the isotropic average of the static Green's function over a radial body), E by 1-D radial integrals.
Usage: python dens2.py [--rc=1] [--profile=..] n1 n2 ...
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path("/home/user/MultipleScatteringCalculations")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import measure_graded_sphere_frequency_series as fs  # noqa: E402
from cubic_scattering import MaterialContrast  # noqa: E402

gs, gfft, lowf, hier = fs.gs, fs.gfft, fs.lowf, fs.hier


def exact_f2():
    a = gs.RADIUS
    s = lambda r: float(gs.smoothstep(np.array([r]))[0])
    br = [gs.CORE]
    m = lambda r: quad(lambda t: s(t) * t * t, 0, r, points=[p for p in br if p < r], epsabs=0, epsrel=1e-13)[0]
    tail = lambda r: quad(lambda t: s(t) * t, r, a, points=[p for p in br if p > r], epsabs=0, epsrel=1e-13)[0]
    phi = lambda r: 4 * math.pi * (m(r) / r + tail(r))
    e = quad(lambda r: 4 * math.pi * r * r * s(r) * phi(r), 0, a, points=br, epsabs=0, epsrel=1e-12, limit=200)[0]
    mu = fs.MU
    g = (1 - (1 - fs.RATIO**2) / 3) / (4 * math.pi * mu)
    return fs.REF.beta**4 * fs.CONTRAST.Drho**2 * g * e


def model_f2(n):
    side, centres, coefs, asm, sols = fs.solve_series(n)
    # rhs_2 at the cells (incident coefficient of k_S^2)
    rhs2 = np.zeros_like(sols[2])
    for c, xc in enumerate(centres):
        kx = float(gs.K_HAT @ xc)
        for pi, p_idx in enumerate(asm.u_list):
            if len(p_idx) > 2:
                continue
            kp_pow = np.prod([gs.K_HAT[ax] for ax in p_idx]) if p_idx else 1.0
            mm = 2 - len(p_idx)
            val = (1j * fs.RATIO) ** 2 * kp_pow * kx**mm / math.factorial(mm)
            for i in fs.AXES:
                rhs2[c, i * asm.nu + pi] += gs.POL[i] * val
    u2 = sols[2] - rhs2
    x1, w1 = np.polynomial.legendre.leggauss(6)
    x1, w1 = 0.5 * side * x1, 0.5 * side * w1
    xi = np.stack(np.meshgrid(x1, x1, x1, indexing="ij"), -1).reshape(-1, 3)
    wts = np.einsum("i,j,k->ijk", w1, w1, w1).ravel()
    mono_u = np.stack([gs.monomial(xi, w) for w in asm.u_list])
    mono_v = np.stack([gs.monomial(xi, v) for v in asm.v_list])
    f = np.zeros(3, dtype=complex)
    for c in range(len(centres)):
        prof = coefs[c] @ mono_v
        u = np.einsum("jw,w,wg->gj", u2[c].reshape(3, asm.nu), asm.coef, mono_u)
        f += (wts * prof) @ u
    return fs.REF.beta**2 * fs.CONTRAST.Drho * f


if __name__ == "__main__":
    args = sys.argv[1:]
    for fl in [x for x in args if x.startswith("--rc=")]:
        gfft.R_C = int(fl.split("=")[1]); args.remove(fl)
    prof = "smoothstep"
    for fl in [x for x in args if x.startswith("--profile=")]:
        prof = fl.split("=")[1]; args.remove(fl)
    lowf.NEAR = -1
    fs.J = 2
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile(prof)
    fs.CONTRAST = MaterialContrast(Dlambda=0.0, Dmu=0.0, Drho=hier.CONTRAST.Drho)
    ex = exact_f2()
    prev = None
    for n in [int(v) for v in args]:
        f = model_f2(n)
        err = abs(f[0] - ex) / abs(ex)
        o = "" if prev is None else f"  order {math.log(prev[1] / err) / math.log(n / prev[0]):.2f}"
        print(f"{prof} rc={gfft.R_C} n={n:2d}  F2 rel err {err:.3e}  (transverse {abs(f[1]):.1e}){o}", flush=True)
        prev = (n, err)
