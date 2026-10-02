#!/usr/bin/env python3
"""The gradient hierarchy of the single site in three dimensions, tested on a ball against the exact sphere.

The hierarchy of degree q expands the displacement inside one scatterer about its centre,

    u_j(x') = sum_{|W| <= q} x'_W U_{j;W} / |W|! ,        U_{j;W} = d_W u_j (0),

takes the strain as its derivative, and imposes the Lippmann-Schwinger equation and its derivatives up to
order q AT THE CENTRE. With g[i,n; D; W] the moment of the background Green's tensor over the scatterer,

    g[i,n; D; W] = < d_D G_in , x_W 1_V >      (field derivatives, distributional: the delta terms kept),

the equation for the derivative multi-index P is

    U_{i;P} - (-1)^|P| omega^2 drho  sum_W g[i,j; P; W] U_{j;W} / |W|!
            + (-1)^|P|               sum_W g[i,n; P+k; W] dc_{nkrj} U_{j;rW} / |W|!   =  d_P u0_i (0),

with |W| <= q in the density sum and |W| <= q - 1 in the stiffness sum. For q = 1 and drho = 0 this is the
first-gradient block of the paper (the uniform-strain closure); for q = 2 it is the closed set with the
six moments; q = 3 is the third-gradient system, which needs the grades (1,3), (3,3), (4,0), (4,2) as well.

WHY A BALL FIRST. The assembly is the same for any centrosymmetric scatterer; only the scalar moments
E[m; D; W] change. For a ball they are elementary (derivatives moved onto the sphere), and the answer is
known exactly: the elastic Mie solution. So the whole tensor assembly, with every moment grade and every
index, is tested here against an external reference before the cube's moments are put into it.

The moments are dynamic: with g_c = exp(i k_c r)/r expanded in powers of r,
    G_in = (1/(4 pi mu)) [ delta_in g_S + (1/k_S^2) d_i d_n (g_S - g_P) ],
every moment is a finite sum of the scalar moments of r^(t-1), t = 0..T.

WHAT IS MEASURED. A P plane wave on a homogeneous sphere of radius a. The far field radiated by the
polynomial source of the hierarchy is compared with the exact Mie displacement at nine scattering angles,
5e8 radii away; the error is the largest component of the difference over the largest exact component.

PREDICTED, before running, from the parity pairs of the layer: with both contrasts the error is
O((k a)^2) for q = 1 and q = 2 and O((k a)^4) for q = 3; with a density contrast alone, O((k a)^4)
already at q = 2.

Run:  python scripts/measure_ball_gradient_hierarchy.py
"""

import functools
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crosscheck_cube_moments_ball_shell import deriv, sphere_monomial, times_monomial  # noqa: E402
from cubic_scattering import MaterialContrast, ReferenceMedium  # noqa: E402
from cubic_scattering.sphere_scattering import compute_elastic_mie, mie_scattered_displacement  # noqa: E402
from gate_sphere_cell_average_vs_mie import obs_points  # noqa: E402

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
RADIUS = 10.0
THETA = np.linspace(0.2, np.pi - 0.2, 9)
R_FAR = 5.0e8 * RADIUS
SERIES = 12  # powers of r kept in exp(i k r)/r
AXES = (0, 1, 2)


# ------------------------------------------------------------ scalar moments of the unit ball
@functools.cache
def ball_unit(m: int, ds: tuple[int, ...], w: tuple[int, ...]) -> float:
    """< d_ds r^m, x_w 1_B > for the UNIT ball, the derivatives peeled onto the sphere."""
    if not ds:
        tot = 0.0
        for (a, b, c, p), coef in times_monomial({(0, 0, 0, m): 1.0}, list(w)).items():
            ang = sphere_monomial(a, b, c)
            if ang != 0.0:
                tot += coef * ang / (a + b + c + p + 3)
        return tot
    q, rest = ds[0], ds[1:]
    kern = {(0, 0, 0, m): 1.0}
    for ax in rest:
        kern = deriv(kern, ax)
    tot = 0.0
    for (a, b, c, _p), coef in times_monomial(kern, list(w)).items():
        e = [a, b, c]
        e[q] += 1
        tot += coef * sphere_monomial(e[0], e[1], e[2])
    for u, wu in enumerate(w):
        if wu == q:
            tot -= ball_unit(m, rest, w[:u] + w[u + 1 :])
    return tot


def scalar_moment(radius: float, m: int, ds: tuple[int, ...], w: tuple[int, ...]) -> float:
    """The same for a ball of this radius: homogeneous of degree m - D + W + 3."""
    if m % 2 == 0 and len(ds) > m:
        return 0.0  # r^m is a polynomial of degree m: more than m derivatives annihilate it
    return radius ** (m - len(ds) + len(w) + 3) * ball_unit(m, tuple(sorted(ds)), tuple(sorted(w)))


def tensor_moment(
    radius: float, omega: float, i: int, n: int, ds: tuple[int, ...], w: tuple[int, ...]
) -> complex:
    """g[i,n; ds; w] = < d_ds G_in, x_w 1_V >, dynamic, by the series of exp(i k r)/r."""
    ks, kp = omega / REF.beta, omega / REF.alpha
    mu = REF.rho * REF.beta**2
    tot = 0.0j
    for t in range(SERIES + 1):
        if i == n:
            tot += (1j * ks) ** t / math.factorial(t) * scalar_moment(radius, t - 1, ds, w)
        if t >= 2:
            coef = ((1j * ks) ** t - (1j * kp) ** t) / math.factorial(t) / ks**2
            tot += coef * scalar_moment(radius, t - 1, (*ds, i, n), w)
    return tot / (4.0 * math.pi * mu)


# ----------------------------------------------------------------------------- the system
def stiffness(contrast: MaterialContrast) -> np.ndarray:
    dc = np.zeros((3, 3, 3, 3))
    for a, b, c, d in itertools.product(AXES, repeat=4):
        dc[a, b, c, d] = contrast.Dlambda * (a == b) * (c == d) + contrast.Dmu * (
            (a == c) * (b == d) + (a == d) * (b == c)
        )
    return dc


def multi_indices(q: int) -> list[tuple[int, ...]]:
    """Sorted derivative multi-indices of length 0..q."""
    return [w for m in range(q + 1) for w in itertools.combinations_with_replacement(AXES, m)]


def solve_site(
    radius: float, omega: float, contrast: MaterialContrast, q: int, k_hat: np.ndarray, pol: np.ndarray
):
    """The hierarchy of degree q for one ball: returns (multi-indices, U[j, index])."""
    idx = multi_indices(q)
    pos = {w: n for n, w in enumerate(idx)}
    nu = len(idx)
    dc = stiffness(contrast)
    a_rho = omega**2 * contrast.Drho
    kp = omega / REF.alpha
    big = np.eye(3 * nu, dtype=complex)
    rhs = np.zeros(3 * nu, dtype=complex)
    for pi, p_idx in enumerate(idx):
        sign = (-1) ** len(p_idx)
        for i in AXES:
            row = i * nu + pi
            # incident plane wave: d_P u0_i (0) = pol_i prod (i k khat_p)
            rhs[row] = pol[i] * np.prod([1j * kp * k_hat[ax] for ax in p_idx]) if p_idx else pol[i]
            for wlen in range(q + 1):
                fact = math.factorial(wlen)
                for w in itertools.product(AXES, repeat=wlen):
                    if (len(p_idx) + wlen) % 2 == 0:  # density: grade (|P|, |W|)
                        col_w = pos[tuple(sorted(w))]
                        for j in AXES:
                            g = tensor_moment(radius, omega, i, j, p_idx, w)
                            big[row, j * nu + col_w] -= sign * a_rho * g / fact
                    if (
                        wlen <= q - 1 and (len(p_idx) + 1 + wlen) % 2 == 0
                    ):  # stiffness: grade (|P| + 1, |W|)
                        for k, n in itertools.product(AXES, repeat=2):
                            g = tensor_moment(radius, omega, i, n, (*p_idx, k), w)
                            if g == 0.0:
                                continue
                            for r, j in itertools.product(AXES, repeat=2):
                                if dc[n, k, r, j] != 0.0:
                                    col = pos[tuple(sorted((r, *w)))]
                                    big[row, j * nu + col] += sign * g * dc[n, k, r, j] / fact
    sol = np.linalg.solve(big, rhs).reshape(3, nu)
    return idx, sol, float(np.linalg.cond(big))


# ----------------------------------------------------------------------------- the far field
def ball_quadrature(radius: float, n: int = 14) -> tuple[np.ndarray, np.ndarray]:
    xr, wr = leggauss(n)
    r = 0.5 * radius * (xr + 1.0)
    wr = 0.5 * radius * wr * r**2
    xc, wc = leggauss(n)
    phi = 2.0 * np.pi * (np.arange(2 * n) + 0.5) / (2 * n)
    pts, wts = [], []
    for rr, w1 in zip(r, wr, strict=True):
        for c, w2 in zip(xc, wc, strict=True):
            s = math.sqrt(1.0 - c * c)
            for ph in phi:
                pts.append([rr * c, rr * s * math.cos(ph), rr * s * math.sin(ph)])
                wts.append(w1 * w2 * 2.0 * np.pi / (2 * n))
    return np.array(pts), np.array(wts)


def far_field(radius, omega, contrast, idx, sol, obs: np.ndarray) -> np.ndarray:
    """Displacement at the far observers radiated by the polynomial source of the hierarchy."""
    pts, wts = ball_quadrature(radius)
    dc = stiffness(contrast)
    u = np.zeros((len(pts), 3), dtype=complex)
    grad = np.zeros((len(pts), 3, 3), dtype=complex)  # grad[:, r, j] = d_r u_j
    for n_w, w in enumerate(idx):
        mult = math.factorial(len(w)) / math.prod(math.factorial(w.count(ax)) for ax in AXES)
        mono = np.prod([pts[:, ax] for ax in w], axis=0) if w else np.ones(len(pts))
        u += mult / math.factorial(len(w)) * mono[:, None] * sol[:, n_w][None, :]
        for r in set(w):
            rest = list(w)
            rest.remove(r)
            dmono = w.count(r) * (
                np.prod([pts[:, ax] for ax in rest], axis=0) if rest else np.ones(len(pts))
            )
            grad[:, r, :] += mult / math.factorial(len(w)) * dmono[:, None] * sol[:, n_w][None, :]
    force = omega**2 * contrast.Drho * u
    tau = np.einsum("nkrj,prj->pnk", dc, grad)
    out = np.zeros((len(obs), 3), dtype=complex)
    for o, x in enumerate(obs):
        dist = float(np.linalg.norm(x))
        rh = x / dist
        for k_c, speed, longitudinal in (
            (omega / REF.alpha, REF.alpha, True),
            (omega / REF.beta, REF.beta, False),
        ):
            phase = np.exp(-1j * k_c * (pts @ rh)) * wts
            amp = phase @ force + 1j * k_c * np.einsum("p,pnk,k->n", phase, tau, rh)
            amp = rh * (rh @ amp) if longitudinal else amp - rh * (rh @ amp)
            out[o] += np.exp(1j * k_c * dist) / (4.0 * math.pi * REF.rho * speed**2 * dist) * amp
    return out


def main() -> int:
    k_hat = np.array([1.0, 0.0, 0.0])
    pol = np.array([1.0, 0.0, 0.0])
    obs = obs_points(R_FAR, THETA)
    kas = [0.4, 0.2, 0.1]
    ok = True
    for label, contrast, want in (
        ("density only", MaterialContrast(0.0, 0.0, CONTRAST.Drho), {1: 2, 2: 4, 3: 4}),
        ("modulus only", MaterialContrast(CONTRAST.Dlambda, CONTRAST.Dmu, 0.0), {1: 2, 2: 2, 3: 4}),
        ("both", CONTRAST, {1: 2, 2: 2, 3: 4}),
    ):
        print(f"\n{label}: relative far-field error against the exact sphere, k_S a = {kas}")
        for q in (1, 2, 3):
            errs, cond = [], 0.0
            for ka in kas:
                omega = ka * REF.beta / RADIUS
                exact = mie_scattered_displacement(compute_elastic_mie(omega, RADIUS, REF, contrast), obs)
                idx, sol, cond = solve_site(RADIUS, omega, contrast, q, k_hat, pol)
                got = far_field(RADIUS, omega, contrast, idx, sol, obs)
                errs.append(float(np.max(np.abs(got - exact)) / np.max(np.abs(exact))))
            orders = [
                math.log(errs[n] / errs[n + 1]) / math.log(kas[n] / kas[n + 1]) for n in range(len(kas) - 1)
            ]
            good = abs(orders[-1] - want[q]) < 0.5
            ok = ok and good
            print(
                f"  q = {q} ({3 * len(idx):2d} unknowns, condition {cond:.1e}):  "
                + "  ".join(f"{e:.2e}" for e in errs)
                + f"   orders {orders[0]:.2f} {orders[1]:.2f}   predicted {want[q]}   "
                + ("PASS" if good else "****FAIL****")
            )
    print("\nALL PASS" if ok else "\nSOME PREDICTIONS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
