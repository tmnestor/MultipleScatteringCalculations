#!/usr/bin/env python3
"""The closed form of the third-order factor F3 of constant cells, from the body's smooth fields alone.

As the cells shrink, constant cells (p = r = 0) give, to second order in the half-width h (the
derivation is in the octree paper, section 3.6),

  T3 - T3_scheme = (h^2/3) int [ grad f . grad (w_ab + w_ba)
                    + f sum_i ( d_i u_a : C : d_i u_b - d_i f (d_i u_a : C : G_ib + G_ia : C : d_i u_b) )
                    + sum_i d_i f ( u_a : C : d_i u_b + d_i u_a : C : u_b )
                    - sum_i (d_i f)^2 ( u_a : C : G_ib + G_ia : C : u_b ) ] dV,

with u_b = Gamma * (f b), u_a = Gamma * (f a),
w_ab = a : Gamma * (f C : u_b), w_ba = b : Gamma * (f C : u_a),
G_ib = Gamma(e_i) b the operator on cube axis i, C the stiffness contrast, a = C : e_out, b = C : e_in,
and E = (h^2/3) int |grad f|^2 / int f^2, so that

  F3 = (T3 - T3_scheme) / (T3 E),      T3 = int f u_a : C : u_b dV.

The same expansion for the second-order term gives F = [2 int grad f . grad (a : Gamma * (f b))
- sum_i int (d_i f)^2 a : G_ib] / (T2 int |grad f|^2 / int f^2), which for a radial profile is
2 - m_ax/m_bar (section 3.4): printed as a check of the machinery.

Only smooth fields enter, so the grid resolves the body, not the cells. Compare with the direct evaluation
on voxels (``measure_t3_tensor_forms.py``) and the scheme (``measure_t3_law_3d.py``).

Run:  python scripts/measure_f3_closed_form.py <grid points across the body> <padding>   e.g. 64 2
"""

import sys
import time

import numpy as np

RADIUS, CORE = 10.0, 5.0
alpha, beta, rho = 5000.0, 3000.0, 2500.0
mu = rho * beta**2
lam = rho * alpha**2 - 2 * mu
dl, dm = 2.0e9, 1.0e9
kappa = (lam + mu) / (lam + 2 * mu)
PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
WEIGHT = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # a : b in Voigt order

nb, pad = int(sys.argv[1]), float(sys.argv[2])
N = int(round(pad * nb / 2)) * 2
dx = 2 * RADIUS / nb
t0 = time.perf_counter()
x1 = (np.arange(N) - N / 2 + 0.5) * dx
X, Y, Z = np.meshgrid(x1, x1, x1, indexing="ij", sparse=True)
r = np.sqrt(X**2 + Y**2 + Z**2)
xs = np.clip((RADIUS - r) / (RADIUS - CORE), 0.0, 1.0)
f = 10 * xs**3 - 15 * xs**4 + 6 * xs**5
del xs, r

k1 = 2 * np.pi * np.fft.fftfreq(N, dx)
KV = np.meshgrid(k1, k1, k1, indexing="ij", sparse=True)
kk = np.sqrt(KV[0] ** 2 + KV[1] ** 2 + KV[2] ** 2)
kk[0, 0, 0] = 1.0
XI = [KV[i] / kk for i in range(3)]


def gamma_hat(tau_hat: list, unit: list) -> list:
    """Gamma(xi) applied to a symmetric tensor (six Voigt components) at unit vectors."""
    t = [sum(tau_hat[PAIRS.index(tuple(sorted((i, j))))] * unit[j] for j in range(3)) for i in range(3)]
    xt = sum(unit[i] * t[i] for i in range(3))
    v = [(t[i] - kappa * unit[i] * xt) / mu for i in range(3)]
    return [(unit[i] * v[j] + unit[j] * v[i]) / 2 for i, j in PAIRS]


# Gamma's average over directions, for the zero wavevector
nf = 20000
kf = np.arange(nf) + 0.5
ph = np.arccos(1 - 2 * kf / nf)
th = np.pi * (1 + 5**0.5) * kf
DIRS = [np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)]


def gamma_field(tau: list) -> list:
    """Gamma * tau for a symmetric tensor field tau (six real arrays)."""
    tau_hat = [np.fft.fftn(c) for c in tau]
    out_hat = gamma_hat(tau_hat, XI)
    means = [complex(c[0, 0, 0]) for c in tau_hat]
    zero = [float(np.mean(c)) for c in gamma_hat([np.full(nf, m) for m in means], DIRS)]
    res = []
    for c, z in zip(out_hat, zero, strict=True):
        c = np.asarray(c).copy()
        c[0, 0, 0] = z
        res.append(np.fft.ifftn(c).real)
    return res


def grad(field: np.ndarray) -> list:
    fh = np.fft.fftn(field)
    return [np.fft.ifftn(1j * KV[i] * fh).real for i in range(3)]


def voigt(t: np.ndarray) -> list:
    return [t[i, j] for i, j in PAIRS]


def dC(e: list) -> list:
    tr = e[0] + e[1] + e[2]
    return [dl * tr + 2 * dm * e[k] if k < 3 else 2 * dm * e[k] for k in range(6)]


def dot(a: list, b: list):
    return sum(WEIGHT[k] * a[k] * b[k] for k in range(6))


def axis_gamma(t: list, i: int) -> list:
    """Gamma(e_i) applied to a constant tensor."""
    e = [float(i == 0), float(i == 1), float(i == 2)]
    return [float(np.asarray(c)) for c in gamma_hat([np.asarray(x) for x in t], e)]


def dc_tensor(e: np.ndarray) -> np.ndarray:
    return dl * np.trace(e) * np.eye(3) + 2 * dm * e


b_t = voigt(dc_tensor(np.outer([1, 0, 0], [1, 0, 0])))
gf = grad(f)
grad2 = sum(g * g for g in gf)
norm_ratio = float(np.sum(grad2) / np.sum(f * f))
u_b = gamma_field([f * x for x in b_t])
du_b = [grad(c) for c in u_b]  # du_b[k][i] = d_i of component k
print(f"grid {N}^3 ({nb} across the body), [{time.perf_counter() - t0:.0f} s]", flush=True)
print(" angle wave    F (closed form, T2)    F3 (closed form)")
for th_obs in (0.2, 0.8853981633974483, np.pi / 2):
    nv = np.array([np.cos(th_obs), np.sin(th_obs), 0.0])
    tv = np.array([-np.sin(th_obs), np.cos(th_obs), 0.0])
    for wave, e_out in (("P", np.outer(nv, nv)), ("S", 0.5 * (np.outer(nv, tv) + np.outer(tv, nv)))):
        a_t = voigt(dc_tensor(e_out))
        # second order
        t2 = float(np.sum(f * dot([np.full(1, x) for x in a_t], u_b)))
        w2 = dot([np.full(1, x) for x in a_t], u_b)
        num2 = float(np.sum(sum(gf[i] * g for i, g in enumerate(grad(w2)))))
        num2 *= 2.0
        num2 -= sum(float(np.sum(gf[i] ** 2)) * float(dot(a_t, axis_gamma(b_t, i))) for i in range(3))
        F2 = num2 / (t2 * norm_ratio)
        # third order
        u_a = gamma_field([f * x for x in a_t])
        du_a = [grad(c) for c in u_a]
        cu_b, cu_a = dC(u_b), dC(u_a)
        w_ab = dot([np.full(1, x) for x in a_t], gamma_field([f * c for c in cu_b]))
        w_ba = dot([np.full(1, x) for x in b_t], gamma_field([f * c for c in cu_a]))
        t3 = float(np.sum(f * dot(u_a, cu_b)))
        g_ab, g_ba = grad(w_ab), grad(w_ba)
        integrand = sum(gf[i] * (g_ab[i] + g_ba[i]) for i in range(3))
        for i in range(3):
            dua = [du_a[k][i] for k in range(6)]
            dub = [du_b[k][i] for k in range(6)]
            ga, gb = axis_gamma(a_t, i), axis_gamma(b_t, i)
            c_gb, c_ga = dC([np.full(1, x) for x in gb]), [np.full(1, x) for x in ga]
            integrand = integrand + f * (dot(dua, dC(dub)) - gf[i] * (dot(dua, c_gb) + dot(c_ga, dC(dub))))
            integrand = integrand + gf[i] * (dot(u_a, dC(dub)) + dot(dua, cu_b))
            integrand = integrand - gf[i] ** 2 * (dot(u_a, c_gb) + dot(c_ga, cu_b))
        F3 = float(np.sum(integrand)) / (t3 * norm_ratio)
        print(f" {th_obs:5.2f}  {wave}       {F2: .4f}                {F3: .4f}", flush=True)
print(f"[{time.perf_counter() - t0:.0f} s]")
