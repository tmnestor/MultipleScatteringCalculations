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

LINEAR CELLS (p = 1, field and contrast in 1, x_i). The scheme's term is int Pi f U_a : C : Pi U_b (one
projection, on the incident side), the residual is quadratic, and every piece reduces to two cell rules:
a cell-scale function meets a smooth one through
B(S, T) = h^4 [sum_i S_ii T_ii / 45 + sum_{i<j} S_ij T_ij / 9],
and the residual's own scattered field carries Gamma on the axes (its parabolas) and M_ij, the average of
Gamma over the lattice of the (i, j) plane (its products), with <u^2 v^2> = 6 G/pi^2 - 2/5. Then

  (T3 - T3_scheme) / h^4 = int [ B(f C u_a, u_b) + B(f, w_ba) - H(f C u_a, f; b) + B(f, w_ab)
        + B(f, u_a C u_b)
        - f K(f; a, u_b) - u_a : C : B(f, u_b) - K2(f; a) : C : u_b ],
  H(S, f; b) = sum_i S_ii : G_ib f_ii / 45 + sum_{i<j} S_ij : M_ij b f_ij / 9,
  K(f; a, u_b) = sum_i f_ii G_ia : C : u_b,ii / 45 + sum_{i<j} f_ij M_ij a : C : u_b,ij / 9,
  K2(f; a) = sum_i f_ii^2 G_ia / 45 + sum_{i<j} f_ij^2 M_ij a / 9,
  E = h^4 int [ sum_i f_ii^2 / 45 + sum_{i<j} f_ij^2 / 9 ] / int f^2,

and for the second term 2 B(f, a : u_b) - a : K2(f; b), whose ratio is the F1 of section 3.4.

Run:  python scripts/measure_f3_closed_form.py <grid points across the body> <padding> [p]   e.g. 64 2 1
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
P_DEG = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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

# ------------------------------------------------------------------ linear cells
CATALAN = 0.915965594177219015054603514932384110774
U2V2 = 6 * CATALAN / np.pi**2 - 0.4
U4 = 0.5 - U2V2
PLANES = [(0, 1), (0, 2), (1, 2)]


def plane_gamma(t: list, i: int, j: int) -> list:
    """M_ij t: Gamma(xi) t averaged over the lattice of the (i, j) plane, weights 1/(m^2 n^2)."""
    tens = np.zeros((3, 3))
    for k, (p, q) in enumerate(PAIRS):
        tens[p, q] = tens[q, p] = t[k]
    m2 = np.zeros((3, 3))
    m2[i, i] = m2[j, j] = 0.5
    m4 = np.zeros((3, 3, 3, 3))
    for p_, q_, r_, s_ in np.ndindex(3, 3, 3, 3):
        idx = sorted((p_, q_, r_, s_))
        if idx == [i] * 4 or idx == [j] * 4:
            m4[p_, q_, r_, s_] = U4
        elif idx == sorted([i, i, j, j]):
            m4[p_, q_, r_, s_] = U2V2
    out = np.zeros((3, 3))
    for p_, q_ in np.ndindex(3, 3):
        quad = (tens[q_] @ m2[p_] + tens[p_] @ m2[q_]) / (2 * mu)
        quart = kappa / mu * np.einsum("rs,rs->", tens, m4[p_, q_])
        out[p_, q_] = quad - quart
    return [out[p, q] for p, q in PAIRS]


def d2(field: np.ndarray) -> dict:
    fh = np.fft.fftn(field)
    return {(i, j): np.fft.ifftn(-KV[i] * KV[j] * fh).real for i in range(3) for j in range(i, 3)}


def const(t: list) -> list:
    return [np.full(1, x) for x in t]


if P_DEG == 1:
    f2 = d2(f)
    e_lin = float(
        np.sum(sum(f2[(i, i)] ** 2 for i in range(3)) / 45 + sum(f2[pq] ** 2 for pq in PLANES) / 9)
        / np.sum(f * f)
    )

    def bform(x: dict, y: dict, tensor: bool) -> np.ndarray:
        """B(X, Y) density from second-derivative dicts (scalar or Voigt-tensor entries)."""
        op = dot if tensor else (lambda p, q: p * q)
        return (
            sum(op(x[(i, i)], y[(i, i)]) for i in range(3)) / 45
            + sum(op(x[pq], y[pq]) for pq in PLANES) / 9
        )

    def d2_tensor(t: list) -> dict:
        comps = [d2(c) for c in t]
        return {key: [comps[k][key] for k in range(6)] for key in comps[0]}

    ub2 = d2_tensor(u_b)
    print(f"\nlinear cells (p = 1): E / h^4 = {e_lin:.4e}", flush=True)
    print(" angle wave    F1 (closed form, T2)    F3 (closed form, linear cells)")
    for th_obs in (0.2, 0.8853981633974483, np.pi / 2):
        nv = np.array([np.cos(th_obs), np.sin(th_obs), 0.0])
        tv = np.array([-np.sin(th_obs), np.cos(th_obs), 0.0])
        for wave, e_out in (("P", np.outer(nv, nv)), ("S", 0.5 * (np.outer(nv, tv) + np.outer(tv, nv)))):
            a_t = voigt(dc_tensor(e_out))
            ga = {i: axis_gamma(a_t, i) for i in range(3)}
            gb = {i: axis_gamma(b_t, i) for i in range(3)}
            ma = {pq: plane_gamma(a_t, *pq) for pq in PLANES}
            mb = {pq: plane_gamma(b_t, *pq) for pq in PLANES}

            def k2(c_axis: dict, c_plane: dict) -> list:
                return [
                    sum(f2[(i, i)] ** 2 * c_axis[i][k] for i in range(3)) / 45
                    + sum(f2[pq] ** 2 * c_plane[pq][k] for pq in PLANES) / 9
                    for k in range(6)
                ]

            # second order
            au = dot(const(a_t), u_b)
            t2 = float(np.sum(f * au))
            num2 = float(np.sum(2 * bform(f2, d2(au), False) - dot(const(a_t), k2(gb, mb))))
            F1 = num2 / (t2 * e_lin)
            # third order
            u_a = gamma_field([f * x for x in a_t])
            cu_b, cu_a = dC(u_b), dC(u_a)
            w_ab = dot(const(a_t), gamma_field([f * c for c in cu_b]))
            w_ba = dot(const(b_t), gamma_field([f * c for c in cu_a]))
            t3 = float(np.sum(f * dot(u_a, cu_b)))
            s_t = [f * c for c in cu_a]
            s2 = d2_tensor(s_t)
            dens = bform(s2, ub2, True)
            dens = dens + bform(f2, d2(w_ba), False)
            dens = dens - (
                sum(dot(s2[(i, i)], const(gb[i])) * f2[(i, i)] for i in range(3)) / 45
                + sum(dot(s2[pq], const(mb[pq])) * f2[pq] for pq in PLANES) / 9
            )
            dens = dens + bform(f2, d2(w_ab), False)
            dens = dens + bform(f2, d2(dot(u_a, cu_b)), False)
            dens = dens - f * (
                sum(f2[(i, i)] * dot(const(ga[i]), dC(ub2[(i, i)])) for i in range(3)) / 45
                + sum(f2[pq] * dot(const(ma[pq]), dC(ub2[pq])) for pq in PLANES) / 9
            )
            dens = dens - (
                sum(f2[(i, i)] * dot(u_a, dC(ub2[(i, i)])) for i in range(3)) / 45
                + sum(f2[pq] * dot(u_a, dC(ub2[pq])) for pq in PLANES) / 9
            )
            dens = dens - dot(k2(ga, ma), cu_b)
            F3 = float(np.sum(dens)) / (t3 * e_lin)
            print(f" {th_obs:5.2f}  {wave}       {F1: .4f}                  {F3: .4f}", flush=True)
    print(f"[{time.perf_counter() - t0:.0f} s]")
