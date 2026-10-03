#!/usr/bin/env python3
"""The 3-D T2 law by a second route: the static quadratic forms of the strain Green operator, by FFT.

In the long-wave limit, with a stiffness contrast f(x) DC and a uniform incident strain, the second-order
term is the quadratic form <f, K f> of the static strain Green operator Gamma, contracted with the
contrast times the outgoing strain (a) and times the incident strain (b): K has the Fourier multiplier
m(xi) = a : Gamma(xi) : b, homogeneous of degree zero. The scheme with field and contrast of degree p in
each cell computes <Pi f, K Pi f>, so with g = f - Pi f

    (T2 - T2_scheme) / (T2 E) = (2 <g, K f> - <g, K g>) / (<f, K f> E),     E = |g|^2 / |f|^2 .

For a radial profile <f, K f> = m_bar |f|^2 exactly (m_bar the angular average of m); the FFT's own value
of it carries a small-xi discretisation error, so the ratio is formed with the exact one: it is column
2<g,Kf>/(m_bar|g|^2) minus column <g,Kg>/(m_bar|g|^2). The profile is sampled at s midpoints per cell
and axis on a box padded by the given factor. The projection onto 1 (p = 0) or 1, eta_i (p = 1) is either
'samples' (default), the discrete one on the samples, which keeps g orthogonal to them and converges as
1/s, or 'exact', the L2 one by Gauss quadrature of the analytic profile, whose sampled residual carries
spurious low moments (harmless for p = 0, large for the small residual of p = 1). Compare with
scripts/measure_t2_law_3d.py, which measures the scheme itself.

Run:  python scripts/measure_t2_quadratic_forms.py <n cells across> <s> <padding> <p> [samples|exact]
      e.g. ... 8 16 2 0     (about 2 s; 32 8 1.5 0 needs about 5 GB)
"""

import sys
import time

import numpy as np

RADIUS, CORE = 10.0, 5.0
alpha, beta, rho = 5000.0, 3000.0, 2500.0
mu = rho * beta**2
lam = rho * alpha**2 - 2 * mu
dl, dm = 2.0e9, 1.0e9
c_bulk = (lam + mu) / (mu * (lam + 2 * mu))
THETA = np.linspace(0.2, np.pi - 0.2, 9)

n, s, pad = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
P_DEG = int(sys.argv[4]) if len(sys.argv) > 4 else 0
PROJ = sys.argv[5] if len(sys.argv) > 5 else "samples"
if PROJ not in ("samples", "exact"):
    raise SystemExit(f"projection {PROJ!r}: use samples or exact")
d = 2 * RADIUS / n
npts_body = n * s
N = int(round(pad * npts_body / 2)) * 2
dx = d / s
t0 = time.perf_counter()
x1 = (np.arange(N) - N / 2 + 0.5) * dx  # midpoints; the body grid [-a, a] is cells 0..n-1 of the middle
X, Y, Z = np.meshgrid(x1, x1, x1, indexing="ij", sparse=True)
r = np.sqrt(X**2 + Y**2 + Z**2)
xs = np.clip((RADIUS - r) / (RADIUS - CORE), 0.0, 1.0)
f = 10 * xs**3 - 15 * xs**4 + 6 * xs**5
del xs, r


# the projection on the voxel grid, cells of s x s x s samples aligned so that the body box is n cells:
# the exact L2 projection of the analytic profile onto 1 (p = 0) and also eta_i (p = 1), its coefficients
# by an 8-point Gauss rule per axis in each cell, evaluated at the samples. E by the same rule.
def profile(rad):
    xs = np.clip((RADIUS - rad) / (RADIUS - CORE), 0.0, 1.0)
    return 10 * xs**3 - 15 * xs**4 + 6 * xs**5


off = (N - npts_body) // 2
gq, gw = np.polynomial.legendre.leggauss(8)
centres1 = (np.arange(n) - (n - 1) / 2) * d
CX, CY, CZ, QX, QY, QZ = np.meshgrid(centres1, centres1, centres1, gq, gq, gq, indexing="ij", sparse=True)
fq = profile(np.sqrt((CX + QX * d / 2) ** 2 + (CY + QY * d / 2) ** 2 + (CZ + QZ * d / 2) ** 2))
w3 = gw[:, None, None] * gw[None, :, None] * gw[None, None, :] / 8.0  # weights of the mean over the cell
c0 = np.einsum("abcijk,ijk->abc", fq, w3)
proj_q = np.broadcast_to(c0[..., None, None, None], fq.shape).copy()
lin = []
if P_DEG >= 1:
    for q in (QX, QY, QZ):
        weight = w3 * np.broadcast_to(q[0, 0, 0], (8, 8, 8))
        ci = np.einsum("abcijk,ijk->abc", fq, weight) * 3.0  # <f, eta> / <eta, eta>, <eta^2> = 1/3
        lin.append(ci)
        proj_q = proj_q + ci[..., None, None, None] * np.broadcast_to(q[0, 0, 0], (8, 8, 8))
E = float(np.einsum("abcijk,ijk->", (fq - proj_q) ** 2, w3) / np.einsum("abcijk,ijk->", fq**2, w3))

g = np.zeros_like(f)
core = f[off : off + npts_body, off : off + npts_body, off : off + npts_body]
blocks6 = core.reshape(n, s, n, s, n, s)
eta = (np.arange(s) - (s - 1) / 2) / (s / 2)  # sub-sample coordinate in [-1, 1]
if PROJ == "samples":
    # the discrete projection on the samples: g is then exactly orthogonal to 1 and eta_i on them, which
    # keeps sampling error out of its low moments; it differs from the L2 projection as 1/s
    c0 = blocks6.mean(axis=(1, 3, 5))
    lin = []
    for ax in range(P_DEG and 3):
        shp = [1, 1, 1, 1, 1, 1]
        shp[2 * ax + 1] = s
        e = eta.reshape(shp)
        lin.append((blocks6 * e).sum(axis=(1, 3, 5)) / (e * e).sum() / (s * s))
fit = np.broadcast_to(c0[:, None, :, None, :, None], blocks6.shape).copy()
for ax, ci in enumerate(lin):
    shp = [1, 1, 1, 1, 1, 1]
    shp[2 * ax + 1] = s
    fit = fit + ci[:, None, :, None, :, None] * eta.reshape(shp)
g_core = (blocks6 - fit).reshape(core.shape)
g[off : off + npts_body, off : off + npts_body, off : off + npts_body] = g_core
outside = f.copy()
outside[off : off + npts_body, off : off + npts_body, off : off + npts_body] = 0.0
assert np.abs(outside).max() == 0.0, "the body must lie inside the voxel box"
if PROJ == "samples":
    E = float(np.sum(g * g) / np.sum(f * f))

F = np.fft.fftn(f)
G = np.fft.fftn(g)
k1 = np.fft.fftfreq(N, dx)
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij", sparse=True)
kk = np.sqrt(KX**2 + KY**2 + KZ**2)
kk[0, 0, 0] = 1.0
u = [KX / kk, KY / kk, KZ / kk]
FF = (F * F.conj()).real
GF = (G.conj() * F).real
GG = (G * G.conj()).real
print(f"n = {n}, s = {s}, grid {N}^3, E = {E:.4e}  [{time.perf_counter() - t0:.0f} s]", flush=True)


def dC(e):
    return dl * np.trace(e) * np.eye(3) + 2 * dm * e


def mult(a, b):
    sab = (a @ b + b @ a) / 2
    q_s = sum(sab[i, j] * u[i] * u[j] for i in range(3) for j in range(3))
    q_a = sum(a[i, j] * u[i] * u[j] for i in range(3) for j in range(3))
    q_b = sum(b[i, j] * u[i] * u[j] for i in range(3) for j in range(3))
    m = q_s / mu - c_bulk * q_a * q_b
    return m


b = dC(np.outer([1, 0, 0], [1, 0, 0]))
print(" angle wave   2<g,Kf>/(m_bar|g|^2)  <g,Kg>/(m_bar|g|^2)  <f,Kf>/(m_bar|f|^2)   F")
for th in THETA:
    nv = np.array([np.cos(th), np.sin(th), 0.0])
    tv = np.array([-np.sin(th), np.cos(th), 0.0])
    for wave, e_out in (("P", np.outer(nv, nv)), ("S", 0.5 * (np.outer(nv, tv) + np.outer(tv, nv)))):
        a = dC(e_out)
        m = mult(a, b)
        # angular average of m: (1/mu)(1/3) tr(S) - c (tr a tr b + 2 a:b) / 15
        sab = (a @ b + b @ a) / 2
        m_bar = np.trace(sab) / (3 * mu) - c_bulk * (np.trace(a) * np.trace(b) + 2 * np.sum(a * b)) / 15
        m[0, 0, 0] = m_bar
        q_ff = np.sum(FF * m)
        q_gf = np.sum(GF * m)
        q_gg = np.sum(GG * m)
        nf, ng = np.sum(FF), np.sum(GG)
        # F with the exact <f, K f> = m_bar |f|^2 of a radial profile, and E by quadrature
        ratio = (2 * q_gf - q_gg) / (m_bar * nf * E)
        print(
            f" {th:5.2f}  {wave}     {2 * q_gf / (m_bar * ng): .4f}              {q_gg / (m_bar * ng): .4f}"
            f"              {q_ff / (m_bar * nf): .4f}        {ratio: .4f}",
            flush=True,
        )
print(f"[{time.perf_counter() - t0:.0f} s]")
