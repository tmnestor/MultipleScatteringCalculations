#!/usr/bin/env python3
"""The 3-D third-order term by a second route: the static strain Green operator applied by FFT, with the
cell means of constant cells (p = r = 0), against the scheme's own T3 (``measure_t3_law_3d.py``).

In the long-wave limit, with a stiffness contrast f(x) DC, the incident strain e_in and the outgoing strain
e_out of the observed wave, a = DC : e_out and b = DC : e_in,

    T2 = int f a : u_b,                 u_b = Gamma * (f b),
    T3 = int f u_a : DC : u_b,          u_a = Gamma * (f a),

Gamma the static strain Green operator (self-adjoint). Constant cells hold every field as its cell mean, so

    T2_scheme = int Pi f a : U_b,     T3_scheme = int Pi f Pi(U_a) : DC : Pi(U_b),     U = Gamma * (Pi f .).

Gamma on a symmetric tensor field tau: in Fourier, t = tau xi, v = (t - kappa xi (xi . t)) / mu,
(Gamma tau)_ij = (xi_i v_j + xi_j v_i) / 2, kappa = (lambda + mu)/(lambda + 2 mu), xi a unit vector;
its value at xi = 0 is set to its average over directions. Printed: (T - T_scheme) / (T E) for T2 and
T3, outgoing P and SV at three angles, E the relative projection error on the samples.

Run:  python scripts/measure_t3_tensor_forms.py <n cells across> <s samples per cell> <padding>
      e.g. ... 8 8 2   (about 4 GB at 256^3)
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

n, s, pad = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
d = 2 * RADIUS / n
nb = n * s
N = int(round(pad * nb / 2)) * 2
dx = d / s
t0 = time.perf_counter()
x1 = (np.arange(N) - N / 2 + 0.5) * dx
X, Y, Z = np.meshgrid(x1, x1, x1, indexing="ij", sparse=True)
r = np.sqrt(X**2 + Y**2 + Z**2)
xs = np.clip((RADIUS - r) / (RADIUS - CORE), 0.0, 1.0)
f = 10 * xs**3 - 15 * xs**4 + 6 * xs**5
del xs, r
off = (N - nb) // 2
sl = (slice(off, off + nb),) * 3


def cell_mean(field: np.ndarray) -> np.ndarray:
    """The field replaced by its mean over each voxel (zero outside the voxel box)."""
    out = np.zeros_like(field)
    core = field[sl].reshape(n, s, n, s, n, s)
    m = core.mean(axis=(1, 3, 5))
    out[sl] = np.broadcast_to(m[:, None, :, None, :, None], core.shape).reshape(nb, nb, nb)
    return out


pf = cell_mean(f)
assert (
    np.abs(
        f[~np.isin(np.arange(N), np.arange(off, off + nb))[:, None, None] * np.ones((1, N, N), bool)]
    ).max()
    == 0.0
)
g = f - pf
E = float(np.sum(g * g) / np.sum(f * f))

k1 = np.fft.fftfreq(N, dx)
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij", sparse=True)
kk = np.sqrt(KX**2 + KY**2 + KZ**2)
kk[0, 0, 0] = 1.0
xi = [KX / kk, KY / kk, KZ / kk]


def gamma_const(tensor: np.ndarray, unit: list) -> list:
    """Gamma(xi) applied to a constant symmetric tensor, as six arrays (Voigt order PAIRS)."""
    t = [sum(tensor[i, j] * unit[j] for j in range(3)) for i in range(3)]
    xt = sum(unit[i] * t[i] for i in range(3))
    v = [(t[i] - kappa * unit[i] * xt) / mu for i in range(3)]
    return [(unit[i] * v[j] + unit[j] * v[i]) / 2 for i, j in PAIRS]


# Gamma's average over directions, for the zero wavevector
nfib = 20000
kf = np.arange(nfib) + 0.5
phi = np.arccos(1 - 2 * kf / nfib)
th = np.pi * (1 + 5**0.5) * kf
dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def gamma_bar(tensor: np.ndarray) -> list:
    vals = gamma_const(tensor, [dirs[:, 0], dirs[:, 1], dirs[:, 2]])
    return [float(np.mean(v)) for v in vals]


def strain_field(profile_hat: np.ndarray, tensor: np.ndarray) -> list:
    """Gamma * (profile tensor): six real arrays."""
    comps = gamma_const(tensor, xi)
    zero = gamma_bar(tensor)
    out = []
    for c, z in zip(comps, zero, strict=True):
        c = np.broadcast_to(c, profile_hat.shape).copy()
        c[0, 0, 0] = z
        out.append(np.fft.ifftn(c * profile_hat).real)
    return out


def dC_voigt(e: list) -> list:
    tr = e[0] + e[1] + e[2]
    return [
        dl * tr + 2 * dm * e[0],
        dl * tr + 2 * dm * e[1],
        dl * tr + 2 * dm * e[2],
        2 * dm * e[3],
        2 * dm * e[4],
        2 * dm * e[5],
    ]


def contract(a: list, b: list) -> np.ndarray:
    """a : b for Voigt-ordered symmetric tensors (shear components counted twice)."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + 2 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])


def dc_tensor(e: np.ndarray) -> np.ndarray:
    return dl * np.trace(e) * np.eye(3) + 2 * dm * e


F_hat, P_hat = np.fft.fftn(f), np.fft.fftn(pf)
b = dc_tensor(np.outer([1, 0, 0], [1, 0, 0]))
ub, Ub = strain_field(F_hat, b), strain_field(P_hat, b)
Ub_mean = [cell_mean(c) for c in Ub]
print(f"n = {n}, s = {s}, grid {N}^3, E = {E:.4e}  [{time.perf_counter() - t0:.0f} s]", flush=True)
print(" angle wave   T2: (T-Ts)/(T E)   T3: (T-Ts)/(T E)   T3/T2")
for th_obs in (0.2, 0.8853981633974483, np.pi / 2):
    nv = np.array([np.cos(th_obs), np.sin(th_obs), 0.0])
    tv = np.array([-np.sin(th_obs), np.cos(th_obs), 0.0])
    for wave, e_out in (("P", np.outer(nv, nv)), ("S", 0.5 * (np.outer(nv, tv) + np.outer(tv, nv)))):
        a = dc_tensor(e_out)
        a_v = [a[i, j] for i, j in PAIRS]
        t2 = float(np.sum(f * contract([np.full(1, x) for x in a_v], ub)))
        t2s = float(np.sum(pf * contract([np.full(1, x) for x in a_v], Ub)))
        ua, Ua = strain_field(F_hat, a), strain_field(P_hat, a)
        t3 = float(np.sum(f * contract(ua, dC_voigt(ub))))
        t3s = float(np.sum(pf * contract([cell_mean(c) for c in Ua], dC_voigt(Ub_mean))))
        print(
            f" {th_obs:5.2f}  {wave}    {(t2 - t2s) / (t2 * E): .4f}          {(t3 - t3s) / (t3 * E): .4f}"
            f"          {t3 / t2: .4f}",
            flush=True,
        )
print(f"[{time.perf_counter() - t0:.0f} s]")
