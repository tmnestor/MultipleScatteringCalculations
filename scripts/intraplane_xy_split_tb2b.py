#!/usr/bin/env python3
"""TB2b — scalar structure constants D[q,s]: x/y-split residue vs Kambe.

Deliverable #1 (cross-code/bridge), irreducible scalar level.  The Kambe
G0^vec is *assembled from* the scalar lattice structure constants D[q,s]
(at kappa_P for L, kappa_S for M/N), so the substantive cross-code statement is

    D_residue[q,s] (directional k_x/k_y-split, Bloch-summed, projected)
        ==  D_Kambe[q,s]  (h_q(kappa R) addition-theorem direct sum)

evaluated in the damped regime, where both sums converge absolutely.

Method (all reuse the validated quadrature scheme from TB2a — dense UNIFORM
grid + damping; Gauss-Legendre under-resolves the 1/kxc propagating disk):
  * g_res(d; kappa): scalar Helmholtz Green's fn via residue in the axis of
    LARGER lateral separation (argmax(|dx|,|dy|)) -> fast evanescent decay.
  * field_res(r) = sum_{R!=0} g_res(r - R) e^{i k_par.R}  on a sphere |r|=rho0.
  * D_res[q,s]   = (1/(i kappa j_q(kappa rho0))) * oint field_res conj(Y_q^s).
  * D_kambe[q,s] = sum_{R!=0} h_q(kappa|R|) conj(Y_q^s(pi/2, phi_R)) e^{i k_par.R}.
    (Addition theorem: projecting field_exact with the same projector returns
    exactly this; verified to 1e-13 in the earlier gate C.)
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1, jv

# ── parameters ──────────────────────────────────────────────────────────────
ALPHA, BETA = 5.0, 3.0
ETA = 0.20  # damping (real phenomenon; chosen for fast lattice convergence)
OMEGA = 2 * np.pi * (1 + 1j * ETA)
KP = OMEGA / ALPHA
KS = OMEGA / BETA

AL = 2.0
KPAR = np.array([0.2, 0.1])
RHO0 = 0.5
QMAX = 4

# residue quadrature (uniform grid).  Proven config: runs foreground in ~1-2 min
# and passes; raise N_TRANS / LR_CMP to tighten the kP residual further.
N_TRANS = 176  # nodes per transverse axis (g_res vs exact ~1e-6)
KMAX_FAC = 6.0  # half-range = KMAX_FAC * |kappa|
# projection sphere
N_U, N_PHI = 10, 20
# lattice truncation for the residue-vs-Kambe compare (same Lr both sides ->
# truncation cancels; the exact-field convergence gate reports physical Lr).
LR_CMP = 5


def sph_j(q: int, z: complex) -> complex:
    z = complex(z)
    return np.sqrt(np.pi / (2 * z)) * jv(q + 0.5, z)


def sph_h1(q: int, z: complex) -> complex:
    z = complex(z)
    return np.sqrt(np.pi / (2 * z)) * hankel1(q + 0.5, z)


def sph_harm_Y(q, s, theta, phi):
    try:
        from scipy.special import sph_harm_y

        return sph_harm_y(q, s, theta, phi)
    except (ImportError, AttributeError):
        from scipy.special import sph_harm

        return sph_harm(s, q, phi, theta)


def g_exact(d: np.ndarray, kappa: complex) -> np.ndarray:
    r = np.sqrt(np.sum(d**2, axis=1) + 0j)
    return np.exp(1j * kappa * r) / (4 * np.pi * r)


def _uniform_nodes(kappa: complex):
    kmax = KMAX_FAC * abs(kappa)
    k = np.linspace(-kmax, kmax, N_TRANS)
    w = np.full(
        N_TRANS, (2 * kmax) / (N_TRANS - 1)
    )  # trapezoid step (ends halved below)
    w[0] *= 0.5
    w[-1] *= 0.5
    return k, w


def g_res(d: np.ndarray, kappa: complex) -> np.ndarray:
    """Scalar Helmholtz Green's fn via directional residue (uniform quadrature).

    Per row, residue axis = argmax(|dx|,|dy|); transverse integral over the
    remaining two wavenumbers by dense uniform grid.
    """
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    out = np.empty(len(d), dtype=complex)
    nodes, wts = _uniform_nodes(kappa)
    ka, kb = np.meshgrid(nodes, nodes, indexing="ij")
    w2 = np.outer(wts, wts)
    kperp = np.sqrt(kappa**2 - ka**2 - kb**2 + 0j)
    kperp = np.where(kperp.imag < 0, -kperp, kperp)
    pref = 1j / (8 * np.pi**2)

    use_x = np.abs(dx) >= np.abs(dy)
    for grp, swept_x in ((use_x, True), (~use_x, False)):
        idx = np.nonzero(grp)[0]
        if idx.size == 0:
            continue
        if swept_x:
            d_sw, d_a, d_b = dx[idx], dy[idx], dz[idx]
        else:
            d_sw, d_a, d_b = dy[idx], dx[idx], dz[idx]
        # chunk over rows to bound memory (each chunk: (rows, N, N) complex)
        for c0 in range(0, idx.size, 64):
            sl = slice(c0, c0 + 64)
            sweep = np.exp(1j * kperp * np.abs(d_sw[sl])[:, None, None]) / kperp
            phase = np.exp(
                1j * (ka * d_a[sl][:, None, None] + kb * d_b[sl][:, None, None])
            )
            out[idx[sl]] = pref * np.sum(w2 * phase * sweep, axis=(1, 2))
    return out


def build_sphere():
    u, wu = np.polynomial.legendre.leggauss(N_U)
    theta = np.arccos(u)
    phi = 2 * np.pi * np.arange(N_PHI) / N_PHI
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    WU = np.broadcast_to(wu[:, None], TH.shape)
    th, ph = TH.ravel(), PH.ravel()
    w = (WU * (2 * np.pi / N_PHI)).ravel()
    sphere = RHO0 * np.column_stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    )
    return sphere, th, ph, w


def lattice_field(field_fn, sphere, kappa, lr):
    acc = np.zeros(len(sphere), dtype=complex)
    for i in range(-lr, lr + 1):
        for j in range(-lr, lr + 1):
            if i == 0 and j == 0:
                continue
            shift = np.array([AL * i, AL * j, 0.0])
            phase = np.exp(1j * AL * (KPAR[0] * i + KPAR[1] * j))
            acc += field_fn(sphere - shift, kappa) * phase
    return acc


def project(field_vals, th, ph, w, kappa):
    out = {}
    for q in range(QMAX + 1):
        jq = sph_j(q, kappa * RHO0)
        for s in range(-q, q + 1):
            Y = sph_harm_Y(q, s, th, ph)
            out[(q, s)] = np.sum(w * field_vals * np.conj(Y)) / (1j * kappa * jq)
    return out


def kambe_direct(kappa, lr):
    iv = [
        (i, j)
        for i in range(-lr, lr + 1)
        for j in range(-lr, lr + 1)
        if not (i == 0 and j == 0)
    ]
    ij = np.array(iv)
    rn = AL * np.sqrt(ij[:, 0] ** 2 + ij[:, 1] ** 2)
    phi_r = np.arctan2(ij[:, 1], ij[:, 0])
    bloch = np.exp(1j * AL * (KPAR[0] * ij[:, 0] + KPAR[1] * ij[:, 1]))
    half = np.full(len(ij), np.pi / 2)
    out = {}
    for q in range(QMAX + 1):
        hq = np.array([sph_h1(q, kappa * r) for r in rn])
        for s in range(-q, q + 1):
            Y = sph_harm_Y(q, s, half, phi_r)
            out[(q, s)] = np.sum(hq * np.conj(Y) * bloch)
    return out


def maxdiff(a, b):
    return max(abs(a[k] - b[k]) for k in a)


def maxabs(a):
    return max(abs(v) for v in a.values())


def main() -> None:
    print("==== TB2b :: scalar structure constants D[q,s] residue vs Kambe ====")
    print(f"  aL={AL} k_par={tuple(KPAR)} rho0={RHO0} Qmax={QMAX} eta={ETA}")
    print(f"  kP={KP:.4f} kS={KS:.4f}  N_trans={N_TRANS}")

    # gate A: residue scalar g vs exact (sphere-relevant displacements, |sep|>=1.5)
    rng = np.random.default_rng(20260625)
    samp = rng.uniform(-6, 6, size=(60, 3))
    samp[:, 2] *= 0.1
    big = np.maximum(np.abs(samp[:, 0]), np.abs(samp[:, 1])) >= 1.5
    samp = samp[big]
    for name, kap in (("kP", KP), ("kS", KS)):
        ge, gr = g_exact(samp, kap), g_res(samp, kap)
        relA = np.max(np.abs(gr - ge) / np.abs(ge))
        print(
            f"  [A] g_res vs exact ({name}): {relA:.2e} -> "
            f"{'PASS' if relA < 1e-4 else 'FAIL'}"
        )

    sphere, th, ph, w = build_sphere()
    for name, kap in (("kP", KP), ("kS", KS)):
        prev = None
        for lr in (4, 6, 8):
            De = project(lattice_field(g_exact, sphere, kap, lr), th, ph, w, kap)
            if prev is not None:
                print(f"  [conv] {name} Lr={lr}: {maxdiff(De, prev) / maxabs(De):.2e}")
            prev = De
        lr = LR_CMP
        De = project(lattice_field(g_exact, sphere, kap, lr), th, ph, w, kap)
        Dr = project(lattice_field(g_res, sphere, kap, lr), th, ph, w, kap)
        Sk = kambe_direct(kap, lr)
        relB = maxdiff(Dr, Sk) / maxabs(Sk)
        relC = maxdiff(De, Sk) / maxabs(Sk)
        print(f"  [C] proj(exact) vs Kambe ({name}): {relC:.2e}")
        print(
            f"  [B] proj(residue) vs Kambe ({name}): {relB:.2e} -> "
            f"{'PASS' if relB < 1e-3 else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
