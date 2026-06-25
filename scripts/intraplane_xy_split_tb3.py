#!/usr/bin/env python3
"""TB3 — the sweep: geometric resummation of the directional lattice sum.

Deliverable #2.  In the directional (k_x-residue) representation the lateral
row-sum is a GEOMETRIC series (the thesis cumulative-phase sweep / FFTProp),
resummed in closed form so the row is never brute-summed.  The y-lattice sum
Poisson-collapses the k_y integral onto the reciprocal lattice
k_y = kpar_y + 2 pi n / aL.  Net: x in closed form, y a fast reciprocal sum,
z a quadrature -- NO spatial truncation.

Row geometric identity (the heart of the sweep), at fixed kxc (Im kxc > 0):
  S(kxc, r_x) = sum_{i in Z} e^{i kxc |r_x - aL i|} e^{i kpar_x aL i}
  split at i0 = floor(r_x / aL):
    i <= i0 :  r_x - aL i >= 0  -> e^{i kxc r_x} p^i,  p = e^{i(kpar_x - kxc) aL}
    i >  i0 :  r_x - aL i <  0  -> e^{-i kxc r_x} q^i, q = e^{i(kpar_x + kxc) aL}
  |p| = e^{Im(kxc) aL} > 1  (sum to -inf converges);  |q| < 1 (sum to +inf).
  S = e^{ i kxc r_x} p^{i0+1}/(p - 1) + e^{-i kxc r_x} q^{i0+1}/(1 - q)

TB3a (this file): verify S_closed == S_brute (large M) over sample (kxc, r_x).
"""

from __future__ import annotations

import numpy as np

AL = 2.0
KPAR = np.array([0.2, 0.1])
ETA = 0.20
KP = 2 * np.pi * (1 + 1j * ETA) / 5.0
KS = 2 * np.pi * (1 + 1j * ETA) / 3.0


def row_brute(kxc: complex, r_x: float, kpar_x: float, m: int) -> complex:
    i = np.arange(-m, m + 1)
    return np.sum(
        np.exp(1j * kxc * np.abs(r_x - AL * i)) * np.exp(1j * kpar_x * AL * i)
    )


def row_closed(kxc: complex, r_x: float, kpar_x: float) -> complex:
    i0 = int(np.floor(r_x / AL))
    p = np.exp(1j * (kpar_x - kxc) * AL)
    q = np.exp(1j * (kpar_x + kxc) * AL)
    left = np.exp(1j * kxc * r_x) * p ** (i0 + 1) / (p - 1.0)
    right = np.exp(-1j * kxc * r_x) * q ** (i0 + 1) / (1.0 - q)
    return left + right


def row_closed_vec(kxc: np.ndarray, r_x: float, kpar_x: float) -> np.ndarray:
    """Vectorised closed-form row sum S(kxc, r_x) over an array of kxc."""
    i0 = int(np.floor(r_x / AL))
    p = np.exp(1j * (kpar_x - kxc) * AL)
    q = np.exp(1j * (kpar_x + kxc) * AL)
    left = np.exp(1j * kxc * r_x) * p ** (i0 + 1) / (p - 1.0)
    right = np.exp(-1j * kxc * r_x) * q ** (i0 + 1) / (1.0 - q)
    return left + right


def field_sweep(
    r: np.ndarray, kappa: complex, nmax: int, kz_max: float, nkz: int
) -> complex:
    """Resummed lattice field at receiver r (R != 0 sites), via the DUAL sweep.

    Sweep the lateral axis with the larger receiver coordinate (argmax(|r_x|,|r_y|))
    -> swept axis in closed form, other axis a reciprocal (Poisson) sum that then
    decays as e^{-|G_n| * r_swept} (fast).  x-sweep:
      F = i/(4 pi aL) sum_n e^{i ky_n r_y} int dkz e^{i kz r_z}
            S_closed(kxc_n, r_x) / kxc_n,   ky_n = kpar_y + 2 pi n / aL.
    y-sweep is the same with x<->y.  Self-site (i=j=0) subtracted as g_exact(r).
    """
    kz = np.linspace(-kz_max, kz_max, nkz)
    dkz = kz[1] - kz[0]
    wz = np.full(nkz, dkz)
    wz[0] *= 0.5
    wz[-1] *= 0.5
    n = np.arange(-nmax, nmax + 1)

    if abs(r[0]) >= abs(r[1]):  # x-sweep: x closed form, y reciprocal
        kg = KPAR[1] + 2 * np.pi * n / AL  # ky_n
        r_sw, r_rec, kpar_sw = r[0], r[1], KPAR[0]
    else:  # y-sweep: y closed form, x reciprocal
        kg = KPAR[0] + 2 * np.pi * n / AL  # kx_m
        r_sw, r_rec, kpar_sw = r[1], r[0], KPAR[1]

    KG, KZ = np.meshgrid(kg, kz, indexing="ij")  # (Nn, Nkz)
    kperp = np.sqrt(kappa**2 - KG**2 - KZ**2 + 0j)
    kperp = np.where(kperp.imag < 0, -kperp, kperp)
    S = row_closed_vec(kperp, r_sw, kpar_sw)
    integ = np.exp(1j * KZ * r[2]) * S / kperp
    kz_int = integ @ wz  # (Nn,)
    F = (1j / (4 * np.pi * AL)) * np.sum(np.exp(1j * kg * r_rec) * kz_int)
    rn = np.sqrt(r @ r + 0j)
    g_self = np.exp(1j * kappa * rn) / (4 * np.pi * rn)
    return F - g_self


def run_tb3b() -> None:
    import scripts.intraplane_xy_split_tb2b as t2

    print("\n==== TB3b :: full sweep-resummed D[q,s] vs brute & Kambe ====")
    nmax, kz_max, nkz, lr = 80, 100.0, 5000, 6
    print(f"  Nmax={nmax} kz_max={kz_max} Nkz={nkz}  (brute Lr={lr})")
    sphere, th, ph, w = t2.build_sphere()
    for name, kap in (("kP", KP), ("kS", KS)):
        f_sweep = np.array([field_sweep(r, kap, nmax, kz_max, nkz) for r in sphere])
        D_sweep = t2.project(f_sweep, th, ph, w, kap)
        D_brute = t2.project(
            t2.lattice_field(t2.g_res, sphere, kap, lr), th, ph, w, kap
        )
        D_kambe = t2.kambe_direct(kap, lr)
        rel_sb = t2.maxdiff(D_sweep, D_brute) / t2.maxabs(D_brute)
        rel_sk = t2.maxdiff(D_sweep, D_kambe) / t2.maxabs(D_kambe)
        print(
            f"  [{name}] sweep vs brute = {rel_sb:.2e}   sweep vs Kambe = {rel_sk:.2e}"
            f"  -> {'PASS' if rel_sk < 5e-3 else 'FAIL'}"
        )


def main() -> None:
    print("==== TB3a :: row geometric resummation (the sweep) vs brute ====")
    print(f"  aL={AL} kpar_x={KPAR[0]} eta={ETA}")
    worst = 0.0
    # representative kxc values: on-shell (ky=kz=0 -> kxc=kappa) and off-shell
    for kap in (KP, KS):
        for ky, kz in ((0.0, 0.0), (0.5, 0.3), (1.0, 1.5)):
            kxc = np.sqrt(kap**2 - ky**2 - kz**2 + 0j)
            if kxc.imag < 0:
                kxc = -kxc
            for r_x in (0.3, 1.7, -0.5, 0.0, -1.3):
                sb = row_brute(kxc, r_x, KPAR[0], 4000)
                sc = row_closed(kxc, r_x, KPAR[0])
                rel = abs(sc - sb) / abs(sb)
                worst = max(worst, rel)
    print(f"  worst |closed - brute(M=4000)| / |brute| = {worst:.2e}")
    print(f"  -> {'PASS' if worst < 1e-6 else 'FAIL'} (tol 1e-6)")

    # convergence of the brute row with M (closed form is the M->inf limit)
    kxc = np.sqrt(KP**2 - 0.5**2 - 0.3**2 + 0j)
    sc = row_closed(kxc, 0.3, KPAR[0])
    print("\n  brute-row convergence to the closed form (kP, ky=.5, kz=.3, r_x=.3):")
    for m in (50, 200, 1000):
        sb = row_brute(kxc, 0.3, KPAR[0], m)
        print(f"    M={m:5d}: |brute - closed|/|closed| = {abs(sb - sc) / abs(sc):.2e}")

    run_tb3b()


if __name__ == "__main__":
    main()
