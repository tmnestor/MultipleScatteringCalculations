#!/usr/bin/env python3
"""Check the thesis's own equations against themselves, numerically.

Transcribed VERBATIM from Thesis_Recompiled_2026/GRepresentations.tex:
  Akdef   the Fourier-domain system matrix A(kx,ky)
  Peigen  |+-,P>   SVeigen |+-,S>   SHeigen |+-,H>
  epsdef  eps_P, eps_S, eps_H
  eigDef  Lambda = i diag[kzP, kzS, kzH, -kzP, -kzS, -kzH]

Claims tested:
  ATdef   A^T(-k) J6 + J6 A(k) = 0
  specA   A D_z = D_z Lambda
  dinv2   ( J6 D_z(-k) )^T D_z(k) = i J6
  D1def   D_z^-1(k) = -i J6 D_z^T(-k) J6
and, separately, Box 5.3's worked explosion example against the jump vector
derived independently in the wrapper work.
"""

import numpy as np
import numpy.linalg as la

RHO, ALPHA, BETA, OM = 2.6, 4.0, 2.22, 2 * np.pi * 12.0
MU = RHO * BETA**2
LAM = RHO * ALPHA**2 - 2 * MU
GAM = LAM / (LAM + 2 * MU)
A_ = 1.0 / (LAM + 2 * MU)
B_ = 1.0 / MU
ZETA = 4 * MU * (LAM + MU) / (LAM + 2 * MU)
CHI = 2 * MU * LAM / (LAM + 2 * MU)

J6 = np.zeros((6, 6))
J6[:3, 3:], J6[3:, :3] = np.eye(3), -np.eye(3)


def khat(c, ky):
    """Kdef."""
    return np.sqrt((OM / c) ** 2 - ky**2 + 0j)


def kz(c, kx, ky):
    """kzcDef."""
    return np.sqrt(khat(c, ky) ** 2 - kx**2 + 0j)


def amat(kx, ky):
    """Akdef, verbatim."""
    return np.array(
        [
            [0, -1j * GAM * kx, -1j * GAM * ky, A_, 0, 0],
            [-1j * kx, 0, 0, 0, B_, 0],
            [-1j * ky, 0, 0, 0, 0, B_],
            [-RHO * OM**2, 0, 0, 0, -1j * kx, -1j * ky],
            [0, -RHO * OM**2 + ZETA * kx**2 + MU * ky**2, kx * ky * (CHI + MU), -1j * kx * GAM, 0, 0],
            [0, kx * ky * (CHI + MU), -RHO * OM**2 + ZETA * ky**2 + MU * kx**2, -1j * ky * GAM, 0, 0],
        ],
        dtype=complex,
    )


def dz(kx, ky):
    """Ddef with Peigen/SVeigen/SHeigen and epsdef, verbatim."""
    kzp, kzs, kzh = kz(ALPHA, kx, ky), kz(BETA, kx, ky), kz(BETA, kx, ky)
    ks_hat, kh_hat = khat(BETA, ky), khat(BETA, ky)
    eps_p = 1.0 / np.sqrt(2 * RHO * OM**2 * kzp)
    eps_s = OM / (BETA * ks_hat * np.sqrt(2 * RHO * OM**2 * kzs))
    eps_h = 1.0 / (kh_hat * np.sqrt(2 * RHO * OM**2 * kzh))

    cols = []
    for sg in (+1, -1):  # + = downgoing, - = upgoing
        cols.append(
            np.array(
                [
                    sg * 1j * kzp,
                    1j * kx,
                    1j * ky,
                    RHO * (2 * BETA**2 * kx**2 + 2 * BETA**2 * ky**2 - OM**2),
                    -sg * 2 * RHO * BETA**2 * kx * kzp,
                    -sg * 2 * RHO * BETA**2 * ky * kzp,
                ],
                dtype=complex,
            )
            * eps_p
        )
    for sg in (+1, -1):
        cols.append(
            np.array(
                [
                    1j * kx,
                    -sg * 1j * kzs,
                    0,
                    -sg * 2 * RHO * BETA**2 * kx * kzs,
                    RHO * (OM**2 - 2 * BETA**2 * kx**2 - BETA**2 * ky**2),
                    -RHO * BETA**2 * kx * ky,
                ],
                dtype=complex,
            )
            * eps_s
        )
    for sg in (+1, -1):
        cols.append(
            np.array(
                [
                    -sg * ky * kzh,
                    -kx * ky,
                    kh_hat**2,
                    2j * ky * RHO * (BETA**2 * kx**2 + BETA**2 * ky**2 - OM**2),
                    -sg * 2j * RHO * BETA**2 * kx * ky * kzh,
                    sg * 1j * kzh * RHO * (OM**2 - 2 * BETA**2 * ky**2),
                ],
                dtype=complex,
            )
            * eps_h
        )
    # Ddef column order: +P, +S, +H, -P, -S, -H
    order = [0, 2, 4, 1, 3, 5]
    return np.column_stack([cols[i] for i in order])


def lam_mat(kx, ky):
    """eigDef."""
    kzp, kzs, kzh = kz(ALPHA, kx, ky), kz(BETA, kx, ky), kz(BETA, kx, ky)
    return 1j * np.diag([kzp, kzs, kzh, -kzp, -kzs, -kzh])


print("Thesis self-consistency (transcribed verbatim from GRepresentations.tex)")
print("=" * 78)
for kx, ky in ((0.9, 0.7), (2.1, -1.3), (0.4, 0.0)):
    a, am = amat(kx, ky), amat(-kx, -ky)
    d, dm = dz(kx, ky), dz(-kx, -ky)
    lm = lam_mat(kx, ky)
    n = la.norm(a)

    at = la.norm(am.T @ J6 + J6 @ a) / n
    sp = la.norm(a @ d - d @ lm) / la.norm(a @ d)
    di = la.norm((J6 @ dm).T @ d - 1j * J6) / la.norm(1j * J6)
    d1 = la.norm(la.inv(d) - (-1j) * J6 @ dm.T @ J6) / la.norm(la.inv(d))
    print(f"\n kx={kx:+.2f} ky={ky:+.2f}")
    print(f"   ATdef  A^T(-k) J6 + J6 A(k) = 0      : {at:.3e}")
    print(f"   specA  A D_z = D_z Lambda            : {sp:.3e}")
    print(f"   dinv2  (J6 D_z(-k))^T D_z(k) = i J6  : {di:.3e}")
    print(f"   D1def  D_z^-1 = -i J6 D_z^T(-k) J6   : {d1:.3e}")

print()
print("=" * 78)
print("BOX 5.3 worked example vs the independently derived jump vector")
print("=" * 78)
kx, ky = 0.9, 0.7
# Thesis:  F_1 + A_S F_2  for a point explosion (M(w) = 1, x_S = 0)
thesis = np.array(
    [1 / (RHO * ALPHA**2), 0, 0, 0, 2j * kx * BETA**2 / ALPHA**2, 2j * ky * BETA**2 / ALPHA**2]
)
# Ours: explosion M_ij = delta_ij, jump vector of the wrapper derivation
m_zz = m_xx = m_yy = 1.0
ours = np.array([A_ * m_zz, 0, 0, 0, 1j * kx * (m_xx - GAM * m_zz), 1j * ky * (m_yy - GAM * m_zz)])
print(f"  thesis F1 + A_S F2 : {np.array2string(thesis, precision=6)}")
print(f"  our jump vector    : {np.array2string(ours, precision=6)}")
print(f"  relative difference: {la.norm(thesis - ours) / la.norm(thesis):.3e}")
