#!/usr/bin/env python
"""Thesis Section 3.1 energy-normalised interface R/T — self-contained Python twin.

A faithful NumPy port of ``Mathematica/ThesisInterfaceRT.wl``: the thesis Section 3.1
(``GRepresentations.tex``) energy-normalised displacement-traction eigenbasis ``D_z`` and the
canonical symplectic ``J6`` for two half-spaces, plus the plane-interface R/T solved via the
symplectic inverse (one 3x3 inverse, no 6x6 solve). Self-contained; run directly.

Conventions (thesis Section 3.1):
    - state vector b = (u_z, u_x, u_y, t_z, t_x, t_y): 3 displacement, 3 traction-on-z-plane.
    - "+" = downgoing, "-" = upgoing; column order of D_z is (+P,+S,+H, -P,-S,-H).
    - TIME convention e^{-i w t} (the THESIS convention, GRepresentations l.29) -- the SAME
      convention CartesianT0 / the Phase-3 code actually use; their "e^{+i w t}" comments are a
      mislabel, since an outgoing h_n^(1) / e^{+ikr} IS e^{-i w t}.
    - traction = physical stress on the +z plane (sigma_zz, sigma_zx, sigma_zy);
      lambda = rho(alpha^2 - 2 beta^2), mu = rho beta^2.
    - SEISMIC UNITS (km/s, g/cm^3 -> moduli in GPa): in these units D_z is well-conditioned
      (cond(D_z) = rho*omega*v ~ 1.9e4), so every check passes in float64. The apparent SI
      "ill-conditioning" (~1.9e10) is a pure units artifact, not a defect; see
      docs/Dz_conditioning_and_nondimensionalisation.md.

Reference: docs/superpowers/specs/2026-06-25-thesis-interface-rt-design.md.
"""

import cmath

import numpy as np

# ---------------------------------------------------------------------------
# 1. Kinematics: velocity per mode, K-hat, vertical wavenumber, energy-norm eps
# ---------------------------------------------------------------------------
MODES = ("P", "S", "H")  # column/mode order within each propagation direction


def _vel(c: str, alpha: float, beta: float) -> float:
    """Mode velocity: P uses alpha, S and H use beta."""
    return alpha if c == "P" else beta


def khat(c: str, alpha: float, beta: float, om: float, ky: float) -> complex:
    """Horizontal-removed wavenumber K_hat_c = sqrt((om/c)^2 - ky^2) (Eq. Kdef)."""
    return cmath.sqrt((om / _vel(c, alpha, beta)) ** 2 - ky**2)


def kz(c: str, alpha: float, beta: float, om: float, kx: float, ky: float) -> complex:
    """Vertical wavenumber k_z,c = sqrt(K_hat_c^2 - kx^2) (Eq. kzcDef)."""
    return cmath.sqrt(khat(c, alpha, beta, om, ky) ** 2 - kx**2)


# ---------------------------------------------------------------------------
# 2. The three energy-normalised eigenvectors (s = +1 down, s = -1 up)
#    Transcribed from Peigen / SVeigen / SHeigen x eps (epsdef).
# ---------------------------------------------------------------------------
def eig_p(
    s: int, alpha: float, beta: float, rho: float, om: float, kx: float, ky: float
) -> np.ndarray:
    """Downgoing(+)/upgoing(-) P eigenvector (6-vector), energy-normalised."""
    b2 = beta**2
    kzc = kz("P", alpha, beta, om, kx, ky)
    eps = 1.0 / cmath.sqrt(2 * rho * om**2 * kzc)
    return eps * np.array(
        [
            s * 1j * kzc,
            1j * kx,
            1j * ky,
            rho * (2 * b2 * kx**2 + 2 * b2 * ky**2 - om**2),
            -s * 2 * rho * b2 * kx * kzc,
            -s * 2 * rho * b2 * ky * kzc,
        ],
        dtype=np.complex128,
    )


def eig_s(
    s: int, alpha: float, beta: float, rho: float, om: float, kx: float, ky: float
) -> np.ndarray:
    """Downgoing(+)/upgoing(-) quasi-SV eigenvector (6-vector), energy-normalised."""
    b2 = beta**2
    ks = khat("S", alpha, beta, om, ky)
    kzc = kz("S", alpha, beta, om, kx, ky)
    eps = om / (beta * ks * cmath.sqrt(2 * rho * om**2 * kzc))
    return eps * np.array(
        [
            1j * kx,
            -s * 1j * kzc,
            0.0,
            -s * 2 * rho * b2 * kx * kzc,
            rho * (om**2 - 2 * b2 * kx**2 - b2 * ky**2),
            -rho * b2 * kx * ky,
        ],
        dtype=np.complex128,
    )


def eig_h(
    s: int, alpha: float, beta: float, rho: float, om: float, kx: float, ky: float
) -> np.ndarray:
    """Downgoing(+)/upgoing(-) quasi-SH eigenvector (6-vector), energy-normalised."""
    b2 = beta**2
    kh = khat("H", alpha, beta, om, ky)
    kzc = kz("H", alpha, beta, om, kx, ky)
    eps = 1.0 / (kh * cmath.sqrt(2 * rho * om**2 * kzc))
    return eps * np.array(
        [
            -s * ky * kzc,
            -kx * ky,
            kh**2,
            2j * ky * rho * (b2 * kx**2 + b2 * ky**2 - om**2),
            -s * 2j * rho * b2 * kx * ky * kzc,
            s * 1j * kzc * rho * (om**2 - 2 * b2 * ky**2),
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# 3. Eigen-matrix D_z = [ +P +S +H  -P -S -H ] (columns) and symplectic J6
# ---------------------------------------------------------------------------
def dz(
    alpha: float, beta: float, rho: float, om: float, kx: float, ky: float
) -> np.ndarray:
    """Energy-normalised eigen-matrix D_z (6x6), columns (+P,+S,+H,-P,-S,-H)."""
    cols = [
        eig_p(1, alpha, beta, rho, om, kx, ky),
        eig_s(1, alpha, beta, rho, om, kx, ky),
        eig_h(1, alpha, beta, rho, om, kx, ky),
        eig_p(-1, alpha, beta, rho, om, kx, ky),
        eig_s(-1, alpha, beta, rho, om, kx, ky),
        eig_h(-1, alpha, beta, rho, om, kx, ky),
    ]
    return np.column_stack(cols)


_I3 = np.eye(3, dtype=np.complex128)
J6 = np.block(
    [
        [np.zeros((3, 3), dtype=np.complex128), _I3],
        [-_I3, np.zeros((3, 3), dtype=np.complex128)],
    ]
)


def dz_inv(
    alpha: float, beta: float, rho: float, om: float, kx: float, ky: float
) -> np.ndarray:
    """Symplectic inverse (Eq. D1def): D_z^{-1}(k) = -i J6 D_z^T(-k) J6 (no elimination)."""
    return -1j * J6 @ dz(alpha, beta, rho, om, -kx, -ky).T @ J6


# ---------------------------------------------------------------------------
# 4. Interface R/T via the symplectic inverse: one 3x3 inverse, no 6x6 solve.
#    Q = D1^{-1} D2; (a_inc; a_refl) = Q.(a_trans; 0) => T = Q11^{-1}, R = Q21.T.
# ---------------------------------------------------------------------------
def interface_rt(
    m1: tuple[float, float, float],
    m2: tuple[float, float, float],
    om: float,
    kx: float,
    ky: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interface reflection/transmission (3x3 each, columns = incident mode P,S,H).

    Args:
        m1: (alpha, beta, rho) of the upper half-space (incidence side).
        m2: (alpha, beta, rho) of the lower half-space.
        om: angular frequency.
        kx: horizontal wavenumber along x.
        ky: horizontal wavenumber along y.

    Returns:
        (R, T): R is down-in -> up-out, T is down-in -> down-out.
    """
    q = dz_inv(*m1, om, kx, ky) @ dz(*m2, om, kx, ky)
    t = np.linalg.inv(q[0:3, 0:3])  # T = Q11^{-1}, the only inverse (3x3)
    r = q[3:6, 0:3] @ t  # R = Q21 . T
    return r, t


def _traction_from_u(
    uvec: np.ndarray, lam: float, mu: float, kx: float, ky: float, kzsig: complex
) -> np.ndarray:
    """Hooke's-law traction (sigma_zz, sigma_zx, sigma_zy) from a plane-wave displacement."""
    uz, ux, uy = uvec[0], uvec[1], uvec[2]
    return np.array(
        [
            lam * (1j * kx * ux + 1j * ky * uy + 1j * kzsig * uz)
            + 2 * mu * (1j * kzsig * uz),
            mu * (1j * kzsig * ux + 1j * kx * uz),
            mu * (1j * kzsig * uy + 1j * ky * uz),
        ],
        dtype=np.complex128,
    )


def _main() -> None:
    """Run the self-checks in seismic units and print PASS/FAIL gates."""
    tol = 1e-7
    # SEISMIC units: km/s, g/cm^3.  kx,ky < om/alpha = 300 -> all six modes propagate.
    m1 = (5.0, 3.0, 2.5)
    m2 = (5.5, 3.3, 2.7)
    om, kx, ky = 1500.0, 100.0, 50.0
    i6 = np.eye(6, dtype=np.complex128)
    print(
        "==== thesis_interface_rt :: half-space (u,t) eigenbasis + J6 (seismic units) ===="
    )
    print(f"  media: HS1 {m1}  HS2 {m2}  (om,kx,ky)=({om},{kx},{ky})")

    # [1] symplectic / energy-normalisation identity (J6 D(-k))^T D(k) == i J6  (dinv2)
    def symp(m: tuple[float, float, float]) -> float:
        return float(
            np.max(np.abs((J6 @ dz(*m, om, -kx, -ky)).T @ dz(*m, om, kx, ky) - 1j * J6))
        )

    s1, s2 = symp(m1), symp(m2)
    print(
        f"  [1] symplectic identity (J6 D(-k))^T D(k)=i J6 : HS1={s1:.3e}, HS2={s2:.3e}"
        f" -> {'PASS' if max(s1, s2) < tol else 'FAIL'}"
    )

    # [1b] inverse consistency D_z D_z^{-1} == I6
    def invres(m: tuple[float, float, float]) -> float:
        return float(np.max(np.abs(dz(*m, om, kx, ky) @ dz_inv(*m, om, kx, ky) - i6)))

    b1, b2 = invres(m1), invres(m2)
    print(
        f"  [1b] D_z . D_z^-1 == I6 : HS1={b1:.3e}, HS2={b2:.3e}"
        f" -> {'PASS' if max(b1, b2) < tol else 'FAIL'}"
    )

    # [2] traction == Hooke(displacement)
    mode_list = [("P", 1), ("S", 1), ("H", 1), ("P", -1), ("S", -1), ("H", -1)]

    def hooke(m: tuple[float, float, float]) -> float:
        al, be, rh = m
        lam, mu = rh * (al**2 - 2 * be**2), rh * be**2
        d = dz(*m, om, kx, ky)
        worst = 0.0
        for j, (c, s) in enumerate(mode_list):
            col = d[:, j]
            kzsig = s * kz(c, al, be, om, kx, ky)
            pred = _traction_from_u(col[0:3], lam, mu, kx, ky, kzsig)
            worst = max(worst, float(np.max(np.abs(pred - col[3:6]))))
        return worst

    h1, h2 = hooke(m1), hooke(m2)
    print(
        f"  [2] traction == Hooke(displacement) : HS1={h1:.3e}, HS2={h2:.3e}"
        f" -> {'PASS' if max(h1, h2) < tol else 'FAIL'}"
    )

    # interface R/T (3x3-via-J) + [3] energy + [3b] regression vs 6x6 solve
    r, t = interface_rt(m1, m2, om, kx, ky)
    print("  --- interface R/T (energy-normalised eigenbasis, mode order P,S,H) ---")
    with np.printoptions(precision=4, suppress=True):
        print("  R (down-in -> up-out) =\n", r)
        print("  T (down-in -> down-out) =\n", t)
    col_power = np.array(
        [np.sum(np.abs(r[:, i]) ** 2) + np.sum(np.abs(t[:, i]) ** 2) for i in range(3)]
    )
    e_dev = float(np.max(np.abs(col_power - 1.0)))
    print(
        f"  [3] per-incident power |R|^2+|T|^2 (P,S,H) = {np.round(col_power.real, 12)}"
        f"  max|.-1|={e_dev:.3e} -> {'PASS' if e_dev < tol else 'FAIL'}"
    )

    d1, d2 = dz(*m1, om, kx, ky), dz(*m2, om, kx, ky)
    mmat = np.hstack([d1[:, 3:6], -d2[:, 0:3]])  # 6x6 (a_refl | a_trans)
    xsol = np.linalg.solve(mmat, -d1[:, 0:3])  # 6x3
    rt_dev = float(np.max(np.abs(np.vstack([r - xsol[0:3], t - xsol[3:6]]))))
    print(
        f"  [3b] 3x3-via-J R/T == 6x6 solve : {rt_dev:.3e} -> {'PASS' if rt_dev < tol else 'FAIL'}"
    )

    assert max(s1, s2) < tol, "symplectic identity failed"
    assert max(b1, b2) < tol, "inverse consistency failed"
    assert max(h1, h2) < tol, "traction/Hooke check failed"
    assert e_dev < tol, "interface energy balance failed"
    assert rt_dev < tol, "3x3-via-J disagrees with 6x6 solve"
    print("  all checks PASS.")


if __name__ == "__main__":
    _main()
