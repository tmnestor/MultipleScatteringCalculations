"""Verify the elastodynamic near-field Green's tensor approximation.

Compares G^NF = -1/(8*pi*mu*r) * [1 - beta^2/alpha^2] * [I - 3*x*x^T/r^2]
against the exact Stokes tensor via two independent formulations:
  1. Kupradze representation (analytic second derivatives)
  2. Kupradze representation (numerical finite-difference check)

Reference: Wu & Ben-Menahem (1985), Geophys. J. R. astr. Soc. 81, 609-621.
"""

import numpy as np


def stokes_tensor_kupradze(
    x: np.ndarray, omega: float, rho: float, lam: float, mu: float
) -> np.ndarray:
    """Exact frequency-domain elastodynamic Green's tensor via Kupradze representation.

    G_ij = (1/(4*pi*mu)) * [delta_ij Phi_beta + (1/k_beta^2) d_i d_j (Phi_beta - Phi_alpha)]

    where Phi(k,R) = exp(-ikR)/R is the outgoing scalar Helmholtz Green's function.
    Satisfies: mu nabla^2 G_ij + (lam+mu) d_i d_k G_kj + rho*omega^2 G_ij = -delta_ij delta(x).
    """
    R = np.linalg.norm(x)
    gamma = x / R
    gg = np.outer(gamma, gamma)
    I3 = np.eye(3)

    alpha = np.sqrt((lam + 2 * mu) / rho)
    beta = np.sqrt(mu / rho)
    k_a = omega / alpha
    k_b = omega / beta

    def phi(k: float, r: float) -> complex:
        """Scalar Helmholtz Green's function: exp(-ikR)/R."""
        return np.exp(-1j * k * r) / r

    def d2_phi(k: float, r: float) -> tuple[complex, complex]:
        """Return (coeff_gg, coeff_delta) where d_i d_j Phi = coeff_gg * gamma_i gamma_j + coeff_delta * delta_ij.

        Phi = exp(-ikR)/R
        Phi' = -(ik + 1/R) exp(-ikR)/R
        Phi'' = (-k^2/R + 2ik/R^2 + 2/R^3) exp(-ikR)
        d_i d_j Phi = [Phi'' - Phi'/R] gamma_i gamma_j + [Phi'/R] delta_ij
        """
        e = np.exp(-1j * k * r)
        phi_prime_over_r = e * (-1j * k / r**2 - 1 / r**3)
        phi_double_prime = e * (-(k**2) / r + 2j * k / r**2 + 2 / r**3)
        coeff_gg = phi_double_prime - phi_prime_over_r
        coeff_delta = phi_prime_over_r
        return coeff_gg, coeff_delta

    # Second derivatives for alpha and beta
    a_gg_a, a_d_a = d2_phi(k_a, R)
    a_gg_b, a_d_b = d2_phi(k_b, R)

    # d_i d_j (Phi_beta - Phi_alpha)  [NOTE: beta MINUS alpha]
    d2_diff = (a_gg_b - a_gg_a) * gg + (a_d_b - a_d_a) * I3

    # Full tensor: (1/(4*pi*mu)) * [delta_ij * Phi_beta + (1/k_beta^2) * d2_diff]
    return (phi(k_b, R) * I3 + d2_diff / k_b**2) / (4 * np.pi * mu)


def stokes_tensor_numerical(
    x: np.ndarray, omega: float, rho: float, lam: float, mu: float, h: float = 1e-4
) -> np.ndarray:
    """Independent check: Kupradze with numerical finite differences for d_i d_j."""
    alpha = np.sqrt((lam + 2 * mu) / rho)
    beta = np.sqrt(mu / rho)
    k_a = omega / alpha
    k_b = omega / beta

    def phi_diff(xx: np.ndarray) -> complex:
        """Phi_beta(R) - Phi_alpha(R)."""
        r = np.linalg.norm(xx)
        return np.exp(-1j * k_b * r) / r - np.exp(-1j * k_a * r) / r

    G = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = 1.0
            ej[j] = 1.0
            # Central difference for d_i d_j
            d2 = (
                phi_diff(x + h * ei + h * ej)
                - phi_diff(x + h * ei - h * ej)
                - phi_diff(x - h * ei + h * ej)
                + phi_diff(x - h * ei - h * ej)
            ) / (4 * h**2)
            G[i, j] = d2 / k_b**2

    # Add scalar (delta_ij * Phi_beta) term
    R = np.linalg.norm(x)
    G += np.exp(-1j * k_b * R) / R * np.eye(3)

    return G / (4 * np.pi * mu)


def near_field_eq5(x: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Wu & Ben-Menahem Eq. (5): leading near-field term only.

    (G)_near ~ -1/(8*pi*mu) * (1/R) * (1 - beta^2/alpha^2) * (I - 3*eR*eR^T)
    """
    r = np.linalg.norm(x)
    eR = x / r
    eReR = np.outer(eR, eR)
    I3 = np.eye(3)
    return (-1.0 / (8 * np.pi * mu * r)) * (1 - (beta / alpha) ** 2) * (I3 - 3 * eReR)


def asymptotic_eq6(x: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Wu & Ben-Menahem Eq. (6): complete R->0 asymptotic with both 1/R terms.

    G ~ -1/(8*pi*mu*R) * (1 - b2/a2)(I - 3 eR eR)
        + 1/(4*pi*mu*R) * [(I - eR eR) + (b2/a2) eR eR]

    Second line simplifies to:
    G ~ 1/(8*pi*mu*R) * [(1 + b2/a2) I + (1 - b2/a2) eR eR]
    """
    r = np.linalg.norm(x)
    eR = x / r
    eReR = np.outer(eR, eR)
    I3 = np.eye(3)
    b2_a2 = (beta / alpha) ** 2

    term1 = (-1.0 / (8 * np.pi * mu * r)) * (1 - b2_a2) * (I3 - 3 * eReR)
    term2 = (1.0 / (4 * np.pi * mu * r)) * ((I3 - eReR) + b2_a2 * eReR)
    return term1 + term2


def main() -> None:
    # Material parameters (granite-like)
    rho = 2700.0
    lam = 2.16e10
    mu = 2.7e10
    alpha = np.sqrt((lam + 2 * mu) / rho)
    beta = np.sqrt(mu / rho)
    omega = 2 * np.pi * 1.0
    k_alpha = omega / alpha
    k_beta = omega / beta
    wavelength_S = 2 * np.pi / k_beta

    print(
        f"alpha = {alpha:.1f} m/s, beta = {beta:.1f} m/s, alpha/beta = {alpha / beta:.4f}"
    )
    print(f"beta^2/alpha^2 = {(beta / alpha) ** 2:.6f}")
    print(f"1 - beta^2/alpha^2 = {1 - (beta / alpha) ** 2:.6f}")
    print(
        f"wavelength_S = {wavelength_S:.1f} m, wavelength_P = {2 * np.pi / k_alpha:.1f} m"
    )

    # --- Test 0: Verify two exact implementations agree ---
    print("\n" + "=" * 75)
    print("TEST 0: Analytic vs Numerical Kupradze (internal consistency)")
    print("=" * 75)
    x_test = np.array([1.0, 2.0, 3.0])
    G_analytic = stokes_tensor_kupradze(x_test, omega, rho, lam, mu)
    G_numerical = stokes_tensor_numerical(x_test, omega, rho, lam, mu, h=1e-3)
    rel_err = np.linalg.norm(G_analytic - G_numerical) / np.linalg.norm(G_analytic)
    print(f"Relative error (analytic vs FD): {rel_err:.2e} (should be < 1e-5)")

    # --- Test 1: Convergence table ---
    x_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

    print("\n" + "=" * 75)
    print("TEST 1: Exact G vs Eq.(5) [NF only] vs Eq.(6) [full asymptotic]")
    print("=" * 75)
    print(
        f"{'R/lam_S':>10} {'k_b*R':>10} {'err(Eq5)':>12} {'err(Eq6)':>12} {'Im/Re':>10}"
    )

    for R_over_lam in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        R = R_over_lam * wavelength_S
        x = x_hat * R

        G_exact = stokes_tensor_kupradze(x, omega, rho, lam, mu)
        G_eq5 = near_field_eq5(x, mu, alpha, beta)
        G_eq6 = asymptotic_eq6(x, mu, alpha, beta)

        norm_ex = np.linalg.norm(G_exact)
        err5 = np.linalg.norm(G_exact - G_eq5) / norm_ex
        err6 = np.linalg.norm(G_exact - G_eq6) / norm_ex
        im_re = np.linalg.norm(G_exact.imag) / np.linalg.norm(G_exact.real)

        print(
            f"{R_over_lam:>10.4f} {k_beta * R:>10.4f}"
            f" {err5:>12.6f} {err6:>12.6f} {im_re:>10.6f}"
        )

    # --- Test 2: Element-wise comparison at deep near field ---
    print("\n" + "=" * 75)
    print("TEST 2: Element-wise at R/lambda_S = 0.0001")
    print("=" * 75)
    R = 0.0001 * wavelength_S
    x = x_hat * R
    G_exact = stokes_tensor_kupradze(x, omega, rho, lam, mu)
    G_eq5 = near_field_eq5(x, mu, alpha, beta)
    G_eq6 = asymptotic_eq6(x, mu, alpha, beta)

    print("\nG_exact (real part):")
    print(np.array2string(G_exact.real, precision=6, suppress_small=True))
    print("\nEq.(5) G_NF:")
    print(np.array2string(G_eq5, precision=6, suppress_small=True))
    print("\nEq.(6) full asymptotic:")
    print(np.array2string(G_eq6, precision=6, suppress_small=True))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_eq6 = np.where(np.abs(G_eq6) > 1e-40, G_exact.real / G_eq6, np.nan)
    print("\nRatio Re(G_exact) / Eq.(6) (should -> 1.0):")
    print(np.array2string(ratio_eq6, precision=6))

    # --- Test 3: Symmetry ---
    print("\n" + "=" * 75)
    print("TEST 3: Symmetry G_ij = G_ji")
    print("=" * 75)
    x = np.array([0.3, 0.5, 0.7])
    G = stokes_tensor_kupradze(x, omega, rho, lam, mu)
    sym_err = np.linalg.norm(G - G.T) / np.linalg.norm(G)
    print(f"||G - G^T|| / ||G|| = {sym_err:.2e}")

    # --- Test 4: Verify identities ---
    print("\n" + "=" * 75)
    print("TEST 4: Algebraic identity check")
    print("=" * 75)
    b2_a2 = (beta / alpha) ** 2
    print(
        f"1 - beta^2/alpha^2 = (lambda+mu)/(lambda+2mu) = {(lam + mu) / (lam + 2 * mu):.6f}"
    )
    print(f"Direct: {1 - b2_a2:.6f}")

    # Eq. (6) second line: should equal (1/(8*pi*mu*R)) * [(1+b2/a2) I + (1-b2/a2) eReR]
    R = 0.0001 * wavelength_S
    x = x_hat * R
    G_eq6 = asymptotic_eq6(x, mu, alpha, beta)
    eR = x / R
    eReR = np.outer(eR, eR)
    I3 = np.eye(3)
    G_eq6_alt = (1.0 / (8 * np.pi * mu * R)) * ((1 + b2_a2) * I3 + (1 - b2_a2) * eReR)
    print(f"Eq.(6) two forms agree: {np.allclose(G_eq6, G_eq6_alt)}")

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()
