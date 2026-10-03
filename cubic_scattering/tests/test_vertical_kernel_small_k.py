"""The dz != 0 spectral plane-to-plane kernel at small kappa: no P-pole / S-polarisation cancellation.

For an evanescent order q the P-pole term k k e^{i k_zP |dz|} / (omega^2 k_zP) and the S-pole
polarisation term -k k e^{i k_zS |dz|} / (omega^2 k_zS) are each about q^2 / (omega^2 |q|), while their
sum is of order one; summed as written they lose digits as eps (q / kappa)^2 (the strain block of the
Bloch-summed kernel at dz = a: 1e-11 at kappa_S a = 0.03, 1e-8 at 1e-3).

The check shares nothing with the construction: with every order evanescent (k_par != 0, |q| > kappa)
the Bloch-summed block is analytic in omega^2, K(omega) = K_0 + K_1 omega^2 + O(omega^4). So the value at
the middle of three small frequencies must equal the linear interpolation in omega^2 of the outer two,
to O((kappa a)^4). Round-off from the cancellation does not interpolate, so the plain sum fails this at
the level it amplifies, which ``test_plain_sum_is_not_smooth`` confirms. Where the cancellation is
benign the new kernel must agree with the plain one.
"""

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sweep_kernels import _assemble_9x9, _branch, vertical_kernel_9x9

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
A = 10.0
B = 2.0 * np.pi / A
N_G = 6
KPAR = (0.2 * np.pi / A, 0.07 * np.pi / A)
PARTS = ((slice(0, 3), slice(0, 3)), (slice(0, 3), slice(3, 9)), (slice(3, 9), slice(3, 9)))


def _orders() -> tuple[np.ndarray, np.ndarray]:
    m, n = np.meshgrid(np.arange(-N_G, N_G + 1), np.arange(-N_G, N_G + 1), indexing="ij")
    return KPAR[0] + B * m.ravel(), KPAR[1] + B * n.ravel()


def _bloch_block(kernel, ka: float, dz: float) -> np.ndarray:
    omega = ka * REF.beta / A
    kx, ky = _orders()
    return sum(kernel(np.array([x]), y, dz, omega)[:, :, 0] for x, y in zip(kx, ky, strict=True))


def _new(kx, ky, dz, omega):
    return vertical_kernel_9x9(kx, ky, dz, omega, REF)


def _plain(kx, ky, dz, omega):
    """The kernel as it was: the three pole groups summed before assembly."""
    rho, alpha, beta = REF.rho, REF.alpha, REF.beta
    n = kx.size
    kh2 = kx**2 + ky**2
    kz_p, kz_s = _branch((omega / alpha) ** 2 - kh2), _branch((omega / beta) ** 2 - kh2)
    sign = 1.0 if dz > 0 else -1.0
    e_p, e_s = np.exp(1j * kz_p * abs(dz)), np.exp(1j * kz_s * abs(dz))
    kv_p = [sign * kz_p, kx.astype(complex), np.full(n, ky, dtype=complex)]
    kv_s = [sign * kz_s, kx.astype(complex), np.full(n, ky, dtype=complex)]
    c_s_iso = (1j / (2 * rho)) * e_s / (beta**2 * kz_s)
    c_p_pol = (1j / (2 * rho)) * e_p / (omega**2 * kz_p)
    c_s_pol = -(1j / (2 * rho)) * e_s / (omega**2 * kz_s)
    g_p, g_iso, g_pol = (np.zeros((3, 3, n), dtype=complex) for _ in range(3))
    for i in range(3):
        g_iso[i, i] = c_s_iso
        for j in range(3):
            g_p[i, j] = kv_p[i] * kv_p[j] * c_p_pol
            g_pol[i, j] = kv_s[i] * kv_s[j] * c_s_pol
    return _assemble_9x9(g_p + g_iso + g_pol, [g_p, g_iso, g_pol], [kv_p, kv_s, kv_s])


def _roughness(kernel, dz: float) -> float:
    """Departure of the middle value from the omega^2 interpolation, worst sub-block, relative."""
    kas = (2.5e-4, 5e-4, 1e-3)
    k1, k2, k3 = (_bloch_block(kernel, ka, dz) for ka in kas)
    u1, u2, u3 = (ka**2 for ka in kas)
    interp = k1 + (k3 - k1) * (u2 - u1) / (u3 - u1)
    return max(float(np.max(np.abs(k2[rc] - interp[rc])) / np.max(np.abs(k2[rc]))) for rc in PARTS)


@pytest.mark.parametrize("dz", [A, -A, 2.0 * A])
def test_kernel_is_smooth_in_omega_squared(dz: float) -> None:
    rough = _roughness(_new, dz)
    # The O(omega^4) curvature at these frequencies is 4.7e-13 (measured: it falls 16-fold per halving
    # of the frequencies, 1.2e-10 at (1, 2, 4) x 1e-3, so it is curvature and not round-off); the
    # round-off of the plain sum is above 1e-8.
    assert rough < 2e-12, f"dz = {dz}: departure from omega^2 smoothness {rough:.2e}"


def test_plain_sum_is_not_smooth() -> None:
    """The discrimination check: the old summation fails the same test."""
    assert _roughness(_plain, A) > 1e-10


@pytest.mark.parametrize("ka", [0.5, 2.0])
@pytest.mark.parametrize("cell", [None, 0.5 * A])
def test_kernel_matches_plain_where_benign(ka: float, cell: float | None) -> None:
    omega = ka * REF.beta / A
    kx, ky = _orders()
    # Only orders where the plain sum is itself trustworthy: its round-off is eps (|q| / kappa_S)^2, so
    # keep (|q| / kappa_S)^2 < 100. Beyond that (|q| a ~ 50 at kappa_S a = 0.5) it is the plain sum
    # that is off, by 1e-12.
    keep = np.hypot(kx, ky) < 10.0 * omega / REF.beta
    for x, y in zip(kx[keep], ky[keep], strict=True):
        new = vertical_kernel_9x9(np.array([x]), y, A, omega, REF, cell)[:, :, 0]
        if cell is None:
            old = _plain(np.array([x]), y, A, omega)[:, :, 0]
            np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12 * np.max(np.abs(old)))
