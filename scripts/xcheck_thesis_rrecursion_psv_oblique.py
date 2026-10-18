"""Cross-check of the P-SV net reflection recursion at OBLIQUE incidence, with mode conversion.

Reference: a 4x4 propagator-matrix solve of the stack, independent of kennett_layers. The state is b =
(u_x, u_z, tau_zx, tau_zz), z down, fields ~ exp(i omega (p x - t)); db/dz = A b from the elastic
equations. Each layer's waves are the eigenvectors of A (down: +i omega xi, +i omega eta; up: their
negatives), normalised by a fixed rule (unit displacement, largest displacement component real
positive). The half-space carries down-going waves only; the solution is carried up through each layer
by expm(-A d), and decomposed in the top layer into down- and up-going amplitudes: R = U D^-1.

The recursion's amplitudes are normalised differently (energy-normalised, with their own polarisation
signs). Since R is referenced in the top medium, the two differ only by mode-wise factors of the top
medium at this slowness: R_code[i, j] = K[i, j] R_ref[i, j]. K is calibrated on ONE interface and must
then hold for multi-layer stacks with the same top medium: any error of the recursion through the
internal layers, including the conversions, breaks it.
"""

import numpy as np
from cubic_scattering.kennett_layers import LayerStack, kennett_layers
from scipy.linalg import expm


def system(alpha, beta, rho, p, w):
    lam2mu = rho * alpha**2
    mu = rho * beta**2
    lam = lam2mu - 2 * mu
    ip = 1j * w * p
    a = np.zeros((4, 4), complex)
    # u_x' = tau_zx / mu - i w p u_z
    a[0, 2], a[0, 1] = 1 / mu, -ip
    # u_z' = (tau_zz - lam i w p u_x) / (lam + 2 mu)
    a[1, 3], a[1, 0] = 1 / lam2mu, -lam * ip / lam2mu
    # tau_zx' = -rho w^2 u_x - i w p tau_xx,  tau_xx = lam2mu i w p u_x + lam u_z'
    a[2, 0] = -rho * w**2 - ip * lam2mu * ip
    a[2, :] += -ip * lam * a[1, :]
    # tau_zz' = -rho w^2 u_z - i w p tau_zx
    a[3, 1] += -rho * w**2
    a[3, 2] += -ip
    return a


def modes(alpha, beta, rho, p, w):
    """Columns: down P, down S, up P, up S, normalised by the fixed rule."""
    a = system(alpha, beta, rho, p, w)
    vals, vecs = np.linalg.eig(a)
    xi = np.sqrt(complex(1 / alpha**2 - p**2))
    eta = np.sqrt(complex(1 / beta**2 - p**2))
    xi = xi if xi.imag >= 0 else -xi
    eta = eta if eta.imag >= 0 else -eta
    out = []
    for target in (1j * w * xi, 1j * w * eta, -1j * w * xi, -1j * w * eta):
        v = vecs[:, np.argmin(np.abs(vals - target))]
        v = v / np.linalg.norm(v[:2])
        k = np.argmax(np.abs(v[:2]))
        v = v * np.exp(-1j * np.angle(v[k]))
        out.append(v)
    return np.array(out).T


def reference(alpha, beta, rho, thick, p, w):
    n = len(alpha) - 1
    m = modes(alpha[n], beta[n], rho[n], p, w)[:, :2]  # down-going only in the half-space
    for j in range(n - 1, 0, -1):
        m = expm(-system(alpha[j], beta[j], rho[j], p, w) * thick[j]) @ m
    coeff = np.linalg.solve(modes(alpha[0], beta[0], rho[0], p, w), m)
    return coeff[2:] @ np.linalg.inv(coeff[:2])  # rows up (P, S), columns down (P, S)


alpha = np.array([3000.0, 3800.0, 2600.0, 4500.0, 5200.0])
beta = np.array([1700.0, 2200.0, 1500.0, 2600.0, 3000.0])
rho = np.array([2200.0, 2400.0, 2100.0, 2600.0, 2800.0])
thick = np.array([400.0, 250.0, 120.0, 300.0, np.inf])
omega = 2 * np.pi * np.array([0.7, 2.0, 5.0])
worst = 0.0
for p in (1.0 / 6000.0, 1.0 / 4200.0):
    # calibrate on one interface: the top medium over the second layer's medium as a half-space
    one = LayerStack.from_arrays(alpha[:2], beta[:2], rho[:2], np.array([thick[0], np.inf]))
    r1 = kennett_layers(one, p, omega[:1]).RD_psv[0]
    k = r1 / reference(alpha[:2], beta[:2], rho[:2], [thick[0], np.inf], p, omega[0])
    for nl in (3, 5):
        stack = LayerStack.from_arrays(alpha[:nl], beta[:nl], rho[:nl], np.r_[thick[: nl - 1], np.inf])
        got = kennett_layers(stack, p, omega).RD_psv
        for i, w in enumerate(omega):
            want = k * reference(alpha[:nl], beta[:nl], rho[:nl], np.r_[thick[: nl - 1], np.inf], p, w)
            err = np.abs(got[i] - want).max() / np.abs(want).max()
            worst = max(worst, err)
        conv = np.abs(got[:, 0, 1]).max()
        print(
            f"p = 1/{1 / p:.0f}, {nl - 1} interfaces: max relative error {worst:.2e}; "
            f"|R_PS| up to {conv:.2f}"
        )
print(f"calibration factors |K| (top medium): {np.round(np.abs(k), 4).tolist()}")
print("PASS" if worst < 1e-9 else "FAIL")
