"""Cross-check of the net reflection recursion (5.51) against a propagator-matrix solve of the same stack.

kennett_layers' recursions (SH and P-SV) return the stack's net downgoing reflection, referenced at the
first interface and seen from the top layer.  The reference here solves the boundary-value problem
directly: a purely downgoing wave in the half-space, (u, tau) matched up through each layer, decomposed
in the top layer into down- and upgoing amplitudes, R = b / a.  SH at oblique incidence (one stack with
an evanescent layer); P-SV at normal incidence, where P and SV decouple, so R_PP and R_SS are scalar
problems with impedances rho alpha and rho beta.
"""

import numpy as np
from cubic_scattering.kennett_layers import LayerStack, kennett_layers


def vertical(s, p):
    """Vertical slowness sqrt(s^2 - p^2), branch Im >= 0 (decaying down), real >= 0 if propagating."""
    q = np.sqrt(complex(s * s - p * p))
    return q if q.imag > 0 or (q.imag == 0 and q.real >= 0) else -q


def propagator_reflection(vel, rho, thick, p, omega):
    """Net reflection of the stack below the first interface, from a direct solve (scalar wave)."""
    out = []
    for w in omega:
        eta = [vertical(1.0 / v, p) for v in vel]
        mu = [r * v * v for r, v in zip(rho, vel, strict=True)]
        # top of the half-space: purely downgoing, u = 1
        n = len(vel) - 1
        u, tau = 1.0 + 0j, 1j * w * mu[n] * eta[n]
        # up through layers n-1, ..., 1: from (u, tau) at a layer's bottom to its top
        for j in range(n - 1, 0, -1):
            e = np.exp(1j * w * eta[j] * thick[j])
            z = 1j * w * mu[j] * eta[j]
            # u_bot = a e + b / e, tau_bot = z (a e - b / e), amplitudes referenced at the layer's top
            a = (u + tau / z) / (2 * e)
            b = (u - tau / z) * e / 2
            u, tau = a + b, z * (a - b)
        z0 = 1j * w * mu[0] * eta[0]
        a0, b0 = (u + tau / z0) / 2, (u - tau / z0) / 2
        out.append(b0 / a0)
    return np.array(out)


omega = 2 * np.pi * np.array([0.5, 1.0, 2.0, 5.0, 11.0])
alpha = np.array([3000.0, 3800.0, 2600.0, 4500.0, 5200.0])
beta = np.array([1700.0, 2200.0, 1500.0, 2600.0, 3000.0])
rho = np.array([2200.0, 2400.0, 2100.0, 2600.0, 2800.0])
thick = np.array([400.0, 250.0, 120.0, 300.0, np.inf])
stack = LayerStack.from_arrays(alpha, beta, rho, thick)

worst = 0.0
# SH, oblique: p below every 1/beta, then p beyond 1/beta in the slow layer (evanescent there)
for p in (1.0 / 4000.0, 1.0 / 1600.0):
    got = kennett_layers(stack, p, omega).RSH
    want = propagator_reflection(beta, rho, thick, p, omega)
    err = np.abs(got - want).max()
    worst = max(worst, err)
    print(
        f"SH, p = 1/{1 / p:.0f} s/m: max |R_recursion - R_propagator| = {err:.2e}"
        f"  (|R| up to {np.abs(want).max():.2f})"
    )

# P-SV at normal incidence: R_PP and R_SS against the scalar solves
res = kennett_layers(stack, 0.0, omega)
for name, got, vel in (("PP", res.RPP, alpha), ("SS", res.RSS, beta)):
    # amplitudes along each wave's own polarisation (Box 4.5: e_{+-,P} = k_{+-,P} / k_P): the
    # upgoing wave points
    # against +z, so the amplitude ratio is minus the displacement ratio of the scalar solve
    want = -propagator_reflection(vel, rho, thick, 0.0, omega)
    err = np.abs(got - want).max()
    worst = max(worst, err)
    print(f"P-SV, p = 0, R_{name}: max |R_recursion - R_propagator| = {err:.2e}")
print(
    f"P-SV, p = 0, mode conversion |R_PS|, |R_SP| = "
    f"{np.abs(res.RPS).max():.1e}, {np.abs(res.RSP).max():.1e}"
)
print("PASS" if worst < 1e-10 else "FAIL")
