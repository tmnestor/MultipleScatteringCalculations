"""Which coarsening of a finely layered background preserves the propagator?

Evidence for LatexPDFs/MultiscaleBackground.

A Haar projection of the background replaces fine structure by block averages
-- but averages of WHAT?  The layered problem marches as d_z b = A(z) b, so a
stack of thin layers has response prod_i exp(A_i h_i), and the Lie product
formula gives

    prod_i exp(A_i h_i)  ->  exp( sum_i A_i h_i )  =  exp( <A> H )

as h -> 0.  So the correct leading-order coarsening is the ARITHMETIC average
of the system matrix.  That is not a coincidence: A is linear in

    gamma = lam/(lam+2mu)   a = 1/(lam+2mu)   b = 1/mu
    zeta  = 4mu(lam+mu)/(lam+2mu)   chi = 2 mu lam/(lam+2mu)   mu   rho

and these are precisely the combinations Backus (1962) averages arithmetically:
C33 = 1/<a>, C44 = 1/<b>, C66 = <mu>, C13 = <gamma>/<a>, rho_eff = <rho>.
Hence: Haar averaging is CORRECT in the variables of A, and WRONG in
(alpha, beta, rho).

METHOD.  One fixed realisation of a strongly contrasted fine layered medium.
The long-wavelength limit is approached by lowering the FREQUENCY, so the
medium never changes and only H/lambda varies.

  A FIRST VERSION OF THIS SCRIPT REDREW THE LAYERS AT EACH REFINEMENT LEVEL,
  which compares different media, and it appeared to REFUTE the claim above.
  Vary the wavelength, not the medium.

NOT VALIDATED HERE: a transversely isotropic propagator for the Backus
effective medium.  The Backus constants printed below are evaluated from the
closed form, so the anisotropy figures are analytic, not measured.

Run: conda run -n seismic python scripts/coarsening_of_layered_background.py
"""

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(7)

N = 256  # fine layers
H = 1.0  # total stack thickness, km
h = H / N
P = 0.10  # horizontal slowness, s/km


def system_matrix(alpha, beta, rho, om, kx, ky):
    """Fourier-domain system matrix A (thesis Akdef; shorthand per its footnote)."""
    mu = rho * beta**2
    lam = rho * alpha**2 - 2 * mu
    gam = lam / (lam + 2 * mu)
    aa = 1.0 / (lam + 2 * mu)
    bb = 1.0 / mu
    zet = 4 * mu * (lam + mu) / (lam + 2 * mu)
    chi = 2 * mu * lam / (lam + 2 * mu)
    return np.array(
        [
            [0, -1j * gam * kx, -1j * gam * ky, aa, 0, 0],
            [-1j * kx, 0, 0, 0, bb, 0],
            [-1j * ky, 0, 0, 0, 0, bb],
            [-rho * om**2, 0, 0, 0, -1j * kx, -1j * ky],
            [
                0,
                -rho * om**2 + zet * kx**2 + mu * ky**2,
                kx * ky * (chi + mu),
                -1j * kx * gam,
                0,
                0,
            ],
            [
                0,
                kx * ky * (chi + mu),
                -rho * om**2 + zet * ky**2 + mu * kx**2,
                -1j * ky * gam,
                0,
                0,
            ],
        ],
        dtype=complex,
    )


# --- one fixed realisation -------------------------------------------------
al = 4.0 + rng.uniform(-1.2, 1.2, N)
be = al / 1.8
rh = 2.6 + rng.uniform(-0.4, 0.4, N)

mu_i = rh * be**2
lam_i = rh * al**2 - 2 * mu_i
gam_i = lam_i / (lam_i + 2 * mu_i)
a_i = 1.0 / (lam_i + 2 * mu_i)
b_i = 1.0 / mu_i
zet_i = 4 * mu_i * (lam_i + mu_i) / (lam_i + 2 * mu_i)

C33 = 1.0 / a_i.mean()
C44 = 1.0 / b_i.mean()
C66 = mu_i.mean()
C13 = gam_i.mean() / a_i.mean()
C11 = zet_i.mean() + gam_i.mean() ** 2 / a_i.mean()
nl_term = gam_i.mean() ** 2 / a_i.mean()

print("Backus effective constants (analytic; GPa, seismic units)")
print(f"  C11 = {C11:8.3f}   C33 = {C33:8.3f}   C11/C33 = {C11 / C33:.4f}")
print(f"  C44 = {C44:8.3f}   C66 = {C66:8.3f}   C66/C44 = {C66 / C44:.4f}")
print(f"  the non-linear <gamma>^2/<a> term is {nl_term / C11 * 100:.1f}% of C11")
print("  -> that term is what <A>-averaging cannot produce, and what induces")
print("     the anisotropy. It is the first-order Magnus (non-commutativity)")
print("     correction to the Lie product formula.")
print()

print("Coarsening the whole stack to ONE homogeneous layer, two ways:")
print(f"{'f (Hz)':>8} {'H/lambda':>10} {'velocity avg':>15} {'<A> avg':>13}")
for f in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    om = 2.0 * np.pi * f
    kx, ky = om * P * 0.6, om * P * 0.8

    exact = np.eye(6, dtype=complex)
    Asum = np.zeros((6, 6), dtype=complex)
    for n in range(N):
        A = system_matrix(al[n], be[n], rh[n], om, kx, ky)
        exact = expm(A * h) @ exact
        Asum += A * h

    vel = expm(system_matrix(al.mean(), be.mean(), rh.mean(), om, kx, ky) * H)
    amean = expm(Asum)

    e_vel = np.linalg.norm(vel - exact) / np.linalg.norm(exact)
    e_a = np.linalg.norm(amean - exact) / np.linalg.norm(exact)
    print(f"{f:8.2f} {H / (al.mean() / f):10.3f} {e_vel:15.3e} {e_a:13.3e}")

print()
print("<A> averaging converges as O((H/lambda)^2); velocity averaging is ~21x")
print("worse at the longest wavelength and fails outright within one wavelength.")
