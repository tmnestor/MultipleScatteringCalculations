"""Cross-check of Algorithms 3.1 and 4.1 (the sampling operators Phi, Phi^-1) as printed.

Upsampling places each voxel's 9-vector at sample ((chi-1) N_Delta + 1, (eta-1) N_Delta + 1) of an N_k x
N_k array; the in-plane coupling is then a product in the transformed domain with the kernel sampled on
the fine grid, and downsampling reads the result back at the same samples.  Checked against the direct
pairwise sum sum_s K(x_r - x_s) f_s, for several N_Delta, with N_k >= (2n - 1) N_Delta so that the
circular convolution of the FFT is the linear one.  N_Delta = 1 is the code's choice (LateralOperator:
padding to 2n - 1).  The round trip Phi^-1 Phi = I is checked too.  The 2.5-D Algorithm 3.1 is the one-
axis, 6-component case.
"""

import numpy as np

rng = np.random.default_rng(7)
n, comps = 5, 9


def kernel(dx, dy):
    """A smooth, decaying 9 x 9 in-plane kernel of the fine-grid offset (dx, dy); any kernel will do."""
    r = np.hypot(dx, dy)
    base = np.exp(1j * 0.7 * r - 0.15 * r) / (1.0 + r)
    mix = np.cos(0.3 * np.arange(comps)[:, None] - 0.2 * np.arange(comps)[None, :])
    return base[..., None, None] * mix


def phi(f, n_delta, n_k):
    """Algorithm 4.1, upsampling: f (9, n, n) -> (n_k, n_k, 9)."""
    g = np.zeros((n_k, n_k, comps), complex)
    for j in range(comps):
        for chi in range(n):
            for eta in range(n):
                g[chi * n_delta, eta * n_delta, j] = f[j, chi, eta]
    return g


def phi_inv(g, n_delta):
    """Algorithm 4.1, downsampling: (n_k, n_k, 9) -> (9, n, n)."""
    f = np.zeros((comps, n, n), complex)
    for j in range(comps):
        for chi in range(n):
            for eta in range(n):
                f[j, chi, eta] = g[chi * n_delta, eta * n_delta, j]
    return f


worst, round_trip = 0.0, 0.0
for n_delta in (1, 2, 4):
    n_k = (2 * n - 1) * n_delta
    f = rng.standard_normal((comps, n, n)) + 1j * rng.standard_normal((comps, n, n))
    round_trip = max(round_trip, np.abs(phi_inv(phi(f, n_delta, n_k), n_delta) - f).max())
    # the kernel on the fine grid, at signed offsets wrapped onto the FFT array
    idx = np.arange(n_k)
    off = np.where(idx <= n_k // 2, idx, idx - n_k)
    kfine = kernel(off[:, None].astype(float), off[None, :].astype(float))  # (n_k, n_k, 9, 9)
    kh = np.fft.fft2(kfine, axes=(0, 1))
    gh = np.fft.fft2(phi(f, n_delta, n_k), axes=(0, 1))
    out = phi_inv(np.fft.ifft2(np.einsum("abij,abj->abi", kh, gh), axes=(0, 1)), n_delta)
    # the direct pairwise sum on the lattice (pitch n_delta fine samples)
    want = np.zeros_like(f)
    for xr in range(n):
        for yr in range(n):
            for xs in range(n):
                for ys in range(n):
                    k = kernel(float((xr - xs) * n_delta), float((yr - ys) * n_delta))
                    want[:, xr, yr] += k @ f[:, xs, ys]
    err = np.abs(out - want).max() / np.abs(want).max()
    worst = max(worst, err)
    print(f"N_Delta = {n_delta}, N_k = {n_k}: FFT product vs pairwise sum, relative error {err:.2e}")
print(f"round trip Phi^-1 Phi = I: max error {round_trip:.1e}")
print("PASS" if worst < 1e-12 and round_trip == 0 else "FAIL")
