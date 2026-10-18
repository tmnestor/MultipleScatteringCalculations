"""Cross-check of Algorithm 6.3 (FGMRES) against cubic_scattering.interface_downup.fgmres.

(a) With a fixed preconditioner P, FGMRES is right-preconditioned GMRES: its residual history must equal
     that of an independent GMRES on H P (dense Arnoldi, modified Gram-Schmidt, as printed in the
     algorithm). (b) With a preconditioner that changes at every application, the residual must still
     decrease monotonically (the spaces span Z_n are nested) and the solution reach the dense solve.
"""

import numpy as np
from cubic_scattering.interface_downup import fgmres

rng = np.random.default_rng(3)
n = 120
# a complex, non-normal, indefinite test matrix: identity minus a scaled random coupling
h = np.eye(n) - 0.9 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2 * n)
b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
x_ref = np.linalg.solve(h, b)
p = np.linalg.inv(np.diag(np.diag(h)) + 0.3 * np.triu(h, 1))  # a fixed, imperfect preconditioner


def gmres_right_reference(a, b, p, m):
    """GMRES on a p (Algorithm 6.3 with P_j = P), modified Gram-Schmidt, relative residuals per step."""
    beta = np.linalg.norm(b)
    v = np.zeros((m + 1, b.size), complex)
    hh = np.zeros((m + 1, m), complex)
    v[0] = b / beta
    out = []
    for j in range(m):
        w = a @ (p @ v[j])
        for i in range(j + 1):
            hh[i, j] = np.vdot(v[i], w)
            w = w - hh[i, j] * v[i]
        hh[j + 1, j] = np.linalg.norm(w)
        v[j + 1] = w / hh[j + 1, j]
        e1 = np.zeros(j + 2, complex)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(hh[: j + 2, : j + 1], e1, rcond=None)
        out.append(np.linalg.norm(e1 - hh[: j + 2, : j + 1] @ y) / beta)
    return np.array(out)


# (a) fixed preconditioner, one cycle (restart larger than the iterations needed)
x_a, n_a, hist_a = fgmres(lambda v: h @ v, lambda v: p @ v, b, tol=1e-12, restart=200)
ref = gmres_right_reference(h, b, p, len(hist_a))
mask = ref > 1e-9
dev_a = np.max(np.abs(np.array(hist_a)[mask] / ref[mask] - 1))
err_a = np.linalg.norm(x_a - x_ref) / np.linalg.norm(x_ref)

# (b) a preconditioner that varies at every application: P perturbed by a fresh random term each time
calls = {"k": 0}


def varying(v):
    calls["k"] += 1
    noise = 0.05 * rng.standard_normal((n, n)) / np.sqrt(n)
    return (p + noise) @ v


x_b, n_b, hist_b = fgmres(lambda v: h @ v, varying, b, tol=1e-12, restart=200)
mono = bool(np.all(np.diff(hist_b) <= 1e-14))
err_b = np.linalg.norm(x_b - x_ref) / np.linalg.norm(x_ref)

print(
    f"(a) fixed P: {len(hist_a)} steps; max relative residual deviation (residual > 1e-9) = {dev_a:.2e}; "
    f"solution error {err_a:.2e}"
)
print(
    f"(b) varying P ({calls['k']} distinct applications): {len(hist_b)} steps; monotone = {mono}; "
    f"solution error {err_b:.2e}"
)
ok = dev_a < 1e-8 and err_a < 1e-10 and mono and err_b < 1e-10
print("PASS" if ok else "FAIL")
