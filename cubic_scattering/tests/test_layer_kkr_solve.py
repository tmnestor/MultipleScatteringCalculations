"""The pieces of the layer-KKR solve that sit either side of the coupling.

  * the incident plane wave's regular-multipole coefficients -- analytic for P,
    by projection for S -- must reconstruct the plane wave;
  * the plane-wave spectrum of one outgoing multipole, every family and general
    m, must reproduce its field by direct Weyl (Sommerfeld) integration.

The solve itself, and its agreement with Mie when the coupling is switched
off, is gated in ``scripts/gate_layer_kkr_sphere_array.py``.

Run:  conda run -n seismic pytest cubic_scattering/tests/test_layer_kkr_solve.py -v
"""

import numpy as np
import pytest

from cubic_scattering import layer_kkr as lk

K_P, K_S = 0.9, 1.5


def _reconstruct(coef: dict, pts: np.ndarray) -> np.ndarray:
    out = np.zeros((len(pts), 3), dtype=complex)
    for (fam, n, m), c in coef.items():
        if fam == "L":
            out += c * lk.l_field(n, m, K_P, pts, "j")
        elif fam == "M":
            out += c * lk.m_field(n, m, K_S, pts, "j")
        else:
            out += c * lk.n_field(n, m, K_S, pts, "j")
    return out


@pytest.mark.parametrize(
    "mode,khat,pol",
    [
        ("P", (0.0, 0.0, 1.0), None),
        ("P", (0.3, -0.4, -np.sqrt(1 - 0.25)), None),
        ("S", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ("S", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ("S", (0.6, 0.0, 0.8), (0.8, 0.0, -0.6)),
    ],
)
def test_incident_coefficients_reconstruct_the_plane_wave(
    mode: str, khat: tuple, pol: tuple | None
) -> None:
    khat_a = np.array(khat) / np.linalg.norm(khat)
    coef = lk.incident_coefficients(mode, khat_a, K_P, K_S, 14, pol=None if pol is None else np.array(pol))
    pts = np.random.default_rng(1).normal(size=(20, 3)) * 0.3
    k = K_P if mode == "P" else K_S
    e = khat_a if mode == "P" else np.array(pol)
    want = np.exp(1j * k * pts @ khat_a)[:, None] * e[None, :]
    got = _reconstruct(coef, pts)
    assert np.max(np.abs(got - want)) < 1e-10


def _weyl(fam: str, n: int, m: int, point: np.ndarray, nq: int = 300, npsi: int = 64) -> np.ndarray:
    """Direct Weyl superposition of lk.plane_wave_amplitude over the whole q-plane."""
    k = K_P if fam == "L" else K_S
    x, y, z = point
    up = z < 0.0
    gl_t, gl_w = np.polynomial.legendre.leggauss(nq)
    psi = 2.0 * np.pi * np.arange(npsi) / npsi
    total = np.zeros(3, dtype=complex)
    # field = Int q dq dpsi uhat(q, psi) e^{i(q.rho + kap |z|)}; each substitution
    # below cancels the 1/kap that uhat carries.
    th = 0.25 * np.pi * (gl_t + 1.0)
    prop = (k * np.sin(th), k * np.cos(th) + 0j, 0.25 * np.pi * gl_w * k**2 * np.sin(th) * np.cos(th))
    tmax = float(np.arccosh(max(40.0 / abs(z), 1.01 * k) / k))
    tt = 0.5 * tmax * (gl_t + 1.0)
    evan = (k * np.cosh(tt), 1j * k * np.sinh(tt), 0.5 * tmax * gl_w * k**2 * np.cosh(tt) * np.sinh(tt))
    for q, kap, wq in (prop, evan):
        for qq, kk, w in zip(q, kap, wq, strict=True):
            kx, ky = qq * np.cos(psi), qq * np.sin(psi)
            amp = lk.plane_wave_amplitude(fam, n, m, kx, ky, K_P, K_S, upward=up)  # (npsi, 3), = uhat
            phase = np.exp(1j * (kx * x + ky * y + kk * abs(z)))
            total += w * np.sum(amp * phase[:, None], axis=0) * (2.0 * np.pi / npsi)
    return total


@pytest.mark.parametrize("fam,n,m", [("L", 2, 1), ("N", 2, -1), ("M", 3, 2), ("N", 1, 0), ("M", 2, -2)])
def test_plane_wave_spectrum_reproduces_the_multipole(fam: str, n: int, m: int) -> None:
    worst = 0.0
    for point in (np.array([0.4, -0.3, 1.2]), np.array([-0.5, 0.2, -1.5])):
        got = _weyl(fam, n, m, point)
        k = K_P if fam == "L" else K_S
        fn = {"L": lk.l_field, "M": lk.m_field, "N": lk.n_field}[fam]
        want = fn(n, m, k, point[None, :], "h")[0]
        worst = max(worst, np.max(np.abs(got - want)) / np.max(np.abs(want)))
    assert worst < 1e-6, f"worst spectrum error {worst:.2e}"
