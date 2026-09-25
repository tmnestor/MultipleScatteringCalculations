#!/usr/bin/env python3
"""GATE A5: the Taylor test -- the gradient against the misfit itself, at finite contrast.

WHY THIS GATE EXISTS
--------------------
A4 showed the adjoint is the transpose of the tangent-linear model; it says
nothing about whether that model is the derivative of the MISFIT. The Taylor
test does. For a random direction dm,

    Rem1(eps) = misfit(m + eps dm) - misfit(m)                        = O(eps)
    Rem2(eps) = misfit(m + eps dm) - misfit(m) - eps (grad misfit . dm)  = O(eps^2)

if and only if grad misfit is right. Halving eps must divide Rem2 by 4: slope 2 in a
log-log fit. Any error in the gradient leaves an O(eps) piece and the slope
falls to 1. Rem2/eps^2 must also settle, to 1/2 dm^T (grad^2 M) dm.

The misfit is evaluated by the full forward model -- T0 recomputed from the
perturbed contrasts, the full Foldy-Lax solve, the receiver map -- so this is
the nonlinear model, not a linearisation. d_obs comes from a different "true"
model, so the residual is non-zero and O(1).

CASES
-----
Physical Rayleigh T0 within its validity reaches spr(G0 T0) ~ 0.3 at most (gate
A4), so, as there:
  [W]  whole space, ka_S = 0.26, 8x the test contrast      spr ~ 0.075
  [S]  stratified fast-slab crust, receiver in the slab     spr ~ 0.33
  [W-stress] whole space, ka_S = 0.52, 40x contrast        spr ~ 0.80
       (outside the Rayleigh cube's validity; the Taylor test is still exact
       for the discrete model, and this case exercises the resolvent)

CHECKS
------
  [A5]  slope of Rem2 in [1.9, 2.1], and Rem2/eps^2 steady to < 1e-2 relative
        across the finest three eps.
  [c1]  CALIBRATION: Rem1 must have slope ~1.
  [c2]  CALIBRATION: a Born-only gradient (residual back-propagated without the
        adjoint solve) must leave slope ~1 as well.
  Both calibration slopes are ASYMPTOTIC: fitted over the finest three eps,
  asserted in [0.8, 1.3]. A first version fitted them over the whole range and
  reported the gate blind. It was not: where grad.dm is small beside the
  curvature (Rem1 even changes sign) or the Born error is small (spr ~ 0.07), the
  O(eps^2) term dominates at large eps, and only the small-eps end shows the
  O(eps) defect these legs exist to expose.

eps runs from 1e-1 down to 1e-1/2^10 (~1e-4) of the base contrast. MEASURED:
no solver floor down there -- Rem2/eps^2 still steady to four digits at 1e-4
(GMRES at rtol 1e-12). The range has to reach that far: in the stress case
grad.dm is only ~0.04 against a curvature of ~2.6, so Rem1 turns linear only for
eps well below 0.017, and a grid stopping at 1e-1/2^6 left the calibration legs
short of their asymptote.

Run:  conda run -n seismic python scripts/gate_a5_taylor.py
Seismic units (km/s, g/cm3, GPa), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import cubic_scattering.sweep_gradient as sg  # noqa: E402
from cubic_scattering.directional_sweeps import (  # noqa: E402
    LayeredBackground,
    build_g0_cache,
    make_sweep_grid,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402

SOLVE = {"tol": 1e-12, "max_iter": 2000}
STEP = np.array([1e-2, 1e-2, 1e-3])
BASE = np.array([2.0, 1.0, 0.1])  # the project's test contrast (Dlambda, Dmu, Drho)
EPS = 1e-1 / 2.0 ** np.arange(11)


def whole_space_setup(freq: float):
    """Caches at +-k_y, receiver map, per-plane media, omega, half-width."""
    ref = ReferenceMedium(alpha=5.0, beta=3.0, rho=2.5)
    omega, pitch, ky = 2 * np.pi * freq * (1.0 + 0.03j), 0.25, 0.6
    caches = [
        build_g0_cache(make_sweep_grid(3, 6, pitch, ky=s * ky, n_kz=128, n_kx=128), ref, omega)
        for s in (1, -1)
    ]
    receiver = sg.whole_space_receiver_map(
        caches[0].grid,
        ref,
        omega,
        receiver_x=np.array([0.1, 0.6, 1.1, 1.4]),
        receiver_dz=-1.0,
        rows=(0, 1, 2),
    )
    return caches, receiver, [ref] * 3, omega, 0.5 * pitch


def stratified_setup():
    """The fast-slab crust, receiver at interface 9 inside the slab."""
    from Kennett_Reflectivity.layer_model import LayerModel

    n_lay, q = 24, 2.0
    al, be, rh = 4.0, 2.22, 2.6
    a = [1.5, *([al] * n_lay), al]
    b = [0.0, *([be] * n_lay), be]
    r = [1.03, *([rh] * n_lay), rh]
    for lay in (9, 10):
        a[lay], b[lay], r[lay] = 6.5, 3.7, 3.3
    model = LayerModel.from_arrays(
        alpha=a,
        beta=b,
        rho=r,
        thickness=[3.0, *([1.0] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e10, *([q] * n_lay), q],
    )
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    planes = (7, 11)
    refs = [ReferenceMedium(1.0 / s_p[p], 1.0 / s_s[p], model.rho[p]) for p in planes]
    omega = 2 * np.pi * 0.3
    bg = LayeredBackground(model=model, plane_ifaces=planes)
    caches = [
        build_g0_cache(
            make_sweep_grid(2, 4, 1.0, ky=s * 0.3, n_kz=64, n_kx=64, kx_max=3.0),
            refs[0],
            omega,
            background=bg,
        )
        for s in (1, -1)
    ]
    receiver = sg.layered_receiver_map(
        caches[0].grid,
        model,
        omega,
        receiver_x=np.array([0.4, 1.6, 2.9]),
        receiver_iface=9,
        plane_ifaces=planes,
        rows=(0, 1, 2),
    )
    return caches, receiver, refs, omega, 0.5


def slope(eps: np.ndarray, err: np.ndarray) -> float:
    """Log-log slope of err against eps."""
    return float(np.polyfit(np.log(eps), np.log(np.abs(err)), 1)[0])


def run(name: str, setup, scale: float) -> bool:
    """Taylor remainders along one random direction."""
    (cache_p, cache_m), receiver, refs, omega, half_width = setup
    n_z, n_x = cache_p.grid.n_z, cache_p.grid.n_x
    rng = np.random.default_rng(20260926)
    psi_inc = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))

    def misfit(m, d_obs):
        t0, dt0 = sg.rayleigh_t0_and_derivative(m, refs, omega, half_width, step=STEP)
        return sg.misfit_and_gradient(cache_p, cache_m, t0, dt0, psi_inc, receiver, d_obs, **SOLVE)

    true_m = scale * BASE * (1.0 + 0.3 * rng.standard_normal((n_z, n_x, 3)))
    d_obs = misfit(true_m, np.zeros(receiver.n_data, complex)).data
    m0 = scale * BASE * (1.0 + 0.3 * rng.standard_normal((n_z, n_x, 3)))
    dm = scale * BASE * rng.standard_normal((n_z, n_x, 3))

    base = misfit(m0, d_obs)
    t0, dt0 = sg.rayleigh_t0_and_derivative(m0, refs, omega, half_width, step=STEP)
    spr = sg_spectral_radius(cache_p, t0)
    g_dm = float(np.sum(base.grad * dm))
    # [c2] Born-only gradient: the residual back-propagated without the adjoint solve.
    psi_adj_born = receiver.apply_transpose(np.conj(base.data - d_obs))
    g_born_dm = float(np.sum(np.einsum("zxi,zxaij,zxj->zxa", psi_adj_born, dt0, base.psi).real * dm))

    e1, e2, e2b = [], [], []
    for eps in EPS:
        diff = misfit(m0 + eps * dm, d_obs).misfit - base.misfit
        e1.append(diff)
        e2.append(diff - eps * g_dm)
        e2b.append(diff - eps * g_born_dm)
    e1, e2, e2b = np.array(e1), np.array(e2), np.array(e2b)
    curv = e2 / EPS**2
    steady = float(np.abs(np.diff(curv[-3:])).max() / abs(curv[-1]))
    s2 = slope(EPS, e2)
    s1, s2b = slope(EPS[-3:], e1[-3:]), slope(EPS[-3:], e2b[-3:])  # asymptotic

    print(f"\n{name}: spr(G0 T0) = {spr:.3f}, misfit = {base.misfit:.4e}")
    print(f"    {'eps':>9} {'Rem1':>11} {'Rem2':>11} {'Rem2/eps^2':>11} {'Rem2 Born grad':>13}")
    for k, eps in enumerate(EPS):
        print(f"    {eps:9.2e} {e1[k]:11.3e} {e2[k]:11.3e} {curv[k]:11.4e} {e2b[k]:13.3e}")
    print(f"    slope: Rem2 {s2:.3f} (all eps);  asymptotic Rem1 {s1:.3f}, Rem2 Born grad {s2b:.3f}")
    print(f"    Rem2/eps^2 steady to {steady:.1e} over the finest three eps")
    ok = 1.9 < s2 < 2.1 and steady < 1e-2
    ok_c = 0.8 < s1 < 1.3 and 0.8 < s2b < 1.3
    print(
        f"    [A5] {'PASS' if ok else 'FAIL'}   [c1, c2] {'fail as they must' if ok_c else 'GATE IS BLIND'}"
    )
    return ok and ok_c


def sg_spectral_radius(cache, t0) -> float:
    """spr(G0 T0), reusing the A4 gate's dense materialisation."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from gate_a4_dot_product import spectral_radius

    return spectral_radius(cache, t0)


def main() -> int:
    """Three cases, weakest to strongest multiple scattering."""
    print("=" * 84)
    print("GATE A5 -- Taylor test: misfit(m + eps dm) - misfit(m) - eps grad.dm = O(eps^2)")
    print("=" * 84)
    ok = run("[W] whole space, ka_S = 0.26, 8x contrast", whole_space_setup(1.0), 8.0)
    ok &= run("[S] stratified, receiver in the fast slab, 8x contrast", stratified_setup(), 8.0)
    ok &= run("[W-stress] whole space, ka_S = 0.52, 40x contrast", whole_space_setup(2.0), 40.0)
    print("\n" + ("GATE A5 PASS" if ok else "GATE A5 FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
