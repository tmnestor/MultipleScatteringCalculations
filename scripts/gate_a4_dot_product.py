#!/usr/bin/env python3
"""GATE A4: the dot-product test -- the adjoint is the transpose of the tangent-linear model.

WHY THIS GATE EXISTS
--------------------
The unit tests compare the gradient with finite differences of the misfit, which
are limited by the step. The dot-product identity has no step: for ANY real
contrast perturbation dm and complex data-space vector q,

    Re(q^H F dm)  =  sum(dm * Re(F^H q)),

where F dm = R (I - T0 G0)^-1 dT0 psi is computed FORWARD (one extra Foldy-Lax
solve, sweep_gradient.frechet_vector_product) and Re(F^H q) is computed by the
ADJOINT route the gradient uses (the solve at -k_y conjugated by W_pm = I_pm W,
sweep_gradient.frechet_adjoint_product). The two share only the forward field
psi and dT0/dm, so the identity holds to solver tolerance only if the adjoint
really is the transpose -- the A1/A1b laws, the receiver transpose, the
conjugate on q and the Re, all at once.

The spectral radius spr(G0 T0) of the assembled operator is measured and
reported, because a dot product at weak contrast is dominated by the Born term
and says little about the (I - T0 G0)^-1 resolvent. MEASURED: with PHYSICAL
Rayleigh T0 inside its validity (ka_S <= 0.3) and the project's contrasts, rho
stays at or below ~0.3 (0.075 whole space at 8x the test contrast, 0.33
stratified). Reaching spr ~ 0.8 needs ka_S ~ 0.5 and Dmu ~ 1.8 mu0, outside the
Rayleigh cube's validity. The identity is algebraic and holds regardless, so a
[W-stress] case is run there, labelled as such, to exercise the resolvent hard.
The rung-5 solver gate's 0.92 used random T0 blocks, not physical ones.

  [A4]  max relative mismatch over five random (dm, q) pairs, asserted < 1e-9
        (GMRES at rtol 1e-12), whole space and stratified, weak to strong contrast.
  [c1]  CALIBRATION: Born-only F dm (multiple scattering dropped) must FAIL, by
        more as rho grows.
  [c2]  CALIBRATION: the adjoint with W in place of W_pm must FAIL.

Run:  conda run -n seismic python scripts/gate_a4_dot_product.py
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
    apply_g0,
    build_g0_cache,
    make_sweep_grid,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402

SOLVE = {"tol": 1e-12, "max_iter": 2000}
STEP = np.array([1e-2, 1e-2, 1e-3])
BASE = np.array([2.0, 1.0, 0.1])  # (Dlambda, Dmu, Drho), the project's test contrast
TOL = 1e-9


def spectral_radius(cache, t0) -> float:
    """spr(G0 T0) of the assembled operator, by dense materialisation."""
    n_z, n_x = cache.grid.n_z, cache.grid.n_x
    size = n_z * n_x * 9
    m = np.zeros((size, size), dtype=complex)
    for c in range(size):
        e = np.zeros(size, dtype=complex)
        e[c] = 1.0
        m[:, c] = apply_g0(np.einsum("zxab,zxb->zxa", t0, e.reshape(n_z, n_x, 9)), cache).ravel()
    return float(np.abs(np.linalg.eigvals(m)).max())


def born_only_jvp(cache_plus, t0, dt0, psi, receiver, dm, **_) -> np.ndarray:
    """[c1] F dm with the multiple-scattering term T0 dpsi dropped."""
    return receiver.apply(np.einsum("zxaij,zxa,zxj->zxi", dt0, dm, psi))


def mismatch(cache_p, cache_m, t0, dt0, psi, receiver, rng, jvp) -> float:
    """Max relative dot-product mismatch over five random (dm, q) pairs."""
    worst = 0.0
    for _ in range(5):
        dm = rng.standard_normal(t0.shape[:2] + (3,)) * np.array([1.0, 1.0, 0.1])
        q = rng.standard_normal(receiver.n_data) + 1j * rng.standard_normal(receiver.n_data)
        lhs = float(np.vdot(q, jvp(cache_p, t0, dt0, psi, receiver, dm, **SOLVE)).real)
        rhs = float(np.sum(dm * sg.frechet_adjoint_product(cache_m, t0, dt0, psi, receiver, q, **SOLVE)))
        worst = max(worst, abs(lhs - rhs) / abs(lhs))
    return worst


def run(name, caches, receiver, refs, omega, half_width, scales) -> bool:
    """Measure the identity and both calibration legs at each contrast scale."""
    cache_p, cache_m = caches
    n_z, n_x = cache_p.grid.n_z, cache_p.grid.n_x
    print(f"\n{name}: {n_z}x{n_x} voxels, {receiver.n_data} data, omega/2pi = {omega / (2 * np.pi):.3f}")
    print(
        f"    {'contrast':>9} {'spr(G0 T0)':>10} {'[A4]':>10} {'[c1] Born F':>12} {'[c2] W for W_pm':>13}"
    )
    ok = True
    for scale in scales:
        rng = np.random.default_rng(20260926)
        m = scale * BASE * (1.0 + 0.3 * rng.standard_normal((n_z, n_x, 3)))
        t0, dt0 = sg.rayleigh_t0_and_derivative(m, refs, omega, half_width, step=STEP)
        psi_inc = rng.standard_normal((n_z, n_x, 9)) + 1j * rng.standard_normal((n_z, n_x, 9))
        psi = sg.solve_sweep_foldy_lax(cache_p, t0, psi_inc, **SOLVE).psi
        spr = spectral_radius(cache_p, t0)

        a4 = mismatch(
            cache_p, cache_m, t0, dt0, psi, receiver, np.random.default_rng(1), sg.frechet_vector_product
        )
        c1 = mismatch(cache_p, cache_m, t0, dt0, psi, receiver, np.random.default_rng(1), born_only_jvp)
        d_saved = sg.W_SIGNED.copy()
        sg.W_SIGNED[:] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        try:
            c2 = mismatch(
                cache_p,
                cache_m,
                t0,
                dt0,
                psi,
                receiver,
                np.random.default_rng(1),
                sg.frechet_vector_product,
            )
        finally:
            sg.W_SIGNED[:] = d_saved
        print(f"    {scale:9.1f} {spr:10.3f} {a4:10.2e} {c1:12.2e} {c2:13.2e}")
        ok &= a4 < TOL and c1 > 1e3 * TOL and c2 > 1e3 * TOL
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def whole_space(freq: float, scales: tuple[float, ...], label: str) -> bool:
    """Whole space, complex frequency, as the rung-5 solver gate."""
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
    return run(label, caches, receiver, [ref] * 3, omega, 0.5 * pitch, scales)


def stratified() -> bool:
    """The fast-slab crust, receiver inside the slab (cross-material)."""
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
    return run(
        "[S] stratified, receiver in the fast slab", caches, receiver, refs, omega, 0.5, (1.0, 4.0, 8.0)
    )


def main() -> int:
    """Whole space, then stratified."""
    print("=" * 84)
    print("GATE A4 -- dot product:  Re(q^H F dm) = dm . Re(F^H q),  F forward vs F^H adjoint")
    print("=" * 84)
    ok = whole_space(1.0, (1.0, 4.0, 8.0), "[W] whole space, ka_S = 0.26 (Rayleigh-valid)")
    ok &= whole_space(2.0, (25.0, 40.0), "[W-stress] whole space, ka_S = 0.52, contrast to Dmu ~ 1.8 mu0")
    ok &= stratified()
    print("\n" + ("GATE A4 PASS" if ok else "GATE A4 FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
