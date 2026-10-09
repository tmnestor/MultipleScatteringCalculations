#!/usr/bin/env python3
"""tau-p seismograms of plane-wave scattering from the graded sphere: Legendre cells against the exact body.

THE PLANE-WAVE SPECTRUM. A plane P wave travels down (+z, axis 0) onto the graded sphere of Paper 2 (radius
a, core a/10, smoothstep shell). Above the body the scattered field is a superposition of upgoing plane
waves (the Weyl representation),

    u(x) = Int d^2k [ c_P e_P exp(i(k.rho + k_z^P |z|)) + c_SV e_SV exp(i(k.rho + k_z^S |z|)) ],

with k_z on the branch Im k_z >= 0, so that |k| > K are evanescent waves that decay away from the body.
The far field holds only the propagating part. On a receiver plane at a finite distance both parts are
present, and the tau-p section is the spectrum itself.

    exact:   c_P, c_SV from the graded sphere's coefficients (``crosscheck_graded_sphere.graded_mie_result``)
             by ``gate_sphere_plane_wave_spectrum.mode_amplitudes``;
    cells:   e^{ikr}/r = (i/2pi) Int d^2k e^{i(k.rho + k_z|z|)}/k_z expands the Green's tensor exactly,
             near field included, so c = (i/(2 pi k_z)) times the cells' radiation pattern at the COMPLEX
             unit vector khat = (k_z dir, k)/K. That is the far-field formula of ``farfield.radiate``,
             continued to complex directions: the cells' polynomial sources take it exactly.

THE SEISMOGRAM, in the seismic convention: z positive down, the receivers on the plane z = 0 and the
sphere's centre at depth z_c (--zr, in radii; the solver's origin is the centre, so the receivers sit at
-z_c there). At horizontal slowness p (in the plane of incidence) the slant stack of the vertical and
horizontal displacement is (2 pi)^2 [c_P e_P e^{i k_z^P z_c} + c_SV e_SV e^{i k_z^S z_c}] at k = omega p.
Time is counted from the incident wave's crossing of the receiver plane, the factor e^{i omega z_c/alpha},
so that tau is a two-way time and reads as depth: the P diffraction from the centre arrives at
tau(p) = z_c (1/alpha + eta_P(p)), eta_P = sqrt(1/alpha^2 - p^2). Times a Ricker wavelet and transformed to
time it is u(tau, p). The wavelet is delayed by --delay/omega_0 (21 ms by default) so that it is causal.

Run:  python -u scripts/taup_graded_sphere.py [--p=1] [--n=8] [--nw=128] [--kmax=3] [--k0=1] [--zr=1.5] [--np=60] [--near=series|quad] [--delay=6.3]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import measure_ball_gradient_hierarchy as hier
import measure_graded_sphere_gradient_hierarchy as gs
from crosscheck_graded_sphere import graded_mie_result
from gate_sphere_plane_wave_spectrum import kz_of, mode_amplitudes

from cubic_scattering.graded_voxel import blocks as gb
from cubic_scattering.graded_voxel.farfield import (
    cell_source_coefficients,
    source_moments,
)
from cubic_scattering.graded_voxel.fft import (
    _transform,
    signed_permutations,
    solve_graded_sphere_fft,
)
from cubic_scattering.graded_voxel.solver import field_sizes
from cubic_scattering.sphere_scattering import _voigt_to_tensor

REF, CONTRAST = hier.REF, hier.CONTRAST
OUT = ROOT / "scratch" / "taup"


def series_blocks(n_sub, h, omega, n_source, n_field, near="series"):
    """K(o) for every offset of the grid from the package's series in k (touching: universal moments; others:
    far coefficients), with the Gauss block where the far series would need more terms than it holds.

    near="quad" takes the touching blocks from the package's quadrature instead (``coupling_block``): for
    cells of degree two the universal moments to the order the top frequency needs cost hours once, the
    quadrature 13 s per frequency for the four touching orbits."""
    canon, out = {}, {}
    for off in __import__("itertools").product(range(-(n_sub - 1), n_sub), repeat=3):
        c = tuple(sorted((abs(o) for o in off), reverse=True))
        if c not in canon:
            if max(c) <= 1:
                if near == "quad":
                    canon[c] = gb.coupling_block(c, h, omega, REF, n_source, n_field)
                else:
                    canon[c] = gb.near_block_series(c, h, omega, REF, n_source, n_field)
            else:
                try:
                    canon[c] = gb.far_block_series(c, h, omega, REF, n_source, n_field)
                except ValueError:
                    canon[c] = gb.coupling_block(c, h, omega, REF, n_source, n_field)
        q = next(
            q
            for q in signed_permutations()
            if np.array_equal(q @ np.array(c, float), np.array(off, float))
        )
        out[off] = _transform(q, canon[c])
    return out


def cell_spectrum(res, omega, q):
    """(c_P, c_SV) of the cells' scattered field at lateral wavenumbers q (in the plane of incidence), upward.

    The radiation pattern at the (complex, when evanescent) unit vector khat needs the sources' moments
    against exp(-i k khat.x): ``farfield.source_moments``, in closed form for every cell (products of 2 i^l
    j_l, the sinc and its derivatives), with no quadrature.
    """
    kp, ks = omega / REF.alpha, omega / REF.beta
    cp, cs = np.zeros(len(q), dtype=complex), np.zeros(len(q), dtype=complex)
    coef = cell_source_coefficients(res)
    for i, qq in enumerate(q):
        for mode, k, speed in (("P", kp, REF.alpha), ("S", ks, REF.beta)):
            kz = kz_of(np.array([qq]), k)[0]
            khat = (
                np.array([-kz, qq, 0.0]) / k
            )  # upward: the z component reversed; complex when evanescent
            mom = source_moments(res, k * khat, coef)
            force, sig = mom[:3], _voigt_to_tensor(mom[3:])
            pref = 1.0 / (4.0 * math.pi * REF.rho * speed**2)
            if mode == "P":
                amp = pref * (force @ khat + 1j * k * (khat @ sig @ khat))
                cp[i] = 1j / (2.0 * math.pi * kz) * amp
            else:
                e_sv = np.array(
                    [-qq / k, -kz / k, 0.0]
                )  # e_theta at the upward direction, as the exact formula
                amp = pref * ((force + 1j * k * (sig @ khat)) @ e_sv)
                cs[i] = 1j / (2.0 * math.pi * kz) * amp
    return cp, cs


def main() -> int:
    opts = dict(
        a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a
    )
    p, n = int(opts.get("p", 1)), int(opts.get("n", 8))
    nw, kmax, k0 = (
        int(opts.get("nw", 128)),
        float(opts.get("kmax", 3.0)),
        float(opts.get("k0", 1.0)),
    )
    zr, n_p = float(opts.get("zr", 1.5)), int(opts.get("np", 60))
    near = opts.get("near", "series" if p == 1 else "quad")
    delay = float(opts.get("delay", 6.3))
    gs.CORE = 0.1 * gs.RADIUS
    gs.set_profile("smoothstep")
    a = gs.RADIUS
    _, n_field, _, n_source = field_sizes(p, p)
    omegas = (np.arange(nw) + 1) * kmax * REF.beta / a / nw
    # cell-centred samples, so that no slowness falls on a branch point (p = 1/alpha or 1/beta, where k_z = 0
    # and the spectrum's 1/k_z is singular; the singularity is integrable, the sample is not)
    slow = (np.arange(n_p) + 0.5) * 1.5 / REF.beta / n_p
    for pb in (1 / REF.alpha, 1 / REF.beta):
        assert np.abs(slow - pb).min() > 0.1 * 1.5 / REF.beta / n_p, (
            "a slowness sample falls on a branch point"
        )

    def prof(pos) -> float:
        return float(gs.smoothstep(np.array([np.linalg.norm(pos)]))[0])

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"p{p}_n{n}"
    store = OUT / f"spectra_{tag}.npz"
    done = dict(np.load(store)) if store.exists() else {}
    ex_p = done.get("ex_p", np.full((nw, n_p), np.nan, dtype=complex))
    ex_s = done.get("ex_s", np.full((nw, n_p), np.nan, dtype=complex))
    vx_p = done.get("vx_p", np.full((nw, n_p), np.nan, dtype=complex))
    vx_s = done.get("vx_s", np.full((nw, n_p), np.nan, dtype=complex))
    t0 = time.perf_counter()
    print(f"tau-p, graded sphere: cells of degree {p}, {n} across; {nw} frequencies to k_S a = {kmax}; "
          f"{n_p} slownesses to 1.5/beta; receivers {zr} a above the centre", flush=True)  # fmt: skip
    for iw, omega in enumerate(omegas):
        if not np.isnan(vx_p[iw]).any():
            continue
        q = omega * slow
        n_max = max(
            8,
            int(
                np.ceil(
                    omega * a / REF.beta + 4 * (omega * a / REF.beta) ** (1 / 3) + 8
                )
            ),
        )
        mie = graded_mie_result(omega, a, gs.CORE, REF, CONTRAST, n_max)
        ex_p[iw], ex_s[iw] = mode_amplitudes(mie, q, upward=True)
        blocks = series_blocks(n, a / n, omega, n_source, n_field, near)
        res = solve_graded_sphere_fft(omega, a, REF, CONTRAST, n, prof, gs.K_HAT, gs.POL, "P", p=p, r=p,
                                      gmres_tol=1e-11, blocks=blocks)  # fmt: skip
        vx_p[iw], vx_s[iw] = cell_spectrum(res, omega, q)
        np.savez(
            store, ex_p=ex_p, ex_s=ex_s, vx_p=vx_p, vx_s=vx_s, omegas=omegas, slow=slow
        )
        if iw % 8 == 0 or iw == nw - 1:
            prop = slow < 1.0 / REF.alpha
            e = (
                np.abs(vx_p[iw][prop] - ex_p[iw][prop]).max()
                / np.abs(ex_p[iw][prop]).max()
            )
            print(f"   k_S a = {omega * a / REF.beta:5.3f}: c_P (propagating) off by {e:.2e}   "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)  # fmt: skip
    # the seismograms
    w0 = k0 * REF.beta / a
    # the Ricker wavelet, delayed by t_d = delay/omega_0 so that it is causal: below 1e-3 of its peak before
    # tau = 0 for delay = 6.3. A diffraction of two-way time T then peaks at T + t_d.
    t_d = delay / w0
    ricker = (omegas / w0) ** 2 * np.exp(-((omegas / w0) ** 2)) * np.exp(1j * omegas * t_d)
    zr_m = zr * a
    sections = {}
    for name, (cp, cs) in (("exact", (ex_p, ex_s)), ("cells", (vx_p, vx_s))):
        uz = np.zeros((nw, n_p), dtype=complex)
        ux = np.zeros((nw, n_p), dtype=complex)
        for iw, omega in enumerate(omegas):
            q = omega * slow
            kp, ks = omega / REF.alpha, omega / REF.beta
            kzp, kzs = kz_of(q, kp), kz_of(q, ks)
            # e_P = (-k_z, q)/k_P and e_SV = (-q, -k_z)/k_S upward, in (z, x)
            uz[iw] = cp[iw] * (-kzp / kp) * np.exp(1j * kzp * zr_m) + cs[iw] * (
                -q / ks
            ) * np.exp(1j * kzs * zr_m)
            ux[iw] = cp[iw] * (q / kp) * np.exp(1j * kzp * zr_m) + cs[iw] * (
                -kzs / ks
            ) * np.exp(1j * kzs * zr_m)
        # time from the incident wave's crossing of the receiver plane, z_c/alpha before it reaches the centre
        shift = np.exp(1j * omegas * zr_m / REF.alpha)[:, None]
        sections[name] = (uz * shift, ux * shift)
    nt = 4096
    dw = omegas[1] - omegas[0]
    tmax = 2 * math.pi / dw
    tau = np.arange(nt) * tmax / nt
    kern = np.exp(-1j * np.outer(tau, omegas))  # e^{-i omega tau}

    def to_time(spec):
        return (
            (2 * math.pi) ** 2 / math.pi * (kern @ (ricker[:, None] * spec)) * dw
        ).real

    out = {"tau": tau, "slow": slow, "omegas": omegas, "t_d": t_d}
    for name, (uz, ux) in sections.items():
        out[f"{name}_uz"], out[f"{name}_ux"] = to_time(uz), to_time(ux)
    np.savez(OUT / f"taup_{tag}.npz", **out)
    for comp in ("uz", "ux"):
        ex, vx = out[f"exact_{comp}"], out[f"cells_{comp}"]
        peak = np.abs(ex).max()
        bands = {"P propagating (p < 1/alpha)": slow < 1 / REF.alpha,
                 "S only (1/alpha < p < 1/beta)": (slow > 1 / REF.alpha) & (slow < 1 / REF.beta),
                 "evanescent (p > 1/beta)": slow > 1 / REF.beta}  # fmt: skip
        parts = [
            f"{k}: {np.abs(vx[:, m] - ex[:, m]).max() / peak:.2e}"
            for k, m in bands.items()
        ]
        print(
            f"   u_{comp[1]}(tau, p): cells against exact, max over tau, relative to the section's peak: "
            + "; ".join(parts),
            flush=True,
        )
    print(f"   wrote {OUT / f'taup_{tag}.npz'}   [{time.perf_counter() - t0:.0f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
