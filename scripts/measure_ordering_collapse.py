"""MEASUREMENT: is the dress-after error a function of rho alone?

`measure_ordering_strong_scattering` established the robust part -- the error
grows from ~10% of the field at rho < 1 to 70-90% at rho > 2.5 -- but over only
six points, and reported it as a RANGE. A range is what you write when you do
not know whether the spread is the axis or the geometry.

THE QUESTION. Plot every configuration against rho(G0 T). If the points collapse
onto one curve, the dress-after error is governed by the scattering strength
alone and "70-90%" becomes a trend with a shape. If they do not, geometry enters
independently and no single number in rho can summarise the error -- which is
itself worth knowing, and would mean the earlier range was hiding a second
variable rather than measurement noise.

TWO REFLECTOR DEPTHS ON PURPOSE. The previous script's ratio result moved from
1.99 to 0.63 when the reflector shifted by one layer, so reflector depth is a
demonstrated second variable. Including two of them is what makes the collapse
test meaningful rather than a restatement of one geometry.

rho VIA A KRYLOV ESTIMATE, not a dense assembly. The dense route needs one
apply_g0_3d per column -- 1728 of them at 3x8x8 -- where ARPACK needs tens of
matvecs for the dominant eigenvalue. That is the only reason the larger lattices
are reachable here at all.

CONTRASTS STAY INSIDE the 52% validity floor of the single-site
renormalisation; the Dlam/lam column is printed so any excursion is visible.

REPORTED, not gated.

Run:  conda run -n seismic python scripts/measure_ordering_collapse.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.pair_propagators import TransverseRule, layered_incident_field  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_tmatrices,
)
from cubic_scattering.sweep_solver import solve_foldy_lax_3d  # noqa: E402

PITCH = 0.25
OM = 2 * np.pi * 6.0
SOURCE_IFACE = 14
RULE = TransverseRule(kr_max=10.0 / PITCH, n_axis=48)
SRC_VEC = np.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=complex)
BG_LAMBDA = 17.5  # rho(alpha^2 - 2 beta^2) = 2.5(25 - 18) GPa


def model(refl_layer: int, n_lay: int = 28, q: float = 1e4):
    """Ocean over elastic layers; the deep contrast starts at `refl_layer`.

    Must stay clear of every scattering plane: a plane ON a discontinuity is
    rejected by assert_interface_continuous, since K is built from the local
    eta_S which is two-valued there.
    """
    import Kennett_Reflectivity.layer_model as lm

    a, b, r = 5.0, 3.0, 2.5
    al, be, rh = [1.5, *([a] * n_lay), a], [0.0, *([b] * n_lay), b], [1.03, *([r] * n_lay), r]
    for j in range(refl_layer, n_lay + 2):
        al[j], be[j], rh[j] = a + 2.0, b + 1.2, r + 0.6
    return lm.LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * n_lay), np.inf],
        Q_alpha=[q] * (n_lay + 2),
        Q_beta=[1e12, *([q] * n_lay), q],
    )


def rho_krylov(cache, t0, shape) -> float:
    """Dominant |eigenvalue| of G0 T, by ARPACK on the matrix-free operator."""
    size = int(np.prod(shape))

    def mv(v):
        psi = v.reshape(shape)
        return apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, psi), cache).ravel()

    op = LinearOperator((size, size), matvec=mv, dtype=complex)
    vals = eigs(op, k=1, which="LM", return_eigenvectors=False, tol=1e-6, maxiter=5000)
    return float(np.abs(vals[0]))


def main() -> int:
    print("=" * 78)
    print("MEASUREMENT -- does the dress-after error collapse onto rho(G0 T)?")
    print("  two reflector depths, four lattices, three contrasts")
    print("=" * 78)

    rows = []
    for refl in (22, 26):
        mod = model(refl)
        s_p, s_s = mod.complex_slowness_p(), mod.complex_slowness_s()
        for n_z, n_x in ((2, 4), (2, 6), (3, 6), (2, 8)):
            planes = tuple(18 + i for i in range(n_z))
            j = planes[0]
            ref = ReferenceMedium(1.0 / s_p[j], 1.0 / s_s[j], mod.rho[j])
            dz_planes = tuple(float(planes[i] - SOURCE_IFACE) * PITCH for i in range(n_z))
            shape = (n_z, n_x, n_x, 9)

            psi_inc = layered_incident_field(
                n_z,
                n_x,
                n_x,
                PITCH,
                OM,
                ref,
                source_xy=(0, 0),
                source_vec=SRC_VEC,
                dz_planes=dz_planes,
                model=mod,
                plane_ifaces=planes,
                source_iface=SOURCE_IFACE,
                transverse=RULE,
            )
            grid = SweepGrid3D(n_z=n_z, n_x=n_x, n_y=n_x, pitch=PITCH)
            c_thesis = build_g0_cache_3d(grid, ref, OM, model=mod, plane_ifaces=planes, transverse=RULE)
            c_dress = build_g0_cache_3d(grid, ref, OM)
            geom = SlabGeometry(M=n_x, N_z=n_z, a=0.5 * PITCH)

            for t_scale in (1.0, 2.0, 3.0):
                rng = np.random.default_rng(20260914)
                s = t_scale * (0.5 + rng.random((n_z, n_x, n_x)))
                material = SlabMaterial(Dlambda=2.0 * s, Dmu=1.0 * s, Drho=0.1 * s, ref=ref)
                t0 = compute_slab_tmatrices(geom, material, OM)
                try:
                    rho = rho_krylov(c_thesis, t0, shape)
                    a = solve_foldy_lax_3d(c_thesis, t0, psi_inc, tol=1e-11).psi
                    b = solve_foldy_lax_3d(c_dress, t0, psi_inc, tol=1e-11).psi
                except (RuntimeError, ArithmeticError) as exc:
                    rows.append((float("nan"), refl, n_z, n_x, t_scale, float("nan"), str(exc)[:28]))
                    continue
                diff = float(np.abs(a - b).max() / np.abs(a).max())
                rows.append((rho, refl, n_z, n_x, t_scale, diff, ""))

    rows.sort(key=lambda r: np.inf if np.isnan(r[0]) else r[0])
    print(f"\n  {'rho(G0T)':>10} {'refl':>5} {'lattice':>9} {'T':>5} {'Dlam/lam':>9} {'diff':>11}")
    for rho, refl, n_z, n_x, t_scale, diff, note in rows:
        frac = 2.0 * 1.5 * t_scale / BG_LAMBDA  # max Dlambda over background
        lat = f"{n_z}x{n_x}x{n_x}"
        if note:
            print(f"  {rho:10.4f} {refl:5d} {lat:>9} {t_scale:5.1f} {frac:9.2f}   {note}")
        else:
            print(f"  {rho:10.4f} {refl:5d} {lat:>9} {t_scale:5.1f} {frac:9.2f} {diff:11.3e}")

    good = [(r[0], r[5]) for r in rows if not np.isnan(r[0]) and not np.isnan(r[5])]
    if len(good) > 3:
        lr = np.array([g[0] for g in good])
        ld = np.array([g[1] for g in good])
        # Spread of diff among points at SIMILAR rho is what refutes a collapse.
        print("\n  collapse check -- spread of diff within rho bands")
        print(f"    {'rho band':>14} {'n':>3} {'min diff':>11} {'max diff':>11} {'ratio':>8}")
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)):
            sel = ld[(lr >= lo) & (lr < hi)]
            if sel.size >= 2:
                print(
                    f"    {f'{lo:.1f}-{hi:.1f}':>14} {sel.size:3d} {sel.min():11.3e} "
                    f"{sel.max():11.3e} {sel.max() / max(sel.min(), 1e-300):8.2f}"
                )

    print("\n  Reading it. If diff is governed by rho alone, points within a band")
    print("  agree and the ratio column sits near 1. A ratio well above 1 means")
    print("  geometry enters independently of rho, and no single curve in rho")
    print("  summarises the dress-after error -- in which case the earlier")
    print("  '70-90%' was hiding a second variable, not measurement spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
