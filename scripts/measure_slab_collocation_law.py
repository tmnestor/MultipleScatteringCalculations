#!/usr/bin/env python3
"""The collocation error law on a lattice of cubes: every reflection of a slab against the exact layer.

THE QUESTION
------------
The Error paper gives the leading error of the collocation scheme on a layer in closed form: at first
order in the contrast the discrete reflection is the exact one times sinc(kh) / sinc(2kh) (its Eq. 16).
That is for one wave at normal incidence. Does the three-dimensional lattice of cubes obey it, and what
is the law for a converted wave, and at oblique incidence?

THE LAW TESTED
--------------
For an incident plane wave of vertical wavenumber k_j and a reflected one of vertical wavenumber k_i,
both of lateral wavenumber k_x, in cubes of half-side h:

    collocation   R_disc / R = c_i / sinc((k_i + k_j) h),     c_m = sinc(k_m h) sinc(k_x h)
    its adjoint   R_disc / R = c_j / sinc((k_i + k_j) h)
    symmetric     R_disc / R = sqrt(c_i c_j) / sinc((k_i + k_j) h)
    first order   R_disc / R = 1: the point sum times sinc((k_i + k_j) h), the first-order term exact
    cell mean     R_disc / R = c_i c_j / sinc((k_i + k_j) h)      (the mean-only Galerkin voxel)

1 / sinc((k_i + k_j) h) is the midpoint rule on the two-way phase exp(i (k_i + k_j) z) of the Born
integral. c is the average of a plane wave over the cube: collocation takes the incident wave at the
cell centre and radiates through the cell (c of the outgoing wave); the adjoint averages the incident
wave over the cell and radiates from a point. The two are each other's transpose, so neither is
reciprocal; their geometric mean is, and it is what ``slab_weyl_amplitudes`` applies.

THE EXPERIMENT
--------------
  medium     alpha = 5000, beta = 3000 m/s, rho = 2500 kg/m^3; the layer differs by dlambda = 2 GPa,
             dmu = 1 GPa, drho = 100 kg/m^3 (the full contrast), or by a hundredth of that (weak)
  lattice    a slab of N_z planes of cubes, periodic laterally (M x M cells; a uniform slab cannot
             depend on M, which the last block checks), exact lattice sums, the source-cell average of
             the propagator, the exact uniform-strain closure as the single site
  reference  the reflection matrix of the uniform layer of the same thickness, by Kennett's recursion
  error      R_disc / R - 1 for each entry of the P-SV reflection matrix, modified convention

Run:  conda run -n seismic python scripts/measure_slab_collocation_law.py
Writes scripts/measure_slab_collocation_law.json and prints the tables.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.point_cell_self import lattice_single_site  # noqa: E402
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_matrix,
    kennett_reference_rpp,
    slab_reflection_matrix,
    slab_rpp_periodic,
    uniform_slab_material,
)

OUT = Path(__file__).with_suffix(".json")
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
FULL = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
WEAK = MaterialContrast(Dlambda=2.0e7, Dmu=1.0e7, Drho=1.0)
K_HAT = np.array([1.0, 0.0, 0.0])
GMRES_TOL = 1e-10


def sinc(x: np.ndarray | float) -> np.ndarray | float:
    """sin(x) / x."""
    return np.sinc(np.asarray(x) / np.pi)


def reim(z: complex) -> list[float]:
    """A complex number as [re, im]."""
    return [float(np.real(z)), float(np.imag(z))]


def kernels(geom: SlabGeometry, omega: float, p: float = 0.0) -> np.ndarray:
    """The periodic kernels on the exact lattice sum, source-cell averaged, at lateral wavenumber omega p."""
    return build_slab_kernels(
        geom, omega, REF, periodic=True, volume_averaged=True, lattice_ewald=True, bloch_k=(omega * p, 0.0)
    )


def normal_incidence() -> list[dict]:
    """R_PP at normal incidence on three planes of 2 m cubes, four frequencies."""
    a, n_z, m = 1.0, 3, 4
    geom = SlabGeometry(M=m, N_z=n_z, a=a)
    thickness = geom.d * n_z
    rows = []
    print("\nNormal incidence, R_PP, three planes of 2 m cubes: R_disc / R - 1")
    print(
        "  k_S d   law (Eq. 16)   weak contrast            full contrast            package T, full   first-order readout, full"
    )
    for ka in (0.1, 0.2, 0.3, 0.5):
        omega = ka * REF.beta / a
        k_p_h = omega / REF.alpha * a
        law = float(sinc(k_p_h) / sinc(2 * k_p_h) - 1.0)
        kern = kernels(geom, omega)
        row: dict = {"k_S_d": 2 * ka, "omega": omega, "law": law}
        for label, contrast in (("weak", WEAK), ("full", FULL)):
            mat = uniform_slab_material(geom, REF, contrast)
            res = compute_slab_scattering(
                geom,
                mat,
                omega,
                K_HAT,
                gmres_tol=GMRES_TOL,
                volume_averaged=True,
                periodic=True,
                kernel_hat=kern,
            )
            exact = kennett_reference_rpp(REF, contrast, H=thickness, omega=omega)
            t_local = compute_slab_tmatrices(geom, mat, omega)
            row[label] = reim(slab_rpp_periodic(res, t_local, readout="symmetric") / exact - 1.0)
            row[label + "_first_order"] = reim(
                slab_rpp_periodic(res, t_local, readout="first_order") / exact - 1.0
            )
        # the package cube T-matrix as the single site, read without the cell factor (it carries its own)
        mat = uniform_slab_material(geom, REF, FULL)
        t_pkg = _sub_cell_tmatrix_9x9(compute_cube_tmatrix(omega, a, REF, FULL), omega, a)
        t_local = np.broadcast_to(t_pkg, (n_z, m, m, 9, 9)).copy()
        res = compute_slab_scattering(
            geom,
            mat,
            omega,
            K_HAT,
            gmres_tol=GMRES_TOL,
            volume_averaged=True,
            periodic=True,
            kernel_hat=kern,
            T_local=t_local,
        )
        exact = kennett_reference_rpp(REF, FULL, H=thickness, omega=omega)
        row["package_full"] = reim(slab_rpp_periodic(res, t_local, readout="point") / exact - 1.0)
        rows.append(row)
        w, f, pk = (complex(*row[key]) for key in ("weak", "full", "package_full"))
        f1 = complex(*row["full_first_order"])
        print(
            f"  {2 * ka:.1f}     {law:.4e}     {w.real:.4e} ({w.real / law:.4f})      "
            f"{f.real:.4e}{f.imag:+.1e}i ({abs(f) / law:.4f})   {abs(pk):.3e}   {abs(f1):.3e}",
            flush=True,
        )
    return rows


def oblique(m: int, n_z: int, a: float, omega: float, p: float, contrasts: tuple) -> dict:
    """The P-SV reflection matrix at slowness p, with each choice of cell factor, against the law."""
    geom = SlabGeometry(M=m, N_z=n_z, a=a)
    thickness = geom.d * n_z
    t0 = time.time()
    kern = kernels(geom, omega, p)
    k_z = omega * np.sqrt(np.array([1.0 / REF.alpha**2, 1.0 / REF.beta**2]) - p**2)
    lateral = float(sinc(omega * p * a))
    c = sinc(k_z * a) * lateral
    midpoint = 1.0 / sinc(np.add.outer(k_z, k_z) * a)
    factors = {
        # first_order: the first-order term integrated exactly over each cube; its law is zero
        "first_order": sinc(np.add.outer(k_z, k_z) * a) * np.ones((2, 2)),
        "collocation": np.outer(c, np.ones(2)),
        "adjoint": np.outer(np.ones(2), c),
        "symmetric": np.sqrt(np.outer(c, c)),
        "cell_mean": np.outer(c, c),
    }
    laws = {name: f * midpoint - 1.0 for name, f in factors.items()}
    out: dict = {
        "M": m,
        "N_z": n_z,
        "d": geom.d,
        "omega": omega,
        "p": p,
        "angle_P_deg": float(np.degrees(np.arcsin(p * REF.alpha))),
        "angle_S_deg": float(np.degrees(np.arcsin(p * REF.beta))),
        "laws": {name: law.tolist() for name, law in laws.items()},
        "measured": {},
    }
    print(
        f"\nOblique, M = {m}, {n_z} planes of {geom.d:g} m cubes, omega = {omega:g}, p = {p:.4e} "
        f"(P at {out['angle_P_deg']:.1f} deg, S at {out['angle_S_deg']:.1f} deg); kernels {time.time() - t0:.0f} s"
    )
    names = (("R_PP", 0, 0), ("R_SS", 1, 1), ("P from S", 0, 1), ("S from P", 1, 0))
    print("  first-order law      " + "   ".join(f"{nm:>9s}" for nm, _, _ in names))
    for name, law in laws.items():
        print(f"    {name:12s}     " + "   ".join(f"{law[i, j]:+.3e}" for _, i, j in names))
    for label, contrast in contrasts:
        mat = uniform_slab_material(geom, REF, contrast)
        sm = slab_reflection_matrix(
            geom,
            mat,
            omega,
            p=p,
            volume_averaged=True,
            kernel_hat=kern,
            gmres_tol=GMRES_TOL,
            readout="symmetric",
        )
        r_sym = sm.to_modified()
        k = kennett_reference_matrix(REF, contrast, H=thickness, omega=omega, p=p)
        exact = np.array([[k.R_PP, k.R_SP], [k.R_PS, k.R_SS]])
        raw = r_sym / factors["symmetric"]
        asym = abs(r_sym[0, 1] - r_sym[1, 0]) / abs(r_sym[0, 1])
        out["measured"][label] = {"asymmetry": float(asym)}
        print(f"  measured, {label} contrast (asymmetry of the symmetric reading {asym:.1e})")
        for name, f in factors.items():
            err = raw * f / exact - 1.0
            out["measured"][label][name] = [[reim(err[i, j]) for j in range(2)] for i in range(2)]
            print(
                f"    {name:12s}     "
                + "   ".join(f"{err[i, j].real:+.3e}" for _, i, j in names)
                + f"     largest |imaginary part| {np.max(np.abs(err.imag)):.1e}",
                flush=True,
            )
    return out


def main() -> int:
    """Run the three blocks and write the JSON.

    Returns:
        0.
    """
    if lattice_single_site() != "exact":
        print("single_site.lattice in cubic_scattering/numerics.yml must be exact for this measurement.")
        return 1
    both = (("weak", WEAK), ("full", FULL))
    full = (("full", FULL),)
    out = {
        "medium": {"alpha": REF.alpha, "beta": REF.beta, "rho": REF.rho},
        "contrast_full": {"dlambda": FULL.Dlambda, "dmu": FULL.Dmu, "drho": FULL.Drho},
        "weak_scale": 1e-2,
        "normal": normal_incidence(),
        "oblique": [
            oblique(4, 3, 1.0, 900.0, 1.0e-4, both),
            oblique(4, 6, 0.5, 900.0, 1.0e-4, both),
            oblique(4, 3, 1.0, 900.0, 1.6e-4, both),
        ],
        # one plane, a wave of which the lattice holds an eighth and a quarter of a lateral wavelength
        "lattice_independence": [
            oblique(m, 1, 1.0, 1500.0, 2 * np.pi / (1500.0 * 16 * 2.0), full) for m in (2, 4)
        ],
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
