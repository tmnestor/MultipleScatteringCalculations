#!/usr/bin/env python3
"""Reference numbers for ``Mathematica/ContinuumLimit_SpecularSums.wl``, notebook 2 of the continuum limit.

Two things the notebook derives independently:

  * the spectral (2-D Fourier) kernel of the point Green's tensor between planes, G^(k_x, k_y; dz),
    from ``sweep_kernels.vertical_kernel_9x9`` at propagating and evanescent lateral wavenumbers;
  * the SPECULAR lattice sums S(m) = sum_R G(R + m d z^) of the POINT propagator over a square lattice
    of pitch d, from the package's exact Ewald kernel (``build_slab_kernels(..., lattice_ewald=True,
    volume_averaged=False)``, whose k = 0 block is the full lattice sum), for m = -4..4 at three pitches.

Every 9x9 in the package's order (u_z, u_x, u_y, e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx), [re, im].

Run:  conda run -n seismic python scripts/continuum_limit_specular.py
Writes Mathematica/ContinuumLimit_specular.json.  SI units, real (lossless) whole-space background.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import SlabGeometry, build_slab_kernels  # noqa: E402
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402

OUT = ROOT / "Mathematica" / "ContinuumLimit_specular.json"
REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OM = 60.0
K_S = OM / REF.beta
PITCHES = (0.5, 1.0, 2.0)
M_MAX = 4


def reim(a: np.ndarray) -> list:
    """Complex array -> nested [re, im]."""
    return np.stack([np.real(a), np.imag(a)], axis=-1).tolist()


def main() -> int:
    """Dump the reference.

    Returns:
        0.
    """
    spectral: list = []
    kx = np.array([0.5 * K_S, 2.0 * K_S, 6.0])  # propagating P and S; evanescent S only; deeply evanescent
    for ky in (0.0, 0.7 * K_S):
        for dz in (0.5, -1.0):
            g = vertical_kernel_9x9(kx, ky, dz, OM, REF)
            for i, kxv in enumerate(kx):
                spectral.append({"kx": float(kxv), "ky": float(ky), "dz": dz, "G": reim(g[:, :, i])})
    lattice: list = []
    for d in PITCHES:
        geom = SlabGeometry(M=1, N_z=M_MAX + 1, a=d / 2)
        kh = build_slab_kernels(geom, OM, REF, periodic=True, lattice_ewald=True, volume_averaged=False)
        for k in range(2 * M_MAX + 1):
            m = k - M_MAX
            lattice.append({"d": d, "m": m, "S": reim(np.asarray(kh[k, 0, 0]))})
    out = {
        "omega": OM,
        "alpha": REF.alpha,
        "beta": REF.beta,
        "rho": REF.rho,
        "spectral": spectral,
        "lattice": lattice,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(spectral)} spectral kernels, {len(lattice)} specular lattice sums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
