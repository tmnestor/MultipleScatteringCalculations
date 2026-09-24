#!/usr/bin/env python3
"""Error against scattering angle, for reflection and transmission together.

The backscatter comparison of ``plot_imbedding_vs_voxel`` is one direction out
of many.  This figure shows the rest.  At k_S a = 2.4 the march is the more
accurate route at every angle, by 7x at the wide angles; at k_S a = 1 on this
900 m period only the specular orders propagate, and there the voxel route is
the better, because the period is short against the wavelength and the array
coupling it costs is largest.

⛔ AN EARLIER VERSION OF THIS FIGURE SHOWED THE MARCH ORDER-OF-UNITY WRONG AT
EVERY WIDE ANGLE, and explained the spread of its points at one theta as the
square array's azimuthal structure.  Both were a scoring defect: the Mie
prediction omitted the phase that moves the sphere from the lateral origin to
the grid centre, where the march puts it.  That phase is 1 on the specular
orders only.  See ``centre_phase`` in ``gate_sphere_vs_impedance_march.py``.

Run:
    conda run -n seismic python scripts/gate_imbedding_vs_foldy_lax.py --dump-angles angles.json
    conda run -n seismic python scripts/plot_angle_resolved.py angles.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "LatexPDFs" / "FirstOrderContrastOperator" / "angle_resolved.pdf"


def main(path: str) -> int:
    """Draw the figure.

    Args:
        path: JSON written by the angle sweep.

    Returns:
        0 on success.
    """
    data = json.loads(Path(path).read_text())
    keys = sorted(data, key=float)

    fig, axes = plt.subplots(1, len(keys), figsize=(5.1 * len(keys), 4.3), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        rows = data[key]
        for kind, marker, label in (("T", "v", "transmitted"), ("R", "^", "reflected")):
            sel = [r for r in rows if r["kind"] == kind]
            if not sel:
                continue
            th = np.array([r["theta"] for r in sel])
            ax.semilogy(
                th,
                [r["march"] for r in sel],
                marker,
                color="#21618c",
                ms=6,
                mfc="none",
                label=f"march, {label}",
            )
            ax.semilogy(
                th,
                [r["voxel"] for r in sel],
                marker,
                color="#c0392b",
                ms=6,
                mfc="none",
                label=f"voxel, {label}",
            )
        ax.axvline(0.0, color="0.8", lw=1.0)
        ax.axvline(180.0, color="0.8", lw=1.0)
        ax.set_title(rf"$k_S a = {float(key):g}$", fontsize=11)
        ax.set_xlabel(r"scattering angle $\theta$ from the incident direction (deg)")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.set_xlim(-12, 192)
    axes[0][0].set_ylabel("relative error against the exact sphere")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.suptitle(
        "Error against the exact sphere, by scattering angle",
        fontsize=11,
        y=0.99,
    )

    fig.savefig(FIGURE, bbox_inches="tight")
    for key in keys:
        rows = data[key]
        spec = [r for r in rows if r["theta"] < 1 or r["theta"] > 179]
        wide = [r for r in rows if 1 <= r["theta"] <= 179]
        for name, grp in (("specular", spec), ("wide-angle", wide)):
            if not grp:
                continue
            m = np.median([r["march"] for r in grp])
            v = np.median([r["voxel"] for r in grp])
            print(f"  k_S a={key:>4}  {name:>11}: march {m:.3e}   voxel {v:.3e}   ratio {v / m:6.2f}")
    print(f"  wrote {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
