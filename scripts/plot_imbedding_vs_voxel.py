#!/usr/bin/env python3
"""Plot the accuracy of the two routes against frequency, and where they cross.

The two errors are different kinds of thing and the figure exists to make that
legible: the voxel route's is the staircase, whose phase error scales with the
cell size measured in wavelengths, so it grows as roughly (k_S a)^2; the march's
is the array's coupling, a property of the arrangement, so it is nearly flat.
A single ratio quoted at one frequency hides both facts, which is what an
earlier version of this comparison did.

Reads the measurements written by the sweep in
``scripts/gate_imbedding_vs_foldy_lax.py`` (part 6 prints them; the JSON is
produced by ``--dump``) and writes a figure next to the note that uses it.

Run:
    conda run -n seismic python scripts/plot_imbedding_vs_voxel.py <data.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in this environment
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURE = ROOT / "LatexPDFs" / "FirstOrderContrastOperator" / "imbedding_vs_voxel.pdf"


def crossing(ka: np.ndarray, ratio: np.ndarray) -> float | None:
    """Where the error ratio passes through one, by log-linear interpolation.

    Args:
        ka: Frequencies, ascending.
        ratio: Voxel error divided by march error at each.

    Returns:
        The k_S a at which the ratio is 1, or None if it never crosses.
    """
    lr = np.log(ratio)
    sign = np.sign(lr)
    idx = np.where(np.diff(sign) != 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    t = -lr[i] / (lr[i + 1] - lr[i])
    return float(np.exp(np.log(ka[i]) + t * (np.log(ka[i + 1]) - np.log(ka[i]))))


def main(path: str) -> int:
    """Draw the figure.

    Args:
        path: JSON produced by the sweep.

    Returns:
        0 on success.
    """
    rows = json.loads(Path(path).read_text())
    ka = np.array([r["ka_s"] for r in rows])
    vox = np.array([r["voxel"] for r in rows])
    mar = np.array([r["march"] for r in rows])
    order = np.argsort(ka)
    ka, vox, mar = ka[order], vox[order], mar[order]

    cross = crossing(ka, vox / mar)
    slope = float(np.polyfit(np.log(ka), np.log(vox), 1)[0])

    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )

    ax.loglog(
        ka,
        vox,
        "o-",
        color="#c0392b",
        lw=1.6,
        ms=5,
        label=rf"voxel Foldy–Lax,  $\propto (k_Sa)^{{{slope:.1f}}}$",
    )
    ax.loglog(ka, mar, "s-", color="#21618c", lw=1.6, ms=5, label="impedance march (array coupling)")
    if cross is not None:
        ax.axvline(cross, color="0.45", ls=":", lw=1.2)
        ax.annotate(
            f"crossover  $k_Sa={cross:.2f}$",
            xy=(cross, vox.min()),
            xytext=(cross * 1.18, vox.min() * 0.93),
            fontsize=9,
            color="0.25",
        )
    ax.set_ylabel("relative error of the backscattered amplitude")
    ax.set_title("Two routes to the same sphere, scored against exact Mie", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    bx.semilogx(ka, vox / mar, "o-", color="0.25", lw=1.4, ms=4)
    bx.axhline(1.0, color="0.55", lw=1.0)
    if cross is not None:
        bx.axvline(cross, color="0.45", ls=":", lw=1.2)
    bx.set_yscale("log")
    # ⚠ a IS THE SPHERE'S RADIUS, NOT THE VOXEL'S.  The voxel sits at
    # k_S a / n_sub, a sixth of this at the n_sub = 6 used throughout, and the
    # two are easy to confuse when the staircase is what is being discussed.
    bx.set_xlabel(r"$k_S a$   ($a$ = sphere radius; voxels are at $k_S a/n_{\rm sub}$)")
    bx.set_ylabel("voxel / march")
    bx.grid(True, which="both", alpha=0.25, lw=0.5)

    fig.savefig(FIGURE, bbox_inches="tight")
    print(f"  voxel error slope: {slope:.2f}")
    print(f"  march error range: {mar.min():.3e} to {mar.max():.3e}  ({mar.max() / mar.min():.1f}x)")
    print(f"  crossover: {'none in range' if cross is None else f'k_S a = {cross:.3f}'}")
    print(f"  wrote {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
