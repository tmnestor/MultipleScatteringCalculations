#!/usr/bin/env python3
"""Figure: far-field error against the grid for the three voxels on the smoothly graded sphere.

Rows: weak contrast (the discretisation alone) and full contrast; columns: k_S a = 0.5 and 1. Data:
the JSON the pilot writes (``scripts/pilot_graded_voxel_sphere.py --summary``). Grey guides show slopes
2 and 4. Colours are the validated categorical order; each arm also has its own marker.

Run:  conda run -n seismic python scripts/plot_graded_voxel_orders.py
"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "LatexPDFs" / "GradedVoxel" / "figures"
OUT = FIG / "fig_orders.pdf"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9,
        "legend.fontsize": 7,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
    }
)

ARMS = {
    "t9": ("#2a78d6", "o", "collocation, uniform field"),
    "g0": ("#eb6834", "s", "Galerkin, uniform field"),
    "g1": ("#1baf7a", "^", "Galerkin, graded first moment"),
}
GUIDE = "#8a8984"


def panel(ax, data: dict, title: str) -> None:
    for arm, (colour, marker, label) in ARMS.items():
        d = data["arms"][arm]
        n, err = np.array(d["n_sub"], float), np.array(d["error"])
        ax.loglog(n, err, marker=marker, color=colour, lw=2.0, ms=6, label=label)
    n_max = max(max(d["n_sub"]) for d in data["arms"].values())
    n0 = np.array([4.0, float(n_max)])
    for p, anchor, name in (
        (2, data["arms"]["t9"]["error"][0], "$h^2$"),
        (4, data["arms"]["g1"]["error"][0], "$h^4$"),
    ):
        y = 0.5 * anchor * (n0 / n0[0]) ** -p
        ax.loglog(n0, y, color=GUIDE, lw=1.0, ls="--")
        ax.text(n0[1] * 1.03, y[1], name, color=GUIDE, fontsize=7, va="center")
    ax.set_xlim(3.7, 19.0)
    ax.set_xticks([4, 6, 8, 12, 16])
    ax.set_xticklabels(["4", "6", "8", "12", "16"])
    ax.minorticks_off()
    ax.set_xlabel("cells across, $n$")
    ax.set_title(title)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.9), layout="constrained", sharex=True)
    for row, (kind, label) in enumerate((("born", "weak contrast"), ("full", "full contrast"))):
        for col, ka in enumerate(("0.5", "1.0")):
            data = json.loads((FIG / f"data_{kind}_ka{ka}.json").read_text())
            panel(axes[row, col], data, f"{label}, $k_Sa={float(ka):g}$")
        axes[row, 0].set_ylabel("far-field error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
