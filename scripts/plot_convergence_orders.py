#!/usr/bin/env python3
"""The convergence figure of the continuum-limit paper: errors, least-squares fits and orders.

Each panel plots the relative error of a voxel scheme against its refinement on log-log axes, with the
least-squares line log(err) = c - p log(n) fitted over the asymptotic points (filled markers; open markers
are plotted but not fitted). The legend gives the order p and its standard error from the fit.

  (a) the uniform layer, normal incidence P (notebook 7, via Mathematica/ContinuumLimit_Figure.wl), with the
      second- and third-moment voxels (notebook 14, Mathematica/ContinuumLimit_SecondMoment.wl);
  (b) the uniform layer at 20 degrees: P to S and SV to S (notebooks 9 and 12, same source);
  (c) the stratified layer, normal incidence P and S, m planes per model layer (notebook 13,
      Mathematica/ContinuumLimit_Heterogeneous.wl);
  (d) the smoothly graded sphere (a core with the contrast, falling to zero across a shell as a C2
      smoothstep), voxelised on n_sub^3 grids, against its exact solution
      (scripts/pilot_graded_sphere_vs_exact.py --summary=...).

Run:  conda run -n seismic python scripts/plot_convergence_orders.py
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MMA = ROOT / "Mathematica"
FIG = ROOT / "LatexPDFs" / "ContinuumLimit" / "figures"
OUT = FIG / "fig_convergence.pdf"

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

COLOURS = {"C": "tab:blue", "G0": "tab:green", "G1": "tab:orange", "G2": "tab:brown", "G3": "tab:pink"}
NAMES = {
    "C": "collocation",
    "G0": "mean only",
    "G1": "first moment",
    "G2": "second moment",
    "G3": "third moment",
}


def fit_order(n: np.ndarray, err: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of log err = c - p log n; returns (p, standard error of p, c)."""
    x, y = np.log(n), np.log(err)
    (slope, c), cov = np.polyfit(x, y, 1, cov=True)
    return -slope, float(np.sqrt(cov[0, 0])), c


def sci(x: float) -> str:
    """One significant figure in mathtext: 0.023 -> 0.02, 3.1e-5 -> 3\\times10^{-5}."""
    if x >= 1e-2:
        return f"{x:.2f}"
    mant, ex = f"{x:.0e}".split("e")
    return f"{mant}\\times10^{{{int(ex)}}}"


def series(ax, n, err, fit_from: float, colour: str, marker: str, label: str, ls: str = "-") -> float:
    """Plot one convergence series with its regression line; returns the fitted order."""
    n, err = np.asarray(n, float), np.asarray(err, float)
    use = n >= fit_from
    p, se, c = fit_order(n[use], err[use])
    ax.plot(n[use], err[use], marker, color=colour, ms=4.5, ls="none")
    ax.plot(n[~use], err[~use], marker, color=colour, ms=4.5, ls="none", mfc="none")
    grid = np.geomspace(n.min(), n.max(), 50)
    ax.plot(grid, np.exp(c) * grid**-p, color=colour, lw=0.9, ls=ls)
    ax.plot([], [], marker, color=colour, ms=4.5, lw=0.9, ls=ls, label=f"{label}: $p={p:.3f}\\pm{sci(se)}$")
    return p


def style(
    ax,
    xlabel: str,
    title: str,
    ticks: list[int],
    legend: str = "lower left",
    top: float = 0,
    bottom: float = 0,
) -> None:
    """Log-log axes, integer refinement ticks and the legend; ``top`` > 0 raises the upper y-limit, or
    ``bottom`` > 0 lowers the lower one, to make room for the legend, with no ticks in that room."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    if top or bottom:
        lo, hi = ax.get_ylim()
        lo, hi = (bottom or lo), (top or hi)
        ax.set_ylim(lo, hi)
        ymin, ymax = ax.dataLim.y0, ax.dataLim.y1
        ax.set_yticks([t for t in ax.get_yticks() if max(lo, ymin / 3) <= t <= min(hi, 3 * ymax)])
        ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative error")
    ax.set_title(title, loc="left")
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.legend(loc=legend, frameon=False, handlelength=2.4)


def main() -> int:
    uni = json.loads((MMA / "ContinuumLimit_figure_data.json").read_text())
    het = json.loads((MMA / "ContinuumLimit_heterogeneous_convergence.json").read_text())
    high = json.loads((MMA / "ContinuumLimit_second_moment_data.json").read_text())
    sph = [json.loads((FIG / f"data_graded_sphere_ka{k}.json").read_text()) for k in ("0.5", "1.0")]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.2), constrained_layout=True)
    orders: dict[str, float] = {}

    ax = axes[0, 0]
    for sc, mk in (("C", "o"), ("G0", "s"), ("G1", "D")):
        n, e = np.array(uni["normal"][sc]).T
        orders[f"uniform normal {sc}"] = series(ax, n, e, 4, COLOURS[sc], mk, NAMES[sc])
    # the second- and third-moment voxels (notebook 14), 40-digit errors down to 1e-22
    for sc, mk in (("G2", "v"), ("G3", "p")):
        e = np.array(high["errors_R_T"][sc])[:, 0]
        orders[f"uniform normal {sc}"] = series(ax, high["n"], e, 2, COLOURS[sc], mk, NAMES[sc])
    title = "(a) uniform layer, P at normal incidence"
    style(ax, "voxel planes $n$", title, [1, 2, 4, 8, 16, 32], "lower right", bottom=1e-33)

    ax = axes[0, 1]
    for key, lab, ls, mk in (("oblique", "P$\\to$S", "-", "s"), ("incidentSV", "SV$\\to$S", "--", "^")):
        for sc in ("G0", "G1"):
            n, e = np.array(uni[key][sc]).T
            orders[f"uniform {key} {sc}"] = series(ax, n, e, 2, COLOURS[sc], mk, f"{lab}, {NAMES[sc]}", ls)
    title = "(b) uniform layer, $20^\\circ$ incidence"
    style(ax, "voxel planes $n$", title, [1, 2, 4, 8], "upper right", 1e1)

    ax = axes[1, 0]
    for wave, ls, mk in (("normal_P", "-", "o"), ("normal_S", "--", "^")):
        m = np.array(het[wave]["m"], float)
        for sc in ("C", "G0", "G1"):
            e = np.array(het[wave]["errors_R_T"][sc])[:, 0]
            lab = f"{wave[-1]}, {NAMES[sc]}"
            orders[f"stratified {wave} {sc}"] = series(ax, m, e, 4, COLOURS[sc], mk, lab, ls)
    title = "(c) stratified layer, normal incidence"
    style(ax, "voxel planes per model layer $m$", title, [1, 2, 4, 8, 16], "upper right", 1e6)

    ax = axes[1, 1]
    for s, colour, mk in zip(sph, ("tab:red", "tab:purple"), ("o", "s"), strict=True):
        lab = f"$k_Sa={s['ka_s']:g}$"
        # fitted from n_sub = 6: at 4 a voxel is as wide as the graded shell
        orders[f"graded sphere ka {s['ka_s']:g}"] = series(
            ax, s["n_sub"], s["error_vs_exact"], 6, colour, mk, lab
        )
    style(ax, "voxels across the sphere $n_{\\rm sub}$", "(d) smoothly graded sphere", [4, 6, 8, 12])

    fig.savefig(OUT)
    for k, v in orders.items():
        print(f"  {k:32s} p = {v:.4f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
