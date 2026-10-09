#!/usr/bin/env python3
"""The tau-p figure of Paper 2 (``LatexPDFs/ExactCouplingIntegrals/figures/fig_taup.pdf``), for greyscale print.

Two rows (vertical and horizontal displacement on the receiver plane z = 0) by up to five columns: the exact
sphere, the degree-one Legendre cells, the difference of the degree-one cells and (when its run exists) of
the degree-two cells, each magnified, and the section's traces. Variable density on the seismic grey scale,
white negative, mid-grey zero, black positive, so that the sign survives greyscale print; one symmetric scale
per row; time running down from the incident wave's crossing of the receiver plane, the wavelet delayed by t_d. The exact panels carry
the two-way times of the centre's P and converted S diffractions plus t_d, the depth proxy.

Run:  python scripts/plot_taup_paper.py [tag1=p1_n8] [tag2=p2_n8]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_taup_graded_sphere import ALPHA, BETA, INK, INK2, MUTED

GREY = LinearSegmentedColormap.from_list("seismic_grey", ["#ffffff", "#7f7f7f", "#000000"])
CELLS = "#8c8c8c"
CLIP = 0.5  # the density is clipped at half the peak, as seismic sections are, so weak arrivals show
ZC = 15.0  # depth of the sphere's centre below the receivers (m): 1.5 a


def load(tag, comp):
    d = np.load(ROOT / "scratch" / "taup" / f"taup_{tag}.npz")
    tau, slow = d["tau"], d["slow"]
    period = tau[1] - tau[0] + tau[-1]
    t = np.where(tau > period / 2, tau - period, tau)
    order = np.argsort(t)
    return t[order], slow, d[f"exact_{comp}"][order], d[f"cells_{comp}"][order], float(d["t_d"])


def sup(mag):
    return f"$\\times10^{{{np.log10(mag):.0f}}}$"


def main() -> int:
    tag1 = sys.argv[1] if len(sys.argv) > 1 else "p1_n8"
    tag2 = sys.argv[2] if len(sys.argv) > 2 else "p2_n8"
    have2 = (ROOT / "scratch" / "taup" / f"taup_{tag2}.npz").exists()
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})  # fmt: skip
    ncol = 5 if have2 else 4
    ratios = [1] * (ncol - 1) + [1.3]
    fig, axes = plt.subplots(2, ncol, figsize=(7.1, 5.6), constrained_layout=True, sharey=True,
                             gridspec_kw={"width_ratios": ratios})  # fmt: skip
    window = (0.0, 60.0)  # ms; the wavelet is delayed to be causal, so nothing of note precedes tau = 0
    for row, comp in enumerate(("uz", "ux")):
        tau, slow, ex, vx, t_d = load(tag1, comp)
        keep = (tau * 1e3 >= window[0]) & (tau * 1e3 <= window[1])
        t_ms, ex, vx = tau[keep] * 1e3, ex[keep], vx[keep]
        peak = np.abs(ex).max()
        ex, vx = ex / peak, vx / peak
        panels = [(ex, "exact sphere"), (vx, "cells, degree one")]
        for tag, label in ((tag1, "one"), (tag2, "two"))[: 2 if have2 else 1]:
            tau2, _, e2, v2, _ = load(tag, comp)  # each run has its own time grid (the period is 2 pi / d omega)
            diff = np.array([np.interp(t_ms, tau2 * 1e3, (v2 - e2)[:, j]) for j in range(e2.shape[1])]).T / peak
            mag = 10 ** np.ceil(np.log10(0.5 * CLIP / np.abs(diff).max()))
            panels.append((diff * mag, f"degree {label} $-$ exact\n{sup(mag)}"))
        p_skm = slow * 1e3
        extent = (p_skm[0], p_skm[-1], t_ms[-1], t_ms[0])
        name = {"uz": "$u_z$", "ux": "$u_x$"}[comp]
        for col, (data, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(data, aspect="auto", extent=extent, cmap=GREY, vmin=-CLIP, vmax=CLIP, interpolation="bilinear",
                           rasterized=True)  # fmt: skip
            ax.set_title(f"{name}: {title}", fontsize=7)
            for pb, lab in ((1e3 / ALPHA, "$1/\\alpha$"), (1e3 / BETA, "$1/\\beta$")):
                ax.axvline(pb, color=INK, lw=0.5, ls=(0, (3, 2)))
                if row == 0:
                    ax.text(pb, window[0], f" {lab}", color=INK, va="top", ha="left", fontsize=6.5,
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])  # fmt: skip
            if col == 0:  # two-way times of the centre's diffractions, the depth proxy
                for v, ls in ((ALPHA, (0, (1, 1))), (BETA, (0, (4, 1.5, 1, 1.5)))):
                    pp = np.linspace(0, 1 / v, 200) * (1 - 1e-9)
                    ax.plot(pp * 1e3, 1e3 * (t_d + ZC * (1 / ALPHA + np.sqrt(1 / v**2 - pp**2))), color=INK, lw=0.7, ls=ls,
                            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])  # fmt: skip
            if row == 1:
                ax.set_xlabel("slowness $p$ (s/km)")
        axes[row, 0].set_ylabel("intercept time $\\tau$ (ms)")
        ax = axes[row, -1]
        gap = p_skm[1] - p_skm[0]  # one trace per slowness sample
        gain = 2.5 * gap  # deflection of the section's peak, in slowness units; neighbours may overlap
        for j in range(len(slow)):
            ax.fill_betweenx(t_ms, p_skm[j], p_skm[j] + gain * ex[:, j], where=ex[:, j] > 0, color=INK, lw=0)
            ax.plot(p_skm[j] + gain * ex[:, j], t_ms, color=INK, lw=0.4)
            ax.plot(p_skm[j] + gain * vx[:, j], t_ms, color=CELLS, lw=0.45, ls=(0, (2, 1.5)))
        for pb in (1e3 / ALPHA, 1e3 / BETA):
            ax.axvline(pb, color=INK, lw=0.5, ls=(0, (3, 2)))
        ax.set_title(f"{name}: traces", fontsize=7)
        ax.set_xlim(p_skm[0] - gap, p_skm[-1] + 2 * gap)
        ax.set_ylim(t_ms[-1], t_ms[0])
        if row == 1:
            ax.set_xlabel("slowness $p$ (s/km)")
            ax.plot([], [], color=INK, lw=0.8, label="exact")
            ax.plot([], [], color=CELLS, lw=0.9, ls=(0, (2, 1.5)), label="cells, degree one")
            ax.legend(loc="lower right", frameon=True, fontsize=6, labelcolor=INK, facecolor="white", edgecolor=MUTED,
                      framealpha=1)  # fmt: skip
    cb = fig.colorbar(im, ax=axes[:, : ncol - 1], location="bottom", shrink=0.6, pad=0.01, aspect=40)
    cb.set_label("displacement / peak of the exact section (differences magnified; clipped at 0.5)", color=INK2)
    cb.outline.set_edgecolor(MUTED)
    cb.outline.set_linewidth(0.6)
    out = ROOT / "LatexPDFs" / "ExactCouplingIntegrals" / "figures" / "fig_taup.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(ROOT / "scratch" / "taup" / "fig_taup.png", dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
